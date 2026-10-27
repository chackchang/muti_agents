"""Persistent ingestion task submission and worker-side PDF execution helpers."""

import asyncio
import base64
import binascii
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

from fastapi import HTTPException

from .ingestion_tasks_db import create_ingestion_task, update_ingestion_task_status
from .rabbitmq_producer import publish_ingestion_task

DOCUMENT_UPLOAD_MAX_BYTES = int(os.getenv("DOCUMENT_UPLOAD_MAX_BYTES", str(30 * 1024 * 1024)))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 上传与 OpenAlex 导入都落在 documents/ 下。逻辑ID由「相对该根目录的路径」推导，
# 这里必须和建目录时用同一个根，否则重试会被当成一篇新文档。
DOCUMENTS_ROOT = Path(os.getenv("INGEST_DOCUMENTS_ROOT", "documents"))

# 子进程入口：PDF_ingest 对外暴露的是 async 函数而不是 CLI 模块，
# 由引导脚本 import 后 asyncio.run，最后把结果打成一行机器可读的 JSON。
# 引导脚本写成文件再执行，避免多行字符串走 Windows 命令行引号解析。
_INGEST_BOOTSTRAP = r"""
import asyncio
import json
import sys

project_root, documents_folder, fast = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
sys.path.insert(0, project_root)

from agent.config import IngestionConfig
from utils.ingestion.PDF_ingest import document_ingest_api

config = IngestionConfig(
    documents_folder=documents_folder,
    include_images=not fast,
    include_tables=not fast,
    verbose=True,
)
results = asyncio.run(document_ingest_api(config)) or []
print("INGEST_RESULT=" + json.dumps([r.model_dump() for r in results], ensure_ascii=False), flush=True)
"""

_INGEST_RESULT_MARKER = "INGEST_RESULT="


def _safe_filename(seed: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (seed or "").strip())
    cleaned = cleaned.strip("._-")
    return cleaned[:120] or f"upload_{uuid.uuid4().hex[:8]}"


def _make_upload_paths(filename_raw: str) -> tuple[str, Path, Path]:
    safe_filename = _safe_filename(filename_raw)
    if not safe_filename.lower().endswith(".pdf"):
        safe_filename = f"{safe_filename}.pdf"
    stem = _safe_filename(Path(safe_filename).stem)
    import_dir = Path("documents") / "ui_uploads" / f"{uuid.uuid4().hex[:8]}_{stem}"
    import_dir.mkdir(parents=True, exist_ok=True)
    target_path = import_dir / safe_filename
    return safe_filename, import_dir, target_path


def _validate_and_decode_upload_payload(payload: Dict[str, Any]) -> tuple[str, bytes, bool]:
    filename_raw = str(payload.get("filename") or "").strip()
    content_base64 = str(payload.get("content_base64") or "").strip()
    fast = bool(payload.get("fast", False))
    if not filename_raw:
        raise HTTPException(status_code=400, detail="filename is required")
    if not content_base64:
        raise HTTPException(status_code=400, detail="content_base64 is required")
    if not filename_raw.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported")
    try:
        data = base64.b64decode(content_base64, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 content") from exc
    if not data:
        raise HTTPException(status_code=400, detail="Empty file content")
    if len(data) > DOCUMENT_UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {DOCUMENT_UPLOAD_MAX_BYTES} bytes)")
    return filename_raw, data, fast


async def _enqueue_file(file_path: Path, *, fast: bool) -> Dict[str, Any]:
    task = await create_ingestion_task(
        task_id=uuid.uuid4().hex,
        document_id=None,
        file_path=str(file_path),
        fast=fast,
        status="queued",
    )
    try:
        await publish_ingestion_task(
            task_id=task["task_id"],
            document_id=None,
            file_path=task["file_path"],
            fast=fast,
        )
    except Exception as exc:
        # 任务刚由 create_ingestion_task 落库为 queued；若期间已被删除则不再改写
        updated = await update_ingestion_task_status(
            task_id=str(task["task_id"]),
            status="failed",
            error_message=str(exc),
            expected_status="queued",
        )
        raise HTTPException(
            status_code=503,
            detail={"message": "Failed to publish ingestion task to RabbitMQ", "task": updated, "error": str(exc)},
        ) from exc
    return task


async def submit_async_ingestion_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    filename_raw, data, fast = _validate_and_decode_upload_payload(payload)
    _, _, target_path = _make_upload_paths(filename_raw)
    target_path.write_bytes(data)
    return await _enqueue_file(target_path, fast=fast)


async def submit_async_ingestion_tasks(payloads: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Create one durable task per PDF; the worker executes them in queue order."""
    if not payloads:
        raise HTTPException(status_code=400, detail="At least one PDF is required")
    if len(payloads) > 100:
        raise HTTPException(status_code=400, detail="A batch can contain at most 100 PDFs")
    return [await submit_async_ingestion_task(payload) for payload in payloads]


def _document_source(path: Path) -> str:
    """逻辑ID的输入：相对 documents/ 的稳定路径，与临时目录、随机前缀都无关。

    用 ``os.path.relpath`` 而不是 ``as_posix``，保证与 pipeline 内部的
    ``os.path.relpath(file_path, documents_folder)`` 逐字符一致（Windows 下是反斜杠）。
    """
    try:
        return os.path.relpath(path.resolve(), DOCUMENTS_ROOT.resolve())
    except ValueError:
        # 跨盘符时退回文件名，同样与物理位置无关
        return path.name


def _build_ingestion_command(script_path: Path, documents_folder: Path, *, fast: bool = False) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(script_path),
        str(PROJECT_ROOT),
        str(documents_folder),
        "1" if fast else "0",
    ]


def _parse_ingest_result(lines: list[str]) -> Dict[str, Any]:
    """从子进程 stdout 里取出 document_ingest_api 的返回值。"""
    for line in reversed(lines):
        if _INGEST_RESULT_MARKER not in line:
            continue
        try:
            payload = json.loads(line.split(_INGEST_RESULT_MARKER, 1)[1])
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, list) or not payload:
            return {}
        first = payload[0]
        if not isinstance(first, dict):
            return {}
        return {
            "document_id": first.get("document_id"),
            "document_title": first.get("title"),
            "chunks_created": first.get("chunks_created"),
            "document_status": first.get("status"),
        }
    return {}


async def ingest_saved_pdf_file(
    file_path: str,
    *,
    fast: bool = False,
    should_interrupt: Optional[callable] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """Execute exactly one queued PDF in an isolated temporary directory.

    暂存目录里按「相对 documents/ 的原路径」镜像摆放文件，再把它当 documents_folder
    传给 document_ingest_api：这样目录里只有一个 PDF，而 pipeline 推导出的 source
    与原路径入库完全一致，逻辑ID/指纹在重试之间保持稳定。
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    source_rel = _document_source(path)
    with tempfile.TemporaryDirectory(prefix="single_ingest_") as tmp_dir:
        run_root = Path(tmp_dir)
        # 扫描根与引导脚本分开放：根目录里只有这 PDF，脚本不会被当成文档
        documents_folder = run_root / "documents"
        script_path = run_root / "_ingest_bootstrap.py"
        script_path.write_text(_INGEST_BOOTSTRAP, encoding="utf-8")
        staged_path = documents_folder / source_rel
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, staged_path)
        # 子进程执行 PDF_ingest 暴露的函数式接口
        process = await asyncio.create_subprocess_exec(
            *_build_ingestion_command(script_path, documents_folder, fast=fast),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        async def read_stream(stream: asyncio.StreamReader, target: list[str]) -> None:
            while raw := await stream.readline():
                line = raw.decode("utf-8", errors="replace")
                target.append(line)
                if progress_callback:
                    await progress_callback(line)

        readers = [asyncio.create_task(read_stream(process.stdout, stdout_lines)), asyncio.create_task(read_stream(process.stderr, stderr_lines))]
        try:
            while True:
                if should_interrupt and await should_interrupt():
                    process.terminate()
                    await process.wait()
                    raise IngestionInterrupted("Task was paused or deleted")
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.5)#防止子进程无限等待
                    break
                except asyncio.TimeoutError:
                    continue
            await asyncio.gather(*readers)
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()
            await asyncio.gather(*readers, return_exceptions=True)
        stdout, stderr = "".join(stdout_lines), "".join(stderr_lines)
        if process.returncode != 0:
            raise RuntimeError((stderr or stdout or "Ingestion process failed").strip()[-2000:])
        return {
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "document_source": source_rel,
            **_parse_ingest_result(stdout_lines),
        }


class IngestionInterrupted(Exception):
    """Raised when a durable task is paused or deleted during processing."""


def _download_binary_file(url: str, target_path: Path) -> None:
    request = Request(url, headers={"User-Agent": "agentic-rag-openalex/1.0"})
    with urlopen(request, timeout=30) as response:
        content_type = str(response.headers.get("Content-Type") or "").lower()
        data = response.read()
    # OpenAlex may expose a publisher landing page even when no OA PDF exists.
    # Reject HTML and non-PDF payloads before they enter the ingestion queue.
    if not data.startswith(b"%PDF-") and "application/pdf" not in content_type:
        raise ValueError(f"URL did not return a PDF (content-type: {content_type or 'unknown'})")
    target_path.write_bytes(data)


async def add_openalex_file_to_kb(file_url: str, title: str, *, fast: bool = False) -> Dict[str, Any]:
    """Download an OpenAlex PDF and submit it to the same durable queue as local uploads."""
    filename = _safe_filename(title)
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"
    import_dir = Path("documents") / "openalex_imports" / f"{uuid.uuid4().hex[:8]}_{_safe_filename(title)}"
    import_dir.mkdir(parents=True, exist_ok=True)
    target_path = import_dir / filename
    try:
        await asyncio.to_thread(_download_binary_file, file_url, target_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAlex PDF download failed: {exc}") from exc
    task = await _enqueue_file(target_path, fast=fast)
    return task
