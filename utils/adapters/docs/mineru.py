"""MinerU 4 命令行提取器。

MinerU 的依赖与主项目冲突（要求 ``openai<3``），因此不放进本项目环境，而是
通过子进程调用独立环境里的 ``mineru-kit``。这里只负责：

1. 以 ``--format zip`` 解析 PDF；
2. 读取 zip 里的 ``markdown.md``；
3. 把 ``images/`` 下的图片单独落盘，并把正文里的图片引用替换为 ``[IMG]``；
4. 从 ``middle_json.json`` 里取图注、页数、表格数。

对上层仍然返回 ``(markdown, metadata)`` 所需的原始材料，具体 metadata 由
``readpdf.PDFExtractor`` 组装，保证与 Docling 分支格式一致。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# zip 里的图片引用，例如 ![](images/page_3_image_2.jpg)
_MARKDOWN_IMAGE = re.compile(
    r"!\[[^\]]*\]\(\s*<?(?P<src>[^)\s>]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)"
)
# 表格体里可能出现的 <img src="images/...">
_HTML_IMAGE = re.compile(
    r"<img\b[^>]*\bsrc\s*=\s*[\"'](?P<src>[^\"']+)[\"'][^>]*>", re.IGNORECASE
)
_IMAGE_NAME_PAGE = re.compile(r"page_(\d+)_")
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass
class MinerUResult:
    """一次 MinerU 解析的结果，字段与 Docling 分支的统计口径对齐。"""

    markdown: str
    assets: List[Dict[str, Any]] = field(default_factory=list)
    pages: int = 0
    texts: int = 0
    tables: int = 0
    raw_image_count: int = 0


def _resolve_binary(binary: Optional[str] = None) -> str:
    """按配置 → 环境变量 → 项目内虚拟环境 → PATH 的顺序找 mineru-kit。"""
    candidates = [
        binary,
        os.environ.get("MINERU_BIN"),
        str(_PROJECT_ROOT / ".venv-mineru" / "Scripts" / "mineru-kit.exe"),
        str(_PROJECT_ROOT / ".venv-mineru" / "bin" / "mineru-kit"),
        shutil.which("mineru-kit"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError(
        "找不到 mineru-kit：请设置 MINERU_BIN 或 PDFExtractionConfig.mineru_bin"
    )


def _iter_blocks(
    node: Any, owner: Optional[Dict[str, Any]] = None
) -> Iterable[tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """遍历 middle_json 的块树，同时带上最近的 image/table/chart 父块。"""
    if isinstance(node, list):
        for item in node:
            yield from _iter_blocks(item, owner)
        return
    if not isinstance(node, dict):
        return

    if node.get("type") in {"image", "table", "chart"}:
        owner = node
    yield node, owner

    for key in ("content", "blocks", "children"):
        child = node.get(key)
        if isinstance(child, (list, dict)):
            yield from _iter_blocks(child, owner)


def _block_text(block: Dict[str, Any]) -> str:
    """取块内文本；caption 的 content 是嵌套的 text 块列表。"""
    content = block.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            _block_text(item) for item in content if isinstance(item, dict)
        ).strip()
    return ""


_CAPTION_LABEL = re.compile(
    r"^(?:fig(?:ure)?|table|chart|algorithm|图|表|算法)\s*\d+", re.IGNORECASE
)


def _pick_caption(parts: List[str]) -> str:
    """一个视觉块可能同时挂 ``(b)`` 这类子图标签和正式图注，优先取后者。"""
    parts = [part for part in parts if part]
    if not parts:
        return ""
    labeled = [part for part in parts if _CAPTION_LABEL.match(part)]
    return max(labeled or parts, key=len)


def _collect_captions(middle_json: Dict[str, Any]) -> Dict[str, str]:
    """把 middle_json 里的图片路径映射到所属父块的图注。"""
    captions: Dict[str, str] = {}
    for page in middle_json.get("pages") or []:
        for node, owner in _iter_blocks(page):
            image_path = node.get("image_path")
            if not image_path:
                continue
            caption = ""
            if owner is not None and owner is not node:
                parts = [
                    _block_text(item)
                    for item, _ in _iter_blocks(owner)
                    if item.get("type") in {"image_caption", "chart_caption"}
                ]
                caption = _pick_caption(parts)
            captions.setdefault(str(image_path), caption)
    return captions


def _count_blocks(middle_json: Dict[str, Any]) -> tuple[int, int, int]:
    """返回 (pages, texts, tables)。"""
    pages = middle_json.get("pages") or []
    texts = 0
    tables = 0
    for page in pages:
        for node, _ in _iter_blocks(page):
            block_type = node.get("type")
            if block_type == "table":
                tables += 1
            elif block_type in {"text", "paragraph_title", "doc_title"}:
                if _block_text(node):
                    texts += 1
    return len(pages), texts, tables


def _caption_from_markdown(lines: List[str], line_index: int, window: int = 3) -> str:
    """middle_json 没有图注时，从图片附近的正文行猜一个。"""
    pattern = re.compile(r"^(?:fig(?:ure)?\.?|表|图)\s*\d+\b", re.IGNORECASE)
    for offset in range(-window, window + 1):
        if offset == 0:
            continue
        index = line_index + offset
        if not 0 <= index < len(lines):
            continue
        text = re.sub(r"^[#>*\-\s]+", "", lines[index]).strip()
        if pattern.match(text):
            return text[:500]
    return ""


def _image_page(name: str) -> Optional[int]:
    match = _IMAGE_NAME_PAGE.search(name)
    return int(match.group(1)) + 1 if match else None


class MinerUExtractor:
    """通过子进程调用 ``mineru-kit parse`` 的提取器。"""

    def __init__(
        self,
        binary: Optional[str] = None,
        tier: str = "standard",
        timeout: int = 3600,
        home: Optional[str] = None,
        model_source: str = "modelscope",
    ) -> None:
        self.binary = _resolve_binary(binary)
        self.tier = tier or "standard"
        self.timeout = int(timeout or 3600)
        self.home = Path(home) if home else _PROJECT_ROOT / ".tmp" / "mineru"
        self.model_source = model_source or "modelscope"

    def _environment(self) -> Dict[str, str]:
        """MinerU 的缓存默认落在用户目录，Windows 上常常不可写，这里改到项目内。"""
        env = os.environ.copy()
        env.setdefault("MINERU_HOME", str(self.home))
        env.setdefault("MODELSCOPE_HOME", str(_PROJECT_ROOT / ".tmp" / "modelscope_home"))
        env.setdefault("MODELSCOPE_CACHE", str(_PROJECT_ROOT / ".tmp" / "modelscope_cache"))
        env.setdefault("MINERU_MODEL_SOURCE", self.model_source)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def parse(
        self,
        pdf_path: str | Path,
        asset_dir: str | Path,
        *,
        include_images: bool = True,
        max_images: Optional[int] = None,
    ) -> MinerUResult:
        """解析 PDF，返回去掉 base64 的 markdown 与已落盘的图片列表。"""
        pdf_path = Path(pdf_path).resolve()
        asset_dir = Path(asset_dir)
        digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]

        with tempfile.TemporaryDirectory(prefix="mineru_parse_") as temp_dir:
            zip_path = Path(temp_dir) / f"{pdf_path.stem}.zip"
            self._run(pdf_path, zip_path)
            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())
                markdown_name = "markdown.md"
                if markdown_name not in names:
                    raise RuntimeError(f"MinerU 输出缺少 {markdown_name}: {sorted(names)[:10]}")
                raw_markdown = archive.read(markdown_name).decode("utf-8", errors="replace")
                middle_json: Dict[str, Any] = {}
                if "middle_json.json" in names:
                    middle_json = json.loads(
                        archive.read("middle_json.json").decode("utf-8", errors="replace")
                    )
                pages, texts, tables = _count_blocks(middle_json)
                captions = _collect_captions(middle_json)
                markdown, assets, image_count = _export_images(
                    archive,
                    raw_markdown,
                    captions,
                    asset_dir=asset_dir,
                    digest=digest,
                    include_images=include_images,
                    max_images=max_images,
                )

        return MinerUResult(
            markdown=markdown,
            assets=assets,
            pages=pages,
            texts=texts,
            tables=tables,
            raw_image_count=image_count,
        )

    def _run(self, pdf_path: Path, zip_path: Path) -> None:
        cmd = [
            self.binary,
            "parse",
            str(pdf_path),
            "-o",
            str(zip_path),
            "-f",
            "zip",
            "--tier",
            self.tier,
        ]
        logger.info("Running MinerU: %s", " ".join(cmd))
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout,
            env=self._environment(),
            # docvortex 会在 cwd 下建临时渲染目录，放到临时目录里一起清理
            cwd=str(zip_path.parent),
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(
                f"MinerU 解析失败（exit {completed.returncode}）：{stderr[-1500:]}"
            )

def _export_images(
    archive: zipfile.ZipFile,
    raw_markdown: str,
    captions: Dict[str, str],
    *,
    asset_dir: Path,
    digest: str,
    include_images: bool,
    max_images: Optional[int],
) -> tuple[str, List[Dict[str, Any]], int]:
    """把 zip 内图片写到 asset_dir，并把正文引用替换成 [IMG]。"""
    archive_names = set(archive.namelist())
    seen: Dict[str, int] = {}
    assets: List[Dict[str, Any]] = []
    if include_images:
        asset_dir.mkdir(parents=True, exist_ok=True)

    def register(src: str, fallback_caption: str = "") -> str:
        """返回替换后的占位符；图片按首次出现顺序编号。"""
        if not src or src.startswith(("http://", "https://", "data:")):
            return "[IMG]"
        name = src.replace("\\", "/").lstrip("./")
        if name not in archive_names or name in seen:
            return "[IMG]"
        extension = Path(name).suffix.lower()
        if extension not in _IMAGE_EXTENSIONS:
            return "[IMG]"

        index = len(seen)
        seen[name] = index
        if include_images and (max_images is None or index < max_images):
            target = asset_dir / f"{digest}-figure-{index + 1}{extension}"
            target.write_bytes(archive.read(name))
            assets.append({
                "index": index,
                "path": str(target),
                "caption": captions.get(name, "") or fallback_caption,
                "page": _image_page(name),
                "asset_kind": "mineru_image",
            })
        return "[IMG]"

    lines = raw_markdown.splitlines()
    for line_index, line in enumerate(lines):
        if not (_MARKDOWN_IMAGE.search(line) or _HTML_IMAGE.search(line)):
            continue
        line_caption = _caption_from_markdown(lines, line_index)
        line = _MARKDOWN_IMAGE.sub(
            lambda m: register(m.group("src"), line_caption), line
        )
        line = _HTML_IMAGE.sub(
            lambda m: register(m.group("src"), line_caption), line
        )
        lines[line_index] = line

    markdown = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")
    # 去掉 MinerU 输出的页脚 HTML 噪声
    markdown = re.sub(
        r"<small>\s*<span[^>]*docvortex-page-footnote[^>]*>.*?</span>\s*</small>",
        "",
        markdown,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # 算法块外层只是排版 div，正文保留
    markdown = re.sub(r"</?div\b[^>]*>", "", markdown)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip("\n")
    return markdown, assets, len(seen)
