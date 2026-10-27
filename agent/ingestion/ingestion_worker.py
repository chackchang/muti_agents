import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from agent.infrastructure.app_config import get_ingestion_max_retries, get_ingestion_queue_name, get_rabbitmq_url
from agent.cache.db_utils import close_database, initialize_database
from .ingestion_jobs import IngestionInterrupted, ingest_saved_pdf_file
from .ingestion_quota import is_quota_exhausted_error, quota_pause_message
from .ingestion_tasks_db import get_ingestion_task, list_ingestion_tasks, update_ingestion_task_status
from .rabbitmq_producer import (
    build_ingestion_dlq_message,
    build_ingestion_invalid_message,
    publish_ingestion_dlq_message,
    publish_ingestion_task,
)

logger = logging.getLogger(__name__)

# worker 可以接管的起始状态：刚入队、上次尝试失败待重试、或进程崩溃后遗留的 processing。
# paused / paused_quota / deleted / done 一律不接管——那是应用侧刚做出的决定。
_CLAIMABLE_STATUSES = ("queued", "failed", "processing")

# 还没被任何 worker 认领过的状态。认领之前的写入（目前只有「消息缺 file_path，放弃本次尝试」）
# 只能用这个集合：显示 processing 的任务要么是别人正在跑（踩不得），要么是崩溃残留
# （启动恢复会把它退回 queued），两种情况都不该由一条坏消息来改。
_UNCLAIMED_STATUSES = ("queued", "failed")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _short_error(exc: Exception, max_len: int = 500) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    if len(text) <= max_len:
        return text
    return text[:max_len]


async def _finish_transition(
    task_id: str,
    previous: Dict[str, Any],
    action: str,
    *,
    error_message: str | None = None,
    **fields: Any,
) -> Dict[str, Any]:
    """写一次「worker 结束本次尝试」的状态跃迁，并带上 processing 前置条件。

    只有任务仍在 processing 时才生效；如果应用侧已经把它暂停/删除/完成，这里不会覆盖，
    也不再触发重投递或 DLQ —— 否则会出现「暂停后被 worker 改回 processing」这类互相踩踏。
    """
    updated = await update_ingestion_task_status(
        task_id=task_id,
        expected_status="processing",
        error_message=error_message,
        **fields,
    )
    if updated is None:
        latest = await get_ingestion_task(task_id)
        logger.warning(
            "Task state changed concurrently, skip %s: task_id=%s status=%s",
            action, task_id, str((latest or {}).get("status") or "missing"),
        )
        return {"action": "skip_conflict", "task": latest or previous, "error_message": error_message}
    return {"action": action, "task": updated, "error_message": error_message}


def _decode_raw_message(body: bytes, max_len: int = 2000) -> str:
    text = body.decode("utf-8", errors="replace")
    if len(text) <= max_len:
        return text
    return text[:max_len]


async def _nack_message(message: Any, requeue: bool = True) -> None:
    nack = getattr(message, "nack", None)
    if callable(nack):
        await nack(requeue=requeue)
        return

    reject = getattr(message, "reject", None)
    if callable(reject):
        await reject(requeue=requeue)
        return

    raise RuntimeError("Queue message does not support nack/reject")


def parse_ingestion_message(body: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON message: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Message payload must be a JSON object")

    task_id = str(payload.get("task_id") or "").strip()
    file_path = str(payload.get("file_path") or "").strip()
    document_id_raw = payload.get("document_id")
    document_id = str(document_id_raw).strip() if document_id_raw else None

    if not task_id:
        raise ValueError("Missing required field: task_id")

    return {
        "task_id": task_id,
        "file_path": file_path,
        "document_id": document_id,
        "fast": bool(payload.get("fast", False)),
    }


async def handle_ingestion_message(payload: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(payload.get("task_id") or "").strip()
    file_path = str(payload.get("file_path") or "").strip()
    message_document_id = payload.get("document_id")
    message_document_id = str(message_document_id).strip() if message_document_id else None
    max_retries = get_ingestion_max_retries()
    if not task_id:
        raise ValueError("Missing required field: task_id")

    current = await get_ingestion_task(task_id)
    if not current:
        logger.warning("Task not found, skip duplicated/orphan message: task_id=%s", task_id)
        return {"action": "skip_not_found", "task_id": task_id}

    current_status = str(current.get("status") or "")
    current_retry = int(current.get("retry_count") or 0)
    current_document_id = str(current.get("document_id") or "").strip() or None
    if current_status == "done":
        return {"action": "skip_done", "task_id": task_id}
    if current_status == "paused_quota":
        return {"action": "skip_paused_quota", "task_id": task_id}
    if current_status in {"paused", "deleted"}:
        return {"action": f"skip_{current_status}", "task_id": task_id}
    if current_status == "failed" and current_retry >= max_retries:
        return {"action": "skip_failed_max", "task_id": task_id}

    queued_tasks = [task for task in await list_ingestion_tasks() if task.get("status") == "queued"] if current.get("queue_order") is not None else []
    if queued_tasks and str(queued_tasks[0].get("task_id") or "") != task_id:
        return {"action": "defer_until_turn", "task": current}

    if not file_path:
        err = "Missing required field: file_path"
        new_retry = current_retry + 1
        will_retry = new_retry < max_retries
        updated = await update_ingestion_task_status(
            task_id=task_id,
            status="queued" if will_retry else "failed",
            error_message=err,
            retry_count=new_retry,
            finished_at=None if will_retry else _now_utc(),
            expected_status=_UNCLAIMED_STATUSES,
        )
        if updated is None:
            latest = await get_ingestion_task(task_id)
            return {"action": "skip_conflict", "task": latest or current, "error_message": err}
        return {"action": "retry" if will_retry else "failed_max", "task": updated, "error_message": err}

    # 抢占任务：只有任务仍处于可执行状态才继续，避免和 pause/delete 或重复投递撞车。
    claimed = await update_ingestion_task_status(
        task_id=task_id,
        status="processing",
        error_message=None,
        started_at=_now_utc(),
        finished_at=None,
        progress_percent=max(1, int(current.get("progress_percent") or 0)),
        progress_stage="正在准备论文文件",
        expected_status=_CLAIMABLE_STATUSES,
    )
    if claimed is None:
        latest = await get_ingestion_task(task_id)
        logger.info(
            "Task is no longer runnable, skip before ingestion: task_id=%s status=%s",
            task_id, str((latest or {}).get("status") or "missing"),
        )
        return {"action": "skip_conflict", "task": latest or current}

    try:
        last_progress = 1

        async def should_interrupt() -> bool:
            latest = await get_ingestion_task(task_id)
            return not latest or str(latest.get("status") or "") in {"paused", "deleted"}

        async def report_progress(line: str) -> None:
            nonlocal last_progress
            text = line.lower()
            explicit_stages = {
                "ingest_stage=vision": (32, "Analyzing figures with caption context"),
                "ingest_stage=structure": (52, "Building sections, algorithms, and tables"),
                "ingest_stage=embed": (68, "Generating GE-M3 embeddings"),
                "ingest_stage=persist": (85, "Persisting paper and artifact evidence"),
                "ingest_stage=graph": (94, "Refreshing paper knowledge graph"),
            }
            for marker, value in explicit_stages.items():
                if marker in text:
                    # 进度只在任务仍为 processing 时写入：被暂停/删除后不能把状态顶回去
                    await update_ingestion_task_status(
                        task_id=task_id, status="processing", progress_percent=max(last_progress, value[0]),
                        progress_stage=value[1], expected_status="processing",
                    )
                    last_progress = max(last_progress, value[0])
                    return
            percent, stage = last_progress, "正在处理论文"
            if "ingest_stage=extract" in text or "extracting content from" in text:
                percent, stage = 10, "正在解析 PDF 版面与正文"
            elif "total chunks created" in text:
                percent, stage = 45, "正在构建章节、算法、表格与图片证据"
            elif "generated embeddings for" in text:
                percent, stage = 65, "正在生成检索向量"
            elif "saved document to postgresql" in text:
                percent, stage = 92, "正在写入知识库与知识图谱"
            if percent > last_progress or stage != "正在处理论文":
                last_progress = max(last_progress, percent)
                await update_ingestion_task_status(
                    task_id=task_id, status="processing", progress_percent=last_progress,
                    progress_stage=stage, expected_status="processing",
                )

        result = await ingest_saved_pdf_file(
            file_path,
            fast=bool(current.get("fast", payload.get("fast", False))),
            should_interrupt=should_interrupt,
            progress_callback=report_progress,
        )
        latest = await get_ingestion_task(task_id)
        if not latest or str(latest.get("status") or "") in {"paused", "deleted"}:
            return {"action": "interrupted", "task": latest or current}
        final_document_id = str(result.get("document_id") or current_document_id or message_document_id or "").strip() or None
        if not final_document_id:
            raise RuntimeError("Ingestion finished without a persisted document ID")
        return await _finish_transition(
            task_id,
            current,
            "done",
            error_message=None,
            status="done",
            document_id=final_document_id,
            finished_at=_now_utc(),
            progress_percent=100,
            progress_stage="入库完成",
        )
    except IngestionInterrupted:
        latest = await get_ingestion_task(task_id)
        return {"action": "interrupted", "task": latest or current}
    except Exception as exc:
        err = _short_error(exc)
        if is_quota_exhausted_error(exc):
            return await _finish_transition(
                task_id,
                current,
                "paused_quota",
                error_message=quota_pause_message(exc),
                status="paused_quota",
                finished_at=_now_utc(),
            )
        new_retry = current_retry + 1
        if new_retry < max_retries:
            return await _finish_transition(
                task_id,
                current,
                "retry",
                error_message=err,
                status="queued",
                retry_count=new_retry,
            )
        return await _finish_transition(
            task_id,
            current,
            "failed_max",
            error_message=err,
            status="failed",
            retry_count=new_retry,
            finished_at=_now_utc(),
        )


async def process_rabbitmq_message(body: bytes) -> Dict[str, Any]:
    payload = parse_ingestion_message(body)
    result = await handle_ingestion_message(payload)
    action = str(result.get("action") or "")
    if action == "retry":
        task = result.get("task") or {}
        await publish_ingestion_task(
            task_id=str(task.get("task_id") or payload.get("task_id") or ""),
            document_id=(str(task.get("document_id") or "").strip() or None),
            file_path=str(task.get("file_path") or payload.get("file_path") or ""),
            fast=bool(payload.get("fast", False)),
        )
        return result
    if action == "defer_until_turn":
        task = result.get("task") or {}
        await asyncio.sleep(0.2)
        await publish_ingestion_task(
            task_id=str(task.get("task_id") or payload.get("task_id") or ""),
            document_id=(str(task.get("document_id") or "").strip() or None),
            file_path=str(task.get("file_path") or payload.get("file_path") or ""),
            fast=bool(task.get("fast", payload.get("fast", False))),
        )
        return result
    if action == "failed_max":
        task = result.get("task") or {}
        failed_at = _now_utc().isoformat()
        dlq_payload = build_ingestion_dlq_message(
            task_id=str(task.get("task_id") or payload.get("task_id") or ""),
            document_id=(str(task.get("document_id") or "").strip() or None),
            file_path=str(task.get("file_path") or payload.get("file_path") or ""),
            retry_count=int(task.get("retry_count") or 0),
            error_message=str(result.get("error_message") or task.get("error_message") or "ingestion failed"),
            failed_at=failed_at,
        )
        await publish_ingestion_dlq_message(dlq_payload)
        return result
    return result


async def process_queue_message_and_ack(message: Any) -> None:
    try:
        await process_rabbitmq_message(message.body)
        await message.ack()
        return
    except ValueError as exc:
        logger.warning("Invalid ingestion message, publishing to DLQ: %s", exc)
        dlq_payload = build_ingestion_invalid_message(
            error_message=str(exc),
            failed_at=_now_utc().isoformat(),
            raw_message=_decode_raw_message(message.body),
        )
        try:
            await publish_ingestion_dlq_message(dlq_payload)
        except Exception as publish_exc:
            logger.exception("Failed to publish invalid ingestion message to DLQ: %s", publish_exc)
            await _nack_message(message, requeue=True)
            return
        await message.ack()
        return
    except Exception as exc:
        logger.exception("Failed to process ingestion message: %s", exc)
        await _nack_message(message, requeue=True)


async def consume_ingestion_tasks() -> None:
    rabbitmq_url = get_rabbitmq_url()
    if not rabbitmq_url:
        raise RuntimeError("RabbitMQ URL is not configured")

    try:
        import aio_pika
    except ImportError as exc:
        raise RuntimeError("aio-pika is not installed") from exc

    queue_name = get_ingestion_queue_name()
    await initialize_database()
    # A process restart abandons the AMQP delivery, but the persisted task can
    # remain in `processing`. Recover it before republishing, so it cannot be
    # stranded indefinitely or require a manual database correction.
    tasks = await list_ingestion_tasks()
    for task in tasks:
        if task.get("status") == "processing" and not task.get("document_id"):
            # 列表是启动时读的，真正动手前再校验一次：期间可能已被暂停/删除/完成
            reset = await update_ingestion_task_status(
                task_id=str(task["task_id"]),
                status="queued",
                progress_percent=0,
                progress_stage="Waiting to resume after worker restart",
                error_message=None,
                finished_at=None,
                expected_status="processing",
            )
            if reset is None:
                logger.info(
                    "Skip restart recovery, task state changed: task_id=%s", str(task["task_id"])
                )
    pending_tasks = [task for task in await list_ingestion_tasks() if task.get("status") == "queued"]
    for task in pending_tasks:
        await publish_ingestion_task(
            task_id=str(task["task_id"]),
            document_id=(str(task.get("document_id") or "").strip() or None),
            file_path=str(task["file_path"]),
            fast=bool(task.get("fast", False)),
        )
    if pending_tasks:
        logger.info("Requeued %s pending ingestion task(s) on worker startup", len(pending_tasks))
    connection = await aio_pika.connect_robust(rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)
    queue = await channel.declare_queue(queue_name, durable=True)
    logger.info("Ingestion worker is consuming queue: %s", queue_name)

    try:
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                await process_queue_message_and_ack(message)
    finally:
        await connection.close()
        await close_database()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(consume_ingestion_tasks())


if __name__ == "__main__":
    main()
