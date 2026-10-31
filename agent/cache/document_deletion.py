"""文档删除编排：active -> deleting -> deleted -> 物理删除。

Qdrant 和 PostgreSQL 没有跨库事务，所以删除不能一步做完，只能分阶段推进：

1. PostgreSQL 把文档标记为 ``deleting`` —— 检索、列表、下载立即看不到它；
2. Qdrant 按 ``document_id`` 删除向量，``wait=True`` 确认落盘；
3. 确认后把文档标记为 ``deleted``；
4. 后台任务物理删除 ``deleted`` 行（chunks/artifacts 等由外键级联清理）。

任何一步失败都停在当前状态，由 :func:`cleanup_pending_document_deletions`
继续推进。Qdrant 删除和状态迁移都是幂等的，重复执行安全。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

from agent.cache.db_utils import (
    begin_document_deletion,
    delete_document_row,
    get_document_status,
    list_documents_by_status,
    mark_document_deleted,
)
from utils.db.qdrant_store import qdrant_store

logger = logging.getLogger(__name__)

DEFAULT_CLEANUP_INTERVAL_SECONDS = 60.0
_CLEANUP_BATCH_SIZE = 50


def get_cleanup_interval_seconds() -> float:
    """后台清理间隔，``DOCUMENT_CLEANUP_INTERVAL_SECONDS`` 可覆盖；最小 5 秒。"""
    raw = str(os.getenv("DOCUMENT_CLEANUP_INTERVAL_SECONDS", "") or "").strip()
    try:
        value = float(raw) if raw else DEFAULT_CLEANUP_INTERVAL_SECONDS
    except ValueError:
        value = DEFAULT_CLEANUP_INTERVAL_SECONDS
    return max(5.0, value)


async def _delete_document_vectors(document_id: str) -> None:
    # qdrant_client 是同步客户端，放到线程里避免阻塞事件循环。
    await asyncio.to_thread(qdrant_store.delete_document, document_id)


async def request_document_deletion(document_id: str) -> Dict[str, Any]:
    """推进一次删除请求。

    返回 ``{"document_id", "status", "deleted", "error"?}``：
    ``status`` 为 None 表示文档不存在；``deleted=True`` 表示 Qdrant 已确认清理且
    PostgreSQL 已标记 deleted（物理行可能由后台任务稍后删除）。
    """
    status = await begin_document_deletion(document_id)
    if status is None:
        return {"document_id": document_id, "status": None, "deleted": False}
    if status == "deleted":
        return {"document_id": document_id, "status": status, "deleted": True}

    # status == "deleting"：可能是本次刚标记的，也可能是上次失败后的重试。
    try:
        await _delete_document_vectors(document_id)
    except Exception as exc:
        logger.warning(
            "Qdrant vector deletion failed, keeping status=deleting for background retry: document_id=%s error=%s",
            document_id,
            exc,
        )
        return {
            "document_id": document_id,
            "status": "deleting",
            "deleted": False,
            "error": str(exc),
        }

    if not await mark_document_deleted(document_id):
        # 行可能已被并发清理，或状态被其他请求改动；以库里实际状态为准。
        latest = await get_document_status(document_id)
        if latest is None:
            return {"document_id": document_id, "status": "deleted", "deleted": True}
        logger.warning(
            "Qdrant vectors deleted but status transition failed: document_id=%s status=%s",
            document_id,
            latest,
        )
        return {"document_id": document_id, "status": latest, "deleted": False}

    logger.info("Document deletion confirmed: document_id=%s", document_id)
    return {"document_id": document_id, "status": "deleted", "deleted": True}


async def cleanup_pending_document_deletions(limit: int = _CLEANUP_BATCH_SIZE) -> Dict[str, int]:
    """推进所有未完成的删除；由后台任务周期性调用，也可在进程启动时跑一次。"""
    stats = {"confirmed": 0, "purged": 0, "failed": 0}

    # deleting：Qdrant 尚未确认，或上次删向量失败 —— 重试后标记 deleted。
    for document in await list_documents_by_status("deleting", limit=limit):
        document_id = str(document["id"])
        try:
            await _delete_document_vectors(document_id)
        except Exception as exc:
            logger.warning(
                "Cleanup sweep: Qdrant vector deletion failed, retrying next round: document_id=%s error=%s",
                document_id,
                exc,
            )
            stats["failed"] += 1
            continue
        if await mark_document_deleted(document_id):
            stats["confirmed"] += 1
        else:
            # 行可能在两次查询之间被并发请求推进到 deleted，或已不存在。
            stats["failed"] += 1

    # deleted：向量已清理，物理删除 PostgreSQL 行（级联清理 chunks/artifacts）。
    for document in await list_documents_by_status("deleted", limit=limit):
        document_id = str(document["id"])
        try:
            if await delete_document_row(document_id):
                stats["purged"] += 1
        except Exception as exc:
            logger.warning(
                "Cleanup sweep: physical document deletion failed, retrying next round: document_id=%s error=%s",
                document_id,
                exc,
            )
            stats["failed"] += 1

    if any(stats.values()):
        logger.info(
            "Document cleanup sweep: confirmed=%s purged=%s failed=%s",
            stats["confirmed"],
            stats["purged"],
            stats["failed"],
        )
    return stats


async def run_document_cleanup_loop(stop_event: asyncio.Event) -> None:
    """API 进程内常驻的清理循环；``stop_event`` 置位后退出。"""
    while not stop_event.is_set():
        try:
            await cleanup_pending_document_deletions()
        except Exception:
            # 单轮失败不能终止后台任务，否则残留的 deleting 行会永远没人处理。
            logger.exception("Document cleanup sweep failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=get_cleanup_interval_seconds())
        except asyncio.TimeoutError:
            continue
