
import os
import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Sequence
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
import logging

import asyncpg
from asyncpg.pool import Pool
from dotenv import load_dotenv

from utils.db.qdrant_store import qdrant_store
from utils.embedding.embedding_runtime import aembed_texts

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class DatabasePool:
    """管理 PostgreSQL 连接池。"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        初始化数据库连接池。
        
        Args:
            database_url: PostgreSQL 连接 URL
        """
        USER = os.getenv("DB_USER", "postgres")
        PASSWORD = os.getenv("DB_PASSWORD", "postgres")
        HOST = os.getenv("DB_HOST", "postgres")
        PORT = os.getenv("DB_PORT", 5432)
        DBNAME = os.getenv("DB_NAME", "postgres")

        self.database_url = database_url  or f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DBNAME}"
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.pool: Optional[Pool] = None
    
    async def initialize(self):
        """创建连接池。"""
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            logger.info("Database connection pool initialized")
    
    async def close(self):
        """关闭连接池。"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def acquire(self):
        """从连接池获取一个连接。"""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            yield connection

# 全局数据库连接池实例
db_pool = DatabasePool()

async def initialize_database():
    """初始化 PostgreSQL 连接池，并确保 Qdrant 集合存在。"""
    await db_pool.initialize()
    await asyncio.to_thread(qdrant_store.initialize)

async def close_database():
    """关闭数据库连接池。"""
    await db_pool.close()


# --------------------------------------------------------------------------
# 分块向量（Qdrant）
# --------------------------------------------------------------------------
async def upsert_chunk_vectors(points: Sequence[Dict[str, Any]]) -> int:
    """把分块向量写入 Qdrant；``points`` 由 ``(dense, sparse)`` + payload 组成。"""
    return await asyncio.to_thread(qdrant_store.upsert_chunks, points)


async def delete_document_vectors(document_id: str) -> None:
    """删除一篇论文在 Qdrant 中的全部分块向量。"""
    await asyncio.to_thread(qdrant_store.delete_document, document_id)


async def clear_vector_store() -> None:
    """清空 Qdrant 集合，用于 --clean / --reset-kb。"""
    await asyncio.to_thread(qdrant_store.clear)


async def get_document_profile_embedding(document_id: str) -> Optional[List[float]]:
    """文档级稠密 profile（非 artifact 分块的均值），供论文关系图使用。"""
    return await asyncio.to_thread(qdrant_store.document_profile, document_id)

async def execute_init_sql(sql_path: str):
    async with db_pool.acquire() as conn:
        # schema.sql only contains idempotent DDL and is also our migration path.
        # Do not skip it merely because the original documents table already exists.
        with open(sql_path, "r", encoding="utf-8") as file:
            sql = file.read()
        await conn.execute(sql)
        logger.info("Schema ensured successfully.")

# 会话管理函数
async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    timeout_minutes: int = 10080
) -> str:
    """
    创建一个新会话。
    
    Args:
        user_id: 可选的用户标识
        metadata: 可选的会话元数据
        timeout_minutes: 会话超时时长（分钟）
    
    Returns:
        会话 ID
    """
    async with db_pool.acquire() as conn:
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)
        session_metadata = {
            "title": "New Chat",
            "title_generated": False,
            "last_message_at": None,
            **(metadata or {})
        }
        
        result = await conn.fetchrow(
            """
            INSERT INTO sessions (user_id, metadata, expires_at)
            VALUES ($1, $2, $3)
            RETURNING id::text
            """,
            user_id,
            json.dumps(session_metadata),
            expires_at
        )
        
        return result["id"]


def _generate_session_title(first_user_message: Optional[str], max_len: int = 22) -> str:
    """Generate a short session title from first user message."""
    if not first_user_message:
        return "New Chat"

    normalized = " ".join(first_user_message.split()).strip()
    if not normalized:
        return "New Chat"

    if len(normalized) <= max_len:
        return normalized

    return normalized[:max_len].rstrip() + "..."

async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    根据 ID 获取会话。
    
    Args:
        session_id: 会话 UUID
    
    Returns:
        会话数据；如果未找到或已过期则返回 None
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                user_id,
                metadata,
                created_at,
                updated_at,
                expires_at
            FROM sessions
            WHERE id = $1::uuid
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id
        )
        
        if result:
            return {
                "id": result["id"],
                "user_id": result["user_id"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat(),
                "expires_at": result["expires_at"].isoformat() if result["expires_at"] else None
            }
        
        return None


async def update_session_metadata(
    session_id: str,
    metadata_updates: Dict[str, Any]
) -> bool:
    """Merge updates into session metadata."""
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE sessions
            SET metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb
            WHERE id = $1::uuid
              AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id,
            json.dumps(metadata_updates)
        )
        return result != "UPDATE 0"


async def get_session_memory_snapshot(session_id: str) -> Dict[str, Any]:
    """Read the structured session snapshot; messages remain the immutable source log."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT version, covered_message_count, summary, updated_at
               FROM session_memory_snapshots WHERE session_id=$1::uuid""",
            session_id,
        )
        if not row:
            return {"version": 0, "covered_message_count": 0, "summary": {}, "updated_at": None}
        return {
            "version": int(row["version"]),
            "covered_message_count": int(row["covered_message_count"]),
            "summary": json.loads(row["summary"]) if isinstance(row["summary"], str) else dict(row["summary"] or {}),
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }


async def save_session_memory_snapshot(
    session_id: str,
    covered_message_count: int,
    summary: Dict[str, Any],
) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO session_memory_snapshots(session_id, version, covered_message_count, summary)
               VALUES ($1::uuid, 1, $2, $3::jsonb)
               ON CONFLICT (session_id) DO UPDATE SET
                   version=session_memory_snapshots.version + 1,
                   covered_message_count=EXCLUDED.covered_message_count,
                   summary=EXCLUDED.summary,
                   updated_at=CURRENT_TIMESTAMP
               RETURNING version, covered_message_count, summary, updated_at""",
            session_id,
            max(0, int(covered_message_count)),
            json.dumps(summary, ensure_ascii=False),
        )
        return {
            "version": int(row["version"]),
            "covered_message_count": int(row["covered_message_count"]),
            "summary": json.loads(row["summary"]) if isinstance(row["summary"], str) else dict(row["summary"] or {}),
            "updated_at": row["updated_at"].isoformat(),
        }


async def save_session_memory_archive(
    session_id: str,
    source_snapshot_version: int,
    covered_message_count: int,
    summary: Dict[str, Any],
) -> bool:
    """Persist the previous structured snapshot as a historical stage summary."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO session_memory_archives(
                   session_id, source_snapshot_version, covered_message_count, summary
               ) VALUES ($1::uuid, $2, $3, $4::jsonb)
               ON CONFLICT (session_id, source_snapshot_version) DO NOTHING
               RETURNING id""",
            session_id,
            max(0, int(source_snapshot_version)),
            max(0, int(covered_message_count)),
            json.dumps(summary, ensure_ascii=False),
        )
        return row is not None


async def get_session_memory_archives(
    session_id: str,
    limit: int = 3,
    query_text: str = "",
) -> List[Dict[str, Any]]:
    """Read a small set of historical summaries, ranked by SQL text relevance then recency."""
    safe_limit = max(1, min(int(limit or 3), 10))
    normalized_query = " ".join(str(query_text or "").split()).strip()
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT source_snapshot_version, covered_message_count, summary, created_at
               FROM session_memory_archives
               WHERE session_id=$1::uuid
               ORDER BY
                   CASE WHEN $3 <> '' THEN
                       ts_rank_cd(
                           to_tsvector('simple', COALESCE(summary::text, '')),
                           plainto_tsquery('simple', $3)
                       )
                   ELSE 0 END DESC,
                   source_snapshot_version DESC
               LIMIT $2""",
            session_id,
            safe_limit,
            normalized_query,
        )
    return [
        {
            "source_snapshot_version": int(row["source_snapshot_version"]),
            "covered_message_count": int(row["covered_message_count"]),
            "summary": json.loads(row["summary"]) if isinstance(row["summary"], str) else dict(row["summary"] or {}),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


async def search_session_memory_messages(
    session_id: str,
    query_text: str,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    """Find a few exact historical messages using PostgreSQL full-text search."""
    normalized_query = " ".join(str(query_text or "").split()).strip()
    if not normalized_query:
        return []
    safe_limit = max(1, min(int(limit or 4), 10))
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, role, content, created_at,
                      ts_rank_cd(
                          to_tsvector('simple', COALESCE(content, '')),
                          plainto_tsquery('simple', $2)
                      ) AS rank
               FROM messages
               WHERE session_id=$1::uuid
                 AND role IN ('user', 'assistant')
                 AND COALESCE(metadata->>'memory_eligible', 'false') = 'true'
                 AND to_tsvector('simple', COALESCE(content, '')) @@ plainto_tsquery('simple', $2)
               ORDER BY rank DESC, created_at DESC
               LIMIT $3""",
            session_id,
            normalized_query,
            safe_limit,
        )
    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


async def save_conversation_query_rewrite(
    *,
    session_id: str,
    user_message_id: str,
    original_query: str,
    rewritten_query: str,
    rewrite_reason: str = "",
    model_used: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist the raw-to-retrieval query mapping for one completed user turn."""
    normalized_original = " ".join(str(original_query or "").split()).strip()[:8000]
    normalized_rewritten = " ".join(str(rewritten_query or "").split()).strip()[:8000]
    if not str(session_id or "").strip() or not str(user_message_id or "").strip():
        raise ValueError("session_id and user_message_id are required for query rewrite audit")
    if not normalized_original or not normalized_rewritten:
        raise ValueError("original_query and rewritten_query are required for query rewrite audit")
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO conversation_query_rewrites(
                   session_id, user_message_id, original_query, rewritten_query,
                   rewrite_reason, model_used, metadata
               ) VALUES ($1::uuid, $2::uuid, $3, $4, $5, $6, $7::jsonb)
               ON CONFLICT (user_message_id) DO UPDATE SET
                   original_query=EXCLUDED.original_query,
                   rewritten_query=EXCLUDED.rewritten_query,
                   rewrite_reason=EXCLUDED.rewrite_reason,
                   model_used=EXCLUDED.model_used,
                   metadata=EXCLUDED.metadata,
                   updated_at=CURRENT_TIMESTAMP
               RETURNING id::text, session_id::text, user_message_id::text,
                         original_query, rewritten_query, rewrite_reason,
                         model_used, metadata, created_at, updated_at""",
            session_id,
            user_message_id,
            normalized_original,
            normalized_rewritten,
            " ".join(str(rewrite_reason or "").split()).strip()[:160],
            bool(model_used),
            json.dumps(metadata or {}, ensure_ascii=False),
        )
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "user_message_id": row["user_message_id"],
        "original_query": row["original_query"],
        "rewritten_query": row["rewritten_query"],
        "rewrite_reason": row["rewrite_reason"],
        "model_used": bool(row["model_used"]),
        "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {}),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


def _normalize_user_memory_id(user_id: Optional[str]) -> str:
    return " ".join(str(user_id or "").split()).strip()[:256]


async def get_user_memory_profile(user_id: Optional[str]) -> Dict[str, Any]:
    """Read the cross-session profile for one user."""
    normalized_user_id = _normalize_user_memory_id(user_id)
    if not normalized_user_id:
        return {"version": 0, "profile": {}, "updated_at": None}
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT version, profile, updated_at
               FROM user_memory_profiles WHERE user_id=$1""",
            normalized_user_id,
        )
    if not row:
        return {"version": 0, "profile": {}, "updated_at": None}
    return {
        "version": int(row["version"] or 0),
        "profile": json.loads(row["profile"]) if isinstance(row["profile"], str) else dict(row["profile"] or {}),
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


async def save_user_memory_profile(user_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Replace a user profile, primarily for correction and migration."""
    normalized_user_id = _normalize_user_memory_id(user_id)
    if not normalized_user_id:
        raise ValueError("user_id is required for user memory")
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                """INSERT INTO user_memory_profiles(user_id, version, profile)
                   VALUES ($1, 1, $2::jsonb)
                   ON CONFLICT (user_id) DO UPDATE SET
                       version=user_memory_profiles.version + 1,
                       profile=EXCLUDED.profile,
                       updated_at=CURRENT_TIMESTAMP
                   RETURNING version, profile, updated_at""",
                normalized_user_id,
                json.dumps(profile, ensure_ascii=False),
            )
            # PUT is a full replacement. Keep the audit rows, but prevent an
            # older item from overriding the newly replaced profile.
            await conn.execute(
                """UPDATE user_memory_items
                   SET status='superseded', superseded_at=CURRENT_TIMESTAMP,
                       updated_at=CURRENT_TIMESTAMP
                   WHERE user_id=$1 AND status='active'""",
                normalized_user_id,
            )
    return {
        "version": int(row["version"] or 0),
        "profile": json.loads(row["profile"]) if isinstance(row["profile"], str) else dict(row["profile"] or {}),
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


async def list_user_memory_items(user_id: Optional[str], limit: int = 32) -> List[Dict[str, Any]]:
    """List active user memory items; deleted/superseded values stay auditable."""
    normalized_user_id = _normalize_user_memory_id(user_id)
    if not normalized_user_id:
        return []
    safe_limit = max(1, min(int(limit or 32), 100))
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT id::text, kind, memory_key, value, status,
                      source_session_id::text, source_message_id::text,
                      metadata, created_at, updated_at
               FROM user_memory_items
               WHERE user_id=$1 AND status='active'
               ORDER BY updated_at DESC, created_at DESC
               LIMIT $2""",
            normalized_user_id,
            safe_limit,
        )
    return [
        {
            "id": row["id"],
            "kind": row["kind"],
            "memory_key": row["memory_key"],
            "value": row["value"],
            "status": row["status"],
            "source_session_id": row["source_session_id"],
            "source_message_id": row["source_message_id"],
            "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {}),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
        }
        for row in rows
    ]


async def add_user_memory_item(
    user_id: str,
    kind: str,
    memory_key: str,
    value: str,
    source_session_id: Optional[str] = None,
    source_message_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Upsert one durable memory item with source provenance."""
    normalized_user_id = _normalize_user_memory_id(user_id)
    normalized_kind = " ".join(str(kind or "").split()).strip().lower()
    normalized_key = " ".join(str(memory_key or "").split()).strip()[:64]
    normalized_value = " ".join(str(value or "").split()).strip()[:240]
    allowed_keys = {
        "preference": {"language", "response_style", "formatting"},
        "continuity": {"research_field"},
    }
    if not normalized_user_id or normalized_kind not in allowed_keys:
        raise ValueError("user_id and a valid memory kind are required")
    if not normalized_key or not normalized_value:
        raise ValueError("memory_key and value are required")
    if normalized_key not in allowed_keys[normalized_kind]:
        raise ValueError("memory_key is not allowed for this memory kind")
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """UPDATE user_memory_items
                   SET status='superseded', superseded_at=CURRENT_TIMESTAMP,
                       updated_at=CURRENT_TIMESTAMP
                   WHERE user_id=$1 AND kind=$2 AND memory_key=$3 AND status='active'""",
                normalized_user_id,
                normalized_kind,
                normalized_key,
            )
            row = await conn.fetchrow(
                """INSERT INTO user_memory_items(
                       user_id, kind, memory_key, value, source_session_id,
                       source_message_id, metadata
                   ) VALUES ($1, $2, $3, $4, $5::uuid, $6::uuid, $7::jsonb)
                   RETURNING id::text, kind, memory_key, value, status,
                             source_session_id::text, source_message_id::text,
                             metadata, created_at, updated_at""",
                normalized_user_id,
                normalized_kind,
                normalized_key,
                normalized_value,
                source_session_id,
                source_message_id,
                json.dumps(metadata or {"origin": "user_managed"}, ensure_ascii=False),
            )
    return {
        "id": row["id"],
        "kind": row["kind"],
        "memory_key": row["memory_key"],
        "value": row["value"],
        "status": row["status"],
        "source_session_id": row["source_session_id"],
        "source_message_id": row["source_message_id"],
        "metadata": json.loads(row["metadata"]) if isinstance(row["metadata"], str) else dict(row["metadata"] or {}),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


async def delete_user_memory_item(user_id: str, item_id: str) -> bool:
    """Soft-delete an item so the audit trail is not lost."""
    normalized_user_id = _normalize_user_memory_id(user_id)
    if not normalized_user_id or not str(item_id or "").strip():
        return False
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """UPDATE user_memory_items
               SET status='deleted', superseded_at=CURRENT_TIMESTAMP,
                   updated_at=CURRENT_TIMESTAMP
               WHERE id=$1::uuid AND user_id=$2 AND status='active'""",
            str(item_id).strip(),
            normalized_user_id,
        )
    return result != "UPDATE 0"


async def refresh_session_metadata(session_id: str) -> None:
    """Refresh title and last_message_at metadata for a session."""
    async with db_pool.acquire() as conn:
        current = await conn.fetchrow(
            """
            SELECT metadata
            FROM sessions
            WHERE id = $1::uuid
              AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """,
            session_id
        )
        if not current:
            return

        metadata = json.loads(current["metadata"]) if current["metadata"] else {}
        updates: Dict[str, Any] = {}

        last_message_at = await conn.fetchval(
            """
            SELECT MAX(created_at)
            FROM messages
            WHERE session_id = $1::uuid
            """,
            session_id
        )
        if last_message_at:
            updates["last_message_at"] = last_message_at.isoformat()

        if not metadata.get("title_generated"):
            first_user_message = await conn.fetchval(
                """
                SELECT content
                FROM messages
                WHERE session_id = $1::uuid
                  AND role = 'user'
                  AND btrim(content) <> ''
                ORDER BY created_at ASC
                LIMIT 1
                """,
                session_id
            )
            updates["title"] = _generate_session_title(first_user_message)
            updates["title_generated"] = True

        if updates:
            await conn.execute(
                """
                UPDATE sessions
                SET metadata = COALESCE(metadata, '{}'::jsonb) || $2::jsonb
                WHERE id = $1::uuid
                """,
                session_id,
                json.dumps(updates)
            )


async def list_recent_sessions(limit: int = 20, days: int = 7) -> List[Dict[str, Any]]:
    """List recent visible sessions with lightweight stats."""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                s.id::text AS session_id,
                s.metadata,
                s.created_at,
                s.updated_at,
                s.expires_at,
                COALESCE(mc.message_count, 0) AS message_count,
                lm.content AS last_message_preview
            FROM sessions s
            LEFT JOIN LATERAL (
                SELECT COUNT(*)::int AS message_count
                FROM messages m
                WHERE m.session_id = s.id
            ) mc ON TRUE
            LEFT JOIN LATERAL (
                SELECT m.content
                FROM messages m
                WHERE m.session_id = s.id
                ORDER BY m.created_at DESC
                LIMIT 1
            ) lm ON TRUE
            WHERE (s.expires_at IS NULL OR s.expires_at > CURRENT_TIMESTAMP)
              AND s.updated_at >= (CURRENT_TIMESTAMP - make_interval(days => $2::int))
            ORDER BY s.updated_at DESC
            LIMIT $1
            """,
            limit,
            days
        )

        sessions: List[Dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            title = metadata.get("title") or "New Chat"
            preview = row["last_message_preview"] or ""
            preview = " ".join(preview.split()).strip()
            if len(preview) > 80:
                preview = preview[:80].rstrip() + "..."

            sessions.append(
                {
                    "session_id": row["session_id"],
                    "title": title,
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                    "expires_at": row["expires_at"].isoformat() if row["expires_at"] else None,
                    "message_count": row["message_count"],
                    "last_message_preview": preview,
                    "recoverable": True
                }
            )

        return sessions


async def delete_session(session_id: str) -> bool:
    """Delete a session and its messages. Returns True if session existed."""
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                DELETE FROM messages
                WHERE session_id = $1::uuid
                """,
                session_id,
            )
            result = await conn.execute(
                """
                DELETE FROM sessions
                WHERE id = $1::uuid
                """,
                session_id,
            )
    return str(result).endswith(" 1")

# 消息管理函数
async def add_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    向会话中添加一条消息。
    
    Args:
        session_id: 会话 UUID
        role: 消息角色（user/assistant/system）
        content: 消息内容
        metadata: 可选的消息元数据
    
    Returns:
        消息 ID
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            INSERT INTO messages (session_id, role, content, metadata)
            VALUES ($1::uuid, $2, $3, $4)
            RETURNING id::text
            """,
            session_id,
            role,
            content,
            json.dumps(metadata or {})
        )
        
        return result["id"]


async def set_message_memory_eligible(message_id: str) -> bool:
    """Promote a persisted final turn into the memory white list after successful completion."""
    async with db_pool.acquire() as conn:
        result = await conn.execute(
            """UPDATE messages
               SET metadata = COALESCE(metadata, '{}'::jsonb) || '{"memory_eligible": true}'::jsonb
               WHERE id=$1::uuid""",
            message_id,
        )
        return result != "UPDATE 0"

async def get_session_messages(
    session_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    获取某个会话的消息。
    
    Args:
        session_id: 会话 UUID
        limit: 返回消息的最大数量
    
    Returns:
        按创建时间排序的消息列表
    """
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                id::text,
                role,
                content,
                metadata,
                created_at
            FROM messages
            WHERE session_id = $1::uuid
            ORDER BY created_at
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = await conn.fetch(query, session_id)
        
        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat()
            }
            for row in results
        ]

# 文档管理函数
async def get_document(document_id: str) -> Optional[Dict[str, Any]]:
    """
    根据 ID 获取文档。
    
    Args:
        document_id: 文档 UUID
    
    Returns:
        文档数据；如果未找到则返回 None
    """
    async with db_pool.acquire() as conn:
        result = await conn.fetchrow(
            """
            SELECT 
                id::text,
                title,
                source,
                content,
                metadata,
                created_at,
                updated_at
            FROM documents
            WHERE id = $1::uuid AND status = 'active'
            """,
            document_id
        )
        
        if result:
            return {
                "id": result["id"],
                "title": result["title"],
                "source": result["source"],
                "content": result["content"],
                "metadata": json.loads(result["metadata"]),
                "created_at": result["created_at"].isoformat(),
                "updated_at": result["updated_at"].isoformat()
            }
        
        return None


async def get_document_pdf(document_id: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT pdf_blob, pdf_media_type FROM documents WHERE id = $1::uuid AND status = 'active'",
            document_id,
        )
        if not row or row["pdf_blob"] is None:
            return None
        return {"content": bytes(row["pdf_blob"]), "media_type": str(row["pdf_media_type"] or "application/pdf")}


async def get_document_pdf_bytes(document_id: str) -> Optional[bytes]:
    pdf = await get_document_pdf(document_id)
    return pdf["content"] if pdf else None


async def delete_document(document_id: str) -> bool:
    """[legacy] 直接物理删除文档并尽力清理向量。

    正式删除流程请用 ``agent.cache.document_deletion.request_document_deletion``：
    它按 active -> deleting -> deleted -> 物理删除 推进，避免 Qdrant 与
    PostgreSQL 状态不一致。这个函数只留给离线脚本。
    """
    async with db_pool.acquire() as conn:
        result = await conn.execute("DELETE FROM documents WHERE id = $1::uuid", document_id)
    if result == "DELETE 0":
        return False
    try:
        await delete_document_vectors(document_id)
    except Exception as exc:
        # PostgreSQL 已经删除；向量残留不应让删除接口失败，但要留下可排查日志。
        logger.warning("Failed to delete Qdrant vectors for %s: %s", document_id, exc)
    return True


async def get_document_translation(document_id: str, target_language: str, source_sha256: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id::text, document_id::text, target_language, content_markdown, model, updated_at
               FROM document_translations WHERE document_id=$1::uuid AND target_language=$2 AND source_sha256=$3""",
            document_id, target_language, source_sha256,
        )
        if not row:
            return None
        return {"id": row["id"], "document_id": row["document_id"], "target_language": row["target_language"], "content_markdown": row["content_markdown"], "model": row["model"], "updated_at": row["updated_at"].isoformat()}


async def save_document_translation(*, document_id: str, target_language: str, source_sha256: str, content_markdown: str, model: str) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO document_translations (document_id,target_language,source_sha256,content_markdown,model)
               VALUES ($1::uuid,$2,$3,$4,$5)
               ON CONFLICT (document_id,target_language,source_sha256) DO UPDATE
               SET content_markdown=EXCLUDED.content_markdown, model=EXCLUDED.model, updated_at=CURRENT_TIMESTAMP
               RETURNING id::text, document_id::text, target_language, content_markdown, model, updated_at""",
            document_id, target_language, source_sha256, content_markdown, model,
        )
        return {"id": row["id"], "document_id": row["document_id"], "target_language": row["target_language"], "content_markdown": row["content_markdown"], "model": row["model"], "updated_at": row["updated_at"].isoformat()}


async def get_translation_profile(document_id: str, target_language: str, source_sha256: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT profile_json, model, updated_at FROM document_translation_profiles
               WHERE document_id=$1::uuid AND target_language=$2 AND source_sha256=$3""",
            document_id, target_language, source_sha256,
        )
    if not row:
        return None
    profile = row["profile_json"] if isinstance(row["profile_json"], dict) else json.loads(row["profile_json"] or "{}")
    return {"profile": profile, "model": row["model"], "updated_at": row["updated_at"].isoformat()}


async def save_translation_profile(*, document_id: str, target_language: str, source_sha256: str, profile: Dict[str, Any], model: str) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO document_translation_profiles(document_id,target_language,source_sha256,profile_json,model)
               VALUES($1::uuid,$2,$3,$4::jsonb,$5)
               ON CONFLICT (document_id,target_language,source_sha256) DO UPDATE
               SET profile_json=EXCLUDED.profile_json, model=EXCLUDED.model, updated_at=CURRENT_TIMESTAMP
               RETURNING profile_json, model, updated_at""",
            document_id, target_language, source_sha256, json.dumps(profile, ensure_ascii=False), model,
        )
    return {"profile": row["profile_json"], "model": row["model"], "updated_at": row["updated_at"].isoformat()}


async def get_selection_translation(document_id: str, target_language: str, source_sha256: str, selection_sha256: str) -> Optional[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """SELECT id::text, source_text, translated_text, model, updated_at FROM document_selection_translations
               WHERE document_id=$1::uuid AND target_language=$2 AND source_sha256=$3 AND selection_sha256=$4""",
            document_id, target_language, source_sha256, selection_sha256,
        )
    return dict(row) if row else None


async def save_selection_translation(*, document_id: str, target_language: str, source_sha256: str, selection_sha256: str, source_text: str, translated_text: str, context_before: str, context_after: str, model: str) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO document_selection_translations(document_id,target_language,source_sha256,selection_sha256,source_text,translated_text,context_before,context_after,model)
               VALUES($1::uuid,$2,$3,$4,$5,$6,$7,$8,$9)
               ON CONFLICT (document_id,target_language,source_sha256,selection_sha256) DO UPDATE
               SET translated_text=EXCLUDED.translated_text, context_before=EXCLUDED.context_before, context_after=EXCLUDED.context_after, model=EXCLUDED.model, updated_at=CURRENT_TIMESTAMP
               RETURNING id::text, source_text, translated_text, model, updated_at""",
            document_id, target_language, source_sha256, selection_sha256, source_text, translated_text, context_before, context_after, model,
        )
    return dict(row)


async def list_document_annotations(document_id: str) -> List[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT id::text, page_number, page_x, page_y, quote, note, color, created_at, updated_at FROM document_annotations WHERE document_id=$1::uuid ORDER BY created_at", document_id)
        return [{"id": r["id"], "page_number": r["page_number"], "page_x": float(r["page_x"]), "page_y": float(r["page_y"]), "quote": r["quote"], "note": r["note"], "color": r["color"], "created_at": r["created_at"].isoformat(), "updated_at": r["updated_at"].isoformat()} for r in rows]


async def create_document_annotation(document_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        page_x = min(1.0, max(0.0, float(payload.get("page_x", 0.5))))
        page_y = min(1.0, max(0.0, float(payload.get("page_y", 0.5))))
        row = await conn.fetchrow("""INSERT INTO document_annotations(document_id,page_number,page_x,page_y,quote,note,color) VALUES($1::uuid,$2,$3,$4,$5,$6,$7) RETURNING id::text,page_number,page_x,page_y,quote,note,color,created_at,updated_at""", document_id, payload.get("page_number"), page_x, page_y, str(payload.get("quote") or ""), str(payload.get("note") or "").strip(), str(payload.get("color") or "yellow"))
        return {"id": row["id"], "page_number": row["page_number"], "page_x": float(row["page_x"]), "page_y": float(row["page_y"]), "quote": row["quote"], "note": row["note"], "color": row["color"], "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat()}


async def update_document_annotation_position(document_id: str, annotation_id: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    page_x = min(1.0, max(0.0, float(payload.get("page_x", 0.5))))
    page_y = min(1.0, max(0.0, float(payload.get("page_y", 0.5))))
    note = payload.get("note")
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""UPDATE document_annotations SET page_number=COALESCE($3,page_number), page_x=$4, page_y=$5,
                                   note=COALESCE($6,note), updated_at=CURRENT_TIMESTAMP
                                   WHERE id=$1::uuid AND document_id=$2::uuid
                                   RETURNING id::text,page_number,page_x,page_y,quote,note,color,created_at,updated_at""", annotation_id, document_id, payload.get("page_number"), page_x, page_y, str(note).strip() if note is not None else None)
        if not row:
            return None
        return {"id": row["id"], "page_number": row["page_number"], "page_x": float(row["page_x"]), "page_y": float(row["page_y"]), "quote": row["quote"], "note": row["note"], "color": row["color"], "created_at": row["created_at"].isoformat(), "updated_at": row["updated_at"].isoformat()}


async def delete_document_annotation(document_id: str, annotation_id: str) -> bool:
    async with db_pool.acquire() as conn:
        result = await conn.execute("DELETE FROM document_annotations WHERE id=$1::uuid AND document_id=$2::uuid", annotation_id, document_id)
        return result != "DELETE 0"


async def list_documents(
    limit: int = 100,
    offset: int = 0,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    列出文档，并可选择按条件过滤。
    
    Args:
        limit: 返回文档的最大数量
        offset: 需要跳过的文档数量
        metadata_filter: 可选的元数据过滤条件
    
    Returns:
        文档列表
    """
    async with db_pool.acquire() as conn:
        query = """
            SELECT 
                d.id::text,
                d.title,
                d.source,
                d.metadata,
                d.created_at,
                d.updated_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
        """
        
        params = []
        # 删除中的文档（deleting/deleted）对外不可见：列表只展示 active。
        conditions = ["d.status = 'active'"]

        if metadata_filter:
            conditions.append(f"d.metadata @> ${len(params) + 1}::jsonb")
            params.append(json.dumps(metadata_filter))
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
            GROUP BY d.id, d.title, d.source, d.metadata, d.created_at, d.updated_at
            ORDER BY d.created_at DESC
            LIMIT $%d OFFSET $%d
        """ % (len(params) + 1, len(params) + 2)
        
        params.extend([limit, offset])
        
        results = await conn.fetch(query, *params)
        
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "source": row["source"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "chunk_count": row["chunk_count"]
            }
            for row in results
        ]


async def list_document_sections(document_ids: List[str]) -> List[Dict[str, Any]]:
    """Return the persisted section catalog for selected documents without loading chunks."""
    normalized_ids = list(
        dict.fromkeys(str(value or "").strip() for value in (document_ids or []) if str(value or "").strip())
    )
    if not normalized_ids:
        return []
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.document_id::text AS document_id,
                d.title AS document_title,
                COALESCE(c.metadata->>'section_path_text', c.metadata->>'section_title', '') AS section_path_text,
                COALESCE(c.metadata->>'section_title', '') AS section_title,
                MIN(COALESCE((c.metadata->>'section_level')::int, 99)) AS section_level,
                MIN(COALESCE((c.metadata->>'section_start_line')::int, 2147483647)) AS section_start_line,
                COUNT(*)::int AS chunk_count
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.document_id = ANY($1::uuid[])
              AND d.status = 'active'
              AND COALESCE(c.metadata->>'section_path_text', c.metadata->>'section_title', '') <> ''
            GROUP BY c.document_id, d.title, COALESCE(c.metadata->>'section_path_text', c.metadata->>'section_title', ''),
                     COALESCE(c.metadata->>'section_title', '')
            ORDER BY c.document_id, MIN(COALESCE((c.metadata->>'section_start_line')::int, 2147483647)),
                     COALESCE(c.metadata->>'section_path_text', c.metadata->>'section_title', '')
            """,
            normalized_ids,
        )
    return [
        {
            "document_id": str(row["document_id"]),
            "document_title": str(row["document_title"] or ""),
            "section_path_text": str(row["section_path_text"] or ""),
            "section_title": str(row["section_title"] or ""),
            "section_level": int(row["section_level"] or 99),
            "section_start_line": int(row["section_start_line"] or 0),
            "chunk_count": int(row["chunk_count"] or 0),
        }
        for row in rows
    ]

# 向量搜索函数（Qdrant）
_ACTIVE_FILTER_OVERFETCH = 3
_ACTIVE_FILTER_MAX_FETCH = 200


def _overfetch_limit(limit: int, cap: int = _ACTIVE_FILTER_MAX_FETCH) -> int:
    """多取一些候选，抵消删除中文档被过滤后的数量损失。"""
    safe_limit = max(1, int(limit or 10))
    return max(safe_limit, min(safe_limit * _ACTIVE_FILTER_OVERFETCH, cap))


async def _filter_active_document_hits(
    hits: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """只保留 PostgreSQL 中 status='active' 的文档命中。

    Qdrant 删向量和 PostgreSQL 改状态不是原子的：文档可能已经进入
    deleting/deleted，但向量尚未删完。检索结果必须回查 PG 状态，
    否则会返回正在删除的论文。
    """
    document_ids = sorted(
        {str(hit.get("document_id") or "") for hit in hits if hit.get("document_id")}
    )
    if not document_ids:
        return hits[:limit]
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id::text FROM documents WHERE id = ANY($1::uuid[]) AND status = 'active'",
            document_ids,
        )
    active_ids = {str(row["id"]) for row in rows}
    return [
        hit for hit in hits if str(hit.get("document_id") or "") in active_ids
    ][:limit]


async def _resolve_query_sparse(
    query_text: str,
    sparse: Optional[Dict[int, float]],
) -> Optional[Dict[int, float]]:
    """没有显式 sparse 时，用同一个 GE-M3 模型对查询编码。"""
    if sparse:
        return sparse
    if not str(query_text or "").strip():
        return None
    _, generated = (await aembed_texts([query_text]))[0]
    return generated or None


async def vector_search(
    embedding: List[float],
    limit: int = 10,
    document_ids: Optional[List[str]] = None,
    sparse: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    """
    执行向量相似度搜索（Qdrant 稠密，可选稀疏）。

    Args:
        embedding: 查询对应的稠密向量
        limit: 返回结果的最大数量
        document_ids: 只在这些文档内检索
        sparse: GE-M3 稀疏向量（token id -> weight）

    Returns:
        按相似度排序的匹配分块列表（最相关的在前）
    """
    safe_limit = max(1, int(limit or 10))
    results = await asyncio.to_thread(
        qdrant_store.search,
        embedding,
        sparse,
        limit=_overfetch_limit(safe_limit),
        document_ids=document_ids,
    )
    hits = [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "content": item["content"],
            "similarity": item["score"],
            "metadata": item["metadata"],
            "document_title": item["document_title"],
            "document_source": item["document_source"],
        }
        for item in results
    ]
    return await _filter_active_document_hits(hits, safe_limit)


async def hybrid_search(
    embedding: List[float],
    query_text: str = "",
    limit: int = 10,
    text_weight: float = 0.3,
    document_ids: Optional[List[str]] = None,
    sparse: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    """
    执行混合搜索（稠密 + 稀疏，Qdrant RRF 融合）。

    ``text_weight`` 仅用于兼容旧签名：RRF 不暴露权重，关键词匹配由
    GE-M3 的稀疏通道承担。未传 ``sparse`` 时用 ``query_text`` 现场编码。
    """
    del text_weight
    safe_limit = max(1, int(limit or 10))
    resolved_sparse = await _resolve_query_sparse(query_text, sparse)
    results = await asyncio.to_thread(
        qdrant_store.search,
        embedding,
        resolved_sparse,
        limit=_overfetch_limit(safe_limit),
        document_ids=document_ids,
    )
    hits = [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "content": item["content"],
            "combined_score": item["score"],
            "metadata": item["metadata"],
            "document_title": item["document_title"],
            "document_source": item["document_source"],
        }
        for item in results
    ]
    return await _filter_active_document_hits(hits, safe_limit)


async def section_search(
    query_text: str,
    limit: int = 10,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    section_query: Optional[str] = None,
    exact_section_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search chunks within section-aware metadata (section_title/section_path_text).
    This does not change vector/hybrid behavior and is a standalone retrieval path.
    """
    async with db_pool.acquire() as conn:
        safe_limit = max(1, min(int(limit or 10), 50))
        query_value = str(query_text or "").strip()
        section_value = str(section_query or "").strip()
        exact_section_value = str(exact_section_path or "").strip()
        normalized_section_value = re.sub(r"[\s\W_]+", "", section_value.lower(), flags=re.UNICODE)
        normalized_exact_section_value = re.sub(r"[\s\W_]+", "", exact_section_value.lower(), flags=re.UNICODE)

        conditions: List[str] = []
        params: List[Any] = []
        idx = 1

        # 删除中的文档不参与检索。
        conditions.append("d.status = 'active'")

        # Prefer section-aware chunks for this tool.
        conditions.append(
            "("
            "COALESCE(c.metadata->>'section_title','') <> '' "
            "OR COALESCE(c.metadata->>'section_path_text','') <> ''"
            ")"
        )

        if exact_section_value:
            conditions.append(
                f"regexp_replace(lower(COALESCE(c.metadata->>'section_path_text','')), '[[:space:][:punct:]_]+', '', 'g') = ${idx}"
            )
            params.append(normalized_exact_section_value)
            idx += 1
        elif section_value:
            section_like = f"%{section_value}%"
            normalized_section_like = f"%{normalized_section_value}%"
            conditions.append(
                "("
                f"LOWER(COALESCE(c.metadata->>'section_title','')) LIKE LOWER(${idx}) "
                f"OR LOWER(COALESCE(c.metadata->>'section_path_text','')) LIKE LOWER(${idx}) "
                f"OR regexp_replace(lower(COALESCE(c.metadata->>'section_title','')), '[[:space:][:punct:]_]+', '', 'g') LIKE ${idx + 1} "
                f"OR regexp_replace(lower(COALESCE(c.metadata->>'section_path_text','')), '[[:space:][:punct:]_]+', '', 'g') LIKE ${idx + 1}"
                ")"
            )
            params.append(section_like)
            params.append(normalized_section_like)
            idx += 2

        if document_id:
            conditions.append(f"d.id = ${idx}::uuid")
            params.append(document_id)
            idx += 1
        elif document_ids:
            normalized_document_ids = list(dict.fromkeys(str(value).strip() for value in document_ids if str(value).strip()))
            if normalized_document_ids:
                conditions.append(f"d.id = ANY(${idx}::uuid[])")
                params.append(normalized_document_ids)
                idx += 1

        score_param_idx: Optional[int] = None
        if query_value and not section_value:
            query_like = f"%{query_value}%"
            conditions.append(
                "("
                f"c.content ILIKE ${idx} "
                f"OR COALESCE(c.metadata->>'section_title','') ILIKE ${idx} "
                f"OR COALESCE(c.metadata->>'section_path_text','') ILIKE ${idx}"
                ")"
            )
            params.append(query_like)
            score_param_idx = idx
            idx += 1
        elif query_value:
            query_like = f"%{query_value}%"
            params.append(query_like)
            score_param_idx = idx
            idx += 1

        where_sql = " AND ".join(conditions) if conditions else "TRUE"

        # query_text only affects light scoring when section_query/document_id already matched.
        if score_param_idx is not None:
            score_expr = (
                f"CASE WHEN c.content ILIKE ${score_param_idx} "
                f"OR COALESCE(c.metadata->>'section_title','') ILIKE ${score_param_idx} "
                f"OR COALESCE(c.metadata->>'section_path_text','') ILIKE ${score_param_idx} "
                "THEN 1.0 ELSE 0.5 END"
            )
        else:
            score_expr = "0.5"

        sql = f"""
            SELECT
                c.id::text AS chunk_id,
                c.document_id::text AS document_id,
                c.content AS content,
                {score_expr}::float8 AS combined_score,
                NULL::float8 AS vector_similarity,
                {score_expr}::float8 AS text_similarity,
                c.metadata AS metadata,
                d.title AS document_title,
                d.source AS document_source
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE {where_sql}
            ORDER BY
                COALESCE((c.metadata->>'section_start_line')::int, 2147483647) ASC,
                COALESCE((c.metadata->>'section_chunk_index')::int, 2147483647) ASC,
                COALESCE(c.chunk_index, 2147483647) ASC
            LIMIT ${idx}
        """
        params.append(safe_limit)
        rows = await conn.fetch(sql, *params)

        out: List[Dict[str, Any]] = []
        for row in rows:
            metadata_raw = row["metadata"]
            if isinstance(metadata_raw, str):
                try:
                    metadata_value = json.loads(metadata_raw)
                except Exception:
                    metadata_value = {}
            elif isinstance(metadata_raw, dict):
                metadata_value = metadata_raw
            else:
                metadata_value = dict(metadata_raw or {})

            out.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "content": row["content"],
                    "combined_score": float(row["combined_score"]),
                    "vector_similarity": row["vector_similarity"],
                    "text_similarity": float(row["text_similarity"]),
                    "metadata": metadata_value,
                    "document_title": row["document_title"],
                    "document_source": row["document_source"],
                }
            )
        return out


async def artifact_search(
    embedding: List[float],
    query_text: str = "",
    limit: int = 10,
    artifact_types: Optional[List[str]] = None,
    document_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    text_weight: float = 0.3,
    sparse: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Search only artifact chunks (table/figure) with dense + sparse relevance.

    ``text_weight`` 仅为兼容旧签名保留：关键词匹配由 GE-M3 稀疏通道承担。
    """
    del text_weight
    resolved_sparse = await _resolve_query_sparse(query_text, sparse)
    normalized_types = [
        str(value or "").strip().lower()
        for value in (artifact_types or [])
        if str(value or "").strip().lower() in {"table", "figure", "algorithm"}
    ]
    safe_limit = max(1, min(int(limit or 10), 50))
    results = await asyncio.to_thread(
        qdrant_store.search,
        embedding,
        resolved_sparse,
        limit=_overfetch_limit(safe_limit, cap=150),
        document_id=document_id,
        document_ids=document_ids,
        artifact_types=normalized_types or None,
        content_type="artifact",
    )
    hits = [
        {
            "chunk_id": item["chunk_id"],
            "document_id": item["document_id"],
            "content": item["content"],
            "combined_score": item["score"],
            "metadata": item["metadata"],
            "document_title": item["document_title"],
            "document_source": item["document_source"],
        }
        for item in results
    ]
    return await _filter_active_document_hits(hits, safe_limit)


async def get_artifact(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Load a complete figure/table evidence record without exposing its binary payload."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT a.id::text, a.document_id::text, a.artifact_type, a.caption, a.page_number,
                   a.section_path, a.context_before, a.context_after, a.raw_content,
                   a.structured_data, a.retrieval_text, a.image_media_type,
                   (a.image_blob IS NOT NULL) AS has_image, d.title AS document_title
            FROM artifacts a
            JOIN documents d ON d.id = a.document_id
            WHERE a.id = $1::uuid
            """,
            artifact_id,
        )
        if not row:
            return None
        structured_data = row["structured_data"]
        if isinstance(structured_data, str):
            structured_data = json.loads(structured_data or "{}")
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "document_title": row["document_title"],
            "artifact_type": row["artifact_type"],
            "caption": row["caption"],
            "page_number": row["page_number"],
            "section_path": row["section_path"],
            "context_before": row["context_before"],
            "context_after": row["context_after"],
            "raw_content": row["raw_content"],
            "structured_data": dict(structured_data or {}),
            "retrieval_text": row["retrieval_text"],
            "has_image": bool(row["has_image"]),
            "image_media_type": row["image_media_type"],
        }


async def get_artifact_image(artifact_id: str) -> Optional[Dict[str, Any]]:
    """Load the original image bytes for a figure evidence record."""
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT image_blob, image_media_type
            FROM artifacts
            WHERE id = $1::uuid AND artifact_type = 'figure' AND image_blob IS NOT NULL
            """,
            artifact_id,
        )
        if not row:
            return None
        return {
            "content": bytes(row["image_blob"]),
            "media_type": str(row["image_media_type"] or "image/png"),
        }

# 分块管理函数
async def get_document_chunks(document_id: str) -> List[Dict[str, Any]]:
    """
    获取某个文档的所有分块。
    
    Args:
        document_id: 文档 UUID
    
    Returns:
        按分块索引排序的分块列表
    """
    async with db_pool.acquire() as conn:
        results = await conn.fetch(
            "SELECT * FROM get_document_chunks($1::uuid)",
            document_id
        )
        
        return [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "metadata": json.loads(row["metadata"])
            }
            for row in results
        ]

async def test_connection() -> bool:
    """
    测试数据库连接。
    
    Returns:
        连接成功时返回 True
    """
    try:
        async with db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False
