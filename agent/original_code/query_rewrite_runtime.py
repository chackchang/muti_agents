"""Conservatively resolve every retrieval query against bounded conversation context."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

from .memory_utils import has_memory_content

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


QUERY_REWRITE_TIMEOUT_SECONDS = float(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "8"))
QUERY_REWRITE_MAX_LENGTH = 800
QUERY_REWRITE_RECENT_MESSAGE_COUNT = int(os.getenv("QUERY_REWRITE_RECENT_MESSAGE_COUNT", "10"))
QUERY_REWRITE_CONCRETE_MESSAGE_LIMIT = int(os.getenv("QUERY_REWRITE_CONCRETE_MESSAGE_LIMIT", "6"))


@dataclass(frozen=True)
class QueryRewriteResult:
    original_query: str
    rewritten_query: str
    model_used: bool
    reason: str


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _format_history_message(message: Dict[str, Any]) -> str:
    role = str(message.get("role") or "").strip().lower()
    label = "用户" if role == "user" else "助手" if role == "assistant" else role or "消息"
    content = _normalize_text(message.get("content"))
    return f"{label}：{content}" if content else ""


def build_query_rewrite_context(
    *,
    history_messages: List[Dict[str, Any]],
    memory_summary: Dict[str, Any] | None,
    memory_archives: List[Dict[str, Any]] | None = None,
    concrete_messages: List[Dict[str, Any]] | None = None,
    history_topic_hint: str = "",
) -> str:
    """Build a raw-message-first context for conservative query rewriting."""
    blocks: List[str] = []

    seen_messages: set[tuple[str, str]] = set()

    def render_unique(message: Dict[str, Any]) -> str:
        role = str(message.get("role") or "").strip().lower()
        content = _normalize_text(message.get("content"))
        identity = (role, content)
        if not content or identity in seen_messages:
            return ""
        seen_messages.add(identity)
        return _format_history_message(message)

    topic_hint = _normalize_text(history_topic_hint)
    if topic_hint:
        seen_messages.add(("user", topic_hint))
        blocks.append(f"指代锚点（最高优先级，来自原始用户消息）：\n用户：{topic_hint}")

    concrete_blocks = [
        rendered
        for rendered in (
            render_unique(message)
            for message in list(concrete_messages or [])[:QUERY_REWRITE_CONCRETE_MESSAGE_LIMIT]
        )
        if rendered
    ]
    if concrete_blocks:
        blocks.append("相关原始历史消息（高优先级）：\n" + "\n".join(concrete_blocks))

    recent_blocks: List[str] = []
    for message in reversed(list(history_messages or [])[-QUERY_REWRITE_RECENT_MESSAGE_COUNT:]):
        rendered = render_unique(message)
        if rendered:
            recent_blocks.insert(0, rendered)
    if recent_blocks:
        blocks.append("最近原始对话（用于补全指代）：\n" + "\n".join(recent_blocks))

    summary = memory_summary if isinstance(memory_summary, dict) else {}
    if has_memory_content(summary):
        summary_text = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        if summary_text:
            blocks.append(f"当前会话摘要（仅在原始消息不足时使用）：\n{summary_text}")

    archives = list(memory_archives or [])[:2]
    if archives:
        archive_text = "\n".join(
            json.dumps(item.get("summary") or {}, ensure_ascii=False, separators=(",", ":"))
            for item in archives
        )
        blocks.append(f"历史阶段摘要（最低优先级）：\n{archive_text}")
    return "\n\n".join(blocks)


def _extract_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    return str(content or "").strip()


def _parse_rewritten_query(response: Any) -> str:
    text = _extract_response_text(response)
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return ""
    try:
        payload = json.loads(match.group(0))
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return " ".join(str(payload.get("rewritten_query") or "").split()).strip()


def get_query_rewrite_model() -> "ChatOpenAI | None":
    """Build the dedicated rewrite client without falling back to the answer model."""
    api_key = os.getenv("QUERY_REWRITE_API_KEY", "").strip()
    base_url = os.getenv("QUERY_REWRITE_BASE_URL", "").strip()
    model = os.getenv("QUERY_REWRITE_MODEL", "").strip()
    if not api_key or not base_url or not model:
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
        timeout=QUERY_REWRITE_TIMEOUT_SECONDS,
        request_timeout=QUERY_REWRITE_TIMEOUT_SECONDS,
    )


def _build_prompt(*, original_query: str, conversation_context: str) -> str:
    return (
        "你是检索问题改写器。请结合会话上下文判断当前问题是否依赖上文。"
        "若当前问题脱离上下文仍完整、明确且可检索，rewritten_query 必须逐字返回当前问题。"
        "只有当当前问题存在指代、省略、短追问或必须依赖上文才能理解的内容时，才改写为可独立检索的简洁问题。"
        "改写时只能消解指代并补齐上下文中明确存在的必要信息；不得扩大问题、不得回答问题、"
        "不得添加上下文中没有的论文、作者、年份、术语、结论或假设，不得改变用户的任务、约束和比较关系。"
        "只输出 JSON：{\"rewritten_query\": \"...\"}。\n\n"
        f"会话上下文：\n{conversation_context}\n\n"
        f"当前问题：{original_query}"
    )


async def rewrite_query_with_conversation(
    *,
    original_query: str,
    conversation_context: str,
    model: Any,
) -> QueryRewriteResult:
    """Return the original query unless the dedicated model can safely resolve it."""
    normalized_original = " ".join(str(original_query or "").split()).strip()
    normalized_context = str(conversation_context or "").strip()
    if not normalized_original:
        return QueryRewriteResult("", "", False, "empty_query")
    if model is None:
        return QueryRewriteResult(normalized_original, normalized_original, False, "rewrite_model_not_configured")

    try:
        response = await asyncio.wait_for(
            model.ainvoke(
                [
                    {"role": "system", "content": "只输出严格 JSON，不输出解释。"},
                    {
                        "role": "user",
                        "content": _build_prompt(
                            original_query=normalized_original,
                            conversation_context=normalized_context,
                        ),
                    },
                ]
            ),
            timeout=QUERY_REWRITE_TIMEOUT_SECONDS,
        )
        rewritten = _parse_rewritten_query(response)
        if not rewritten or len(rewritten) > QUERY_REWRITE_MAX_LENGTH:
            return QueryRewriteResult(normalized_original, normalized_original, False, "invalid_model_output_fallback")
        reason = "model_rewrite" if rewritten != normalized_original else "model_kept_original"
        return QueryRewriteResult(normalized_original, rewritten, True, reason)
    except Exception:
        return QueryRewriteResult(normalized_original, normalized_original, False, "model_rewrite_failed_fallback")
