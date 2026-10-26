"""Scoped, structured conversation memory built only from approved final turns."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


MEMORY_CONTEXT_BUDGET_TOKENS = int(os.getenv("MEMORY_CONTEXT_BUDGET_TOKENS", "32000"))
MEMORY_COMPACTION_TRIGGER_TOKENS = int(os.getenv("MEMORY_COMPACTION_TRIGGER_TOKENS", "24000"))
RECENT_MESSAGE_COUNT = int(os.getenv("MEMORY_RECENT_TURNS", "8"))
MEMORY_SUMMARY_KEYS = (
    "schema_version",
    "conversation_background",
    "current_topic",
    "current_question",
    "goal",
    "discussed_points",
    "constraints",
    "confirmed_decisions",
    "open_questions",
    "uncertainties",
    "next_step",
    "last_turn_summary",
)


@dataclass
class SessionMemoryState:
    version: int = 0
    covered_message_count: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextBuildResult:
    full_prompt: str
    compression_used: bool
    memory_updated: bool
    memory_state: SessionMemoryState


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def empty_memory_summary() -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "conversation_background": "",
        "current_topic": "",
        "current_question": "",
        "goal": "",
        "discussed_points": [],
        "constraints": [],
        "confirmed_decisions": [],
        "open_questions": [],
        "uncertainties": [],
        "next_step": "",
        "last_turn_summary": "",
        "discussion_index": {"papers": [], "questions": []},
    }


def normalize_memory_summary(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    result = empty_memory_summary()
    result["schema_version"] = 2
    for key in (
        "conversation_background",
        "current_topic",
        "current_question",
        "goal",
        "next_step",
        "last_turn_summary",
    ):
        max_length = 600 if key == "goal" else 800
        result[key] = " ".join(str(source.get(key) or "").split())[:max_length]
    for key in ("discussed_points", "constraints", "confirmed_decisions", "open_questions", "uncertainties"):
        raw_items = source.get(key, [])
        if not isinstance(raw_items, list):
            raw_items = []
        result[key] = [" ".join(str(item).split())[:360] for item in raw_items if str(item).strip()][:12]
    raw_index = source.get("discussion_index")
    if isinstance(raw_index, dict):
        papers = raw_index.get("papers") if isinstance(raw_index.get("papers"), list) else []
        questions = raw_index.get("questions") if isinstance(raw_index.get("questions"), list) else []
        result["discussion_index"] = {
            "papers": [
                {
                    "title": " ".join(str(item.get("title") or "").split())[:240],
                    "status": " ".join(str(item.get("status") or "").split())[:120],
                    "last_discussed": " ".join(str(item.get("last_discussed") or "").split())[:360],
                }
                for item in papers
                if isinstance(item, dict) and str(item.get("title") or "").strip()
            ][:8],
            "questions": [
                {
                    "question": " ".join(str(item.get("question") or "").split())[:360],
                    "status": " ".join(str(item.get("status") or "").split())[:120],
                    "related_papers": [
                        " ".join(str(paper).split())[:160]
                        for paper in (item.get("related_papers") if isinstance(item.get("related_papers"), list) else [])
                        if str(paper).strip()
                    ][:8],
                }
                for item in questions
                if isinstance(item, dict) and str(item.get("question") or "").strip()
            ][:12],
        }
    return result


def has_memory_content(summary: Dict[str, Any]) -> bool:
    """Return whether a normalized summary contains usable memory fields."""
    normalized = normalize_memory_summary(summary)
    for key, value in normalized.items():
        if key == "schema_version":
            continue
        if key == "discussion_index":
            if any(value.get(bucket) for bucket in ("papers", "questions")):
                return True
            continue
        if value not in ("", [], {}, None):
            return True
    return False


def normalize_memory_state(snapshot: Optional[Dict[str, Any]]) -> SessionMemoryState:
    value = snapshot or {}
    return SessionMemoryState(
        version=max(0, int(value.get("version") or 0)),
        covered_message_count=max(0, int(value.get("covered_message_count") or 0)),
        summary=normalize_memory_summary(value.get("summary")),
    )


def is_memory_eligible_message(message: Dict[str, Any]) -> bool:
    role = str(message.get("role") or "").strip().lower()
    content = str(message.get("content") or "").strip()
    metadata = message.get("metadata") or {}
    return role in {"user", "assistant"} and bool(metadata.get("memory_eligible")) and bool(content)


def memory_eligible_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    eligible: List[Dict[str, Any]] = []
    for message in messages or []:
        if not is_memory_eligible_message(message):
            continue
        item: Dict[str, Any] = {
            "role": str(message["role"]),
            "content": str(message["content"]).strip(),
        }
        eligible.append(item)
    return eligible


def _messages_to_text(messages: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in messages:
        lines.append(f"{item['role']}: {item['content']}")
    return "\n".join(lines)


def build_memory_update_prompt(
    previous_summary: Dict[str, Any],
    turns: List[Dict[str, str]],
    current_question: str = "",
) -> str:
    return (
        "Update a structured research conversation memory. Return JSON only. "
        "Schema: {schema_version:2, conversation_background:string, current_topic:string, "
        "current_question:string, goal:string, discussed_points:string[], constraints:string[], "
        "confirmed_decisions:string[], open_questions:string[], uncertainties:string[], "
        "next_step:string, last_turn_summary:string, discussion_index:{papers:object[], questions:object[]}}. "
        "The summary must preserve the overall conversation background, the current topic, "
        "the active question, important discussed points, and the next actionable step. "
        "Only keep durable conversation coordination: user goal, explicit constraints, "
        "user-confirmed decisions, follow-up questions, and uncertainty boundaries. "
        "In discussion_index, record only paper titles/aliases, question topics, and their current status; "
        "never store paper claims, numbers, citations, evidence snippets, or retrieval results. "
        "Never store paper claims, numerical results, citations, source snippets, retrieval results, "
        "tool calls, model reasoning, debug data, errors, or assistant guesses as memory. "
        "When a newer explicit user statement conflicts with an older one, keep the newer statement "
        "and remove the obsolete one. Do not merge contradictory values. Omit empty or stale items.\n\n"
        f"Previous structured memory:\n{json.dumps(normalize_memory_summary(previous_summary), ensure_ascii=False)}\n\n"
        f"Eligible completed turns:\n{_messages_to_text(turns)}\n\n"
        f"Current user question (active context, not yet answered):\n{current_question.strip()}"
    )


def should_update_memory(
    history_messages: List[Dict[str, str]],
    memory_state: SessionMemoryState,
    current_question: str,
) -> bool:
    pending_messages = history_messages[max(0, memory_state.covered_message_count) :]
    projected = _messages_to_text(pending_messages + [{"role": "user", "content": current_question}])
    return estimate_tokens(projected) >= MEMORY_COMPACTION_TRIGGER_TOKENS


def messages_for_memory_update(
    history_messages: List[Dict[str, str]],
    covered_message_count: int,
) -> List[Dict[str, str]]:
    return history_messages[max(0, covered_message_count) :]


def build_context(
    history_messages: List[Dict[str, str]],
    current_question: str,
    memory_state: SessionMemoryState,
    memory_archives: Optional[List[Dict[str, Any]]] = None,
    concrete_memories: Optional[List[Dict[str, Any]]] = None,
) -> ContextBuildResult:
    recent = history_messages[-RECENT_MESSAGE_COUNT:]
    blocks: List[str] = []
    use_snapshot = bool(memory_state.version and has_memory_content(memory_state.summary))
    context_messages = recent if use_snapshot else history_messages
    archive_block = ""
    archives = list(memory_archives or [])[:3]
    if archives:
        archive_items = []
        for archive in archives:
            archive_items.append(
                f"stage_version={int(archive.get('source_snapshot_version') or 0)} "
                f"covered_messages={int(archive.get('covered_message_count') or 0)} "
                f"summary={json.dumps(normalize_memory_summary(archive.get('summary')), ensure_ascii=False)}"
            )
        archive_block = "Historical stage summaries (lower priority than current memory):\n" + "\n".join(archive_items)
    if use_snapshot:
        blocks.append(f"Structured conversation memory:\n{json.dumps(memory_state.summary, ensure_ascii=False)}")
    if context_messages:
        label = "Recent eligible conversation" if use_snapshot else "Eligible conversation"
        blocks.append(f"{label}:\n{_messages_to_text(context_messages)}")
    if archive_block:
        blocks.append(archive_block)
    concrete_items = list(concrete_memories or [])[:4]
    if concrete_items:
        concrete_block = "Relevant concrete historical messages (lower priority than current memory):\n" + "\n".join(
            f"{str(item.get('role') or 'message')}: {str(item.get('content') or '').strip()}"
            for item in concrete_items
            if str(item.get("content") or "").strip()
        )
        if concrete_block.count("\n") >= 1:
            blocks.append(concrete_block)
    blocks.append(f"Current question: {current_question}")
    prompt = "\n\n".join(blocks)
    if estimate_tokens(prompt) > MEMORY_CONTEXT_BUDGET_TOKENS:
        bounded_recent: List[Dict[str, Any]] = []
        for item in reversed(context_messages):
            candidate = [item] + bounded_recent
            candidate_blocks = []
            if use_snapshot:
                candidate_blocks.append(
                    f"Structured conversation memory:\n{json.dumps(memory_state.summary, ensure_ascii=False)}"
                )
            candidate_blocks.append(f"Recent eligible conversation:\n{_messages_to_text(candidate)}")
            candidate_blocks.append(f"Current question: {current_question}")
            if estimate_tokens("\n\n".join(candidate_blocks)) > MEMORY_CONTEXT_BUDGET_TOKENS:
                break
            bounded_recent = candidate
        blocks = []
        if use_snapshot:
            blocks.append(f"Structured conversation memory:\n{json.dumps(memory_state.summary, ensure_ascii=False)}")
        if bounded_recent:
            label = "Recent eligible conversation" if use_snapshot else "Eligible conversation"
            blocks.append(f"{label}:\n{_messages_to_text(bounded_recent)}")
        blocks.append(f"Current question: {current_question}")
        prompt = "\n\n".join(blocks)
    return ContextBuildResult(
        full_prompt=prompt,
        compression_used=use_snapshot,
        memory_updated=False,
        memory_state=memory_state,
    )
