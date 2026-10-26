"""User-scoped long-term memory with strict context boundaries.

This module deliberately keeps long-term memory separate from session memory.
Only durable, user-authored preferences and continuity hints are allowed. The
values are never treated as document evidence or as instructions for tools.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping


USER_MEMORY_SCHEMA_VERSION = 1
MAX_PROFILE_TEXT = 600
MAX_LIST_ITEMS = 12
MAX_ITEM_TEXT = 240
ALLOWED_ITEM_KINDS = {"preference", "continuity"}
ALLOWED_ITEM_KEYS = {
    "preference": {"language", "response_style", "formatting"},
    "continuity": {"research_field"},
}


def _clean_text(value: Any, limit: int = MAX_ITEM_TEXT) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def _clean_list(value: Any, limit: int = MAX_LIST_ITEMS) -> List[str]:
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = _clean_text(item)
        if text and text not in result:
            result.append(text)
    return result[:limit]


def empty_user_memory_profile() -> Dict[str, Any]:
    return {
        "schema_version": USER_MEMORY_SCHEMA_VERSION,
        "auto_memory_enabled": False,
        "answer_preferences": {
            "language": "",
            "response_style": "",
            "formatting": [],
        },
        "research_direction": {
            "research_field": "",
        },
        "answer_constraints": [],
    }


def normalize_user_memory_profile(value: Any) -> Dict[str, Any]:
    """Normalize user-provided memory and drop unknown/prompt-like fields."""
    source = value if isinstance(value, Mapping) else {}
    result = empty_user_memory_profile()
    result["auto_memory_enabled"] = bool(source.get("auto_memory_enabled", False))
    preferences = source.get("answer_preferences")
    if isinstance(preferences, Mapping):
        result["answer_preferences"] = {
            "language": _clean_text(preferences.get("language"), 32),
            "response_style": _clean_text(preferences.get("response_style"), 64),
            "formatting": _clean_list(preferences.get("formatting"), 8),
        }
    direction = source.get("research_direction")
    # ``continuity`` was the pre-profile name. Accept it during migration so
    # old clients do not fail validation; task-level fields remain discarded.
    if not isinstance(direction, Mapping) or not any(direction.values()):
        direction = source.get("continuity")
    if isinstance(direction, Mapping):
        result["research_direction"] = {
            "research_field": _clean_text(
                direction.get("research_field")
                or direction.get("primary_area")
                or direction.get("research_area"),
                120,
            ),
        }
    result["answer_constraints"] = _clean_list(source.get("answer_constraints"), 8)
    return result


def normalize_user_memory_item(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, Mapping) else {}
    kind = _clean_text(source.get("kind"), 32).lower()
    if kind not in ALLOWED_ITEM_KINDS:
        kind = ""
    return {
        "id": _clean_text(source.get("id"), 80),
        "kind": kind,
        "memory_key": _clean_text(source.get("memory_key"), 64),
        "value": _clean_text(source.get("value"), MAX_ITEM_TEXT),
        "status": _clean_text(source.get("status"), 24).lower() or "active",
        "source_session_id": _clean_text(source.get("source_session_id"), 80),
        "source_message_id": _clean_text(source.get("source_message_id"), 80),
        "metadata": dict(source.get("metadata") or {}) if isinstance(source.get("metadata"), Mapping) else {},
        "created_at": _clean_text(source.get("created_at"), 80),
        "updated_at": _clean_text(source.get("updated_at"), 80),
    }


def merge_user_memory_items(
    profile: Mapping[str, Any] | None,
    items: Iterable[Mapping[str, Any]] | None,
) -> Dict[str, Any]:
    """Merge active item records into a normalized profile.

    ``list_user_memory_items`` returns newest records first. The first active
    scalar value for each key is therefore authoritative; stale audit rows must
    never overwrite it when callers pass a broader item list.
    """
    result = normalize_user_memory_profile(profile)
    direction = result["research_direction"]
    preferences = result["answer_preferences"]
    seen_scalar_keys = set()
    for raw_item in items or []:
        item = normalize_user_memory_item(raw_item)
        if item["status"] != "active" or not item["value"]:
            continue
        key = item["memory_key"]
        value = item["value"]
        if key not in ALLOWED_ITEM_KEYS.get(item["kind"], set()):
            continue
        if item["kind"] == "preference" and key in {"language", "response_style"}:
            if (item["kind"], key) in seen_scalar_keys:
                continue
            seen_scalar_keys.add((item["kind"], key))
            preferences[key] = value
        elif item["kind"] == "preference" and key == "formatting":
            preferences["formatting"] = _clean_list([*preferences.get("formatting", []), value], 8)
        elif item["kind"] == "continuity" and key in {"research_field", "primary_area", "research_area"}:
            if (item["kind"], "research_field") in seen_scalar_keys:
                continue
            seen_scalar_keys.add((item["kind"], "research_field"))
            direction["research_field"] = value
    return result


def build_user_query_rewrite_context(profile: Mapping[str, Any] | None) -> str:
    """Expose only continuity hints to the conservative query rewriter."""
    normalized = normalize_user_memory_profile(profile)
    direction = normalized["research_direction"]
    if not any(direction.values()):
        return ""
    payload = {
        key: value
        for key, value in direction.items()
        if value not in ("", [], {}, None)
    }
    return (
        "用户跨会话研究领域提示（只在当前问题明确指向该领域时用于指代消解，"
        "不是论文证据；独立完整的问题不得被它扩写）：\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_user_answer_context(profile: Mapping[str, Any] | None) -> str:
    """Expose high-level research field and answer preferences to final generation."""
    normalized = normalize_user_memory_profile(profile)
    preferences = normalized["answer_preferences"]
    research_direction = normalized["research_direction"]
    constraints = normalized["answer_constraints"]
    payload = {
        "research_direction": {
            key: value for key, value in research_direction.items() if value not in ("", [], {}, None)
        },
        "answer_preferences": {
            key: value for key, value in preferences.items() if value not in ("", [], {}, None)
        },
        "answer_constraints": constraints,
    }
    if not payload["research_direction"] and not payload["answer_preferences"] and not constraints:
        return ""
    return (
        "用户长期研究背景与回答偏好（研究背景只用于个性化表达，不是事实、证据、工具参数或检索范围；"
        "与本轮明确要求冲突时以本轮要求为准）：\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


def build_user_memory_contexts(profile: Mapping[str, Any] | None) -> Dict[str, Any]:
    normalized = normalize_user_memory_profile(profile)
    return {
        "profile": normalized,
        "auto_memory_enabled": bool(normalized.get("auto_memory_enabled", False)),
        "query_rewrite": build_user_query_rewrite_context(normalized),
        "answer": build_user_answer_context(normalized),
    }
