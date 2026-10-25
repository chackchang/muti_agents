"""Conservative extraction of durable, user-authored long-term memory."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List


_DURABLE_MARKER_RE = re.compile(
    r"(?:以后|之后|今后|后续|默认|长期|平时|一直|从现在起|未来|always|from now on|by default|in future)",
    re.IGNORECASE,
)
_TEMPORARY_MARKER_RE = re.compile(
    r"(?:本次|这次|当前|这轮|这一轮|暂时|今天|这篇|该篇|不要记住|别记|do not remember|this time|for now)",
    re.IGNORECASE,
)
_TASK_SCOPE_RE = re.compile(
    r"(?:这篇|该篇|论文|paper|数据集|dataset|实验|experiment|benchmark|代码|code|实现|implement|如何|什么|为什么|吗|？|\?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LongTermMemoryCandidate:
    kind: str
    memory_key: str
    value: str
    confidence: float
    reason: str

    def metadata(self) -> dict[str, object]:
        return {
            "origin": "automatic",
            "extractor": "durable_user_statement_v1",
            "confidence": self.confidence,
            "reason": self.reason,
        }


def automatic_memory_allowed(*, administrator_enabled: bool, user_enabled: bool, user_id: str | None) -> bool:
    """Apply the global kill switch and the user's explicit opt-in."""
    return bool(administrator_enabled and user_enabled and str(user_id or "").strip())


def _clean(value: str, limit: int) -> str:
    return " ".join(str(value or "").split()).strip(" ，,。.!！?？:：;；")[:limit]


def _is_explicitly_durable(message: str) -> bool:
    return bool(_DURABLE_MARKER_RE.search(message)) and not bool(_TEMPORARY_MARKER_RE.search(message))


def _extract_language(message: str) -> str:
    patterns = (
        r"(?:用|使用|以|switch to)\s*(中文|汉语|英文|英语|english|chinese).{0,20}(?:回答|回复|交流|respond|reply)",
        r"(?:回答|回复|respond|reply)\s*(?:时|请|用|in)?\s*(中文|汉语|英文|英语|english|chinese)",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1).lower()
        if raw in {"中文", "汉语", "chinese"}:
            return "中文"
        if raw in {"英文", "英语", "english"}:
            return "英文"
    return ""


def _extract_response_style(message: str) -> str:
    patterns = (
        (r"(?:简洁|精简|concise|brief).{0,12}(?:回答|回复|respond|reply)", "简洁"),
        (r"(?:详细|展开|深入|detailed|in depth).{0,12}(?:回答|回复|respond|reply)", "详细"),
        (r"(?:学术|专业|academic|professional).{0,12}(?:回答|回复|respond|reply)", "专业"),
    )
    for pattern, value in patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return value
    return ""


def _extract_research_field(message: str) -> str:
    patterns = (
        r"(?:我的?|本人)?(?:研究方向|研究领域|主要研究方向)\s*(?:是|为|:|：)\s*([^。！？!?；;\n]{2,120})",
        r"我(?:主要|长期)研究\s*([^。！？!?；;\n]{2,120})",
        r"我(?:是)?做\s*([^。！？!?；;\n]{2,120})(?:方向)?",
    )
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if not match:
            continue
        value = _clean(match.group(1), 120)
        value = re.sub(r"(?:方向)?(?:的)?$", "", value).strip()
        if value and not _TASK_SCOPE_RE.search(value):
            return value
    return ""


def extract_user_long_term_memory_candidates(user_message: str) -> List[LongTermMemoryCandidate]:
    """Return only explicit, durable facts stated by the user in this turn.

    The extractor is deliberately deterministic and never reads assistant text,
    retrieved evidence, or session summaries. This prevents paper facts and
    one-turn instructions from becoming cross-session user memory.
    """
    message = _clean(user_message, 4000)
    if not message or _TEMPORARY_MARKER_RE.search(message):
        return []

    candidates: List[LongTermMemoryCandidate] = []
    research_field = _extract_research_field(message)
    if research_field:
        candidates.append(
            LongTermMemoryCandidate(
                kind="continuity",
                memory_key="research_field",
                value=research_field,
                confidence=0.96,
                reason="explicit_research_field",
            )
        )

    if _is_explicitly_durable(message):
        language = _extract_language(message)
        if language:
            candidates.append(
                LongTermMemoryCandidate(
                    kind="preference",
                    memory_key="language",
                    value=language,
                    confidence=0.98,
                    reason="explicit_durable_language_preference",
                )
            )
        response_style = _extract_response_style(message)
        if response_style:
            candidates.append(
                LongTermMemoryCandidate(
                    kind="preference",
                    memory_key="response_style",
                    value=response_style,
                    confidence=0.94,
                    reason="explicit_durable_response_style",
                )
            )

    unique: List[LongTermMemoryCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        identity = (candidate.kind, candidate.memory_key)
        if identity not in seen:
            seen.add(identity)
            unique.append(candidate)
    return unique
