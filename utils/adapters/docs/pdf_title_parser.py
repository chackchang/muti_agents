"""Derive stable paper titles from parsed PDF content."""

import re
from pathlib import Path


_STAGING_PREFIX = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)
_HEADING = re.compile(r"^#{1,2}\s+(.+?)\s*$")
_SECTION = re.compile(
    r"^(abstract|introduction|keywords?|authors?|references|contents?)\b",
    re.IGNORECASE,
)
_ABSTRACT = re.compile(r"^(abstract|keywords?)\b", re.IGNORECASE)
_AUTHOR_MARKERS = re.compile(r"(?:@|\b(university|institute|department|laboratory|email)\b)", re.IGNORECASE)
# 期刊页眉：“A R T I C L E I N F O”这类逐字母排版的元数据
_SPACED_UPPER = re.compile(r"^(?:[A-Z]\s+){3,}[A-Z]$")
_JOURNAL_CONTEXT = re.compile(
    r"journal homepage|contents lists available|sciencedirect", re.IGNORECASE
)


def filename_title(file_path: str) -> str:
    """Return a readable file-derived fallback without upload staging prefixes."""
    stem = Path(file_path).stem.strip()
    stem = _STAGING_PREFIX.sub("", stem)
    return re.sub(r"[_\s]+", " ", stem).strip() or "Untitled paper"


def _clean_candidate(value: str) -> str:
    value = re.sub(r"^#{1,6}\s*", "", str(value or "")).strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" -_\t")


def _looks_like_title(value: str) -> bool:
    if not 12 <= len(value) <= 360:
        return False
    if _SECTION.match(value) or _AUTHOR_MARKERS.search(value):
        return False
    readable = re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", value)
    if len(readable) / len(value) < 0.6:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9*+/-]*", value)
    return len(words) >= 3 and len(words) <= 40


def _heading_title(raw_lines: list[str], lines: list[str]) -> str | None:
    """从标题行里挑正文标题，跳过期刊名和“A R T I C L E I N F O”这类元数据。"""
    boundary = len(lines)
    for index, line in enumerate(lines):
        if _ABSTRACT.match(line):
            boundary = index
            break

    candidates: list[str] = []
    for index in range(min(boundary, len(raw_lines))):
        heading = _HEADING.match(raw_lines[index])
        if not heading:
            continue
        candidate = _clean_candidate(heading.group(1))
        if not _looks_like_title(candidate) or _SPACED_UPPER.match(candidate):
            continue
        context = " ".join(lines[max(0, index - 1) : index + 2])
        if _JOURNAL_CONTEXT.search(context):
            continue
        candidates.append(candidate)

    # 同一页里正文标题通常比期刊名、栏目标题更长
    return max(candidates, key=len) if candidates else None


def extract_document_title(content: str, file_path: str) -> str:
    """Prefer the paper's Markdown title, then a credible first-page line."""
    fallback = filename_title(file_path)
    raw_lines = str(content or "").splitlines()[:80]
    lines = [_clean_candidate(line) for line in raw_lines]

    title = _heading_title(raw_lines, lines)
    if title:
        return title

    for index, line in enumerate(lines[:30]):
        if not _looks_like_title(line):
            continue
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if _looks_like_title(next_line) and not _AUTHOR_MARKERS.search(next_line):
            combined = f"{line} {next_line}"
            if _looks_like_title(combined):
                return combined
        return line
    return fallback