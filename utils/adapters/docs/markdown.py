"""Markdown 文档适配器：只做格式清理，不改写正文内容。"""

from __future__ import annotations

import re
from pathlib import Path

_HEADING = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def clean_markdown_content(content: str) -> str:
    """统一换行、去行尾空格、压缩多余空行；保留标题和正文结构。"""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    content = "\n".join(line.rstrip() for line in content.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", content).strip()


def _extract_title(content: str, path: str) -> str:
    match = _HEADING.search(content)
    if not match:
        return Path(path).stem
    # 站点生成的标题常带 [#](url) 锚点，去掉链接只留文字
    title = _MD_LINK.sub(lambda m: "" if m.group(1).strip("# ") == "" else m.group(1),
                         match.group(1))
    return title.strip(" #\t-") or Path(path).stem


def read_markdown_document(path: str) -> tuple[str, dict]:
    """读取 markdown 文件，返回 (markdown, metadata)。"""
    source = Path(path)
    content = clean_markdown_content(source.read_text(encoding="utf-8"))
    metadata = {
        "source": str(source),
        "adapter": "markdown",
        "source_type": "docs",
        "content_type": "text/markdown",
        "title": _extract_title(content, path),
    }
    return content, metadata
