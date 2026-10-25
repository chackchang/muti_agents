"""牛客面经页面的文本适配器。

针对 ``data/text1`` 的布局：

    标题行
      面经 01
      面试岗位：…… ｜ 面试时间：……
      问题 1
      问题 2
    面经 02
      ……
    剩余60%内容，订阅专栏后可继续查看……   <- 从这里开始是页面噪音

提取时保留标题、每个面经小节的岗位/时间和问题列表，遇到付费墙、
评论区、相关推荐等噪音就停止。
"""

from __future__ import annotations

import re
from pathlib import Path

_SECTION = re.compile(r"^面经\s*\d+\s*$")
_FIELD = re.compile(r"^(面试岗位|面试时间)\s*[:：]\s*(.*)$")
_COMPANY = re.compile(r"^(?P<company>[^\s面经]{1,12})面经")

# 命中任意一条就说明正文结束，后面是付费提示、评论或推荐流
_NOISE = (
    re.compile(r"剩余\s*\d+%\s*内容"),
    re.compile(r"订阅专栏|文章被收录于专栏"),
    re.compile(r"^提示\s+评论"),
    re.compile(r"^更多真题"),
    re.compile(r"^一键发评"),
    re.compile(r"^相关推荐"),
    re.compile(r"^全站热榜"),
    re.compile(r"^正在热议"),
    re.compile(r"分享\s*转发到动态"),
)


def _is_noise(line: str) -> bool:
    return any(pattern.search(line) for pattern in _NOISE)


def _split_fields(line: str) -> dict[str, str]:
    """解析“面试岗位：X ｜ 面试时间：Y”，也兼容分行的写法。"""
    fields: dict[str, str] = {}
    for part in re.split(r"[｜|]", line):
        match = _FIELD.match(part.strip())
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def parse_niuke_text(text: str) -> tuple[str, dict]:
    """把牛客面经页面的可见文本转成 markdown。"""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]

    title = ""
    sections: list[dict] = []
    current: dict | None = None
    truncated = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _is_noise(stripped):
            truncated = True
            break
        if not title:
            title = stripped
            continue

        if _SECTION.match(stripped):
            current = {"name": stripped, "position": "", "time": "", "questions": []}
            sections.append(current)
            continue

        fields = _split_fields(stripped)
        if fields:
            if current is None:
                current = {"name": "正文", "position": "", "time": "", "questions": []}
                sections.append(current)
            if "面试岗位" in fields:
                current["position"] = fields["面试岗位"]
            if "面试时间" in fields:
                current["time"] = fields["面试时间"]
            continue

        if current is None:
            current = {"name": "正文", "position": "", "time": "", "questions": []}
            sections.append(current)
        current["questions"].append(stripped)

    parts = [f"# {title}"] if title else []
    for section in sections:
        parts.append(f"## {section['name']}")
        if section["position"]:
            parts.append(f"**面试岗位**：{section['position']}")
        if section["time"]:
            parts.append(f"**面试时间**：{section['time']}")
        if section["questions"]:
            parts.append("\n".join(f"- {question}" for question in section["questions"]))

    company = _COMPANY.match(title)
    metadata = {
        "source_type": "web",
        "content_type": "text/markdown",
        "title": title,
        "company": company.group("company") if company else None,
        "sections": [
            {
                "name": section["name"],
                "position": section["position"],
                "time": section["time"],
                "question_count": len(section["questions"]),
            }
            for section in sections
        ],
        "question_count": sum(len(section["questions"]) for section in sections),
        "truncated": truncated,
    }
    return "\n\n".join(parts).strip() + "\n", metadata


def read_niuke_document(path: str) -> tuple[str, dict]:
    """读取牛客页面文本，返回 (markdown, metadata)。"""
    source = Path(path)
    content, metadata = parse_niuke_text(source.read_text(encoding="utf-8"))
    metadata["source"] = str(source)
    metadata["adapter"] = "niuke"
    return content, metadata
