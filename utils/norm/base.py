"""归一化层：把适配器产出的 markdown 统一为 Document 对象。

当前规则（保守）：
- 全角数字/英文字母转半角，全角空格转普通空格，去掉零宽/无法映射字符；
- 数字与中文单位之间去空格（``8 月 23 日`` -> ``8月23日``）；
- 合并行内多余空格和连续空行，保留缩进与代码块；
- 英文大小写保持原样，避免破坏 RAG、MCP、AI 这类缩写。
"""

from __future__ import annotations

import hashlib
import re

# 零宽字符，以及 PDFium 无法映射字形时输出的 U+FFFE/U+FFFF
_INVISIBLE = re.compile(r"[\u200b\u200c\u200d\ufeff\ufffe\uffff]")
_INLINE_SPACES = re.compile(r"[ \t]{2,}")
_UNIT_SPACE = re.compile(
    r"(?<=\d)[ \t]+(?=[年月日时分秒岁天周])|(?<=[年月日时分秒岁天周])[ \t]+(?=\d)"
)
_BLANK_LINES = re.compile(r"\n{3,}")


def _to_halfwidth(text: str) -> str:
    """只转换全角数字、英文字母和全角空格，不动中文标点。"""
    chars = []
    for char in text:
        code = ord(char)
        if (
            0xFF10 <= code <= 0xFF19
            or 0xFF21 <= code <= 0xFF3A
            or 0xFF41 <= code <= 0xFF5A
        ):
            chars.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)


def normalize_markdown(content: str) -> str:
    """归一化 markdown 正文，保留标题、缩进和代码块内容。"""
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    in_fence = False

    for raw in content.split("\n"):
        line = _INVISIBLE.sub("", raw).replace("\u00a0", " ")
        if line.strip().startswith("```"):
            in_fence = not in_fence
            lines.append(line.rstrip())
            continue
        if in_fence:
            lines.append(line.rstrip())
            continue

        line = _to_halfwidth(line)
        match = re.match(r"^(\s*)(.*)$", line)
        indent, body = match.group(1), match.group(2)
        body = _INLINE_SPACES.sub(" ", body).rstrip()
        body = _UNIT_SPACE.sub("", body)
        lines.append(indent + body)

    # 只去首尾空行，保留首行缩进（缩进代码块有意义）
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip("\n")


def normalize_metadata(metadata: dict | None) -> dict:
    """递归归一化元数据里的字符串值，保留原始键名。"""
    def norm(value):
        if isinstance(value, str):
            return _UNIT_SPACE.sub("", re.sub(r"\s+", " ", _to_halfwidth(value)).strip())
        if isinstance(value, list):
            return [norm(item) for item in value]
        if isinstance(value, dict):
            return {key: norm(item) for key, item in value.items()}
        return value

    return norm(dict(metadata or {}))


class Document(dict):
    """统一文档对象：id + title + content(markdown) + content_type + metadata。"""

    def __init__(
        self,
        content: str = "",
        metadata: dict | None = None,
        title: str | None = None,
        doc_id: str | None = None,
    ):
        metadata = dict(metadata or {})
        title = title or metadata.get("title") or ""
        if not doc_id:
            source = str(metadata.get("source", ""))
            digest = hashlib.sha256(f"{source}\n{content}".encode("utf-8")).hexdigest()
            doc_id = digest[:16]
        super().__init__(
            id=doc_id,
            title=title,
            content=content,
            content_type="text/markdown",
            metadata=metadata,
        )


class Normalizer:
    """把已经提取好的 markdown 规范化为 Document。

    只依赖文本和元数据，不认识任何适配器。
    """

    def normalize(self, content: str, metadata: dict | None = None) -> Document:
        """归一化已经提取好的 markdown。"""
        return Document(
            content=normalize_markdown(content),
            metadata=normalize_metadata(metadata),
        )
