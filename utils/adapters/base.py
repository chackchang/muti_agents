"""适配层总接口：把不同来源的原始文件提取为 markdown。

每个适配器返回 ``(markdown, metadata)``，归一化层只认这一种中间格式。
"""

from __future__ import annotations

from .docs.markdown import read_markdown_document
from .docs.readpdf import read_pdf_document
from .web.niuke import read_niuke_document

READERS = {
    "markdown": read_markdown_document,
    "pdf": read_pdf_document,
    "niuke": read_niuke_document,
}


class Adapter:
    """按类型选择适配器，统一返回 markdown 与元数据。"""

    def read_document(self, path: str, type: str) -> tuple[str, dict]:
        try:
            reader = READERS[type]
        except KeyError:
            supported = "、".join(self.supported_types())
            raise ValueError(f"不支持的类型：{type}；可选：{supported}") from None

        content, metadata = reader(path)
        metadata = dict(metadata or {})
        metadata.setdefault("adapter", type)
        return content, metadata

    @staticmethod
    def supported_types() -> tuple[str, ...]:
        return tuple(sorted(READERS))
