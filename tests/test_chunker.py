"""PDF 分块：章节层级、标题去重、exporter JSON 输入。

chunker 依赖 langchain，而项目基础依赖里没有它；缺少依赖时整组测试跳过。
"""

from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_experimental")
pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_text_splitters")
pytest.importorskip("agent.embedding_runtime")

from utils.ingestion.chunker import ChunkingConfig, PDFSemanticChunker  # noqa: E402

TITLE = "A Study of Wire and Arc Additive Manufacturing Defects"


def make_chunker() -> PDFSemanticChunker:
    return PDFSemanticChunker(
        ChunkingConfig(use_semantic_splitting=False, chunk_size=400, chunk_overlap=50)
    )


class TestSectionHierarchy:
    CONTENT = f"""## {TITLE}

## 1 Introduction

Intro body.

## 2 Method

Method body.

## 2.1 System

System body.

## 2.1.1 Scanner

Scanner body.

## 3 Results

Results body.
"""

    def test_numbered_headings_infer_levels(self):
        sections = make_chunker()._split_markdown_sections(self.CONTENT, TITLE)
        levels = {section.title: section.level for section in sections}
        assert levels[TITLE] == 0
        assert levels["1 Introduction"] == 1
        assert levels["2 Method"] == 1
        assert levels["2.1 System"] == 2
        assert levels["2.1.1 Scanner"] == 3
        assert levels["3 Results"] == 1

    def test_path_follows_numbering(self):
        sections = make_chunker()._split_markdown_sections(self.CONTENT, TITLE)
        paths = {section.title: section.path for section in sections}
        assert paths["2.1.1 Scanner"] == ["2 Method", "2.1 System", "2.1.1 Scanner"]
        assert paths["3 Results"] == ["3 Results"]

    def test_unnumbered_heading_nests_under_previous(self):
        content = (
            "## 2 Method\n\nBody.\n\n"
            "## Algorithm Details\n\nDetails.\n\n"
            "## 2.1 Step\n\nStep body."
        )
        sections = make_chunker()._split_markdown_sections(content)
        paths = {section.title: section.path for section in sections}
        assert paths["Algorithm Details"] == ["2 Method", "Algorithm Details"]
        assert paths["2.1 Step"] == ["2 Method", "2.1 Step"]


class TestSectionDedup:
    def test_duplicate_title_keeps_latest(self):
        content = (
            f"## {TITLE}\n\ncover junk\n\n"
            f"## {TITLE}\n\nreal body\n\n"
            "## 1 Introduction\n\nintro"
        )
        sections = make_chunker()._split_markdown_sections(content, TITLE)
        title_sections = [section for section in sections if section.title == TITLE]
        assert len(title_sections) == 1
        assert "real body" in title_sections[0].content
        assert "cover junk" not in title_sections[0].content

    def test_repeated_previous_text_keeps_latest(self):
        body = "A" * 100
        content = (
            f"## 1 Introduction\n\n{body}\n\n"
            f"## 2 Method\n\n{body}\n\nnext"
        )
        sections = make_chunker()._split_markdown_sections(content)
        assert [section.title for section in sections] == ["2 Method"]

    def test_distinct_sections_are_kept(self):
        content = "## 1 Introduction\n\nfirst body\n\n## 2 Method\n\nsecond body"
        sections = make_chunker()._split_markdown_sections(content)
        assert [section.title for section in sections] == ["1 Introduction", "2 Method"]


class TestJsonInput:
    def test_accepts_exporter_dict(self):
        payload = {
            "id": "abc123def456",
            "title": TITLE,
            "content": f"## {TITLE}\n\n## 1 Introduction\n\n" + "intro text " * 30,
            "content_type": "text/markdown",
            "metadata": {"source": "paper.pdf", "adapter": "pdf"},
        }
        chunks = make_chunker().chunk_pdf_documents([payload])
        assert chunks
        metadata = chunks[0].metadata
        assert metadata["title"] == TITLE
        assert metadata["document_id"] == "abc123def456"
        assert metadata["source"] == "paper.pdf"

    def test_promotes_top_level_keywords_only(self):
        payload = {
            "title": TITLE,
            "content": "## 1 Introduction\n\n" + "intro text " * 30,
            "keywords": ["wire arc", "defect detection"],
            "abstract": "摘要由 readpdf 保留在正文里，不再写入 metadata",
            "metadata": {},
        }
        chunks = make_chunker().chunk_pdf_documents([payload])
        assert chunks[0].metadata["keywords"] == ["wire arc", "defect detection"]
        assert "abstract" not in chunks[0].metadata

    def test_accepts_langchain_document(self):
        from langchain_core.documents import Document

        document = Document(
            page_content="## 1 Introduction\n\n" + "intro text " * 30,
            metadata={"title": TITLE, "source": "paper.pdf"},
        )
        chunks = make_chunker().chunk_pdf_documents([document])
        assert chunks
        assert chunks[0].metadata["title"] == TITLE

    def test_title_only_section_does_not_emit_chunk(self):
        payload = {
            "title": TITLE,
            "content": f"## {TITLE}\n\n## 1 Introduction\n\n" + "intro text " * 30,
            "metadata": {},
        }
        chunks = make_chunker().chunk_pdf_documents([payload])
        assert chunks
        assert all(chunk.metadata["section_title"] != TITLE for chunk in chunks)
