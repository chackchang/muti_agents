"""清洗管线测试：adapter -> norm -> JSON exporter。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utils.adapters.base import Adapter
from utils.adapters.docs.markdown import read_markdown_document
from utils.adapters.docs.pdf_title_parser import extract_document_title
from utils.adapters.docs.readpdf import extract_front_matter
from utils.adapters.web.niuke import parse_niuke_text, read_niuke_document
from utils.exporters.json import JsonExporter, Pipeline, export, export_json
from utils.norm.base import Document, normalize_markdown, normalize_metadata

ROOT = Path(__file__).resolve().parent.parent
TEXT1 = ROOT / "data" / "text1"
TEXT3 = ROOT / "data" / "text3"


# ---------------------------------------------------------------- markdown 适配器

class TestMarkdownAdapter:
    def test_keeps_structure(self, tmp_path):
        source = tmp_path / "note.md"
        source.write_text(
            "# 主标题\n\n## 小节\n\n正文 **加粗**\n\n```python\nprint(1)\n```\n",
            encoding="utf-8",
        )
        content, metadata = read_markdown_document(str(source))
        assert content.startswith("# 主标题")
        assert "## 小节" in content and "**加粗**" in content
        assert "```python" in content
        assert metadata["title"] == "主标题"
        assert metadata["source_type"] == "docs"

    def test_title_strips_anchor_link(self, tmp_path):
        source = tmp_path / "anchor.md"
        source.write_text("# [#](https://x.test#a) 真正的标题\n", encoding="utf-8")
        _, metadata = read_markdown_document(str(source))
        assert metadata["title"] == "真正的标题"

    def test_text3_keeps_headings_and_code(self):
        content, metadata = read_markdown_document(str(TEXT3))
        assert metadata["adapter"] == "markdown"
        assert "# " in content and "## " in content
        assert "```" in content
        assert "\\#" not in content


class TestPdfTitle:
    DOCLING_LIKE = """<!-- image -->

Contents lists available at ScienceDirect

## Optics and Laser Technology

[journal homepage: www.elsevier.com/locate/optlastec](https://www.elsevier.com/locate/optlastec)

Full length article

## In-situ surface inspection for wire-arc directed energy deposition integrating 3D topography reconstruction, defect detection and roughness measurement

Hao Wang a, Songyu Qi b

## A R T I C L E I N F O

Keywords: Wire-arc directed energy deposition
"""

    def test_skips_journal_heading(self):
        title = extract_document_title(self.DOCLING_LIKE, "WAAM.pdf")
        assert title.startswith("In-situ surface inspection")
        assert title != "Optics and Laser Technology"

    def test_plain_text_uses_first_credible_line(self):
        content = (
            "Full length article\n"
            "In-situ surface inspection for wire-arc directed energy deposition\n"
            "Hao Wang a\n"
        )
        assert extract_document_title(content, "WAAM.pdf").startswith("Full length article")

    def test_falls_back_to_filename(self):
        assert extract_document_title("", "some_paper.pdf") == "some paper"


class TestPdfFrontMatter:
    """前言字段拆到 metadata，并从正文中删除。"""

    SAMPLE = """<!-- image -->

## Journal of Testing

ISSN: 1234-5678 Journal homepage: www.example.org

## A Study of Wire and Arc Additive Manufacturing Defects

## A Study of Wire and Arc Additive Manufacturing Defects

Mengru Liu a, Xingwang Bai b

a School of Mechanical Engineering, Example University, City, China

## ABSTRACT

This paper studies defects in wire and arc additive manufacturing.

## 1 Introduction

First introduction paragraph.

## ARTICLE HISTORY

Received 8 August 2023

## KEYWORDS

Wire arc; defect detection; surface curvature

Second introduction paragraph continues here.

## 2 Method

Body.
"""

    TITLE = "A Study of Wire and Arc Additive Manufacturing Defects"

    def test_extracts_kv_fields(self):
        _, metadata = extract_front_matter(self.SAMPLE, self.TITLE)
        assert metadata["journal"] == "Journal of Testing"
        assert metadata["authors"] == "Mengru Liu a, Xingwang Bai b"
        assert "Example University" in metadata["affiliations"]
        assert metadata["keywords"] == [
            "Wire arc",
            "defect detection",
            "surface curvature",
        ]
        assert metadata["article_history"] == "Received 8 August 2023"

    def test_removes_front_matter_from_content(self):
        content, _ = extract_front_matter(self.SAMPLE, self.TITLE)
        for removed in (
            "Journal of Testing",
            "ISSN",
            "Mengru Liu a, Xingwang Bai b",
            "Example University",
            "ARTICLE HISTORY",
            "KEYWORDS",
        ):
            assert removed not in content
        # Introduction 的后半段不能被 Keywords 区块吞掉
        assert "First introduction paragraph." in content
        assert "Second introduction paragraph continues here." in content
        assert "## 2 Method" in content

    def test_keeps_abstract_in_content_only(self):
        content, metadata = extract_front_matter(self.SAMPLE, self.TITLE)
        assert "## ABSTRACT" in content
        assert "This paper studies defects" in content
        # 摘要只保留在正文里，不再重复写入 metadata
        assert "abstract" not in metadata

    def test_no_match_leaves_content_unchanged(self):
        content, metadata = extract_front_matter("## 1 Introduction\n\nBody text.", "Other")
        assert metadata == {}
        assert content == "## 1 Introduction\n\nBody text."

    def test_plain_text_fallback_is_untouched(self):
        text = "Abstract\n\nThis is not markdown.\n\nKeywords: a; b"
        content, metadata = extract_front_matter(text, "T")
        assert metadata == {}
        assert content == text


# ---------------------------------------------------------------- 牛客页面适配器

class TestNiukeAdapter:
    def test_text1_layout(self):
        content, metadata = read_niuke_document(str(TEXT1))
        assert metadata["title"].startswith("美团面经")
        assert metadata["company"] == "美团"
        assert [s["name"] for s in metadata["sections"]] == ["面经 01", "面经 02"]
        assert metadata["sections"][0]["position"] == "美团 AI Agent开发岗"
        assert metadata["sections"][0]["time"] == "8 月 23 日"  # adapter 不做归一化
        assert metadata["sections"][0]["question_count"] == 15
        assert metadata["sections"][1]["question_count"] == 22
        assert metadata["question_count"] == 37
        assert metadata["truncated"] is True

        assert content.startswith("# 美团面经")
        assert "## 面经 01" in content and "## 面经 02" in content
        assert "**面试岗位**：美团 AI Agent开发岗" in content
        assert content.count("\n- ") == 37

    def test_noise_is_excluded(self):
        content, _ = read_niuke_document(str(TEXT1))
        for noise in ("剩余60%内容", "订阅专栏", "相关推荐", "全站热榜", "更多真题", "一键发评"):
            assert noise not in content

    def test_synthetic_page(self):
        content, metadata = parse_niuke_text(
            "某公司面经测试\n"
            "面经 01\n"
            "面试岗位：后端 ｜ 面试时间：9 月 1 日\n"
            "问题甲\n"
            "问题乙\n"
            "剩余50%内容，订阅专栏后可继续查看\n"
            "相关推荐\n"
        )
        assert metadata["question_count"] == 2
        assert metadata["truncated"] is True
        assert metadata["sections"][0]["time"] == "9 月 1 日"  # adapter 不做归一化
        assert "剩余50%" not in content and "相关推荐" not in content
        assert "- 问题甲" in content and "- 问题乙" in content


# ---------------------------------------------------------------- 归一化

class TestNormalizeMarkdown:
    def test_fullwidth_digits_and_letters(self):
        assert normalize_markdown("ＡＩ１２３ ＲＡＧ") == "AI123 RAG"

    def test_keeps_english_case(self):
        assert normalize_markdown("RAG 与 MCP 和 Function Calling") == "RAG 与 MCP 和 Function Calling"

    def test_removes_space_around_chinese_units(self):
        assert normalize_markdown("8 月 23 日 12 时") == "8月23日12时"

    def test_collapses_inline_spaces(self):
        assert normalize_markdown("a   b\t\tc") == "a b c"

    def test_removes_zero_width_and_nbsp(self):
        text = "a" + chr(0x200B) + "b" + chr(0xFEFF) + "c" + chr(0x00A0) + "d"
        assert normalize_markdown(text) == "abc d"

    def test_removes_pdfium_unmapped_placeholder(self):
        text = "in" + chr(0xFFFE) + "spection"
        assert normalize_markdown(text) == "inspection"

    def test_code_fence_untouched(self):
        source = "```python\nx  =  1  # Ａ１２\n```"
        assert normalize_markdown(source) == source

    def test_blank_lines_collapsed(self):
        assert normalize_markdown("a\n\n\n\n\nb") == "a\n\nb"

    def test_preserves_indentation(self):
        assert normalize_markdown("  - item") == "  - item"


class TestNormalizeMetadata:
    def test_normalizes_string_values(self):
        metadata = normalize_metadata({"time": "8 月 23 日", "items": [{"x": "Ａ１"}]})
        assert metadata["time"] == "8月23日"
        assert metadata["items"][0]["x"] == "A1"

    def test_none(self):
        assert normalize_metadata(None) == {}


class TestDocument:
    def test_shape(self):
        document = Document(content="# 标题\n正文", metadata={"source": "a.md", "title": "标题"})
        assert document["title"] == "标题"
        assert document["content"] == "# 标题\n正文"
        assert document["content_type"] == "text/markdown"
        assert len(document["id"]) == 16

    def test_id_is_stable_and_content_sensitive(self):
        first = Document(content="同一内容", metadata={"source": "a.md"})
        second = Document(content="同一内容", metadata={"source": "a.md"})
        other = Document(content="不同内容", metadata={"source": "a.md"})
        assert first["id"] == second["id"]
        assert first["id"] != other["id"]


# ---------------------------------------------------------------- 总接口与导出

class TestAdapterInterface:
    def test_supported_types(self):
        assert set(Adapter.supported_types()) == {"markdown", "pdf", "niuke"}

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="不支持的类型"):
            Adapter().read_document(str(TEXT1), "unknown")

    def test_metadata_gets_adapter(self):
        _, metadata = Adapter().read_document(str(TEXT1), "niuke")
        assert metadata["adapter"] == "niuke"


class TestPipeline:
    def test_run_composes_adapter_and_norm(self):
        document = Pipeline().run(str(TEXT1), "niuke")
        assert document["metadata"]["adapter"] == "niuke"

    def test_run_applies_norm(self):
        document = Pipeline().run(str(TEXT1), "niuke")
        assert "8月23日" in document["content"]
        assert "8 月 23 日" not in document["content"]
        assert document["metadata"]["sections"][0]["time"] == "8月23日"


class TestJsonExporter:
    def test_export_writes_json(self, tmp_path):
        target = tmp_path / "out" / "text1.json"
        document = export(str(TEXT1), "niuke", str(target))
        assert target.exists()
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload == document
        assert payload["metadata"]["question_count"] == 37
        assert payload["content_type"] == "text/markdown"

    def test_export_json_round_trips(self):
        payload = json.loads(export_json(str(TEXT3), "markdown"))
        assert payload["title"]
        assert payload["metadata"]["adapter"] == "markdown"

    def test_dumps_round_trips(self):
        document = Pipeline().run(str(TEXT1), "niuke")
        assert json.loads(JsonExporter.dumps(document)) == document
