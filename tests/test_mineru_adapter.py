"""MinerU 适配器的纯函数与 zip 后处理测试（不调用 CLI）。"""

from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path

import pytest

from utils.adapters.docs.mineru import (
    _block_text,
    _caption_from_markdown,
    _collect_captions,
    _count_blocks,
    _export_images,
    _pick_caption,
    _pid_alive,
    _temp_workspace,
    cleanup_stale_workspaces,
)

MARKDOWN = """# Demo paper

![](images/page_0_image_1.jpg)

Fig. 1. Demo figure.

<img src="images/page_1_chart_2.png">

<small><span class="docvortex-page-footnote" data-block-type="page_footnote">footnote</span></small>

Body text.
"""

MIDDLE_JSON = {
    "pages": [
        {
            "page_idx": 0,
            "blocks": [
                {
                    "type": "image",
                    "index": 1,
                    "content": [
                        {
                            "type": "image_body",
                            "index": 1,
                            "image_path": "images/page_0_image_1.jpg",
                            "content": "",
                        },
                        {
                            "type": "image_caption",
                            "index": 2,
                            "content": [{"type": "text", "content": "Fig. 1. Demo figure."}],
                        },
                    ],
                },
                {"type": "text", "index": 3, "content": "Body text."},
                {"type": "table", "index": 4, "content": [{"type": "table_body", "content": ""}]},
            ],
        },
        {
            "page_idx": 1,
            "blocks": [
                {
                    "type": "chart",
                    "index": 2,
                    "content": [
                        {
                            "type": "chart_body",
                            "index": 2,
                            "image_path": "images/page_1_chart_2.png",
                            "content": "",
                        },
                        {
                            "type": "chart_caption",
                            "index": 3,
                            "content": [
                                {"type": "text", "content": "(b)"},
                                {"type": "text", "content": "Fig. 2. Demo chart."},
                            ],
                        },
                    ],
                }
            ],
        },
    ]
}


def make_archive(tmp_path: Path) -> zipfile.ZipFile:
    path = tmp_path / "out.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("markdown.md", MARKDOWN)
        archive.writestr("middle_json.json", json.dumps(MIDDLE_JSON))
        archive.writestr("images/page_0_image_1.jpg", b"jpg-0")
        archive.writestr("images/page_1_chart_2.png", b"png-1")
        archive.writestr("images/page_9_equation_3.jpg", b"unused")
    return zipfile.ZipFile(path)


class TestBlockHelpers:
    def test_block_text_reads_nested_caption(self):
        block = MIDDLE_JSON["pages"][0]["blocks"][0]["content"][1]
        assert _block_text(block) == "Fig. 1. Demo figure."

    def test_pick_caption_prefers_figure_label(self):
        assert _pick_caption(["(b)", "Fig. 4. Hand-eye calibration."]) == "Fig. 4. Hand-eye calibration."

    def test_pick_caption_falls_back_to_longest(self):
        assert _pick_caption(["", "(b)"]) == "(b)"

    def test_count_blocks(self):
        # 正文 1 条 + 图片 caption 1 条 + 图表 caption 2 条
        assert _count_blocks(MIDDLE_JSON) == (2, 4, 1)

    def test_collect_captions_includes_charts(self):
        captions = _collect_captions(MIDDLE_JSON)
        assert captions["images/page_0_image_1.jpg"] == "Fig. 1. Demo figure."
        assert captions["images/page_1_chart_2.png"] == "(b) Fig. 2. Demo chart."

    def test_caption_from_markdown_neighbor(self):
        lines = ["![](images/a.jpg)", "", "Fig. 3. Nearby caption.", "text"]
        assert _caption_from_markdown(lines, 0) == "Fig. 3. Nearby caption."

    def test_caption_from_markdown_ignores_body(self):
        lines = ["![](images/a.jpg)", "", "ordinary paragraph"]
        assert _caption_from_markdown(lines, 0) == ""


class TestWorkspaceCleanup:
    """临时目录的清理：正常/异常退出、以及被硬中断后的下次扫描。"""

    def test_removed_on_normal_exit(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        with _temp_workspace() as root:
            (root / "docvortex-render-x" / "page-1.pixels").parent.mkdir()
            (root / "docvortex-render-x" / "page-1.pixels").write_bytes(b"1")
            assert root.is_dir()
        assert not root.exists()

    def test_removed_when_body_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        with pytest.raises(RuntimeError):
            with _temp_workspace() as root:
                raise RuntimeError("boom")
        assert not root.exists()

    def test_owner_marker_written(self, monkeypatch, tmp_path):
        monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
        with _temp_workspace() as root:
            assert (root / ".owner").read_text() == str(os.getpid())

    def test_sweep_removes_dead_owner(self, tmp_path):
        stale = tmp_path / "mineru_parse_dead"
        stale.mkdir()
        (stale / ".owner").write_text("999999")
        assert cleanup_stale_workspaces(tmp_path) == 1
        assert not stale.exists()

    def test_sweep_keeps_live_owner(self, tmp_path):
        live = tmp_path / "mineru_parse_live"
        live.mkdir()
        (live / ".owner").write_text(str(os.getpid()))
        assert cleanup_stale_workspaces(tmp_path) == 0
        assert live.is_dir()

    def test_sweep_keeps_fresh_unmarked_dir(self, tmp_path):
        fresh = tmp_path / "mineru_parse_fresh"
        fresh.mkdir()
        assert cleanup_stale_workspaces(tmp_path) == 0
        assert fresh.is_dir()

    def test_sweep_removes_old_unmarked_dir(self, tmp_path):
        old = tmp_path / "mineru_parse_old"
        old.mkdir()
        stale_time = time.time() - 3600
        os.utime(old, (stale_time, stale_time))
        assert cleanup_stale_workspaces(tmp_path) == 1
        assert not old.exists()

    def test_sweep_ignores_unrelated_dirs(self, tmp_path):
        other = tmp_path / "other_dir"
        other.mkdir()
        assert cleanup_stale_workspaces(tmp_path) == 0
        assert other.is_dir()

    def test_pid_alive_detects_self_and_missing(self):
        assert _pid_alive(os.getpid()) is True
        assert _pid_alive(999999) is False


class TestExportImages:
    def test_writes_assets_and_strips_base64_style_refs(self, tmp_path):
        with make_archive(tmp_path) as archive:
            markdown, assets, count = _export_images(
                archive,
                MARKDOWN,
                _collect_captions(MIDDLE_JSON),
                asset_dir=tmp_path / "assets",
                digest="abc",
                include_images=True,
                max_images=None,
            )

        assert count == 2
        assert len(assets) == 2
        assert "[IMG]" in markdown and "](images/" not in markdown
        assert "docvortex-page-footnote" not in markdown
        # 只导出正文引用过的图片，equation 裁图不会落盘
        assert len(list((tmp_path / "assets").iterdir())) == 2
        assert assets[0]["caption"] == "Fig. 1. Demo figure."
        assert assets[0]["page"] == 1
        assert assets[1]["page"] == 2
        assert Path(assets[1]["path"]).is_file()

    def test_max_images_keeps_placeholder_for_skipped(self, tmp_path):
        with make_archive(tmp_path) as archive:
            markdown, assets, count = _export_images(
                archive,
                MARKDOWN,
                {},
                asset_dir=tmp_path / "assets",
                digest="abc",
                include_images=True,
                max_images=1,
            )

        assert count == 2
        assert len(assets) == 1
        assert markdown.count("[IMG]") == 2
        assert len(list((tmp_path / "assets").iterdir())) == 1

    def test_include_images_false_writes_nothing(self, tmp_path):
        with make_archive(tmp_path) as archive:
            markdown, assets, count = _export_images(
                archive,
                MARKDOWN,
                {},
                asset_dir=tmp_path / "assets",
                digest="abc",
                include_images=False,
                max_images=None,
            )

        assert count == 2
        assert assets == []
        assert not (tmp_path / "assets").exists()
        assert markdown.count("[IMG]") == 2

    def test_duplicate_reference_registered_once(self, tmp_path):
        markdown = "![](images/page_0_image_1.jpg)\n\n![](images/page_0_image_1.jpg)"
        with make_archive(tmp_path) as archive:
            result, assets, count = _export_images(
                archive,
                markdown,
                {},
                asset_dir=tmp_path / "assets",
                digest="abc",
                include_images=True,
                max_images=None,
            )

        assert count == 1
        assert len(assets) == 1
        assert result.count("[IMG]") == 2
