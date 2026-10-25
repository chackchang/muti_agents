"""main.py 纯函数与 CLI 逻辑的单元测试（不启动浏览器）。"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import utils.crawler as crawler


# ---------------------------------------------------------------- clean_text

class TestCleanText:
    def test_crlf_and_cr_to_lf(self):
        assert crawler.clean_text("a\r\nb\rc\nd") == "a\nb\nc\nd"

    def test_invisible_chars(self):
        text = crawler.clean_text("x\u00a0y\u3000z\u200bw\ufeffv")
        assert "\u00a0" not in text and "\u3000" not in text
        assert "\u200b" not in text and "\ufeff" not in text
        #  /　 替换为空格；​/﻿ 直接删除
        assert text == "x y zwv"

    def test_trailing_spaces_stripped(self):
        assert crawler.clean_text("hello   \nworld  \n") == "hello\nworld"

    def test_collapse_blank_lines(self):
        assert crawler.clean_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_around_spaces_untouched(self):
        assert crawler.clean_text("  a  b  ") == "a  b"

    def test_empty(self):
        assert crawler.clean_text("") == ""


# -------------------------------------------------------- wait_for_content

class FakePage:
    """按顺序返回内容签名；滚动脚本单独计数。"""

    def __init__(self, signatures):
        self._signatures = list(signatures)
        self._index = 0
        self.scrolls = 0
        self.waits: list[int] = []

    async def evaluate(self, script):
        if script == crawler.SIG_JS:
            sig = self._signatures[min(self._index, len(self._signatures) - 1)]
            self._index += 1
            return list(sig)
        assert script == crawler.SCROLL_JS
        self.scrolls += 1
        return None

    async def wait_for_timeout(self, ms):
        self.waits.append(ms)


class TestWaitForContent:
    def _wait(self, page, scrolls, pause):
        return asyncio.run(crawler.wait_for_content(page, scrolls, pause))

    def test_stable_after_two_unchanged_rounds(self):
        page = FakePage([(100, 50)])
        assert self._wait(page, 2, 100) is True
        assert page.scrolls == 2

    def test_change_resets_counter(self):
        # 基线 -> 相同 -> 变化 -> 相同 -> 相同
        page = FakePage([(100, 50), (100, 50), (120, 80), (120, 80), (120, 80)])
        assert self._wait(page, 4, 100) is True
        assert page.scrolls == 4

    def test_never_stable_returns_false_after_grace(self):
        page = FakePage([(i, i) for i in range(10)])
        assert self._wait(page, 2, 10) is False
        # 滚动预算 2 轮 + 宽限期 3 轮；宽限期不滚动
        assert page.scrolls == 2
        assert len(page.waits) == 5

    def test_zero_scrolls_never_scrolls_but_can_settle(self):
        page = FakePage([(100, 50)])
        assert self._wait(page, 0, 100) is True
        assert page.scrolls == 0

    def test_grace_period_rescues_late_settle(self):
        page = FakePage([(100, 50), (100, 60), (120, 60), (120, 60), (120, 60)])
        assert self._wait(page, 2, 100) is True
        assert page.scrolls == 2

    def test_grace_waits_at_least_grace_ms(self):
        page = FakePage([(100, 50)])
        self._wait(page, 1, 100)
        assert page.waits[-1] >= crawler.GRACE_MS

    def test_negative_arguments_are_clamped(self):
        page = FakePage([(100, 50)])
        assert self._wait(page, -3, -5) is True
        assert page.scrolls == 0
        assert all(ms >= 0 for ms in page.waits)


# ------------------------------------------------------------ 临时目录回退

class TestFallbackTemp:
    def test_sets_env_and_creates_dir(self, monkeypatch, tmp_path, capsys):
        # crawler.py 位于 utils/ 下，父目录的父目录才是项目根
        monkeypatch.setattr(crawler, "__file__", str(tmp_path / "utils" / "crawler.py"))
        for key in ("TMPDIR", "TMP", "TEMP"):
            monkeypatch.setenv(key, "unused")
        crawler.use_fallback_temp()
        fallback = tmp_path / ".tmp"
        assert os.environ["TMP"] == str(fallback)
        assert os.environ["TEMP"] == str(fallback)
        assert fallback.is_dir()
        assert "系统临时目录不可写" in capsys.readouterr().err

    def test_quiet_suppresses_log(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(crawler, "__file__", str(tmp_path / "utils" / "crawler.py"))
        for key in ("TMPDIR", "TMP", "TEMP"):
            monkeypatch.setenv(key, "unused")
        crawler.use_fallback_temp(quiet=True)
        assert capsys.readouterr().err == ""


class TestRenderRetry:
    @staticmethod
    def _args():
        return SimpleNamespace(headed=False, timeout=1000, quiet=True, scrolls=0, pause=0)

    def test_retries_with_fallback_on_mkdtemp_error(self, monkeypatch):
        calls = []

        async def fake_render(url, args):
            calls.append("render")
            if len(calls) == 1:
                raise crawler.PlaywrightError(
                    "BrowserType.launch: EPERM: operation not permitted, "
                    "mkdtemp 'C:\\Users\\x\\Temp\\playwright-artifacts-XXX'"
                )
            return "html", "dom", True

        monkeypatch.setattr(crawler, "_render", fake_render)
        monkeypatch.setattr(
            crawler, "use_fallback_temp", lambda quiet=False: calls.append("fallback"))
        assert asyncio.run(crawler.render("http://x", self._args())) == ("html", "dom", True)
        assert calls == ["render", "fallback", "render"]

    def test_other_launch_errors_exit(self, monkeypatch):
        async def fake_render(url, args):
            raise crawler.PlaywrightError("boom")

        monkeypatch.setattr(crawler, "_render", fake_render)
        with pytest.raises(SystemExit) as exc:
            asyncio.run(crawler.render("http://x", self._args()))
        assert "浏览器访问失败" in str(exc.value)

    def test_timeout_exits(self, monkeypatch):
        async def fake_render(url, args):
            raise crawler.PlaywrightTimeoutError("slow")

        monkeypatch.setattr(crawler, "_render", fake_render)
        with pytest.raises(SystemExit) as exc:
            asyncio.run(crawler.render("http://x", self._args()))
        assert "页面加载超时" in str(exc.value)


# -------------------------------------------------------------- extract_text

ARTICLE_HTML = """
<html><head><title>测试文章</title></head><body>
<article>
<h1>测试文章标题</h1>
<p>第一段正文，包含标记 KEY_VALUE_1，以及足够多的文字让提取器认为这是主要内容部分。这里是更多的填充文字，用于满足最小长度要求。</p>
<p>第二段正文，还有一个链接 <a href="https://example.org/page">点我</a> 以及一个表格。</p>
<table><tr><th>项目</th><th>数量</th></tr><tr><td>苹果</td><td>3</td></tr></table>
<p>第三段结尾。再补充一些文字，确保正文足够长可以被识别为主内容而不是样板信息。</p>
</article>
</body></html>
"""

URL = "https://example.com"


class TestExtractText:
    def test_recall_keeps_body_table_and_link(self):
        text, source = crawler.extract_text(ARTICLE_HTML, URL, "")
        assert "第一段正文" in text
        assert "| 苹果 |" in text  # markdown 表格
        assert "[点我](https://example.org/page)" in text  # markdown 链接
        assert source == "trafilatura"

    def test_underscore_escape_restored(self):
        text, _ = crawler.extract_text(ARTICLE_HTML, URL, "")
        assert "KEY_VALUE_1" in text
        assert "\\_" not in text

    def test_empty_result_falls_back_to_dom(self):
        text, source = crawler.extract_text("", URL, "DOM 文本")
        assert (text, source) == ("DOM 文本", "dom")

    def test_short_result_falls_back_to_dom(self):
        dom = "D" * 10_000
        text, source = crawler.extract_text(ARTICLE_HTML, URL, dom)
        assert (text, source) == (dom, "dom")

    def test_ratio_boundary_keeps_trafilatura(self):
        base, _ = crawler.extract_text(ARTICLE_HTML, URL, "")
        dom = "D" * (len(base) * 2)  # len(base) == DOM_RATIO * len(dom)
        text, source = crawler.extract_text(ARTICLE_HTML, URL, dom)
        assert text == base and source == "trafilatura"

    def test_dom_fallback_is_cleaned(self):
        text, _ = crawler.extract_text("", URL, "a\u00a0b")
        assert text == "a b"

    def test_all_empty(self):
        assert crawler.extract_text("", URL, "")[0] == ""


# ------------------------------------------------------------------ 参数解析

class TestParser:
    def test_defaults(self):
        args = crawler.build_parser().parse_args(["http://x"])
        assert args.positional_url == "http://x"
        assert args.out is None
        assert args.timeout == 45000
        assert args.scrolls == 6
        assert args.pause == 1200
        assert args.headed is False
        assert args.quiet is False

    def test_all_flags(self):
        args = crawler.build_parser().parse_args([
            "--url", "http://y", "--out", "o.txt", "--headed",
            "--timeout", "1000", "--scrolls", "2", "--pause", "50", "--quiet",
        ])
        assert args.url_flag == "http://y"
        assert (args.out, args.timeout, args.scrolls, args.pause) == ("o.txt", 1000, 2, 50)
        assert args.headed is True and args.quiet is True

    @pytest.mark.parametrize("flag", [
        "--format", "--strategy", "--min-ratio", "--expand-more", "--save-html",
        "--no-tables", "--no-links", "--user-agent", "--storage-state", "--wait-login",
    ])
    def test_removed_flags_rejected(self, flag):
        with pytest.raises(SystemExit):
            crawler.build_parser().parse_args(["http://x", flag, "value"])


# ---------------------------------------------------------------- main() 输出

def workspace_tmp() -> Path:
    """沙箱禁写系统临时目录（创建链接会 WinError 5），改用工作区内目录。"""
    d = Path(__file__).resolve().parent.parent / ".tmp" / f"pytest-{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestMainOutput:
    def _run(self, monkeypatch, argv, *, text="第一行\n第二行", stable=True):
        calls = {}

        async def fake_render(url, args):
            calls["url"] = url
            calls["args"] = args
            return "<html>", "DOM", stable

        def fake_extract(html, url, dom_text):
            calls["dom"] = dom_text
            return text, "trafilatura"

        monkeypatch.setattr(crawler, "render", fake_render)
        monkeypatch.setattr(crawler, "extract_text", fake_extract)
        monkeypatch.setattr("sys.argv", argv)
        return crawler.crawler(), calls

    def test_text_format(self, monkeypatch, capsys):
        self._run(monkeypatch, ["crawler.py", "http://x"])
        assert capsys.readouterr().out == "第一行\n第二行\n"

    def test_out_writes_file(self, monkeypatch, capsys):
        out_file = workspace_tmp() / "out.txt"
        self._run(monkeypatch, ["crawler.py", "http://x", "--out", str(out_file)])
        assert out_file.read_text(encoding="utf-8") == "第一行\n第二行"
        assert capsys.readouterr().out == ""

    def test_flag_url_wins_over_positional(self, monkeypatch):
        _, calls = self._run(monkeypatch, ["crawler.py", "http://pos", "--url", "http://flag"])
        assert calls["url"] == "http://flag"

    def test_negative_scrolls_and_pause_clamped(self, monkeypatch):
        _, calls = self._run(
            monkeypatch, ["crawler.py", "http://x", "--scrolls", "-1", "--pause", "-5"])
        assert calls["args"].scrolls == 0 and calls["args"].pause == 0

    def test_no_url_exits_with_error(self, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["crawler.py"])
        with pytest.raises(SystemExit) as exc:
            crawler.crawler()
        assert exc.value.code == 2
        assert "必须提供 URL" in capsys.readouterr().err

    def test_empty_text_returns_one(self, monkeypatch, capsys):
        code, _ = self._run(monkeypatch, ["crawler.py", "http://x"], text="")
        captured = capsys.readouterr()
        assert code == 1
        assert captured.out == ""
        assert "正文为空" in captured.err

    def test_thin_text_warns_but_succeeds(self, monkeypatch, capsys):
        code, _ = self._run(monkeypatch, ["crawler.py", "http://x"], text="短")
        captured = capsys.readouterr()
        assert code == 0
        assert captured.out == "短\n"
        assert "正文只有" in captured.err

    def test_unstable_content_is_reported(self, monkeypatch, capsys):
        self._run(monkeypatch, ["crawler.py", "http://x"], text="长" * 300, stable=False)
        assert "内容未完全稳定" in capsys.readouterr().err

    def test_quiet_suppresses_out_log(self, monkeypatch, capsys):
        out_file = workspace_tmp() / "o.txt"
        self._run(monkeypatch, ["crawler.py", "http://x", "--out", str(out_file), "--quiet"])
        assert capsys.readouterr().err == ""

    def test_out_write_error_returns_one(self, monkeypatch, capsys):
        # 父路径已存在同名文件，mkdir 会失败
        blocker = workspace_tmp() / "blocker"
        blocker.write_text("x", encoding="utf-8")
        target = blocker / "o.txt"
        code, _ = self._run(
            monkeypatch, ["crawler.py", "http://x", "--out", str(target)], text="长" * 300)
        assert code == 1
        assert "写入失败" in capsys.readouterr().err


# ------------------------------------------------------------- log / stdio

class TestLogAndStdio:
    def test_log_writes_stderr(self, capsys):
        crawler.log("诊断")
        assert capsys.readouterr().err == "诊断\n"

    def test_log_quiet(self, capsys):
        crawler.log("诊断", quiet=True)
        assert capsys.readouterr().err == ""

    def test_configure_stdio_survives_bad_stream(self, monkeypatch):
        class NoReconfigure:
            pass
        monkeypatch.setattr(crawler.sys, "stdout", NoReconfigure())
        monkeypatch.setattr(crawler.sys, "stderr", NoReconfigure())
        crawler.configure_stdio()  # 不应抛出
