"""端到端测试：本地 HTTP 服务器 + 真实 Chromium，子进程完整运行爬虫 CLI。

覆盖：页面渲染、懒加载滚动、异步填充、空正文/拦截页告警、
stdout/--out/--quiet、错误路径。
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
TMP = ROOT / ".tmp"

FAST = ["--scrolls", "2", "--pause", "300"]  # 夹具页足够，缩短测试时间


@pytest.fixture(scope="module")
def base_url():
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(FIXTURES), **kwargs)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def run_cli(*args, timeout=180):
    TMP.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["TMP"] = str(TMP)
    env["TEMP"] = str(TMP)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "utils" / "crawler.py"), *[str(a) for a in args]],
        cwd=str(ROOT), capture_output=True, timeout=timeout, env=env,
        encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


# ------------------------------------------------------------------ 正常抓取

def test_default_text_output(base_url):
    code, out, err = run_cli(f"{base_url}/article.html", *FAST)
    assert code == 0
    for marker in ("SENTENCE_ONE", "SENTENCE_TWO", "SENTENCE_THREE", "SENTENCE_FOUR"):
        assert marker in out, f"{marker} 缺失:\n{out[:800]}"
    assert "TABLE_VALUE_1" in out          # 默认保留表格
    assert "https://example.org/target" in out  # 默认保留链接
    assert "\\_" not in out                # markdown 转义已还原
    for stage in ("[1/3]", "[2/3]", "[3/3]"):
        assert stage in err


def test_positional_url_equivalent(base_url):
    code, out, _ = run_cli(f"{base_url}/article.html", *FAST)
    code2, out2, _ = run_cli("--url", f"{base_url}/article.html", *FAST)
    assert code == code2 == 0
    assert out == out2


def test_out_writes_file(base_url):
    target = TMP / "e2e_out.txt"
    if target.exists():
        target.unlink()
    code, out, err = run_cli(f"{base_url}/article.html", "--out", target, *FAST)
    assert code == 0
    assert out == ""  # 正文只进文件
    body = target.read_text(encoding="utf-8")
    assert "SENTENCE_ONE" in body and "SENTENCE_FOUR" in body
    assert "已写入" in err


def test_quiet_suppresses_diagnostics(base_url):
    code, out, err = run_cli(f"{base_url}/article.html", "--quiet", *FAST)
    assert code == 0
    assert "[1/3]" not in err and "[3/3]" not in err
    assert "SENTENCE_ONE" in out  # 正文仍走 stdout


# ------------------------------------------------------------------ 动态行为

def test_lazy_load_scroll(base_url):
    code, out, err = run_cli(
        f"{base_url}/lazy.html", "--scrolls", "6", "--pause", "500")
    assert code == 0, err
    assert "LAZY_BASE" in out
    for i in range(1, 7):
        assert f"LAZY_ITEM_{i}" in out, f"懒加载第 {i} 段没拿到"


def test_lazy_load_without_scroll(base_url):
    # --scrolls 0：完全不滚动，不会触发依赖 scroll 事件的懒加载
    code, out, _ = run_cli(f"{base_url}/lazy.html", "--scrolls", "0", "--pause", "300")
    assert code == 0
    assert "LAZY_BASE" in out
    assert "LAZY_ITEM_1" not in out


def test_late_content_keeps_waiting(base_url):
    # 文档高度不变，只有文本长度在 600ms 后变化：验证签名不只看高度
    code, out, err = run_cli(
        f"{base_url}/late.html", "--scrolls", "1", "--pause", "900")
    assert code == 0, err
    assert "LATE_BASE" in out
    assert "LATE_ITEM" in out
    assert "内容未完全稳定" not in err


# ------------------------------------------------------------------ 异常页面

def test_blocked_page_warns_but_succeeds(base_url):
    code, out, err = run_cli(f"{base_url}/blocked.html", *FAST)
    assert code == 0
    assert "安全验证" in out
    assert "正文只有" in err


def test_empty_page_returns_one(base_url):
    code, out, err = run_cli(f"{base_url}/empty.html", *FAST)
    assert code == 1
    assert out == ""
    assert "正文为空" in err


# ------------------------------------------------------------------ 错误路径

def test_connection_refused():
    code, _, err = run_cli("http://127.0.0.1:1/", "--timeout", "5000", *FAST)
    assert code != 0
    assert "浏览器访问失败" in err or "页面加载超时" in err


def test_missing_url():
    code, _, err = run_cli()
    assert code == 2
    assert "必须提供 URL" in err


# ------------------------------------------------------------------ 真实网络

@pytest.mark.network
def test_example_com_smoke():
    code, out, err = run_cli("https://example.com/", "--timeout", "30000",
                             "--scrolls", "1", "--pause", "200")
    assert code == 0, err
    assert "Example Domain" in out
