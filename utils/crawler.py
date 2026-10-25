"""抓取动态网页，等页面真正渲染完再提取正文。

抓不全的根源通常是“取快照的时机”：
1. 页面还没渲染完 / 懒加载没触发 -> 等网络安静，滚动触发懒加载，
   再用内容签名（文档高度 + 可见文本长度）判定页面是否稳定。
2. trafilatura 默认偏向干净正文，会丢掉列表、表格和链接 -> 用召回优先 + markdown；
   结果为空或明显少于页面可见文本时，退回剪枝后的 DOM 文本。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

import trafilatura
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# 内容签名：文档高度 + 可见文本长度。只看高度会漏掉“固定高度容器里异步填字”，
# 只看文本长度会漏掉追加的空块和图片占位。
SIG_JS = """() => [
  document.documentElement.scrollHeight,
  document.body ? document.body.innerText.length : 0,
]"""

# instant 避免页面的 scroll-behavior: smooth 让滚动还没到底就开始采样
SCROLL_JS = (
    "window.scrollTo({top: document.documentElement.scrollHeight, behavior: 'instant'})"
)

# 去掉结构性噪音后再取可见文本，作为 trafilatura 之外的兜底数据源
PRUNE_JS = """
() => {
  const drop = ['script','style','noscript','template','svg','iframe','canvas',
                'header','nav','footer','aside','form','button','select','option'];
  const root = document.querySelector('main, article, [role=main]') || document.body;
  if (!root) return '';
  const clone = root.cloneNode(true);
  clone.querySelectorAll(drop.join(',')).forEach(n => n.remove());
  clone.querySelectorAll('[role=navigation],[role=banner],[role=contentinfo],[aria-hidden=true]')
       .forEach(n => n.remove());
  clone.querySelectorAll('[class*=ad-],[class*=ads],[class*=cookie],[class*=banner],[class*=sidebar]')
       .forEach(n => n.remove());
  return clone.innerText || clone.textContent || '';
}
"""

NETWORK_IDLE_MS = 5000   # 网络空闲只是附加等待：长轮询页面可能永远等不到
STABLE_ROUNDS = 2        # 连续几轮签名不变才算稳定
GRACE_ROUNDS = 3         # 滚动预算用完后，再观察几轮
GRACE_MS = 800           # 宽限期单轮最短等待
DOM_RATIO = 0.5          # trafilatura 结果少于 DOM 文本的这个比例时改用 DOM
THIN_TEXT = 200          # 少于这个字符数时告警


def configure_stdio() -> None:
    """Windows 控制台/重定向默认用 GBK，正文里的 nbsp、生僻字会抛
    UnicodeEncodeError 导致输出为空，所以强制 stdout/stderr 走 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def log(msg: str, quiet: bool = False) -> None:
    """诊断信息走 stderr，正文走 stdout，方便重定向到文件。"""
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


def clean_text(text: str) -> str:
    """统一换行、清理不可见字符、去掉行尾空格和多余空行。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    for src, dst in (("\u00a0", " "), ("\u3000", " "), ("\u200b", ""), ("\ufeff", "")):
        text = text.replace(src, dst)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def use_fallback_temp(quiet: bool = False) -> None:
    """把 TMP/TEMP 指到项目内的 .tmp。

    Playwright 的 Node 驱动直接用系统 TEMP 建目录（不像 Python 的 tempfile
    会自动回退到当前目录），所以必须在启动驱动之前改好环境变量。
    """
    Base_dir = Path(__file__).resolve().parent
    fallback = Base_dir.parent / ".tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = str(fallback)
    tempfile.tempdir = None  # 让 Python 侧也重新解析
    log(f"系统临时目录不可写，改用 {fallback}", quiet)


async def content_signature(page) -> tuple[int, int]:
    """页面当前的内容指纹：(文档高度, 可见文本长度)。"""
    height, length = await page.evaluate(SIG_JS)
    return height, length


async def wait_for_content(page, max_scrolls: int, pause_ms: int) -> bool:
    """滚动触发懒加载，直到内容签名连续 STABLE_ROUNDS 轮不变。

    滚动次数用完后仍不稳定时，不立刻取快照，而是再观察 GRACE_ROUNDS 轮
    （不滚动），给异步请求和 setTimeout 渲染留出时间。
    返回 True 表示已稳定；False 表示宽限期结束仍在变化，调用方按当前快照抓取。
    """
    pause_ms = max(0, pause_ms)
    previous = await content_signature(page)  # 基线样本，本身不算“稳定了一轮”
    unchanged = 0

    async def sample(wait_ms: int) -> bool:
        nonlocal previous, unchanged
        await page.wait_for_timeout(wait_ms)
        current = await content_signature(page)
        unchanged = unchanged + 1 if current == previous else 0
        previous = current
        return unchanged >= STABLE_ROUNDS

    for _ in range(max(0, max_scrolls)):
        await page.evaluate(SCROLL_JS)
        if await sample(pause_ms):
            return True

    for _ in range(GRACE_ROUNDS):
        if await sample(max(pause_ms, GRACE_MS)):
            return True
    return False


def extract_text(html: str, url: str, dom_text: str) -> tuple[str, str]:
    """召回优先提取；结果为空或明显短于 DOM 文本时退回 DOM。

    返回 (正文, 来源)，来源用于诊断日志。
    """
    text = clean_text(
        trafilatura.extract(
            html, url=url, favor_recall=True,
            include_tables=True, include_links=True,
            output_format="markdown",
        ) or ""
    ).replace("\\_", "_")  # markdown 会转义下划线，正文里不该出现反斜杠

    if dom_text and (not text or len(text) < len(dom_text) * DOM_RATIO):
        return clean_text(dom_text), "dom"
    return text, "trafilatura"


async def _render(url: str, args) -> tuple[str, str, bool]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=not args.headed,
            # 去掉 --enable-automation，减少最基础的自动化特征
            ignore_default_args=["--enable-automation"],
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            context = await browser.new_context(
                user_agent=DEFAULT_UA,
                locale="zh-CN",
                timezone_id="Asia/Shanghai",
                viewport={"width": 1440, "height": 900},
                extra_http_headers={
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Upgrade-Insecure-Requests": "1",
                },
            )
            await context.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
                "window.chrome = window.chrome || {runtime: {}};"
            )
            page = await context.new_page()

            log(f"[1/3] 访问: {url}", args.quiet)
            await page.goto(url, wait_until="domcontentloaded", timeout=args.timeout)
            try:
                await page.wait_for_load_state(
                    "networkidle", timeout=min(args.timeout, NETWORK_IDLE_MS)
                )
            except PlaywrightTimeoutError:
                log("      网络未安静，改用内容稳定性判断", args.quiet)

            log(f"[2/3] 等待渲染（最多滚动 {max(0, args.scrolls)} 次）", args.quiet)
            stable = await wait_for_content(page, args.scrolls, args.pause)
            html = await page.content()
            dom_text = clean_text(await page.evaluate(PRUNE_JS))
            return html, dom_text, stable
        finally:
            await browser.close()


async def render(url: str, args) -> tuple[str, str, bool]:
    """渲染页面，返回 (HTML, 剪枝后的可见文本, 内容是否已稳定)。

    系统临时目录不可写时（Playwright 报 EPERM: mkdtemp），改用项目内的
    .tmp 并重试一次——必须在 Playwright 驱动启动前改环境变量。
    """
    for attempt in (1, 2):
        try:
            return await _render(url, args)
        except PlaywrightTimeoutError:
            raise SystemExit(f"页面加载超时（{args.timeout}ms）：{url}")
        except PlaywrightError as exc:
            if attempt == 1 and "mkdtemp" in str(exc):
                use_fallback_temp(args.quiet)
                continue
            raise SystemExit(f"浏览器访问失败：{exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="渲染动态网页并提取尽可能完整的正文",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("positional_url", nargs="?", help="要抓取的网页 URL")
    parser.add_argument("--url", dest="url_flag", help="要抓取的网页 URL（与位置参数等价）")
    parser.add_argument("--out", help="把正文写入文件（默认打印到 stdout）")
    parser.add_argument("--headed", action="store_true", help="显示浏览器窗口，便于排查")
    parser.add_argument("--timeout", type=int, default=45000, help="页面加载超时（毫秒）")
    parser.add_argument("--scrolls", type=int, default=6,
                        help="最多滚动几次以触发懒加载；0 表示不滚动")
    parser.add_argument("--pause", type=int, default=1200, help="每轮滚动后的等待（毫秒）")
    parser.add_argument("--quiet", action="store_true", help="不打印诊断信息")
    return parser


def crawler() -> int:
    configure_stdio()
    parser = build_parser()
    args = parser.parse_args()
    url = args.url_flag or args.positional_url
    if not url:
        parser.error("必须提供 URL，例如：python crawler.py --url https://example.com")
    args.scrolls = max(0, args.scrolls)
    args.pause = max(0, args.pause)

    html, dom_text, stable = asyncio.run(render(url, args))
    text, source = extract_text(html, url, dom_text)

    log(
        f"[3/3] 完成: HTML {len(html)} 字符 | 可见文本 {len(dom_text)} | "
        f"提取 {len(text)} | 来源 {source}"
        + ("" if stable else " | 内容未完全稳定（已按当前快照提取）"),
        args.quiet,
    )

    if not text:
        log(
            "正文为空：页面可能没渲染完、需要登录或被反爬拦截。\n"
            "  - 加 --headed 看浏览器里到底显示了什么\n"
            "  - 加大 --pause / --scrolls 再试",
            args.quiet,
        )
        return 1

    if len(text) < THIN_TEXT:
        log(f"⚠ 正文只有 {len(text)} 字符：可能抓到了拦截页或残缺页面。", args.quiet)

    if args.out:
        
        try:
            parent_dir=Path(__file__).resolve().parent.parent
            target_file=parent_dir/args.out
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.write_text(text, encoding="utf-8")
        except OSError as exc:
            log(f"写入失败：{exc}", args.quiet)
            return 1
        log(f"已写入 {args.out}（{len(text)} 字符）", args.quiet)
        return 0

    try:
        print(text)
    except UnicodeEncodeError:
        # 极端情况下控制台编码仍不可写，退回转义输出，别让正文凭空消失
        print(text.encode("utf-8", "replace").decode("ascii", "backslashreplace"))
    return 0


if __name__ == "__main__":
    sys.exit(crawler())
