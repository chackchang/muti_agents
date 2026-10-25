# agents

用 Playwright 渲染动态网页，等页面内容真正稳定后，再用 trafilatura 提取尽可能完整的正文。

## 安装

```powershell
uv sync                                  # 安装 Python 依赖（含 playwright）
uv run playwright install chromium       # 下载浏览器内核，只做一次
```

> 若不用 uv：`pip install playwright trafilatura` + `playwright install chromium`。

## 用法

```powershell
uv run python -m utils.crawler --url https://example.com
uv run python -m utils.crawler https://example.com --out article.md   # 位置参数写法也可以
uv run python -m utils.crawler https://example.com --headed           # 打开窗口排查
uv run python -m utils.crawler https://example.com --scrolls 12 --pause 2000
```

正文走 stdout，诊断信息走 stderr，所以 `... > out.txt` 得到的文件只有正文。

## 参数

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--url` / 位置参数 | — | 要抓取的网页 URL |
| `--out` | 无 | 把正文写入文件（默认打印到 stdout） |
| `--headed` | 关 | 显示浏览器窗口，便于排查 |
| `--timeout` | `45000` | 页面加载超时（毫秒） |
| `--scrolls` | `6` | 最多滚动几次以触发懒加载；`0` 表示不滚动 |
| `--pause` | `1200` | 每轮滚动后的等待（毫秒） |
| `--quiet` | 关 | 不打印诊断信息 |

返回码：`0` 成功；`1` 正文为空或写入失败；`2` 参数错误。

## 清洗管线

抓到的网页文本、markdown 文档和 PDF 可以继续清洗为统一的 JSON：

```
输入文件
  → adapters/      按来源提取为 markdown（docs/ 文档，web/ 网页）
  → norm/          数字/空白归一化，封装为 Document
  → exporters/json.py  Document → JSON

pipeline.py 只负责编排上面两层：adapter -> norm。
adapters、norm、exporters 之间互不依赖，编排集中在 pipeline.py。
```

| 适配器 | 输入 | 说明 |
| --- | --- | --- |
| `markdown` | `.md` 等文本 | 保留标题、表格、代码块，只清理格式 |
| `pdf` | `.pdf` | 默认 Docling，可切换 MinerU；缺依赖时用 PDFium 兜底 |
| `niuke` | 牛客面经页面文本 | 当前针对 `data/text1` 的布局，遇到付费墙/评论/推荐即停止 |

> **Docling 首次运行**：会下载布局/表格模型到 HuggingFace 缓存（约 500MB）。
> 默认缓存目录不可写时会自动改用项目内 `.tmp/hf`。如果 `huggingface.co`
> 无法直连（例如系统代理中断 TLS），先设置镜像并绕过代理：
>
> ```powershell
> $env:HF_ENDPOINT = "https://hf-mirror.com"
> $env:NO_PROXY = "hf-mirror.com"
> $env:HF_HUB_DISABLE_XET = "1"
> ```
>
> 模型缓存后再次运行会直接复用，不再下载（项目内 `.tmp/hf` 已有模型时会自动优先使用）。
> 缓存约 500MB 且已被 gitignore；**删除 `.tmp` 会连带删除模型缓存**，下次需要重新下载，
> 否则 Docling 会自动回退到 PDFium（命令仍能输出 JSON）。

> **MinerU（可选，公式更完整）**：MinerU 4 要求 `openai<3`，与主项目冲突，
> 因此装在独立环境 `.venv-mineru`，由 `utils/adapters/docs/mineru.py` 以子进程调用。
> 把 `agent/config.py` 里的 `PDFExtractionConfig.mineru` 设为 `True` 即可切换：
> 解析使用 `mineru-kit parse --format zip`，图片单独写到 `documents/.vision_assets_2/`，
> 正文里的图片引用替换为 `[IMG]`。首次运行下载约 2GB 模型到 `.tmp/mineru`，
> 之后 20 页论文约 4–6 分钟（CPU）。

命令行（`json.py` 就是入口）：

```powershell
uv run python -m utils.exporters.json data/text1 niuke --out out/text1.json
uv run python -m utils.exporters.json data/text3 markdown      # 不给 --out 时打印到 stdout
```

作为工具调用：

```python
from utils.exporters.json import Pipeline, export, export_json

document = export("data/text1", "niuke", "out/text1.json")  # 返回 Document，并写文件
payload = export_json("data/text1", "niuke")                # 直接返回 JSON 字符串
document = Pipeline().run("data/text1", "niuke")            # 只要 Document，不写文件
```

输出结构：

```json
{
  "id": "469074099cbc3998",
  "title": "美团面经 美团AI Agent开发岗面经-03",
  "content": "# 美团面经 ...",
  "content_type": "text/markdown",
  "metadata": { "question_count": 37, "...": "..." }
}
```

`id` 由 `sha256(source + content)` 前 16 位生成，同样输入得到同样的 id。

归一化是保守的：全角数字/字母转半角、全角空格转普通空格、去掉零宽字符、
数字与中文单位之间去空格（`8 月 23 日` → `8月23日`）、合并多余空格和空行；
**英文大小写保持原样**，避免破坏 RAG、MCP、AI 这类缩写；代码块内容不做改写。

## 怎么解决“抓不全”

1. **等页面真正渲染完，而不是等一个固定时间。**
   `domcontentloaded` 之后，先用最多 5 秒等网络空闲（超时不算失败，长轮询页面
   本来就不会空闲）。随后滚动到底触发懒加载，并持续采样**内容签名**：

   ```
   (document.documentElement.scrollHeight, document.body.innerText.length)
   ```

   只有高度和文本长度**连续两轮都不变**才取快照。这样既能发现追加段落，
   也能发现“文档高度不变、容器里异步填字”的 SPA。
   滚动次数用完后仍不稳定时，会再观察 3 轮（每轮至少 800ms）才放弃；
   最终仍不稳定就按当前快照提取，并在 stderr 标注。

2. **召回优先提取。** `trafilatura.extract(favor_recall=True)` + markdown，
   保留列表、表格和链接。结果为空或少于页面可见文本的一半时，退回剪枝后的
   DOM 可见文本（去掉 nav/header/footer/aside、脚本、表单、广告和侧边栏）。

3. **失败时给出可操作信息。** stderr 会打印 HTML 长度、可见文本长度、采用的
   数据源；正文为空或少于 200 字符时提示加 `--headed`、加大 `--pause/--scrolls`。

## 已知限制

- 页面内部独立滚动容器（不是文档滚动）里的懒加载内容抓不到；当前只滚动文档。
- 异步内容晚于 `2 × pause` 才出现时，仍可能被提前判定为稳定；慢站点请加大
  `--pause`。持续变化的页面（计时器、轮播）会在宽限期后按快照抓取。
- 知乎、微博、小红书等反爬站点返回验证页时拿不到正文，只能用 `--headed`
  确认浏览器里显示了什么。需要登录的页面目前不支持复用登录态。
- Playwright 需要命名管道和临时目录。系统临时目录不可写时（`EPERM ... mkdtemp`），
  脚本会自动把 `TMP/TEMP` 指到项目内的 `.tmp` 并重试一次；若仍报 `WinError 5`，
  请在普通终端运行。
- 抓取请遵守目标站点的 robots.txt 与服务条款。

## 测试

```powershell
uv run pytest tests/test_unit.py -q          # 纯函数与 CLI 逻辑
uv run pytest tests/test_pipeline.py -q      # adapter → norm → JSON
uv run pytest tests/test_e2e.py -q -m "not network"   # 本地夹具 + 真实 Chromium
```
