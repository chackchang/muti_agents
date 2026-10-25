# Ingest 运行记录与复现指南

本文记录一次完整 ingest 的实测结论，并回答三个问题：

1. 那次运行是 CPU 还是 GPU？谁在 CPU 上跑？为什么模型能在 CPU 上跑？
2. HF 模型下载的到底是「权重」还是「模型代码」？两者怎么触发？
3. 完整流程的 CLI 顺序是什么？怎么自己复现？检索结果怎么看？

---

## 一、CPU / GPU：那次是纯 CPU，而且你的 GPU 目前用不上

### 实测环境

| 项 | 值 |
|---|---|
| 显卡 | NVIDIA GeForce GTX 1650（4GB 显存），驱动 CUDA 12.6 |
| 项目内 torch | `2.14.1+cpu`（**CPU 版**） |
| `torch.cuda.is_available()` | `False` |

> 关键：**显卡存在，但装的是 CPU 版 torch**。`torch 2.14.1+cpu` 里的 `+cpu` 表示这个轮子没编译 CUDA，所以即使有显卡，`torch.cuda.is_available()` 也恒为 `False`。

因此 `.env` 里 `EMBEDDING_DEVICE=cuda` 现在**开了不但不会加速，反而会让 ingest 在编码阶段崩溃**（`model.to("cuda")` 报 `Torch not compiled with CUDA enabled`）。
**在装好 CUDA 版 torch 之前，先把 `.env` 的 `EMBEDDING_DEVICE=cuda` 改回注释或 `cpu`。**

### 那次 ingest 里，各环节跑在什么上

| 环节 | 跑在 | 说明 |
|---|---|---|
| MinerU PDF 解析 | CPU | 独立子进程（`.venv-mineru` 环境里的 `mineru-kit.exe`），自带 layout/OCR/table 模型，默认 CPU |
| GE-M3 嵌入编码 | CPU | 日志 `device=auto, fp16=False`：`auto` 检测不到 CUDA，退回 CPU |
| PostgreSQL / Qdrant | CPU | 跑在 Docker 容器里，Qdrant 的向量计算也是 CPU 算的 |

### 为什么模型能在 CPU 上跑

PyTorch 模型本质是一堆张量运算（矩阵乘法、注意力等），**CPU 和 GPU 都能算，只是速度不同**：

- GPU：几千个核心并行，矩阵乘法快几十倍；
- CPU：通用核心也能算，只是慢。

嵌入后端的 `device=auto` 逻辑是「有 CUDA 就用 GPU，没有就退回 CPU」。因为 torch 是 CPU 版，`torch.cuda.is_available()` 恒为 `False`，所以每次都在 CPU 上跑。

### 想让 GE-M3 走 GPU

1. 装 CUDA 版 torch（替换当前 `+cpu` 版）：
   ```
   uv pip install torch --index-url https://download.pytorch.org/whl/cu126
   ```
   （国内下载慢，需要走代理；GTX 1650 是 Turing 架构，现代 CUDA torch 支持。）
2. `.env` 保持 `EMBEDDING_DEVICE=cuda`。
3. 代码里 `EMBEDDING_DEVICE=cuda` 会自动开 fp16（见 `embedding_runtime.py` 的 `use_fp16 = device != "cpu"`），GTX 1650 的 4GB 显存能放下 GE-M3（fp16 约 1.1GB）。

---

## 二、HF 下载的到底是什么：代码 vs 权重

### 一句话结论

- **模型代码**（Python 类、前向逻辑、transformer 架构）= 通过 `uv add flagembedding` 装进 `.venv` 的 site-packages，**不是** HF 缓存下载的。
- **模型权重 + 配置 + 分词器** = 从 HuggingFace Hub 下载，存在缓存里。

两者是**分开的、由不同命令触发**。

### 分开触发

| 内容 | 谁提供 | 触发方式 | 落盘位置 |
|---|---|---|---|
| 代码（`BGEM3FlagModel` 类、transformer 实现） | pip/uv 包（`flagembedding`、`transformers`） | `uv add flagembedding`（手动装包） | `.venv/Lib/site-packages/` |
| 权重（`model.safetensors` 等） | HuggingFace Hub | **首次 `BGEM3FlagModel("BAAI/bge-m3")` 自动触发** | HF 缓存（本项目 `.hf_cache/`） |
| 配置/分词器（`config.json`、tokenizer 等） | HuggingFace Hub | 同上，自动 | 同上 |

### 权重具体下载了哪些（bge-m3 共 30 个文件）

- `config.json`、`tokenizer.json`、`tokenizer_config.json`、`sentencepiece.bpe.model` → 模型结构 + 分词器
- `model.safetensors` → **主权重，约 2.2GB**（大文件，下载最慢）
- `colbert_linear.safetensors`、`sparse_linear.safetensors` → bge-m3 额外的 ColBERT / 稀疏输出头
- `1_Pooling/config.json` 等 → sentence-transformers 兼容文件

日志里两个进度条的含义：

- `Fetching 30 files: 100%` → 下载这 30 个文件
- `Loading weights: 391/391` → 从 safetensors 读 391 个张量进内存（是加载，不是下载）

### 自动 vs 手动

- **自动**：第一次实例化模型时自动下载。本项目就是 ingest → 编码阶段首次 `_load_model()` 触发。
- **手动预下载**（效果相同，可选）：
  ```
  .venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"
  ```

### 缓存目录结构（本项目）

```
.hf_cache/hub/models--BAAI--bge-m3/
  ├── snapshots/<commit_hash>/   # 实际文件（模型、配置、分词器）
  ├── blobs/                     # 内容寻址存储（跨模型去重）
  └── refs/main                  # 指向当前版本的指针
```

缓存位置由 `.env` 的 `HF_HOME` 决定。默认是 `~/.cache/huggingface`，本项目改到了项目内 `.hf_cache/`（已 gitignore）。下载镜像由 `HF_ENDPOINT=https://hf-mirror.com` 指定。

---

## 三、完整 CLI 复现顺序

### 0. 启动数据库（Docker）

```
docker compose up -d        # 启动 postgres + qdrant
docker compose ps           # 确认两个容器都是 Up
```

> Windows git bash 里若 `docker` 找不到：确认 Docker Desktop 的 bin 目录在 PATH 上（`C:\Users\magic\AppData\Local\Programs\DockerDesktop\resources\bin`），或重开一个终端。

### 1. 装依赖（首次）

```
uv sync --extra embedding    # 主依赖 + flagembedding
```

### 2. 建表（ingest 会自动做，无需单独执行）

ingest 启动时 `initialize()` 会自动执行 `sql/schema.sql`，所以正常不用手动建表。

### 3. 跑 ingest

```
.venv/Scripts/python.exe -m utils.ingestion.PDF_ingest  --documents data/test1/ --no-images
```

- `--documents data`：PDF 在 `data/`（CLI 默认是 `documents`，别漏）。
- `--no-images`：跳过图表视觉分析（不需要 OpenAI key）。
- 想连表格一起关：加 `--no-tables`，或直接 `--fast`。

### 4. 查数据

PostgreSQL（正文 / 分块 / 图）：

```
docker exec agents-postgres psql -U postgres -c "SELECT count(*) FROM documents;"
docker exec agents-postgres psql -U postgres -c "SELECT count(*) FROM chunks;"
docker exec agents-postgres psql -U postgres -c "SELECT id, title FROM documents;"
```

Qdrant（向量）：

```
curl http://localhost:6333/collections/paper_chunks
curl http://localhost:6333/collections/paper_chunks/points/count
```

### 5. 检索（自己复现检索结果）

写个脚本（或 heredoc）跑，会用 GE-M3 给查询编码（CPU 约 1-2 分钟）：

```python
import asyncio
from agent.db.db_utils import initialize_database, close_database, vector_search, hybrid_search
from agent.embedding.embedding_runtime import aembed_texts

async def main():
    await initialize_database()
    query = "wire arc additive manufacturing defect detection"
    (dense, sparse) = (await aembed_texts([query]))[0]

    print("[1] 纯稠密 vector_search:")
    for r in await vector_search(dense, limit=5):
        print(f"  {r['similarity']:.4f}  {r['content'][:60]}")

    print("[2] 混合 hybrid_search (RRF):")
    for r in await hybrid_search(dense, query_text=query, limit=5, sparse=sparse):
        print(f"  {r['combined_score']:.4f}  {r['content'][:60]}")

    await close_database()

asyncio.run(main())
```

> 只想看库里数据、不想加载模型，用第 4 步的 `curl`/`psql` 即可。

---

## 附：本次实测结果（供对照）

| 文档 | chunk 数 | 耗时 | PostgreSQL ID |
|---|---|---|---|
| WAAM.pdf（In-situ surface inspection…） | 142 | 757.7s | `7ba69575-…` |
| WAAM2.pdf（Detection and quantitative evaluation…） | 86 | 351.2s | `59c93afb-…` |

- 2 个文档、228 个 chunk，Qdrant 228 个 point（dense 1024 维 + sparse）。
- 检索验证：纯稠密 `similarity` 与混合 RRF `combined_score` 均能返回同篇论文语义相关 chunk。
