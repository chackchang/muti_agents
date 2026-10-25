import os
import sys
import time

# 脚本放在 out/ 下，需把项目根目录加入 sys.path 才能 import agent 包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 必须先于任何 huggingface_hub 导入设置镜像端点与缓存目录
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))

t0 = time.time()

from utils.embedding.embedding_runtime import (  # noqa: E402
    embed_texts,
    get_embedding_backend_name,
    get_embedding_model_name,
    supports_sparse_vectors,
)

print(
    f"[{time.time() - t0:6.1f}s] backend={get_embedding_backend_name()} "
    f"model={get_embedding_model_name()} sparse={supports_sparse_vectors()}",
    flush=True,
)
print(f"[{time.time() - t0:6.1f}s] 加载模型（首次需下载约 2GB，走代理）...", flush=True)

vecs = embed_texts(
    ["wire arc additive manufacturing process monitoring for defect detection"]
)
dense, sparse = vecs[0]
print(f"[{time.time() - t0:6.1f}s] dense={len(dense)}d sparse_terms={len(sparse)}", flush=True)
print("SPARSE_SAMPLE", list(sparse.items())[:6], flush=True)
print("DONE_OK", flush=True)
