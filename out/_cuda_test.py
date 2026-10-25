import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
from utils.embedding.embedding_runtime import (
    embed_texts,
    get_embedding_backend_name,
    get_embedding_model_name,
    supports_sparse_vectors,
)

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("backend", get_embedding_backend_name(), "model", get_embedding_model_name(), "sparse", supports_sparse_vectors())

t0 = time.time()
vecs = embed_texts(["wire arc additive manufacturing process monitoring for defect detection"])
t1 = time.time()
dense, sparse = vecs[0]
print(f"dense={len(dense)}d sparse_terms={len(sparse)}")
print(f"load+embed_seconds={t1 - t0:.2f}")
print("cuda_allocated_MB", round(torch.cuda.memory_allocated() / 1024 / 1024, 1))
print("cuda_max_allocated_MB", round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1))
print("DONE_OK")
