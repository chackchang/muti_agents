"""utils 包入口（门面）。

对外只暴露一组稳定的名字：爬虫导出、以及**唯一**的嵌入入口。外部代码
（``agent``、``evals``、脚本）一律 ``from utils import ...``，不要直接
``utils.embedding.embedding_runtime``——这样以后换模型、换后端、拆文件都不会
波及调用方。

用 PEP 562 的模块级 ``__getattr__`` 惰性导入：``import utils`` 本身不加载
任何重依赖（FlagEmbedding、httpx、导出器），只有真正取用某个名字时才加载。
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "JsonExporter": ("utils.exporters.json", "JsonExporter"),
    "Pipeline": ("utils.exporters.json", "Pipeline"),
    # 嵌入运行时：稠密 + 稀疏，所有语言共用同一个模型，没有语言路由。
    "DENSE_VECTOR_NAME": ("utils.embedding.embedding_runtime", "DENSE_VECTOR_NAME"),
    "SPARSE_VECTOR_NAME": ("utils.embedding.embedding_runtime", "SPARSE_VECTOR_NAME"),
    "GE_M3_DIMENSIONS": ("utils.embedding.embedding_runtime", "GE_M3_DIMENSIONS"),
    "EmbeddingVector": ("utils.embedding.embedding_runtime", "EmbeddingVector"),
    "aembed_texts": ("utils.embedding.embedding_runtime", "aembed_texts"),
    "aembed_texts_batched": ("utils.embedding.embedding_runtime", "aembed_texts_batched"),
    "embed_texts": ("utils.embedding.embedding_runtime", "embed_texts"),
    "get_embedding_backend_name": (
        "utils.embedding.embedding_runtime",
        "get_embedding_backend_name",
    ),
    "get_embedding_dimensions": ("utils.embedding.embedding_runtime", "get_embedding_dimensions"),
    "get_embedding_model_name": ("utils.embedding.embedding_runtime", "get_embedding_model_name"),
    "supports_sparse_vectors": ("utils.embedding.embedding_runtime", "supports_sparse_vectors"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value  # 缓存，之后的访问不再走 __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
