"""统一的本地嵌入运行时（GE-M3，稠密 + 稀疏）。

所有语言共用同一个模型与同一套接口，不再做语言判断与模型路由。

一次 ``encode`` 返回 ``(dense, sparse)`` 元组列表：

* ``dense``：``List[float]``，长度 1024，写入 Qdrant 的 ``dense`` 命名向量；
* ``sparse``：``Dict[int, float]``，键是 token id，写入 Qdrant 的 ``sparse``
  稀疏向量。

后端通过 ``EMBEDDING_BACKEND`` 选择：

* ``flagembedding``（默认）：在本进程内用 ``BGEM3FlagModel.encode`` 推理，
  可以同时拿到 dense 与 sparse，维度与词表完全对齐 GE-M3。
* ``ollama``：调用 Ollama 的 ``/api/embed``。Ollama 只返回 dense 向量，
  因此这条路径没有 sparse；此时 Qdrant 集合只创建 dense 向量。

环境变量
--------
``EMBEDDING_BACKEND``      ``flagembedding`` | ``ollama``
``EMBEDDING_MODEL``        本地模型名/路径，默认 ``BAAI/bge-m3``
``EMBEDDING_OLLAMA_MODEL`` Ollama 模型名，默认 ``bge-m3``
``OLLAMA_BASE_URL``        默认 ``http://127.0.0.1:11434``
``EMBEDDING_DIMENSIONS``   dense 维度，默认 1024
``EMBEDDING_BATCH_SIZE``   FlagEmbedding 单批条数，默认 16
``EMBEDDING_MAX_LENGTH``   单条最大 token 数，默认 8192（GE-M3 上限）
``EMBEDDING_DEVICE``       ``cuda`` / ``cpu``，默认交给 torch 自动选择
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
GE_M3_DIMENSIONS = 1024

# (dense, sparse)
EmbeddingVector = Tuple[List[float], Dict[int, float]]


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; falling back to %s", name, raw, default)
        return default


def get_embedding_backend_name() -> str:
    return (os.getenv("EMBEDDING_BACKEND") or "flagembedding").strip().lower()


def get_embedding_model_name() -> str:
    """当前后端实际使用的模型名。"""
    if get_embedding_backend_name() == "ollama":
        return (os.getenv("EMBEDDING_OLLAMA_MODEL") or "bge-m3").strip()
    return (os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3").strip()


# 旧调用点仍可能引用这个名字
def get_embedding_model() -> str:
    return get_embedding_model_name()


def get_embedding_dimensions() -> int:
    return _env_int("EMBEDDING_DIMENSIONS", GE_M3_DIMENSIONS)


def _normalize_sparse(weights: Any) -> Dict[int, float]:
    """把 FlagEmbedding 的 ``lexical_weights`` 转成 token_id -> weight。

    ``lexical_weights`` 的键可能是字符串或 numpy 整数，统一转成非负 int，
    这样可以直接作为 Qdrant 稀疏向量的 indices。
    """
    sparse: Dict[int, float] = {}
    if not isinstance(weights, dict):
        return sparse
    for key, value in weights.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if index < 0:
            continue
        try:
            sparse[index] = float(value)
        except (TypeError, ValueError):
            continue
    return sparse


class EmbeddingBackend:
    """嵌入后端的统一接口。"""

    supports_sparse = True

    def encode(self, texts: Sequence[str]) -> List[EmbeddingVector]:
        raise NotImplementedError

    async def aencode(self, texts: Sequence[str]) -> List[EmbeddingVector]:
        return await asyncio.to_thread(self.encode, texts)


class FlagEmbeddingBackend(EmbeddingBackend):
    """进程内 GE-M3：一次前向同时产出 dense 与 sparse。"""

    supports_sparse = True

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
        use_fp16: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name or get_embedding_model_name()
        self.batch_size = batch_size or _env_int("EMBEDDING_BATCH_SIZE", 16)
        self.max_length = max_length or _env_int("EMBEDDING_MAX_LENGTH", 8192)
        self.device = device or (os.getenv("EMBEDDING_DEVICE") or "").strip() or None
        self.use_fp16 = use_fp16
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._encode_lock = threading.Lock()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model

            from FlagEmbedding import BGEM3FlagModel  # 延迟导入，未安装时可给出可读错误

            use_fp16 = self.use_fp16
            if use_fp16 is None:
                if self.device:
                    use_fp16 = self.device != "cpu"
                else:
                    import torch

                    use_fp16 = bool(torch.cuda.is_available())

            kwargs: Dict[str, Any] = {"use_fp16": use_fp16}
            if self.device:
                kwargs["device"] = self.device

            logger.info(
                "Loading embedding model %s (device=%s, fp16=%s)",
                self.model_name,
                self.device or "auto",
                use_fp16,
            )
            self._model = BGEM3FlagModel(self.model_name, **kwargs)
            return self._model

    def encode(self, texts: Sequence[str]) -> List[EmbeddingVector]:
        items = [str(text or "") for text in texts]
        if not items:
            return []

        model = self._load_model()
        output = model.encode(
            items,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vectors = output["dense_vecs"].tolist()
        lexical_weights = output.get("lexical_weights") or [{} for _ in items]
        return [
            (dense, _normalize_sparse(weights))
            for dense, weights in zip(dense_vectors, lexical_weights)
        ]


class OllamaBackend(EmbeddingBackend):
    """Ollama 本地服务。``/api/embed`` 只返回 dense 向量。"""

    supports_sparse = False

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434").rstrip("/")
        self.model = model or get_embedding_model_name()
        self.timeout = float(timeout or os.getenv("OLLAMA_TIMEOUT") or 180)

    def _payload(self, texts: Sequence[str]) -> Dict[str, Any]:
        return {"model": self.model, "input": list(texts), "truncate": True}

    @staticmethod
    def _parse(response: httpx.Response, count: int) -> List[EmbeddingVector]:
        response.raise_for_status()
        embeddings = response.json().get("embeddings") or []
        if len(embeddings) != count:
            raise RuntimeError(
                f"Ollama 返回了 {len(embeddings)} 条向量，期望 {count} 条"
            )
        return [(list(map(float, vector)), {}) for vector in embeddings]

    def encode(self, texts: Sequence[str]) -> List[EmbeddingVector]:
        items = [str(text or "") for text in texts]
        if not items:
            return []
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json=self._payload(items),
            timeout=self.timeout,
        )
        return self._parse(response, len(items))

    async def aencode(self, texts: Sequence[str]) -> List[EmbeddingVector]:
        items = [str(text or "") for text in texts]
        if not items:
            return []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/embed", json=self._payload(items)
            )
        return self._parse(response, len(items))


_backend: Optional[EmbeddingBackend] = None
_backend_lock = threading.Lock()


def _build_backend() -> EmbeddingBackend:
    backend_name = get_embedding_backend_name()
    if backend_name == "ollama":
        return OllamaBackend()
    if backend_name not in {"flagembedding", "local", "bgem3"}:
        logger.warning(
            "Unknown EMBEDDING_BACKEND=%r; falling back to flagembedding", backend_name
        )
    return FlagEmbeddingBackend()


def get_embedding_backend() -> EmbeddingBackend:
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = _build_backend()
    return _backend


def reset_embedding_backend() -> None:
    """清空进程内单例，测试或切换配置时使用。"""
    global _backend
    with _backend_lock:
        _backend = None


def supports_sparse_vectors() -> bool:  # noqa: F811 - 最终实现
    return bool(get_embedding_backend().supports_sparse)


def embed_texts(texts: Sequence[str]) -> List[EmbeddingVector]:
    """同步嵌入，返回 ``[(dense, sparse), ...]``。"""
    return get_embedding_backend().encode(texts)


async def aembed_texts(texts: Sequence[str]) -> List[EmbeddingVector]:
    """异步嵌入，返回 ``[(dense, sparse), ...]``。"""
    return await get_embedding_backend().aencode(texts)


async def aembed_texts_batched(
    texts: Sequence[str],
    batch_size: Optional[int] = None,
) -> List[EmbeddingVector]:
    """按批异步嵌入，顺序与输入一致。"""
    size = max(1, int(batch_size or _env_int("EMBEDDING_BATCH_SIZE", 16)))
    vectors: List[EmbeddingVector] = []
    for start in range(0, len(texts), size):
        vectors.extend(await aembed_texts(texts[start:start + size]))
    return vectors
