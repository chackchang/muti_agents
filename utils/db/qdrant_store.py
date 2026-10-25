"""Qdrant 分块向量库。

PostgreSQL 仍然是文档、分块正文、图/表证据的来源；Qdrant 只负责向量检索。
每个分块是 Qdrant 里的一个 point：

* ``id``：PostgreSQL ``chunks.id``（UUID 字符串）；
* ``dense``：GE-M3 稠密向量；
* ``sparse``：GE-M3 稀疏向量（token id -> weight）；
* ``payload``：``chunk_id`` / ``document_id`` / ``content`` / ``metadata`` 等。

环境变量
--------
``QDRANT_URL``        默认 ``http://127.0.0.1:6333``；测试可设为 ``:memory:``
``QDRANT_API_KEY``    可选
``QDRANT_COLLECTION`` 默认 ``paper_chunks``
``QDRANT_TIMEOUT``    默认 60 秒
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

from dotenv import load_dotenv

from utils.embedding.embedding_runtime import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_embedding_dimensions,
    supports_sparse_vectors,
)

load_dotenv()

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 128
_SEARCH_PREFETCH_FACTOR = 4
_SEARCH_MIN_PREFETCH = 20

_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _is_loopback(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOOPBACK_HOSTS


class QdrantChunkStore:
    """分块向量的读写入口（同步客户端，异步调用方用 ``asyncio.to_thread``）。"""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: Optional[str] = None,
        client: Any = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.url = (url or os.getenv("QDRANT_URL") or "http://127.0.0.1:6333").strip()
        self.api_key = api_key or (os.getenv("QDRANT_API_KEY") or "").strip() or None
        self.collection = (
            collection or os.getenv("QDRANT_COLLECTION") or "paper_chunks"
        ).strip()
        self.timeout = float(timeout or os.getenv("QDRANT_TIMEOUT") or 60)
        self._client = client
        self._ready = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 连接
    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            if self.url in {":memory:", "memory"}:
                self._client = QdrantClient(":memory:")
            else:
                kwargs: Dict[str, Any] = {
                    "url": self.url,
                    "api_key": self.api_key,
                    "timeout": self.timeout,
                }
                if _is_loopback(self.url):
                    # 本地 Qdrant 必须直连：httpx 默认 trust_env=True 会把 127.0.0.1
                    # 也转发给系统代理（如 Clash），导致 502 Bad Gateway。
                    kwargs["trust_env"] = False
                    # 版本兼容检查会用独立的 httpx 客户端（不继承 trust_env），
                    # 本地 Docker 场景无需它，关掉以免每次启动都报一次 502。
                    kwargs["check_compatibility"] = False
                self._client = QdrantClient(**kwargs)
        return self._client

    def initialize(self, *, recreate: bool = False) -> None:
        """确保集合与 payload 索引存在。"""
        if self._ready and not recreate:
            return
        from qdrant_client import models

        with self._lock:
            exists = self.client.collection_exists(self.collection)
            if recreate and exists:
                self.client.delete_collection(self.collection)
                exists = False

            if exists:
                self._warn_dimension_mismatch()
            else:
                sparse_config = None
                if supports_sparse_vectors():
                    sparse_config = {
                        SPARSE_VECTOR_NAME: models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    }
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config={
                        DENSE_VECTOR_NAME: models.VectorParams(
                            size=get_embedding_dimensions(),
                            distance=models.Distance.COSINE,
                        )
                    },
                    sparse_vectors_config=sparse_config,
                )
                logger.info(
                    "Created Qdrant collection %s (dense=%s, sparse=%s)",
                    self.collection,
                    get_embedding_dimensions(),
                    bool(sparse_config),
                )

            for field_name, schema in (
                ("document_id", models.PayloadSchemaType.KEYWORD),
                ("content_type", models.PayloadSchemaType.KEYWORD),
                ("artifact_type", models.PayloadSchemaType.KEYWORD),
            ):
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection,
                        field_name=field_name,
                        field_schema=schema,
                        wait=True,
                    )
                except Exception as exc:  # 索引已存在或服务端不支持时不影响读写
                    logger.debug("Qdrant payload index %s skipped: %s", field_name, exc)

            self._ready = True

    def _warn_dimension_mismatch(self) -> None:
        try:
            info = self.client.get_collection(self.collection)
            dense = info.config.params.vectors[DENSE_VECTOR_NAME]
            size = int(dense.size)
        except Exception:
            return
        expected = get_embedding_dimensions()
        if size != expected:
            logger.warning(
                "Qdrant collection %s dense size=%s, but EMBEDDING_DIMENSIONS=%s",
                self.collection,
                size,
                expected,
            )

    # ------------------------------------------------------------------ 写入
    def _build_point(self, point: Dict[str, Any]) -> Any:
        from qdrant_client import models

        dense = point.get("dense")
        if not dense:
            raise ValueError(f"Point {point.get('id')} has no dense vector")
        expected = get_embedding_dimensions()
        if len(dense) != expected:
            raise ValueError(
                f"Point {point.get('id')} dense dimension {len(dense)} != {expected}"
            )

        vector: Dict[str, Any] = {DENSE_VECTOR_NAME: [float(value) for value in dense]}
        sparse = point.get("sparse") or {}
        if sparse and supports_sparse_vectors():
            indices = sorted(int(index) for index in sparse)
            vector[SPARSE_VECTOR_NAME] = models.SparseVector(
                indices=indices,
                values=[float(sparse[index]) for index in indices],
            )

        return models.PointStruct(
            id=str(point["id"]),
            vector=vector,
            payload=dict(point.get("payload") or {}),
        )

    def upsert_chunks(self, points: Sequence[Dict[str, Any]]) -> int:
        """写入分块向量，返回写入条数。"""
        if not points:
            return 0
        self.initialize()
        built = [self._build_point(point) for point in points]
        for start in range(0, len(built), _UPSERT_BATCH_SIZE):
            self.client.upsert(
                collection_name=self.collection,
                points=built[start:start + _UPSERT_BATCH_SIZE],
                wait=True,
            )
        logger.info("Upserted %s chunk vectors into Qdrant %s", len(built), self.collection)
        return len(built)

    def delete_document(self, document_id: str) -> None:
        """删除一篇论文的全部分块向量。"""
        from qdrant_client import models

        if not str(document_id or "").strip():
            return
        self.initialize()
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
            wait=True,
        )

    def clear(self) -> None:
        """清空集合（重建集合与索引）。"""
        self.initialize(recreate=True)
        logger.info("Cleared Qdrant collection %s", self.collection)

    # ------------------------------------------------------------------ 查询
    @staticmethod
    def _sparse_vector(values: Dict[int, float]) -> Any:
        from qdrant_client import models

        indices = sorted(int(index) for index in values)
        return models.SparseVector(
            indices=indices,
            values=[float(values[index]) for index in indices],
        )

    def _build_filter(
        self,
        *,
        document_id: Optional[str] = None,
        document_ids: Optional[Sequence[str]] = None,
        artifact_types: Optional[Sequence[str]] = None,
        content_type: Optional[str] = None,
        exclude_artifacts: bool = False,
    ) -> Any:
        from qdrant_client import models

        must: List[Any] = []
        must_not: List[Any] = []

        normalized_ids = [
            str(value).strip()
            for value in (document_ids or [])
            if str(value).strip()
        ]
        if document_id:
            must.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=str(document_id))
                )
            )
        elif normalized_ids:
            must.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=normalized_ids)
                )
            )

        normalized_types = [
            str(value).strip().lower()
            for value in (artifact_types or [])
            if str(value).strip()
        ]
        if normalized_types:
            must.append(
                models.FieldCondition(
                    key="artifact_type", match=models.MatchAny(any=normalized_types)
                )
            )
        if content_type:
            must.append(
                models.FieldCondition(
                    key="content_type", match=models.MatchValue(value=str(content_type))
                )
            )
        if exclude_artifacts:
            must_not.append(
                models.FieldCondition(
                    key="content_type", match=models.MatchValue(value="artifact")
                )
            )

        if not must and not must_not:
            return None
        return models.Filter(must=must or None, must_not=must_not or None)

    def _to_result(self, point: Any) -> Dict[str, Any]:
        payload = dict(point.payload or {})
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "chunk_id": str(payload.get("chunk_id") or point.id),
            "document_id": str(payload.get("document_id") or ""),
            "content": str(payload.get("content") or payload.get("text") or ""),
            "score": float(point.score or 0.0),
            "metadata": metadata,
            "document_title": str(payload.get("document_title") or ""),
            "document_source": str(payload.get("document_source") or ""),
        }

    def search(
        self,
        dense: Sequence[float],
        sparse: Optional[Dict[int, float]] = None,
        *,
        limit: int = 10,
        document_id: Optional[str] = None,
        document_ids: Optional[Sequence[str]] = None,
        artifact_types: Optional[Sequence[str]] = None,
        content_type: Optional[str] = None,
        exclude_artifacts: bool = False,
    ) -> List[Dict[str, Any]]:
        """稠密 + 稀疏混合检索；没有稀疏向量时退化为纯稠密检索。"""
        from qdrant_client import models

        result_limit = max(1, min(int(limit or 10), 100))
        query_filter = self._build_filter(
            document_id=document_id,
            document_ids=document_ids,
            artifact_types=artifact_types,
            content_type=content_type,
            exclude_artifacts=exclude_artifacts,
        )
        self.initialize()

        if sparse and supports_sparse_vectors():
            prefetch_limit = max(result_limit * _SEARCH_PREFETCH_FACTOR, _SEARCH_MIN_PREFETCH)
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(
                        query=[float(value) for value in dense],
                        using=DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                    models.Prefetch(
                        query=self._sparse_vector(sparse),
                        using=SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                        filter=query_filter,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                limit=result_limit,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection,
                query=[float(value) for value in dense],
                using=DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=result_limit,
                with_payload=True,
            )

        return [self._to_result(point) for point in response.points]

    def document_profile(
        self,
        document_id: str,
        *,
        exclude_artifacts: bool = True,
    ) -> Optional[List[float]]:
        """文档级 profile：该论文所有非 artifact 分块稠密向量的均值。"""
        query_filter = self._build_filter(
            document_id=document_id, exclude_artifacts=exclude_artifacts
        )
        self.initialize()

        vectors: List[Sequence[float]] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=[DENSE_VECTOR_NAME],
            )
            for point in points:
                vector = point.vector
                if isinstance(vector, dict):
                    vector = vector.get(DENSE_VECTOR_NAME)
                if vector:
                    vectors.append(vector)
            if offset is None:
                break

        if not vectors:
            return None
        dimensions = len(vectors[0])
        count = len(vectors)
        return [
            sum(float(vector[index]) for vector in vectors) / count
            for index in range(dimensions)
        ]


# 全局单例
qdrant_store = QdrantChunkStore()
