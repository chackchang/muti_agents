import argparse
import asyncio
from collections import Counter
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from warnings import filterwarnings
from utils.exporters import JsonExporter
from utils.db.db_utils import (
    clear_vector_store,
    close_database,
    db_pool,
    delete_document_vectors,
    execute_init_sql,
    initialize_database,
    upsert_chunk_vectors,
)

from utils.embedding.embedding_runtime import aembed_texts_batched, get_embedding_model_name
from agent.graph.graph_runtime import refresh_paper_graph
from agent.config import ChunkingConfig, IngestionConfig, IngestionResult
from utils.chunker.chunker import DocumentChunk, create_chunker
from utils.norm.base import Document, Normalizer
from utils.adapters.base import Adapter
from utils.chunker.vision_client import ArkVisionClient, figure_context

filterwarnings("ignore", category=UserWarning)

# 加载环境变量
load_dotenv()

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """将文档导入 PostgreSQL（正文）+ Qdrant（向量）的流水线。"""

    def __init__(
        self,
        config: IngestionConfig | None = None,
        normalizer: Normalizer | None = None,
    ):
        """初始化导入流水线；所有可调项都从 IngestionConfig 解析。

        ``normalizer`` 是运行时依赖（可注入替身用于测试），不属于配置，单独传。
        """
        self.config = config or IngestionConfig()
        self.documents_folder = self.config.documents_folder
        self.clean_before_ingest = self.config.clean_before_ingest
        self.reset_kb_before_ingest = self.config.reset_kb_before_ingest
        self.sql_schema_path = self.config.sql_schema_path
        self.include_images = self.config.include_images
        self.include_tables = self.config.include_tables
        self.exporter_out_path = self.config.exporter_out_path

        # 分块配置由 IngestionConfig 派生，CLI 的 --chunk-size/--chunk-overlap 才真正生效
        self.chunker_config = ChunkingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            max_chunk_size=self.config.max_chunk_size,
        )

        # 创建提取器和分块器
        self.extractor = Adapter()
        # 配置归一化层
        self.normalizer = normalizer or Normalizer()
        # 导出层
        self.exporter = JsonExporter()
        self.chunker = create_chunker(self.chunker_config)
        self._initialized = False

    async def initialize(self):
        """初始化数据库连接。"""
        if self._initialized:
            return

        logger.info("Initializing ingestion pipeline...")
        await initialize_database()
        await execute_init_sql(self.sql_schema_path)

        self._initialized = True
        logger.info("Ingestion pipeline initialized")

    async def close(self):
        """关闭数据库连接。"""
        if self._initialized:
            await close_database()
            self._initialized = False

    async def _clean_databases(self):
        """清理数据库表中的已有数据。"""
        if os.getenv("ALLOW_DESTRUCTIVE_INGEST_CLEAN", "").strip().lower() != "true":
            raise RuntimeError(
                "--clean 会删除 sessions/messages/documents/chunks。"
                "如确认要执行，请设置环境变量 ALLOW_DESTRUCTIVE_INGEST_CLEAN=true 后重试。"
            )
        logger.warning("Cleaning existing data from databases...")

        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM messages")
                await conn.execute("DELETE FROM sessions")
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")

        await clear_vector_store()
        logger.info("Cleaned PostgreSQL database and Qdrant collection")

    async def _reset_knowledge_base(self):
        if os.getenv("ALLOW_KB_RESET", "").strip().lower() != "true":
            raise RuntimeError(
                "--reset-kb 会删除 documents/chunks 并要求重新入库。"
                "如确认要执行，请设置环境变量 ALLOW_KB_RESET=true 后重试。"
            )

        logger.warning("Resetting knowledge base: deleting chunks and documents only...")
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM chunks")
                await conn.execute("DELETE FROM documents")
        await clear_vector_store()
        logger.info("Knowledge base reset complete; sessions/messages were preserved.")

    def _compute_identity(self, pdf_bytes: bytes, document_source: str) -> tuple[str, str]:
        """从 PDF 原文推导幂等标识。

        逻辑ID只看 source（相对路径），内容修改后保持不变，用于定位“同一篇文档”。
        指纹用 PDF 原始字节（确定性）+ 会改变入库结果的配置：MinerU 抽取是非确定性的，
        同一 PDF 每次抽出的 markdown 都有细微差异（公式、标题格式），不能拿抽取后的
        content 当指纹，否则每次都会误判“内容已变”。
        """
        logical_id = hashlib.sha256(document_source.strip().encode("utf-8")).hexdigest()[:16]
        config_signature = json.dumps(
            {
                "include_images": self.include_images,
                "include_tables": self.include_tables,
                "chunk_size": self.chunker_config.chunk_size,
                "chunk_overlap": self.chunker_config.chunk_overlap,
                "embedding_model": get_embedding_model_name(),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(pdf_bytes + b"\n" + config_signature).hexdigest()
        return logical_id, fingerprint

    async def _find_current_document(self, logical_id: str) -> Optional[Dict[str, Any]]:
        """按逻辑ID查当前版本（is_current 且版本号最大者）。"""
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id::text, fingerprint, version, title
                FROM documents
                WHERE logical_id = $1 AND is_current = true
                ORDER BY version DESC
                LIMIT 1
                """,
                logical_id,
            )
        if not row:
            return None
        return {
            "id": row["id"],
            "fingerprint": row["fingerprint"],
            "version": int(row["version"]),
            "title": row["title"],
        }

    async def _ingest_single_document(self, file_path: str) -> IngestionResult:
        """导入单个文档：先做逻辑ID/指纹幂等把关，再交给共用落库流程。"""
        start_time = datetime.now()

        # 指纹用 PDF 原始字节：MinerU 抽取非确定，抽取后的 content 不能当指纹
        pdf_bytes = Path(file_path).read_bytes()
        document_source = os.path.relpath(file_path, self.documents_folder)
        # 幂等把关：解析前先算逻辑ID + 指纹，查库决定 append / update / skip。
        logical_id, fingerprint = self._compute_identity(pdf_bytes, document_source)
        existing = await self._find_current_document(logical_id)
        if existing is not None and existing["fingerprint"] == fingerprint:
            # title 取自库中当前版本；兼容旧行 title 为空的情况
            skipped_title = str(existing.get("title") or document_source).strip()
            logger.info("Document unchanged (logical_id=%s), skip: %s", logical_id, skipped_title)
            return IngestionResult(
                document_id=existing["id"],
                title=skipped_title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                status="skipped",
            )

        return await self._extract_and_persist(
            file_path,
            pdf_bytes=pdf_bytes,
            document_source=document_source,
            logical_id=logical_id,
            fingerprint=fingerprint,
            existing=existing,
        )

    async def _extract_and_persist(
        self,
        file_path: str,
        *,
        pdf_bytes: bytes,
        document_source: str,
        logical_id: str,
        fingerprint: str,
        existing: Optional[Dict[str, Any]],
    ) -> IngestionResult:
        """抽取 → 归一化 → 分块 → 向量化 → 落库。

        ``existing`` 为空时 append；非空时走 update：版本号 +1，
        新版本在 PostgreSQL 与 Qdrant 都落库成功后删除旧版本。
        """
        start_time = datetime.now()
        mode = "update" if existing is not None else "append"
        next_version = int(existing["version"]) + 1 if existing else 1

        print("INGEST_STAGE=extract", flush=True)
        raw_content, raw_metadata = self.extractor.read_document(file_path, "pdf")

        # 归一化后后续流程只认这一个 JSON：{id, title, content, content_type, metadata}。
        # source 在归一化前换成相对路径：分块层会把它复制进每个 chunk，且 Document 的 id
        # 基于 source + content 计算，这样 id 与最终 metadata 保持一致。
        document = self.normalizer.normalize(
            raw_content, {**raw_metadata, "source": document_source}
        )
        # title 为空时退回相对路径，保证文档表、chunk metadata 和导出 JSON 用同一个标题
        document_title = str(document.get("title") or document_source).strip()
        document["title"] = document_title
        document["metadata"].setdefault("title", document_title)

        logger.info(
            "Document %s (logical_id=%s, mode=%s): %s",
            document_title,
            logical_id,
            mode,
            "fingerprint changed" if mode == "update" else "new document",
        )

        print("INGEST_STAGE=export", flush=True)
        self._write_document_json(document)

        document_metadata = document["metadata"]
        document_metadata["include_artifacts"] = bool(self.include_images or self.include_tables)

        # 图的上下文从归一化后的正文里取，和最终入库的 content 保持一致
        print("INGEST_STAGE=vision", flush=True)
        vision_chunks = await self._build_vision_chunks(document["content"], document_metadata, document_title, document_source)

        logger.info(f"Processing document: {document_title}")
        logger.info(
            f"Found {document_metadata.get('pictures', 0)} images and {document_metadata.get('tables', 0)} tables"
        )

        print("INGEST_STAGE=structure", flush=True)
        main_chunks = self.chunker.chunk_pdf_documents([document])
        main_chunks.extend(vision_chunks)
        # 视觉分析结果只放在文档 metadata 里；放在分块前会被复制进每个 chunk
        if vision_chunks:
            document_metadata["vision_analyses"] = [chunk.metadata.get("vision_analysis") for chunk in vision_chunks]

        if not main_chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                status="failed",
            )

        # 视觉证据块在普通文本分块之后追加，重新编号保证数据库排序稳定。
        total_chunks = len(main_chunks)
        for index, chunk in enumerate(main_chunks):
            chunk.index = index
            chunk.metadata.update({"chunk_index": index, "total_chunks": total_chunks})
        logger.info(f"Total chunks created: {len(main_chunks)}")
        method_counts = Counter(
            str((chunk.metadata or {}).get("chunk_method") or "unknown")
            for chunk in main_chunks
        )
        section_titles = {
            str((chunk.metadata or {}).get("section_title") or "").strip()
            for chunk in main_chunks
            if str((chunk.metadata or {}).get("section_title") or "").strip()
        }
        logger.info("Chunk method distribution for %s: %s", document_title, dict(method_counts))
        logger.info("Detected %s unique sections for %s", len(section_titles), document_title)

        print("INGEST_STAGE=embed", flush=True)
        embedded_chunks = await self.aembed_chunks(
            chunks=main_chunks,
        )
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

        print("INGEST_STAGE=persist", flush=True)
        document_id = await self._save_to_postgres(
            document_title,
            document_source,
            document["content"],
            embedded_chunks,
            document_metadata,
            pdf_bytes=pdf_bytes,
            logical_id=logical_id,
            fingerprint=fingerprint,
            version=next_version,
            mode=mode,
        )

        try:
            print("INGEST_STAGE=graph", flush=True)
            await refresh_paper_graph(document_id)
        except Exception as exc:
            logger.exception("Paper graph refresh failed for %s: %s", document_id, exc)

        logger.info(f"Saved document to PostgreSQL with ID: {document_id}")

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        return IngestionResult(
            document_id=document_id,
            title=document_title,
            chunks_created=len(main_chunks),
            processing_time_ms=processing_time,
            status="updated" if mode == "update" else "created",
        )



    def _write_document_json(self, document: Document) -> Optional[Path]:
        """把归一化 JSON 落盘，供分块层或排查时直接复用。

        ``exporter_out_path`` 以 ``.json`` 结尾时当成文件路径，否则当成目录，
        目录下按 ``<id>.json`` 命名。导出失败只告警，不阻断入库。
        """
        try:
            target = Path(self.exporter_out_path)
            if target.suffix.lower() != ".json":
                target = target / f"{document['id']}.json"
            return self.exporter.write(document, str(target))
        except Exception as exc:
            logger.warning(
                "Failed to export normalized JSON for %s: %s", document.get("title"), exc
            )
            return None

    async def upgrade_document_to_full(self, file_path: str) -> IngestionResult:
        """人工更新：用磁盘上的 PDF 重新抽取并覆盖同一篇已入库文档。

        与批量入库的 update 分支完全一致：
        1. 逻辑ID由 ``source`` 推导，指纹由 PDF 字节 + 配置推导；
        2. 文档未入库 → 拒绝（升级只针对已有文档）；
        3. 指纹一致 → ``skipped``，不碰任何库；
        4. 指纹变化 → 生成新版本，PostgreSQL + Qdrant 都成功后再删除旧版本。
        """
        start_time = datetime.now()

        if not Path(file_path).is_file():
            raise FileNotFoundError(f"PDF 不存在：{file_path}")

        pdf_bytes = Path(file_path).read_bytes()
        document_source = os.path.relpath(file_path, self.documents_folder)
        logical_id, fingerprint = self._compute_identity(pdf_bytes, document_source)
        existing = await self._find_current_document(logical_id)
        if existing is None:
            raise ValueError(
                f"文档尚未入库，无法升级：source={document_source} "
                f"(logical_id={logical_id})。请先通过 document_ingest_api 入库。"
            )

        if existing["fingerprint"] == fingerprint:
            title = str(existing.get("title") or document_source).strip()
            logger.info("Document unchanged (logical_id=%s), skip: %s", logical_id, title)
            return IngestionResult(
                document_id=existing["id"],
                title=title,
                chunks_created=0,
                processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
                status="skipped",
            )

        return await self._extract_and_persist(
            file_path,
            pdf_bytes=pdf_bytes,
            document_source=document_source,
            logical_id=logical_id,
            fingerprint=fingerprint,
            existing=existing,
        )

    async def _build_vision_chunks(
        self,
        markdown: str,
        metadata: Dict[str, Any],
        title: str,
        source: str,
    ) -> List[DocumentChunk]:
        """将已导出的论文图逐张转为带图注与邻近语境的检索证据块。"""
        if not self.include_images:
            return []
        assets = metadata.get("vision_assets") or []
        if not assets:
            return []
        chunks: List[DocumentChunk] = []
        failures: List[str] = []
        for asset in assets:
            caption = str(asset.get("caption") or "").strip()
            context_before, context_after = figure_context(markdown, caption)
            try:
                analysis = await ArkVisionClient().analyze_figure(
                    asset["path"], caption=caption, context_before=context_before, context_after=context_after
                )
            except Exception as exc:
                failures.append(str(exc)[:300])
                logger.warning("Vision analysis skipped for %s: %s", source, exc)
                continue
            analysis_dict = analysis.to_dict()
            figure_number = int(asset.get("index", len(chunks))) + 1
            body = (
            "[Artifact: Figure]\n"
            f"[Document: {title}]\n"
            f"[Caption: {caption or '[无图注]'}]\n"
            f"[Context before: {context_before or '[无]'}]\n"
            f"[Context after: {context_after or '[无]'}]\n\n"
            f"Visual summary: {analysis.summary or '[无]'}\n"
            f"Figure type: {analysis.figure_type}\n"
            f"Research purpose: {analysis.research_purpose or '[无]'}\n"
            f"Experimental task: {analysis.experimental_task or '[无]'}\n"
            f"Axes and units: {'; '.join(analysis.axes_and_units) or '[无]'}\n"
            f"Series or methods: {'; '.join(analysis.series_or_methods) or '[无]'}\n"
            f"Visible text: {'; '.join(analysis.visible_text) or '[无]'}\n"
            f"Observations: {'; '.join(analysis.observations) or '[无]'}\n"
            f"Quantitative findings: {'; '.join(analysis.quantitative_findings) or '[无]'}\n"
            f"Comparative claims: {'; '.join(analysis.comparative_claims) or '[无]'}\n"
            f"Visual tags: {'; '.join(analysis.visual_tags) or '[无]'}\n"
            f"Evidence confidence: {analysis.evidence_confidence}\n"
                f"Limitations: {'; '.join(analysis.limitations) or '[无]'}"
            )
            chunks.append(DocumentChunk(
            content=body,
            index=0,
            start_char=0,
            end_char=len(body),
            metadata={
                "title": title,
                "source": source,
                "content_type": "artifact",
                "artifact_type": "figure",
                "chunk_method": "vision_figure",
                "retrieval_title": caption or f"{title} figure {figure_number}",
                "caption": caption,
                "image_asset_path": asset["path"],
                "vision_analysis": analysis_dict,
                "vision_context_before": context_before,
                "vision_context_after": context_after,
                "figure_number": figure_number,
                "page_number": asset.get("page"),
            },
            ))
        metadata["vision_analysis_status"] = "success" if chunks else "failed"
        metadata["vision_model"] = chunks[0].metadata["vision_analysis"]["model"] if chunks else None
        if failures:
            metadata["vision_analysis_errors"] = failures
        return chunks

    @staticmethod
    def _media_type_for_path(path: str) -> str:
        suffix = Path(path).suffix.lower()
        return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(
            suffix, "image/png"
        )

    async def _save_visual_artifacts(self, conn: Any, document_id: str, chunks: List[DocumentChunk]) -> None:
        """Persist figures and tables as complete research evidence records.

        Only figure/table chunks are promoted. Algorithms deliberately keep the
        existing text-chunk representation because their original text is already
        directly searchable and citable.
        """
        for chunk in chunks:
            metadata = dict(chunk.metadata or {})
            artifact_type = str(metadata.get("artifact_type") or "").lower()
            if metadata.get("content_type") != "artifact" or artifact_type not in {"figure", "table"}:
                continue

            image_path = str(metadata.get("image_asset_path") or "")
            image_blob = None
            image_media_type = None
            if artifact_type == "figure" and image_path:
                path = Path(image_path)
                if path.is_file():
                    image_blob = path.read_bytes()
                    image_media_type = self._media_type_for_path(image_path)

            structured_data = dict(metadata.get("vision_analysis") or {})
            structured_data.update(
                {
                    "artifact_index": metadata.get("artifact_index"),
                    "figure_number": metadata.get("figure_number"),
                    "artifact_start_line": metadata.get("artifact_start_line"),
                    "artifact_end_line": metadata.get("artifact_end_line"),
                }
            )
            artifact_row = await conn.fetchrow(
                """
                INSERT INTO artifacts (
                    document_id, artifact_type, caption, page_number, section_path,
                    context_before, context_after, raw_content, structured_data,
                    retrieval_text, image_blob, image_media_type
                )
                VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11, $12)
                RETURNING id::text
                """,
                document_id,
                artifact_type,
                str(metadata.get("caption") or ""),
                metadata.get("page") or metadata.get("page_number"),
                str(metadata.get("section_path_text") or ""),
                str(metadata.get("vision_context_before") or metadata.get("context_before") or ""),
                str(metadata.get("vision_context_after") or metadata.get("context_after") or ""),
                str(metadata.get("raw_artifact_content") or chunk.content),
                json.dumps(structured_data, ensure_ascii=False),
                chunk.content,
                image_blob,
                image_media_type,
            )
            metadata["artifact_id"] = artifact_row["id"]
            chunk.metadata = metadata

    async def ingest_documents(self, progress_callback: Optional[callable] = None) -> List[IngestionResult]:
        """导入文档目录中的所有文档。"""
        if not self._initialized:
            await self.initialize()

        if self.clean_before_ingest and self.reset_kb_before_ingest:
            raise ValueError("Use either --clean or --reset-kb, not both.")

        if self.clean_before_ingest:
            await self._clean_databases()

        if self.reset_kb_before_ingest:
            await self._reset_knowledge_base()

        pdf_files = self._find_pdfs_in_directory(self.documents_folder)

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_folder}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        results = []
        for i, file_path in enumerate(pdf_files):
            try:
                logger.info(f"Processing file {i + 1}/{len(pdf_files)}: {file_path}")
                result = await self._ingest_single_document(file_path)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(pdf_files))
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(
                    IngestionResult(
                        document_id="",
                        title=os.path.basename(file_path),
                        chunks_created=0,
                        processing_time_ms=0,
                        status="failed",
                    )
                )

        failed_results = [result for result in results if not str(result.document_id or "").strip()]
        for r in results:
            print(f"DEBUG -> Title: {getattr(r, 'title', 'unknown')}, ID: {getattr(r, 'document_id', None)}, Raw: {r}")
        if failed_results:
            failed_titles = ", ".join(str(result.title or "unknown") for result in failed_results[:3])
            raise RuntimeError(f"{len(failed_results)} document(s) were not persisted: {failed_titles}")

        total_chunks = sum(r.chunks_created for r in results)
        logger.info(f"Ingestion complete: {len(results)} documents, {total_chunks} chunks")
        return results

    async def aembed_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> List[DocumentChunk]:
        """用同一个本地 GE-M3 模型批量生成 (dense, sparse) 向量。

        所有语言共用同一个模型，不再按语言分组或路由。
        """
        if not chunks:
            return []

        vectors = await aembed_texts_batched([chunk.content for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"Embedding returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        generated_at = datetime.now().isoformat()
        model_name = get_embedding_model_name()
        embedded_chunks: List[DocumentChunk] = []
        for chunk, (dense, sparse) in zip(chunks, vectors):
            embedded_chunk = DocumentChunk(
                content=chunk.content,
                index=chunk.index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                metadata={
                    **chunk.metadata,
                    "embedding_model": model_name,
                    "embedding_generated_at": generated_at,
                },
                token_count=chunk.token_count,
            )
            embedded_chunk.embedding = dense
            embedded_chunk.sparse_embedding = sparse
            embedded_chunks.append(embedded_chunk)
        return embedded_chunks

    @staticmethod
    def _vector_point(
        chunk_id: str,
        document_id: str,
        title: str,
        source: str,
        chunk: DocumentChunk,
        chunk_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """构造 Qdrant point：dense + sparse + 检索所需的 payload。"""
        dense = getattr(chunk, "embedding", None)
        if not dense:
            return None
        metadata = dict(chunk_metadata or chunk.metadata or {})
        return {
            "id": chunk_id,
            "dense": dense,
            "sparse": getattr(chunk, "sparse_embedding", None) or {},
            "payload": {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "content": chunk.content,
                "metadata": metadata,
                "content_type": str(metadata.get("content_type") or "pdf"),
                "artifact_type": str(metadata.get("artifact_type") or "").lower(),
                "document_title": title,
                "document_source": source,
            },
        }

    async def _delete_failed_document(self, document_id: str) -> None:
        """向量写入失败时回滚文档，避免出现“有正文但检索不到”的半成品。"""
        try:
            async with db_pool.acquire() as conn:
                await conn.execute("DELETE FROM documents WHERE id=$1::uuid", document_id)
            await delete_document_vectors(document_id)
        except Exception as exc:
            logger.warning("Failed to roll back document %s after vector error: %s", document_id, exc)

    def _find_pdfs_in_directory(self, directory: str, recursive: bool = True) -> List[str]:
        """查找目录中的所有 PDF 文件。"""
        directory_path = Path(directory)

        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(f"Directory not found or not a directory: {directory_path}")

        if recursive:
            pdf_files = list(directory_path.rglob("*.pdf"))
        else:
            pdf_files = list(directory_path.glob("*.pdf"))

        pdf_paths = [str(pdf.resolve()) for pdf in pdf_files if pdf.is_file()]
        logger.info(f"Found {len(pdf_paths)} PDF files in {directory_path}")
        return pdf_paths

    async def _save_to_postgres(
        self,
        title: str,
        source: str,
        content: str,
        chunks: List[DocumentChunk],
        metadata: Dict[str, Any],
        pdf_bytes: Optional[bytes] = None,
        *,
        logical_id: str = "",
        fingerprint: str = "",
        version: int = 1,
        mode: str = "append",
    ) -> str:
        """将文档与分块正文保存到 PostgreSQL，向量写入 Qdrant。

        ``mode`` 由切片前的幂等比对决定：
        - ``append``：全新文档，直接落库；
        - ``update``：同一逻辑ID内容已变，先落新版，双库确认成功后再删旧版。
        """
        vector_points: List[Dict[str, Any]] = []
        document_id = ""
        async with db_pool.acquire() as conn:
            async with conn.transaction():
                document_result = await conn.fetchrow(
                    """
                    INSERT INTO documents
                        (logical_id, fingerprint, version, is_current, title, source, content, metadata, pdf_blob, pdf_media_type)
                    VALUES ($1, $2, $3, true, $4, $5, $6, $7, $8, 'application/pdf')
                    RETURNING id::text
                    """,
                    logical_id,
                    fingerprint,
                    version,
                    title,
                    source,
                    content,
                    json.dumps(metadata, ensure_ascii=False),
                    pdf_bytes,
                )

                document_id = document_result["id"]

                await self._save_visual_artifacts(conn, document_id, chunks)

                for chunk in chunks:
                    chunk_metadata = {
                        **chunk.metadata,
                        "chunk_type": chunk.metadata.get("content_type", "text"),
                    }

                    row = await conn.fetchrow(
                        """
                        INSERT INTO chunks (document_id, content, chunk_index, metadata, token_count)
                        VALUES ($1::uuid, $2, $3, $4::jsonb, $5)
                        RETURNING id::text
                        """,
                        document_id,
                        chunk.content,
                        chunk.index,
                        json.dumps(chunk_metadata, ensure_ascii=False),
                        chunk.token_count if hasattr(chunk, "token_count") else len(chunk.content.split()),
                    )
                    point = self._vector_point(
                        row["id"], document_id, title, source, chunk, chunk_metadata
                    )
                    if point:
                        vector_points.append(point)

        # PostgreSQL 提交后再写向量；向量失败时回滚新文档，避免半成品进入知识库。
        try:
            await upsert_chunk_vectors(vector_points)
        except Exception:
            await self._delete_failed_document(document_id)
            raise

        # update 分支：新版已在两个库都落库成功，删除同逻辑ID下版本号更小的旧数据。
        if mode == "update":
            removed = await self._remove_old_versions(logical_id, version)
            logger.info(
                "Replaced %s old version(s) of logical_id=%s with version=%s (%s)",
                removed,
                logical_id,
                version,
                document_id,
            )

        return document_id

    async def _remove_old_versions(self, logical_id: str, new_version: int) -> int:
        """删除同一逻辑ID下所有 version < new_version 的行及其 chunks/向量。

        按版本号删而不是按某一行的 UUID 删：既只清理旧版，也能顺带清掉历史残留
        （例如上次更新删旧版失败留下的行）。新版本行 version == new_version，
        永远不会落入删除范围。
        """
        if new_version <= 1:
            return 0

        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id::text FROM documents WHERE logical_id = $1 AND version < $2",
                logical_id,
                new_version,
            )
            old_document_ids = [row["id"] for row in rows]
            if not old_document_ids:
                return 0
            # chunks 通过 ON DELETE CASCADE 一并删除
            await conn.execute(
                "DELETE FROM documents WHERE logical_id = $1 AND version < $2",
                logical_id,
                new_version,
            )

        for old_document_id in old_document_ids:
            try:
                await delete_document_vectors(old_document_id)
            except Exception as exc:
                # 向量残留只会造成检索重复，不应让本次更新失败；留下日志供排查。
                logger.warning(
                    "Failed to remove old vectors %s: %s", old_document_id, exc
                )
        return len(old_document_ids)

async def document_upgrade_api(
    file_path: str,
    config: IngestionConfig | None = None,
) -> IngestionResult:
    """人工更新接口：按 PDF 路径重新抽取并更新已入库的同一篇文档。

    幂等规则与批量入库的 update 分支完全一致（逻辑ID + 指纹）：
    - 文档未入库 → ``ValueError``；
    - 指纹未变 → ``status="skipped"``，不做任何写操作；
    - 指纹变化 → 新版本写入 PostgreSQL + Qdrant 后删除旧版本，``status="updated"``。

    ``file_path`` 需要位于 ``config.documents_folder`` 之下：逻辑ID由相对
    source 推导，路径不在其中会推导出不同的 source，从而被判定为“未入库”。
    """
    pipeline = DocumentIngestionPipeline(config=config)
    await pipeline.initialize()
    try:
        return await pipeline.upgrade_document_to_full(file_path)
    finally:
        await pipeline.close()


async def document_ingest_api(config: IngestionConfig):
    '''学术文档pdf入库接口'''
    log_level = logging.DEBUG if config.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if config.include_images or config.include_tables:
        logger.warning(
            "Image/table extraction can be slow. Use --fast or --no-images --no-tables for faster text-only ingestion."
        )

    pipeline = DocumentIngestionPipeline(config=config)

    def progress_callback(current: int, total: int):
        logger.info(f"Progress: {current}/{total} documents processed")

    try:
        start_time = datetime.now()
        results = await pipeline.ingest_documents(progress_callback)
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        logger.info("\n" + "=" * 60)
        logger.info("INGESTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Documents processed: {len(results)}")
        logger.info(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        logger.info(f"Images extracted: {pipeline.include_images}")
        logger.info(f"Tables extracted: {pipeline.include_tables}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info("=" * 60)

        for result in results:
            if result.status == "skipped":
                logger.info(f"{result.title}: skipped (unchanged)")
            elif result.status == "updated":
                logger.info(f"{result.title}: {result.chunks_created} chunks (updated, {result.processing_time_ms/1000:.1f}s)")
            elif result.chunks_created > 0:
                logger.info(f"{result.title}: {result.chunks_created} chunks ({result.processing_time_ms/1000:.1f}s)")
            else:
                logger.warning(f"{result.title}: Failed to process")

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()

async def main():
    """运行导入流程的主函数。"""
    parser = argparse.ArgumentParser(description="Document ingestion with table/image processing")
    parser.add_argument("--documents", "-d", default="documents", help="Documents folder path")
    parser.add_argument("--clean", "-c", action="store_true", help="Clean existing data before ingestion")
    parser.add_argument(
        "--reset-kb",
        action="store_true",
        help="Delete existing documents and chunks before ingestion, while preserving sessions and messages.",
    )
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting documents")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--sql-schema-path", "-sql", default="sql/schema.sql", help="Path to SQL schema file")
    parser.add_argument("--no-images", action="store_true", help="Skip image extraction")
    parser.add_argument("--no-tables", action="store_true", help="Skip table extraction")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast text-only ingestion: disables image and table extraction",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    no_images = bool(args.no_images or args.fast)
    no_tables = bool(args.no_tables or args.fast)

    config = IngestionConfig(
        documents_folder=args.documents,
        clean_before_ingest=args.clean,
        reset_kb_before_ingest=args.reset_kb,
        sql_schema_path=args.sql_schema_path,
        include_images=not no_images,
        include_tables=not no_tables,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    if config.include_images or config.include_tables:
        logger.warning(
            "Image/table extraction can be slow. Use --fast or --no-images --no-tables for faster text-only ingestion."
        )

    pipeline = DocumentIngestionPipeline(config=config)

    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")

    try:
        start_time = datetime.now()
        results = await pipeline.ingest_documents(progress_callback)
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        print("\n" + "=" * 60)
        print("INGESTION SUMMARY")
        print("=" * 60)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Images extracted: {pipeline.include_images}")
        print(f"Tables extracted: {pipeline.include_tables}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print("=" * 60)

        for result in results:
            if result.status == "skipped":
                logger.info(f"{result.title}: skipped (unchanged)")
            elif result.status == "updated":
                logger.info(f"{result.title}: {result.chunks_created} chunks (updated, {result.processing_time_ms/1000:.1f}s)")
            elif result.chunks_created > 0:
                logger.info(f"{result.title}: {result.chunks_created} chunks ({result.processing_time_ms/1000:.1f}s)")
            else:
                logger.warning(f"{result.title}: Failed to process")

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
