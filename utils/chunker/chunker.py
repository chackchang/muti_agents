import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from agent.config import ChunkingConfig

logger = logging.getLogger(__name__)

load_dotenv()

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# “## 3.1.2 xxx” 的层级由编号里的点决定：3 -> 1 级，3.1 -> 2 级，3.1.2 -> 3 级
_NUMBERED_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+\S")
_STANDARD_SECTION_PREFIX = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:abstract|introduction|related works?|background"
    r"|methods?|methodology|approach|experiments?|experimental|evaluation|results?"
    r"|discussion|conclusions?|references|bibliography|appendix|acknowledg(?:e)?ments?)"
    r"\b",
    re.IGNORECASE,
)
_TOP_LEVEL_HEADINGS = {
    "abstract",
    "article history",
    "article info",
    "data availability statement",
    "disclosure statement",
    "funding",
    "key words",
    "keywords",
    "acknowledgements",
    "acknowledgments",
    "appendix",
}


# @dataclass
# class ChunkingConfig:
#     """PDF 分块的基础配置。"""

#     chunk_size: int = 1000
#     chunk_overlap: int = 200
#     min_chunk_size: int = 100
#     max_chunk_size: int = 2000
#     use_semantic_splitting: bool = True

#     def __post_init__(self):
#         """校验配置。"""
#         if self.chunk_overlap >= self.chunk_size:
#             raise ValueError("Chunk overlap must be less than chunk size")
#         if self.min_chunk_size <= 0:
#             raise ValueError("Minimum chunk size must be positive")


@dataclass
class DocumentChunk:
    """表示一个文档分块。"""

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None
    # GE-M3 稠密向量；稀疏向量为 token id -> weight
    embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Dict[int, float]] = None

    def __post_init__(self):
        """如果未提供 token 数，则自动计算。"""
        if self.token_count is None:
            self.token_count = len(self.content) // 4


@dataclass
class MarkdownSection:
    title: str
    level: int
    path: List[str]
    content: str
    start_line: int
    end_line: int


@dataclass
class ArtifactCandidate:
    artifact_type: str
    start_line_idx: int
    end_line_idx: int
    caption: str
    content: str


class PDFSemanticChunker:
    """PDF 文档的语义分块器。"""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.embeddings = None
        self.semantic_splitter = None

        # 递归切分器
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
        )

    #项目语义分块器，根据内容自动选择模型和参数
    # def _semantic_splitter_for_content(self, content: str) -> SemanticChunker:
    #     embeddings = OpenAIEmbeddings(...)
    #     return SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile")

    def _is_standard_paper_section_title(self, title: str) -> bool:
        """判断是否是论文的常规一级章节。

        只匹配章节名的开头；“RANSAC Line Fitting Algorithm” 这类正文小标题
        虽然含 algorithm，但不能当成一级章节。
        """
        normalized = str(title or "").strip()
        if not normalized:
            return False
        return bool(_STANDARD_SECTION_PREFIX.match(normalized))

    def _build_chunk_title(
        self,
        section: MarkdownSection,
        section_chunk_index: int = 0,
        section_chunk_count: int = 1,
    ) -> str:
        base = " > ".join(section.path).strip() or str(section.title or "").strip() or "Section"
        if section_chunk_count > 1:
            base = f"{base} · Part {section_chunk_index + 1}/{section_chunk_count}"
        if len(base) > 180:
            base = base[:177].rstrip() + "..."
        return base

    @staticmethod
    def _normalize_heading(title: str) -> str:
        """去掉 ``#``、合并空白，用于标题比较和去重。"""
        text = re.sub(r"^#{1,6}\s*", "", str(title or ""))
        text = re.sub(r"\s+", " ", text)
        return text.strip(" :.-_").casefold()

    def _matches_document_title(self, title: str, document_title: Optional[str]) -> bool:
        """判断某个 markdown 标题是否就是 JSON 里的论文标题。"""
        if not document_title:
            return False
        heading = self._normalize_heading(title)
        expected = self._normalize_heading(document_title)
        if not heading or not expected:
            return False
        if heading == expected:
            return True
        # PDF 转换时标题可能被折行或带上多余后缀，长标题允许包含关系
        if len(heading) > 30 and len(expected) > 30:
            return heading in expected or expected in heading
        return False

    def _infer_section_level(
        self,
        title: str,
        markdown_level: int,
        document_title: Optional[str],
        previous_level: int,
    ) -> int:
        """从编号推断章节层级。

        exporter 产出的 markdown 只有 ``##``，不能直接拿 markdown 的 ``#`` 数量
        当层级；论文编号（``3`` / ``3.1`` / ``3.1.2``）才是可靠信号。
        论文标题本身记为 0 级，作为路径根。
        """
        if self._matches_document_title(title, document_title):
            return 0
        numbered = _NUMBERED_HEADING_RE.match(title)
        if numbered:
            return min(numbered.group(1).count(".") + 1, 6)
        if self._is_standard_paper_section_title(title):
            return 1
        if self._normalize_heading(title) in _TOP_LEVEL_HEADINGS:
            return 1
        if markdown_level == 1:
            return 1
        # 无编号标题（如 “RANSAC Line Fitting Algorithm”）挂在上一个章节下面
        return min(max(previous_level, 1) + 1, 6)

    @staticmethod
    def _section_fingerprint(section: MarkdownSection) -> str:
        """段落正文指纹：去掉标题行、空白差异后用于判重。"""
        body = re.sub(r"^#{1,6}\s+.*$", "", section.content, flags=re.MULTILINE)
        return re.sub(r"\s+", " ", body).strip().casefold()

    def _dedupe_sections(self, sections: List[MarkdownSection]) -> List[MarkdownSection]:
        """标题或正文重复时只保留最新的一份。

        Docling 会先输出期刊封面（含一次论文标题），再在正文开头重复标题和
        作者信息；这里记录上一段正文并在发现重复时丢弃旧的那份。
        """
        if len(sections) <= 1:
            return sections

        titles = [self._normalize_heading(section.title) for section in sections]
        texts = [self._section_fingerprint(section) for section in sections]
        dropped: set[int] = set()

        # 同一标题出现多次（封面标题 + 正文标题）时保留后出现的版本
        seen_titles: Dict[str, int] = {}
        for idx, key in enumerate(titles):
            if not key:
                continue
            if key in seen_titles:
                dropped.add(seen_titles[key])
            seen_titles[key] = idx

        seen_texts: Dict[str, int] = {}
        for idx, key in enumerate(texts):
            if not key:
                continue
            if key in seen_texts:
                dropped.add(seen_texts[key])
            seen_texts[key] = idx

        # 新段落开头把上一段完整重复了一遍：保留最新的一份
        for idx in range(1, len(sections)):
            previous, current = texts[idx - 1], texts[idx]
            if len(previous) >= 80 and current.startswith(previous):
                dropped.add(idx - 1)

        return [section for idx, section in enumerate(sections) if idx not in dropped]

    def _split_markdown_sections(
        self,
        content: str,
        document_title: Optional[str] = None,
    ) -> List[MarkdownSection]:
        lines = content.splitlines()
        matches = []
        for idx, line in enumerate(lines, start=1):
            match = _HEADING_RE.match(line)
            if not match:
                continue
            title = str(match.group(2) or "").strip()
            if not title:
                continue
            matches.append((idx, len(match.group(1)), title))

        if not matches:
            return []
        # 处理文档前言
        sections: List[MarkdownSection] = []
        first_heading_line = matches[0][0]
        if first_heading_line > 1:
            preamble_lines = lines[: first_heading_line - 1]
            preamble_text = "\n".join(preamble_lines).strip()
            if preamble_text:
                sections.append(
                    MarkdownSection(
                        title="Document Preamble",
                        level=0,
                        path=["Document Preamble"],
                        content=preamble_text,
                        start_line=1,
                        end_line=first_heading_line - 1,
                    )
                )

        path_stack: List[str] = []
        previous_level = 0
        for idx, (start_line, markdown_level, title) in enumerate(matches):
            level = self._infer_section_level(
                title,
                markdown_level,
                document_title,
                previous_level,
            )
            previous_level = level

            if level == 0:
                path_stack = [title]
            else:
                if len(path_stack) >= level:
                    path_stack = path_stack[: level - 1]
                path_stack.append(title)

            end_line = matches[idx + 1][0] - 1 if idx + 1 < len(matches) else len(lines)
            section_lines = lines[start_line - 1 : end_line]
            section_text = "\n".join(section_lines).strip()
            if not section_text:
                continue
            sections.append(
                MarkdownSection(
                    title=title,
                    level=level,
                    path=list(path_stack),
                    content=section_text,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        return self._dedupe_sections(sections)

    # def _chunk_whole_document(
    #     self,
    #     content: str,
    #     base_metadata: Dict[str, Any],
    # ) -> List[Document]:
    #     doc = Document(page_content=content, metadata=base_metadata)
    #     if self.config.use_semantic_splitting and len(content) > self.config.chunk_size:
    #         try:
    #             chunks = self._semantic_splitter_for_content(content).split_documents([doc])
    #             for chunk in chunks:
    #                 chunk.metadata["chunk_method"] = "semantic"
    #         except Exception as e:
    #             logger.warning("Semantic chunking failed, falling back: %s", e)
    #             chunks = self.fallback_splitter.split_documents([doc])   # 失败就抛，不遮
    #             for chunk in chunks:
    #                 chunk.metadata["chunk_method"] = "fallback"
    #     else:
    #         chunks = self.fallback_splitter.split_documents([doc])
    #         for chunk in chunks:
    #             chunk.metadata["chunk_method"] = "recursive"
    #     return chunks

    def _chunk_section(
        self,
        section: MarkdownSection,
        base_metadata: Dict[str, Any],
        title: str,
        source: str,
        artifact_index_start: int = 0,
    ) -> List[DocumentChunk]:
        # 标题行本身不算正文；只剩标题的段落（论文标题、空的父章节）不单独成块
        body_text = re.sub(
            r"^#{1,6}\s+.*(?:\n|$)", "", section.content, count=1
        ).strip()

        section_metadata = {
            **base_metadata,
            "title": title,
            "source": source,
            "section_title": section.title,
            "section_level": section.level,
            "section_path": list(section.path),
            "section_path_text": " > ".join(section.path),
            "section_start_line": section.start_line,
            "section_end_line": section.end_line,
        }

        artifact_candidates = (
            self._extract_artifacts(section.content)
            if bool(section_metadata.get("include_artifacts", True))
            else []
        )
        artifact_chunks = self._build_artifact_chunks(
            section=section,
            section_metadata=section_metadata,
            artifact_candidates=artifact_candidates,
            artifact_index_start=artifact_index_start,
        )
        artifact_aware_text = self._replace_artifacts_with_placeholders(
            section.content, artifact_candidates
        ).strip()

        if len(body_text) < self.config.min_chunk_size and not artifact_chunks:
            return []
        if len(artifact_aware_text) < self.config.min_chunk_size:
            return artifact_chunks

        section_chunks: List[DocumentChunk] = []
        if len(artifact_aware_text) <= self.config.max_chunk_size:
            section_chunk_count = 1
            chunk_title = self._build_chunk_title(section, 0, section_chunk_count)
            section_path_text = " > ".join(section.path).strip()
            prefix_parts: List[str] = []
            if section_path_text:
                prefix_parts.append(f"[Section: {section_path_text}]")
            prefix = "\n".join(prefix_parts).strip()
            enhanced_content = (
                f"{prefix}\n\n{artifact_aware_text}".strip() if prefix else artifact_aware_text
            )
            section_chunks.append(
                DocumentChunk(
                    content=enhanced_content,
                    index=0,
                    start_char=0,
                    end_char=len(enhanced_content),
                    metadata={
                        **section_metadata,
                        "chunk_method": "section",
                        "raw_chunk_size": len(artifact_aware_text),
                        "chunk_title": chunk_title,
                        "retrieval_title": chunk_title,
                    },
                )
            )
        else:
            docs = self.fallback_splitter.split_documents(
                [Document(page_content=artifact_aware_text, metadata=section_metadata)]
            )
            filtered_docs = []
            for doc_chunk in docs:
                text = doc_chunk.page_content.strip()
                if len(text) >= self.config.min_chunk_size:
                    filtered_docs.append(text)
            section_chunk_count = len(filtered_docs)
            for idx, text in enumerate(filtered_docs):
                chunk_title = self._build_chunk_title(section, idx, section_chunk_count)
                section_path_text = " > ".join(section.path).strip()
                prefix_parts: List[str] = []
                if section_path_text:
                    prefix_parts.append(f"[Section: {section_path_text}]")
                if section_chunk_count > 1:
                    prefix_parts.append(f"[Chunk: {chunk_title}]")
                prefix = "\n".join(prefix_parts).strip()
                enhanced_content = f"{prefix}\n\n{text}".strip() if prefix else text
                section_chunks.append(
                    DocumentChunk(
                        content=enhanced_content,
                        index=idx,
                        start_char=0,
                        end_char=len(enhanced_content),
                        metadata={
                            **section_metadata,
                            "chunk_method": "section_recursive",
                            "section_chunk_index": idx,
                            "section_chunk_count": section_chunk_count,
                            "raw_chunk_size": len(text),
                            "chunk_title": chunk_title,
                            "retrieval_title": chunk_title,
                        },
                    )
                )
            return artifact_chunks + section_chunks

        section_chunk_count = len(section_chunks)
        for idx, chunk in enumerate(section_chunks):
            chunk.index = idx
            chunk.metadata["section_chunk_index"] = idx
            chunk.metadata["section_chunk_count"] = section_chunk_count
        return artifact_chunks + section_chunks

    def _extract_caption(self, artifact_type: str, lines: List[str], start_idx: int) -> str:
        current = lines[start_idx].strip() if 0 <= start_idx < len(lines) else ""
        window = lines[start_idx : min(len(lines), start_idx + 3)]
        caption_re = re.compile(
            r"^\s*(table|fig(?:ure)?|algorithm)\s*[\.:]?\s*\d+[\w\.\-: ]*",
            re.IGNORECASE,
        )
        for line in window:
            candidate = line.strip()
            if not candidate:
                continue
            if caption_re.match(candidate):
                return candidate[:160]
        if artifact_type == "figure":
            for line in window:
                candidate = line.strip()
                if candidate.lower().startswith(("fig.", "figure")):
                    return candidate[:160]
        if current:
            return current[:120]
        defaults = {
            "table": "Table artifact",
            "figure": "Figure artifact",
            "algorithm": "Algorithm artifact",
        }
        return defaults.get(artifact_type, "Artifact")

    def _extract_artifacts(self, section_text: str) -> List[ArtifactCandidate]:
        lines = section_text.splitlines()
        artifacts: List[ArtifactCandidate] = []
        used = [False] * len(lines)

        def mark_used(start: int, end: int) -> None:
            for i in range(start, end + 1):
                if 0 <= i < len(used):
                    used[i] = True

        # Table blocks
        i = 0
        while i < len(lines):
            if used[i]:
                i += 1
                continue
            if "|" in lines[i] and i + 1 < len(lines) and re.search(r"\|\s*:?-{3,}", lines[i + 1]):
                start = i
                j = i + 2
                while j < len(lines) and "|" in lines[j].strip():
                    j += 1
                end = j - 1
                block = "\n".join(lines[start : end + 1]).strip()
                if block:
                    artifacts.append(
                        ArtifactCandidate(
                            artifact_type="table",
                            start_line_idx=start,
                            end_line_idx=end,
                            caption=self._extract_caption("table", lines, max(0, start - 1)),
                            content=block,
                        )
                    )
                    mark_used(start, end)
                i = j
                continue
            i += 1

        # Figure blocks
        for idx, line in enumerate(lines):
            if used[idx]:
                continue
            stripped = line.strip()
            if (
                "<!-- image -->" in stripped.lower()
                or stripped.startswith("![")
                or re.match(r"^\s*(fig\.|figure)\s*\d+", stripped, re.IGNORECASE)
            ):
                start = idx
                end = idx
                if "<!-- image -->" in stripped.lower() or stripped.startswith("!["):
                    next_idx = idx + 1
                    if next_idx < len(lines) and not lines[next_idx].strip():
                        next_idx += 1
                    if next_idx < len(lines) and re.match(
                        r"^\s*(fig\.|figure)\s*\d+", lines[next_idx].strip(), re.IGNORECASE
                    ):
                        end = next_idx
                elif idx + 1 < len(lines) and re.match(
                    r"^\s*(fig\.|figure)\s*\d+", lines[idx + 1].strip(), re.IGNORECASE
                ):
                    end = idx + 1
                block = "\n".join(lines[start : end + 1]).strip()
                if block:
                    artifacts.append(
                        ArtifactCandidate(
                            artifact_type="figure",
                            start_line_idx=start,
                            end_line_idx=end,
                            caption=self._extract_caption("figure", lines, start),
                            content=block,
                        )
                    )
                    mark_used(start, end)

        # Algorithm blocks
        strong_algo_trigger = re.compile(r"(algorithm\s*\d+|pseudo-?code)", re.IGNORECASE)
        io_signal = re.compile(r"^\s*(input|output)\s*:", re.IGNORECASE)
        step_signal = re.compile(
            r"^\s*(step\s*\d+[:\.\)]|(\d+[\.\)]\s+\S+)|([ivxlcdm]+[\.\)]\s+\S+))",
            re.IGNORECASE,
        )

        def build_algorithm_block(start: int) -> ArtifactCandidate:
            j = start + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    break
                if re.match(r"^#{1,6}\s+", next_line):
                    break
                if re.search(r"\|\s*:?-{3,}", next_line):
                    break
                if next_line.startswith("![") or "<!-- image -->" in next_line.lower():
                    break
                j += 1
            end = max(start, j - 1)
            block = "\n".join(lines[start : end + 1]).strip()
            return ArtifactCandidate(
                artifact_type="algorithm",
                start_line_idx=start,
                end_line_idx=end,
                caption=self._extract_caption("algorithm", lines, start),
                content=block,
            )

        # Strong trigger: Algorithm N / Pseudocode
        for idx, line in enumerate(lines):
            if used[idx]:
                continue
            if strong_algo_trigger.search(line):
                candidate = build_algorithm_block(idx)
                if candidate.content:
                    artifacts.append(candidate)
                    mark_used(candidate.start_line_idx, candidate.end_line_idx)

        # Weak trigger requires BOTH IO clues and step-like structure within a local window
        window_size = 8
        for start in range(len(lines)):
            if used[start]:
                continue
            end = min(len(lines), start + window_size)
            window_lines = lines[start:end]
            io_count = sum(1 for ln in window_lines if io_signal.search(ln.strip()))
            step_count = sum(1 for ln in window_lines if step_signal.search(ln.strip()))
            if io_count >= 1 and step_count >= 2:
                first_signal = None
                for idx in range(start, end):
                    if used[idx]:
                        continue
                    striped = lines[idx].strip()
                    if io_signal.search(striped) or step_signal.search(striped):
                        first_signal = idx
                        break
                if first_signal is None:
                    continue
                candidate = build_algorithm_block(first_signal)
                if candidate.content:
                    artifacts.append(candidate)
                    mark_used(candidate.start_line_idx, candidate.end_line_idx)

        artifacts.sort(key=lambda item: item.start_line_idx)
        return artifacts

    def _make_context(self, text: str, max_chars: int = 300) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return ""
        if len(normalized) <= max_chars:
            return normalized
        return normalized[:max_chars].rstrip() + "..."

    def _build_artifact_chunks(
        self,
        section: MarkdownSection,
        section_metadata: Dict[str, Any],
        artifact_candidates: List[ArtifactCandidate],
        artifact_index_start: int,
    ) -> List[DocumentChunk]:
        lines = section.content.splitlines()
        chunks: List[DocumentChunk] = []
        label_map = {"table": "Table", "figure": "Figure", "algorithm": "Algorithm"}
        method_map = {
            "table": "artifact_table",
            "figure": "artifact_figure",
            "algorithm": "artifact_algorithm",
        }
        for local_idx, artifact in enumerate(artifact_candidates):
            before_text = "\n".join(lines[max(0, artifact.start_line_idx - 3) : artifact.start_line_idx])
            after_text = "\n".join(
                lines[artifact.end_line_idx + 1 : min(len(lines), artifact.end_line_idx + 4)]
            )
            context_before = self._make_context(before_text, max_chars=300)
            context_after = self._make_context(after_text, max_chars=300)
            section_path_text = " > ".join(section.path).strip()
            header = (
                f"[Artifact: {label_map.get(artifact.artifact_type, 'Artifact')}]\n"
                f"[Section: {section_path_text or section.title}]\n"
                f"[Caption: {artifact.caption}]"
            )
            body = (
                f"{header}\n\n"
                f"Context before:\n{context_before or '[None]'}\n\n"
                f"Artifact content:\n{artifact.content}\n\n"
                f"Context after:\n{context_after or '[None]'}"
            )
            global_artifact_index = artifact_index_start + local_idx
            chunks.append(
                DocumentChunk(
                    content=body.strip(),
                    index=local_idx,
                    start_char=0,
                    end_char=len(body),
                    metadata={
                        **section_metadata,
                        "content_type": "artifact",
                        "artifact_type": artifact.artifact_type,
                        "chunk_method": method_map.get(artifact.artifact_type, "artifact"),
                        "artifact_index": global_artifact_index,
                        "caption": artifact.caption,
                        "raw_artifact_content": artifact.content,
                        "artifact_start_line": section.start_line + artifact.start_line_idx,
                        "artifact_end_line": section.start_line + artifact.end_line_idx,
                        "context_before": context_before,
                        "context_after": context_after,
                        "retrieval_title": artifact.caption,
                        "raw_chunk_size": len(artifact.content),
                    },
                )
            )
        return chunks

    def _replace_artifacts_with_placeholders(
        self,
        section_text: str,
        artifact_candidates: List[ArtifactCandidate],
    ) -> str:
        lines = section_text.splitlines()
        if not artifact_candidates:
            return section_text
        artifact_map = {a.start_line_idx: a for a in artifact_candidates}
        skip_until = -1
        output: List[str] = []
        for i, line in enumerate(lines):
            if i <= skip_until:
                continue
            artifact = artifact_map.get(i)
            if artifact:
                label = artifact.artifact_type.capitalize()
                output.append(f"[{label} omitted. See artifact chunk: {artifact.caption}]")
                skip_until = artifact.end_line_idx
            else:
                output.append(line)
        return "\n".join(output)

    def chunk_content(
        self,
        content: str,
        title: str = "PDF Document",
        source: str = "pdf",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """将 PDF 内容切分为语义片段，并转换为 DocumentChunk 对象。"""
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "content_type": "pdf",
            **(metadata or {}),
        }
        try:
            sections = self._split_markdown_sections(content, document_title=title)
            if sections:
                final_chunks: List[DocumentChunk] = []
                artifact_index = 0
                for section in sections:
                    section_chunks = self._chunk_section(
                        section=section,
                        base_metadata=base_metadata,
                        title=title,
                        source=source,
                        artifact_index_start=artifact_index,
                    )
                    artifact_index += sum(
                        1 for c in section_chunks if c.metadata.get("content_type") == "artifact"
                    )
                    final_chunks.extend(section_chunks)
                if final_chunks:
                    total_chunks = len(final_chunks)
                    for idx, chunk in enumerate(final_chunks):
                        text = chunk.content.strip()
                        chunk.index = idx
                        chunk.metadata.update(
                            {
                                "chunk_index": idx,
                                "total_chunks": total_chunks,
                                "chunk_size": len(text),
                            }
                        )
                    return final_chunks
        except Exception as e:
            logger.warning(f"Section-aware chunking failed, falling back to whole-document splitting: {e}")

        # chunks = self._chunk_whole_document(content, base_metadata)

        # final_chunks = []
        # for i, chunk in enumerate(chunks):
        #     text = chunk.page_content.strip()
        #     if len(text) >= self.config.min_chunk_size:
        #         chunk.metadata.update(
        #             {
        #                 "chunk_index": i,
        #                 "total_chunks": len(chunks),
        #                 "chunk_size": len(text),
        #             }
        #         )
        #         final_chunks.append(
        #             DocumentChunk(
        #                 content=text,
        #                 index=i,
        #                 start_char=0,
        #                 end_char=len(text),
        #                 metadata=chunk.metadata,
        #             )
        #         )

        # return final_chunks

        # 没有任何 markdown 章节（例如 PDFium 兜底的纯文本）时，明确返回空列表；
        # 否则函数会隐式返回 None，调用方 extend(None) 会抛 TypeError。
        logger.warning(
            "No markdown sections found for %s; no chunks produced", title
        )
        return []

    @staticmethod
    def _coerce_pdf_document(item: Any) -> Tuple[str, Dict[str, Any]]:
        """把 exporter JSON 或 LangChain Document 统一成 (content, metadata)。"""
        if isinstance(item, dict):
            content = item.get("content") or item.get("page_content") or ""
            metadata = dict(item.get("metadata") or {})
            if item.get("title") and not metadata.get("title"):
                metadata["title"] = item["title"]
            if item.get("id") and not metadata.get("document_id"):
                metadata["document_id"] = item["id"]
            if item.get("content_type") and not metadata.get("content_type"):
                metadata["content_type"] = item["content_type"]
            if item.get("keywords") and not metadata.get("keywords"):
                metadata["keywords"] = item["keywords"]
            return str(content), metadata

        content = getattr(item, "page_content", None)
        if content is None and isinstance(item, str):
            content = item
        metadata = dict(getattr(item, "metadata", None) or {})
        return str(content or ""), metadata

    def chunk_pdf_documents(self, documents: List[Any]) -> List[DocumentChunk]:
        """对 PDF 文档分块。

        既接受 ``utils.exporters.json`` 输出的 dict（``title`` / ``content`` /
        ``metadata``），也接受 LangChain ``Document``，可以直接接在 exporter 后面。
        """
        all_chunks: List[DocumentChunk] = []

        for item in documents or []:
            content, metadata = self._coerce_pdf_document(item)
            if not content.strip():
                continue

            chunks = self.chunk_content(
                content=content,
                title=str(metadata.get("title") or "PDF Document"),
                source=str(metadata.get("source") or "pdf"),
                metadata=metadata,
            )
            all_chunks.extend(chunks)

        return all_chunks


def create_chunker(config: ChunkingConfig) -> PDFSemanticChunker:
    """使用基础配置创建 PDF 分块器。"""
    return PDFSemanticChunker(config)
