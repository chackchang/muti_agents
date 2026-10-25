import hashlib
import html
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .pdf_title_parser import extract_document_title


def _ensure_hf_cache(preferred: Optional[str] = None) -> None:
    """确保 HuggingFace 有可写缓存目录。

    Windows 上用户目录的 ``.cache`` 可能因权限或沙箱不可写；此时退回项目内
    ``.tmp/hf``。必须在导入 huggingface_hub / docling 之前调用。
    """
    if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
        return

    fallback = Path(__file__).resolve().parents[3] / ".tmp" / "hf"
    candidates = [Path(preferred)] if preferred else []
    # 项目内已经下载过模型时优先复用，避免重复下载或依赖系统缓存
    if (fallback / "hub").is_dir():
        candidates.append(fallback)
    candidates.append(Path.home() / ".cache" / "huggingface")
    if fallback not in candidates:
        candidates.append(fallback)

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            continue
        os.environ["HF_HOME"] = str(candidate)
        if candidate == fallback:
            logger.warning("HuggingFace 默认缓存不可写，改用 %s", candidate)
        return

    raise RuntimeError("没有可写的 HuggingFace 缓存目录，请设置 HF_HOME")


def _load_docling_dependencies():
    """Load Docling only for an actual extraction attempt.

    Docling imports Torch transitively. Keeping that import out of module load
    preserves the PDFium fallback on hosts where the local Torch DLL is absent.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    return DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions

logger = logging.getLogger(__name__)


# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # 禁用符号链接警告


from agent.config import PDFExtractionConfig
# @dataclass
# class PDFExtractionConfig:
#     """支持 GPU 的 PDF 提取配置。"""
#     enable_ocr: bool = True
#     images_scale: float = 2.0
#     include_images: bool = True
#     include_tables: bool = True 
#     image_output_dir: str = "documents/.vision_assets_2"
#     max_images: Optional[int] = None
#     hf_home: Optional[str] = None

_FRONT_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ANY_HEADING = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_ABSTRACT_LABEL = re.compile(r"^(?:#{1,6}\s*)?abstract\b[.:]?\s*(.*)$", re.IGNORECASE)
_KEYWORDS_LABEL = re.compile(
    r"^(?:#{1,6}\s*)?(?:key\s*words?|index\s+terms)\b[.:]?\s*(.*)$", re.IGNORECASE
)
_ARTICLE_HISTORY_LABEL = re.compile(
    r"^(?:#{1,6}\s*)?(?:article\s+history"
    r"|a\s*r\s*t\s*i\s*c\s*l\s*e\s+h\s*i\s*s\s*t\s*o\s*r\s*y)\b[.:]?\s*(.*)$",
    re.IGNORECASE,
)
_ARTICLE_INFO_LABEL = re.compile(
    r"^(?:#{1,6}\s*)?a\s*r\s*t\s*i\s*c\s*l\s*e\s+i\s*n\s*f\s*o\s*$", re.IGNORECASE
)
_INTRO_LABEL = re.compile(r"^(?:#{1,6}\s*)?(?:\d+\.?\s*)?introduction\b", re.IGNORECASE)
_JOURNAL_CONTEXT = re.compile(
    r"issn|journal homepage|contents lists available|sciencedirect|tandfonline"
    r"|informa|elsevier|springer|wiley",
    re.IGNORECASE,
)
_AFFILIATION = re.compile(
    r"(university|institute|department|school of|college|laborator|academy"
    r"|faculty|centre|center for|hospital)",
    re.IGNORECASE,
)
_AUTHOR_SPLIT = re.compile(r"\s*(?:,|;|&|\band\b)\s*")
_AUTHOR_NAME = re.compile(
    r"^[A-Z][A-Za-z.\-']*(?:\s+[A-Z][A-Za-z.\-']*)+(?:\s*[a-z\d*†‡]+)?$"
)
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)
_COVER_NOISE = re.compile(
    r"^(?:to cite this article|to link to this article|published online"
    r"|submit your article|article views|citing articles|view related articles"
    r"|view crossmark|crossmark|open access)\b",
    re.IGNORECASE,
)
_FRONT_LABELS = (
    _ABSTRACT_LABEL,
    _KEYWORDS_LABEL,
    _ARTICLE_HISTORY_LABEL,
    _ARTICLE_INFO_LABEL,
)


def _heading_text(line: str) -> Optional[str]:
    match = _FRONT_HEADING.match(line)
    return match.group(2).strip() if match else None


def _normalize_title_text(value: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", str(value or ""))
    text = re.sub(r"\s+", " ", text)
    return text.strip(" :.-_").casefold()


def _matches_title(text: str, title: str) -> bool:
    heading = _normalize_title_text(text)
    expected = _normalize_title_text(title)
    if not heading or not expected:
        return False
    if heading == expected:
        return True
    return len(heading) > 30 and len(expected) > 30 and (
        heading in expected or expected in heading
    )


def _is_front_label(line: str) -> bool:
    stripped = line.strip()
    return any(pattern.match(stripped) for pattern in _FRONT_LABELS)


def _extract_label_block(
    lines: List[str],
    pattern: re.Pattern,
    limit: int,
    max_chars: int = 3000,
    stop_at_blank: bool = False,
) -> Optional[Tuple[int, int, str]]:
    """找到 "Abstract/Keywords" 这类标签，返回 (起始行, 结束行, 正文)。

    ``stop_at_blank`` 用于 Keywords / Article history：它们通常只有一行，
    后面的空行就是正文开始的位置，不能把整个 Introduction 吞进来。
    """
    for idx in range(min(limit, len(lines))):
        match = pattern.match(lines[idx].strip())
        if not match:
            continue
        inline = match.group(1).strip()
        body = [inline] if inline else []
        collected = len(inline)
        end = idx + 1
        while end < min(limit, len(lines)):
            stripped = lines[end].strip()
            if not stripped:
                if stop_at_blank and body:
                    break
                end += 1
                continue
            if _FRONT_HEADING.match(lines[end]) or _is_front_label(stripped):
                break
            body.append(stripped)
            collected += len(stripped)
            end += 1
            if collected >= max_chars:
                break
        return idx, end, "\n".join(body).strip()
    return None


def _find_title_line(
    lines: List[str],
    title: str,
    limit: int,
    abstract_idx: Optional[int] = None,
) -> Optional[int]:
    """找到正文里的论文标题行，优先取摘要之前最后一次出现。"""
    if not title:
        return None
    matches = []
    for idx in range(min(limit, len(lines))):
        text = _heading_text(lines[idx]) or lines[idx].strip()
        if _matches_title(text, title):
            matches.append(idx)
    if not matches:
        return None
    if abstract_idx is not None:
        before = [idx for idx in matches if idx < abstract_idx]
        if before:
            return before[-1]
    return matches[-1]


def _looks_like_author_line(line: str) -> bool:
    if not 3 <= len(line) <= 400 or not re.search(r"[A-Za-z]", line):
        return False
    if _JOURNAL_CONTEXT.search(line) or _URL.search(line):
        return False
    parts = [part.strip() for part in _AUTHOR_SPLIT.split(line) if part.strip()]
    if len(parts) < 2:
        return False
    name_like = sum(1 for part in parts if _AUTHOR_NAME.match(part))
    return name_like >= max(2, len(parts) // 2)


def extract_front_matter(content: str, title: str = "") -> Tuple[str, Dict[str, Any]]:
    """把 PDF 前言中的期刊、作者、单位、关键词拆成 metadata。

    只处理论文开头（Introduction 前后一段），返回 ``(清理后的正文, metadata 增量)``。
    期刊/作者/单位/关键词/收稿信息会从正文里删除；摘要保留在正文中，分块后
    作为一个独立章节参与检索，不重复写入 metadata。
    """
    text = str(content or "")
    # PDFium 兜底输出的是无标题的纯文本，此时不做前言切分，避免误删正文
    if not _ANY_HEADING.search(text):
        return text, {}
    lines = text.splitlines()

    intro_idx = next(
        (idx for idx, line in enumerate(lines) if _INTRO_LABEL.match(line.strip())),
        None,
    )
    # Docling 的双栏排版常把 ARTICLE HISTORY / KEYWORDS 排到 Introduction
    # 标题之后，所以前言窗口要给 Introduction 留一段余量。
    limit = min(len(lines), max((intro_idx + 40) if intro_idx is not None else 0, 120))

    abstract_block = _extract_label_block(
        lines, _ABSTRACT_LABEL, limit, max_chars=4000
    )
    keywords_block = _extract_label_block(
        lines, _KEYWORDS_LABEL, limit, max_chars=600, stop_at_blank=True
    )
    history_block = _extract_label_block(
        lines, _ARTICLE_HISTORY_LABEL, limit, max_chars=300, stop_at_blank=True
    )
    article_info_idx = next(
        (idx for idx in range(limit) if _ARTICLE_INFO_LABEL.match(lines[idx].strip())),
        None,
    )

    abstract_idx = abstract_block[0] if abstract_block else None
    title_idx = _find_title_line(lines, title, limit, abstract_idx)

    remove: set[int] = set()
    # 摘要保留在正文里（chunker 会把它切成独立 chunk），其余前言字段删除
    for block in (keywords_block, history_block):
        if block:
            remove.update(range(block[0], block[1]))
    if article_info_idx is not None:
        remove.add(article_info_idx)

    # 期刊名通常和 ISSN / journal homepage 出现在同一页眉区域
    journal = ""
    journal_end = title_idx if title_idx is not None else limit
    for idx in range(min(journal_end, limit)):
        if not _FRONT_HEADING.match(lines[idx]):
            continue
        heading = _heading_text(lines[idx]) or ""
        context = " ".join(lines[max(0, idx - 3) : idx + 4])
        if heading and not _matches_title(heading, title) and _JOURNAL_CONTEXT.search(context):
            journal = re.sub(r"\s+", " ", heading)
            remove.add(idx)
            break
    # 只清理论文标题之前的页眉信息（ISSN / journal homepage），避免误删正文
    for idx in range(min(journal_end, limit)):
        if _JOURNAL_CONTEXT.search(lines[idx]):
            remove.add(idx)

    authors: List[str] = []
    affiliations: List[str] = []
    if title_idx is not None:
        region_end = limit
        for block in (abstract_block, keywords_block, history_block):
            if block and block[0] > title_idx:
                region_end = min(region_end, block[0])
        for idx in range(title_idx + 1, region_end):
            if idx in remove:
                continue
            stripped = lines[idx].strip()
            if not stripped or stripped.startswith("<!--") or "©" in stripped:
                continue
            if _URL.search(stripped) or _COVER_NOISE.match(stripped):
                continue
            if _AFFILIATION.search(stripped):
                affiliations.append(stripped)
                remove.add(idx)
                continue
            if _looks_like_author_line(stripped):
                authors.append(stripped.rstrip(" ,;"))
                remove.add(idx)

    metadata: Dict[str, Any] = {}
    if journal:
        metadata["journal"] = html.unescape(journal)
    if authors:
        metadata["authors"] = html.unescape(" ".join(authors))
    if affiliations:
        metadata["affiliations"] = html.unescape("\n".join(affiliations))
    if keywords_block and keywords_block[2]:
        metadata["keywords"] = [
            html.unescape(term.strip())
            for term in re.split(r"[;,]", keywords_block[2])
            if term.strip()
        ]
    if history_block and history_block[2]:
        metadata["article_history"] = html.unescape(
            re.sub(r"\s+", " ", history_block[2])
        ).strip()

    cleaned_content = re.sub(
        r"\n{3,}",
        "\n\n",
        "\n".join(line for idx, line in enumerate(lines) if idx not in remove),
    ).strip("\n")
    return cleaned_content, metadata


class PDFExtractor:
    """使用 Docling 的 PDF 内容提取器，支持 GPU。"""
    
    def __init__(self, config: PDFExtractionConfig = None):
        self.config = config or PDFExtractionConfig()
        self.setup_converter()
    
    def setup_converter(self):
        """按 ``config.mineru`` 选择提取后端，最终都落在 ``self.converter`` 上。"""
        self.converter = None
        self.docling_error: Optional[str] = None
        self.converter_error: Optional[str] = None
        if self.config.mineru:
            self._setup_mineru_converter()
        else:
            self._setup_docling_converter()

    def _setup_mineru_converter(self):
        """MinerU 依赖与主项目冲突，这里只创建子进程调用器。"""
        try:
            from .mineru import MinerUExtractor

            self.converter = MinerUExtractor(
                binary=self.config.mineru_bin,
                tier=self.config.mineru_tier,
                timeout=self.config.mineru_timeout,
                home=self.config.mineru_home,
                model_source=self.config.mineru_model_source,
            )
        except Exception as exc:  # 可执行文件缺失等
            self.converter_error = str(exc)
            logger.warning(
                "MinerU is unavailable; PDFium fallback will be used: %s", exc
            )
            self.converter = None

    def _setup_docling_converter(self):
        """使用选项配置 Docling 文档转换器。"""
        try:
            _ensure_hf_cache(self.config.hf_home)
            DocumentConverter, PdfFormatOption, InputFormat, PdfPipelineOptions = _load_docling_dependencies()
        except Exception as exc:  # pragma: no cover - depends on the host Torch runtime
            self.docling_error = str(exc)
            self.converter_error = str(exc)
            logger.warning("Docling is unavailable; PDFium fallback will be used: %s", self.docling_error)
            return
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.config.enable_ocr
        # 图片由方舟视觉适配器解释，避免 Docling 再启动一套未配置的视觉模型。
        pipeline_options.do_picture_description = False
        pipeline_options.do_table_structure = self.config.include_tables
        pipeline_options.images_scale = self.config.images_scale
        pipeline_options.do_formula_enrichment = self.config.enable_formula
        # 不加这一项时 doc.pictures 有记录，但 get_image() 返回 None。
        pipeline_options.generate_picture_images = self.config.include_images
        try:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )
        except Exception as e:
            logger.warning("Docling initialization failed; PDFium fallback will be used: %s", e)
            self.docling_error = str(e)
            self.converter_error = str(e)
            self.converter = None

    def extract_pdf_content(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """从单个 PDF 文件中提取内容。"""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting content from: {pdf_path.name}")
        start_time = time.time()
                
        try:
            if self.config.mineru:
                content_text, metadata = self._extract_with_mineru(pdf_path, start_time)
                doc = None
            else:
                if self.converter is None:
                    raise RuntimeError(
                        self.converter_error or "Docling converter is unavailable"
                    )
                # 调用 Docling 时加载模型权重，自动去 HuggingFace 缓存目录找
                result = self.converter.convert(str(pdf_path))
                end_time = time.time()
                doc = result.document

                content_text = doc.export_to_markdown()
                metadata = {
                    "source": str(pdf_path),
                    "title": extract_document_title(content_text, str(pdf_path)),
                    "title_source": "parsed_pdf_content",
                    "processing_time": round(end_time - start_time, 2),
                    "pages": len(doc.pages),
                    "texts": len(doc.texts),
                    "pictures": len(doc.pictures),
                    "tables": len(doc.tables),
                    "extraction_method": "docling",
                    "content_type": "pdf",
                }
        except Exception as exc:
            backend = "MinerU" if self.config.mineru else "Docling"
            logger.warning(
                "%s extraction failed for %s; using PDFium fallback: %s",
                backend,
                pdf_path.name,
                exc,
            )
            content_text, metadata = self._extract_with_pdfium_fallback(pdf_path, start_time)
            doc = None
        # 作者/单位/期刊/关键词拆到 metadata；摘要只保留在正文里供检索
        content_text, front_matter = extract_front_matter(
            content_text, str(metadata.get("title") or "")
        )
        metadata.update(front_matter)
        # MinerU 分支的图片已在 _extract_with_mineru 里单独落盘
        if (
            not self.config.mineru
            and self.config.include_images
            and doc is not None
            and doc.pictures
        ):
            asset_dir = Path(self.config.image_output_dir)
            asset_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
            saved = []
            for index, picture in enumerate(doc.pictures[: self.config.max_images]):
                try:
                    image = picture.get_image(doc)
                    if image is None:
                        continue
                    image_path = asset_dir / f"{digest}-figure-{index + 1}.png"
                    image.save(image_path, format="PNG")
                    saved.append({
                        "index": index,
                        "path": str(image_path),
                        "caption": picture.caption_text(doc).strip(),
                    })
                except Exception as exc:
                    logger.warning("Failed to export figure %s from %s: %s", index + 1, pdf_path.name, exc)
            metadata["vision_assets"] = saved
            metadata["vision_asset_count"] = len(saved)
        return content_text, metadata

    def _extract_with_mineru(
        self, pdf_path: Path, start_time: float
    ) -> Tuple[str, Dict[str, Any]]:
        """调用独立环境里的 MinerU CLI，返回与 Docling 分支同构的 markdown/metadata。"""
        if self.converter is None:
            raise RuntimeError(self.converter_error or "MinerU extractor is unavailable")

        result = self.converter.parse(
            pdf_path,
            asset_dir=self.config.image_output_dir,
            include_images=self.config.include_images,
            max_images=self.config.max_images,
        )
        metadata: Dict[str, Any] = {
            "source": str(pdf_path),
            "title": extract_document_title(result.markdown, str(pdf_path)),
            "title_source": "parsed_pdf_content",
            "processing_time": round(time.time() - start_time, 2),
            "pages": result.pages,
            "texts": result.texts,
            "pictures": result.raw_image_count,
            "tables": result.tables,
            "extraction_method": "mineru",
            "content_type": "pdf",
        }
        if result.assets:
            metadata["vision_assets"] = result.assets
            metadata["vision_asset_count"] = len(result.assets)
        return result.markdown, metadata

    def _extract_with_pdfium_fallback(self, pdf_path: Path, start_time: float) -> Tuple[str, Dict[str, Any]]:
        """Docling 模型不可用时，使用已安装的 PDFium 读取文本并渲染一页图像。

        这不是 Docling 的替代品，只保证单图视觉入库仍可完成，避免首次模型下载阻塞任务。
        """
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(pdf_path))
        page_texts = []
        target_page = 0
        caption = ""
        figure_pattern = re.compile(r"(?:fig(?:ure)?\.?\s*\d+[^\n]{0,240})", re.IGNORECASE)
        for index in range(len(document)):
            page = document.get_page(index)
            text = page.get_textpage().get_text_range()
            page_texts.append(text)
            match = figure_pattern.search(text)
            if match and not caption:
                target_page = index
                caption = re.sub(r"\s+", " ", match.group(0)).strip()

        metadata = {
            "source": str(pdf_path),
            "title": extract_document_title("\n".join(page_texts), str(pdf_path)),
            "title_source": "parsed_pdf_content",
            "processing_time": round(time.time() - start_time, 2),
            "pages": len(document),
            "texts": sum(bool(value.strip()) for value in page_texts),
            "pictures": 0,
            "tables": 0,
            "extraction_method": "pdfium_fallback",
            "content_type": "pdf",
        }
        if self.config.include_images:
            asset_dir = Path(self.config.image_output_dir)
            asset_dir.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
            asset_path = asset_dir / f"{digest}-figure-page-{target_page + 1}.png"
            page = document.get_page(target_page)
            page.render(scale=2).to_pil().save(asset_path, format="PNG")
            metadata["vision_assets"] = [{
                "index": 0,
                "path": str(asset_path),
                "caption": caption or f"PDF page {target_page + 1} visual evidence",
                "page": target_page + 1,
                "asset_kind": "rendered_page_fallback",
            }]
            metadata["vision_asset_count"] = 1
        return "\n\n".join(page_texts), metadata
    

def create_pdf_extractor(config: PDFExtractionConfig = None) -> PDFExtractor:
    """创建 PDF 提取器实例。"""
    return PDFExtractor(config)


def read_pdf_document(path: str) -> tuple[str, dict]:
    """读取 PDF，返回 (markdown, metadata)。

    Docling / PDFium 都是可选依赖，只有真正解析 PDF 时才加载。
    """
    try:
        content, metadata = PDFExtractor().extract_pdf_content(path)
    except ImportError as exc:
        raise RuntimeError(
            "PDF 提取需要安装 docling 或 pypdfium2，例如：pip install docling"
        ) from exc

    metadata = dict(metadata or {})
    metadata.update(
        {
            "adapter": "pdf",
            "source_type": "docs",
            "content_type": "text/markdown",
        }
    )
    return content, metadata

   