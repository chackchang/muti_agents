from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from dataclasses import dataclass, field
from typing import Optional
@dataclass
class PDFExtractionConfig:
    """支持 GPU 的 PDF 提取配置。"""
    enable_ocr: bool = True
    images_scale: float = 2.0
    include_images: bool = True
    include_tables: bool = True
    image_output_dir: str = "documents/.vision_assets_mineru"
    max_images: Optional[int] = None
    hf_home: Optional[str] = None
    enable_formula: bool = True

    # True 时走独立环境里的 MinerU CLI（子进程），否则走 Docling
    mineru: bool = True
    mineru_bin: Optional[str] = None
    mineru_tier: str = "standard"
    mineru_timeout: int = 3600
    mineru_home: Optional[str] = None
    mineru_model_source: str = "modelscope"


@dataclass
class ChunkingConfig:
    """PDF 分块的基础配置。"""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    # use_semantic_splitting: bool = True

    def __post_init__(self):
        """校验配置。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")
        

class IngestionConfig(BaseModel):
    # 文档来源与运行开关
    documents_folder: str = Field(default="data")
    clean_before_ingest: bool = Field(default=False)
    reset_kb_before_ingest: bool = Field(default=False)
    sql_schema_path: str = Field(default="sql/schema.sql")
    exporter_out_path: str = Field(default="exported_documents")
    # 提取选项
    include_images: bool = Field(default=True)
    include_tables: bool = Field(default=True)
    # 分块选项（默认值与 ChunkingConfig / 既有库保持一致，改动会改变指纹）
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


class IngestionResult(BaseModel):
    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    # 本次导入对这篇文档做了什么：created / updated / skipped / failed
    status: str = "created"
