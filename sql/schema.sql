-- 幂等建表脚本：由 utils.ingest 在初始化时通过 execute_init_sql 执行。
-- 向量检索已迁到 Qdrant；PostgreSQL 只保留文档/会话/分块正文/图证据/论文图。
-- paper_graph_nodes.embedding 仍用 pgvector（论文级 profile 的余弦相似度在 SQL 里算）。

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- 会话与会话记忆
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     text,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at  timestamptz
);

CREATE TABLE IF NOT EXISTS messages (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        text NOT NULL,
    content     text NOT NULL,
    metadata    jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages (session_id, created_at);

CREATE TABLE IF NOT EXISTS session_memory_snapshots (
    session_id            uuid PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    version               integer NOT NULL DEFAULT 0,
    covered_message_count integer NOT NULL DEFAULT 0,
    summary               jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at            timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_memory_archives (
    id                      uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id              uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_snapshot_version integer NOT NULL,
    covered_message_count   integer NOT NULL DEFAULT 0,
    summary                 jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at              timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (session_id, source_snapshot_version)
);

CREATE TABLE IF NOT EXISTS conversation_query_rewrites (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_message_id uuid NOT NULL UNIQUE,
    original_query  text NOT NULL,
    rewritten_query text NOT NULL,
    rewrite_reason  text NOT NULL DEFAULT '',
    model_used      boolean NOT NULL DEFAULT false,
    metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ---------------------------------------------------------------------------
-- 跨会话用户记忆
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_memory_profiles (
    user_id    text PRIMARY KEY,
    version    integer NOT NULL DEFAULT 0,
    profile    jsonb NOT NULL DEFAULT '{}'::jsonb,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_memory_items (
    id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            text NOT NULL,
    kind               text NOT NULL,
    memory_key         text NOT NULL,
    value              text NOT NULL,
    status             text NOT NULL DEFAULT 'active',
    source_session_id  uuid,
    source_message_id  uuid,
    metadata           jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at         timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    superseded_at      timestamptz
);
CREATE INDEX IF NOT EXISTS idx_user_memory_items_user
    ON user_memory_items (user_id, kind, memory_key);

-- ---------------------------------------------------------------------------
-- 文档与分块
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    -- 幂等三件套：逻辑ID（稳定，标识“哪篇文档”）、指纹（内容+配置哈希）、版本号。
    logical_id     text NOT NULL DEFAULT '',
    fingerprint    text NOT NULL DEFAULT '',
    version        integer NOT NULL DEFAULT 1,
    is_current     boolean NOT NULL DEFAULT true,
    title          text NOT NULL DEFAULT '',
    source         text NOT NULL DEFAULT '',
    content        text NOT NULL DEFAULT '',
    metadata       jsonb NOT NULL DEFAULT '{}'::jsonb,
    pdf_blob       bytea,
    pdf_media_type text,
    created_at     timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at     timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents (created_at);

-- 迁移：为已存在的 documents 表补齐幂等字段（幂等 DDL，反复执行安全）。
ALTER TABLE documents ADD COLUMN IF NOT EXISTS logical_id  text NOT NULL DEFAULT '';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS fingerprint text NOT NULL DEFAULT '';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS version     integer NOT NULL DEFAULT 1;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS is_current  boolean NOT NULL DEFAULT true;

-- 存量行回填：逻辑ID取自 source，指纹取自 content（content 指纹与 Python 侧
-- 配置指纹不同，首次重跑会走 update 替换，但不会重复入库）。
UPDATE documents
SET logical_id = CASE
        WHEN source IS NOT NULL AND source <> ''
            THEN substr(encode(digest(source, 'sha256'), 'hex'), 1, 16)
        ELSE substr(id::text, 1, 16)
    END
WHERE logical_id = '';

UPDATE documents
SET fingerprint = encode(digest(COALESCE(content, ''), 'sha256'), 'hex')
WHERE fingerprint = '';

-- 幂等由应用层保证（查询 is_current + 版本号最大者，再决定 skip/update），
-- 不建唯一索引：更新路径有“新版本已写入、旧版本待删除”的短暂共存窗口，
-- 且存量库可能存在重复行，唯一索引会卡住迁移。
CREATE INDEX IF NOT EXISTS idx_documents_logical_id ON documents (logical_id);

-- 分块正文；向量不再存这里，改存 Qdrant（同名 dense + sparse 命名向量）。
CREATE TABLE IF NOT EXISTS chunks (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content      text NOT NULL,
    chunk_index  integer NOT NULL DEFAULT 0,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    token_count  integer
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_index
    ON chunks (document_id, chunk_index);

-- 图/表证据（图片二进制与结构化分析）。
CREATE TABLE IF NOT EXISTS artifacts (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    artifact_type   text NOT NULL,
    caption         text NOT NULL DEFAULT '',
    page_number     integer,
    section_path    text,
    context_before  text,
    context_after   text,
    raw_content     text,
    structured_data jsonb NOT NULL DEFAULT '{}'::jsonb,
    retrieval_text  text,
    image_blob      bytea,
    image_media_type text
);
CREATE INDEX IF NOT EXISTS idx_artifacts_document ON artifacts (document_id);

-- ---------------------------------------------------------------------------
-- 翻译与标注
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_translations (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_language text NOT NULL,
    source_sha256   text NOT NULL,
    content_markdown text NOT NULL,
    model           text NOT NULL DEFAULT '',
    created_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, target_language, source_sha256)
);

CREATE TABLE IF NOT EXISTS document_translation_profiles (
    document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_language text NOT NULL,
    source_sha256   text NOT NULL,
    profile_json    jsonb NOT NULL DEFAULT '{}'::jsonb,
    model           text NOT NULL DEFAULT '',
    updated_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, target_language, source_sha256)
);

CREATE TABLE IF NOT EXISTS document_selection_translations (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id      uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_language  text NOT NULL,
    source_sha256    text NOT NULL,
    selection_sha256 text NOT NULL,
    source_text      text NOT NULL,
    translated_text  text NOT NULL,
    context_before   text,
    context_after    text,
    model            text NOT NULL DEFAULT '',
    created_at       timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, target_language, source_sha256, selection_sha256)
);

CREATE TABLE IF NOT EXISTS document_annotations (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number integer,
    page_x      double precision NOT NULL DEFAULT 0.5,
    page_y      double precision NOT NULL DEFAULT 0.5,
    quote       text NOT NULL DEFAULT '',
    note        text NOT NULL DEFAULT '',
    color       text NOT NULL DEFAULT 'yellow',
    created_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_document_annotations_document
    ON document_annotations (document_id);

-- ---------------------------------------------------------------------------
-- 论文知识图
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS paper_graph_nodes (
    document_id   uuid PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    abbreviation  text NOT NULL DEFAULT '',
    profile_text  text NOT NULL DEFAULT '',
    embedding     vector,
    metadata      jsonb NOT NULL DEFAULT '{}'::jsonb,
    graph_version integer NOT NULL DEFAULT 1,
    created_at    timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS paper_graph_localizations (
    document_id     uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_language text NOT NULL DEFAULT 'zh',
    source_sha256   text NOT NULL,
    schema_version  integer NOT NULL,
    status          text NOT NULL DEFAULT 'pending',
    model           text,
    content         jsonb,
    validation      jsonb,
    error_message   text,
    updated_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_id, target_language, source_sha256, schema_version)
);

CREATE TABLE IF NOT EXISTS paper_graph_edges (
    source_document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    target_document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    relation_type      text NOT NULL,
    score              double precision NOT NULL DEFAULT 0,
    evidence           jsonb NOT NULL DEFAULT '{}'::jsonb,
    graph_version      integer NOT NULL DEFAULT 1,
    created_at         timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_document_id, target_document_id, relation_type)
);
CREATE INDEX IF NOT EXISTS idx_paper_graph_edges_source
    ON paper_graph_edges (source_document_id);
CREATE INDEX IF NOT EXISTS idx_paper_graph_edges_target
    ON paper_graph_edges (target_document_id);

-- ---------------------------------------------------------------------------
-- SQL 函数
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_document_chunks(doc_id uuid)
RETURNS TABLE(chunk_id text, content text, chunk_index integer, metadata jsonb)
LANGUAGE sql
STABLE
AS $$
    SELECT id::text, content, chunk_index, metadata
    FROM chunks
    WHERE document_id = doc_id
    ORDER BY chunk_index;
$$;
