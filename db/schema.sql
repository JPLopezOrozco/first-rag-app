CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents(
    id              bigserial PRIMARY KEY,
    filename        text NOT NULL UNIQUE,
    content_type    text NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT now() 
);

CREATE TABLE IF NOT EXISTS chunks(
    id          text PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk       int  NOT NULL,
    content     text NOT NULL,
    n_tokens    int,
    metadata    jsonb NOT NULL DEFAULT '{}',
    embedding   vector(1024) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('spanish', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_document_idx  ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx       ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);