CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    attributes JSONB NOT NULL DEFAULT '{}',
    name_embedding vector(1024),
    mention_count INT NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    weight FLOAT NOT NULL DEFAULT 1.0,
    attributes JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS communities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    level INT NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    summary_embedding vector(1024),
    entity_count INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS community_members (
    community_id UUID NOT NULL REFERENCES communities(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (community_id, entity_id)
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    content_embedding vector(1024),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW indexes for vector similarity search
CREATE INDEX IF NOT EXISTS idx_entities_name_embedding
    ON entities USING hnsw (name_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS idx_communities_summary_embedding
    ON communities USING hnsw (summary_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS idx_messages_content_embedding
    ON conversation_messages USING hnsw (content_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- Trigram index for fuzzy entity name matching
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm
    ON entities USING gin (name gin_trgm_ops);

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_entities_namespace_type
    ON entities (namespace, entity_type);

CREATE INDEX IF NOT EXISTS idx_relationships_source_target
    ON relationships (source_id, target_id);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON conversation_messages (conversation_id, created_at);
