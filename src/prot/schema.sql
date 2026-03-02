CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Drop old schema (fresh start per design doc)
DROP TABLE IF EXISTS community_members CASCADE;
DROP TABLE IF EXISTS communities CASCADE;
DROP TABLE IF EXISTS relationships CASCADE;
DROP TABLE IF EXISTS entities CASCADE;
DROP TABLE IF EXISTS conversation_messages CASCADE;
DROP TABLE IF EXISTS semantic_memories CASCADE;
DROP TABLE IF EXISTS episodic_memories CASCADE;
DROP TABLE IF EXISTS emotional_memories CASCADE;
DROP TABLE IF EXISTS procedural_memories CASCADE;

-- Layer 1: Semantic Memory (facts, knowledge, preferences)
CREATE TABLE IF NOT EXISTS semantic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'compaction',
    mention_count INT NOT NULL DEFAULT 1,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (subject, predicate, object)
);

CREATE INDEX IF NOT EXISTS idx_semantic_embedding
    ON semantic_memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_semantic_subject
    ON semantic_memories USING gin (subject gin_trgm_ops);

-- Layer 2: Episodic Memory (conversation episodes)
CREATE TABLE IF NOT EXISTS episodic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    summary TEXT NOT NULL,
    topics TEXT[] NOT NULL DEFAULT '{}',
    emotional_tone TEXT,
    significance FLOAT NOT NULL DEFAULT 0.5,
    duration_turns INT NOT NULL DEFAULT 0,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_episodic_embedding
    ON episodic_memories USING hnsw (embedding vector_cosine_ops);

-- Layer 3: Emotional Memory (emotional context, bonding)
CREATE TABLE IF NOT EXISTS emotional_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    emotion TEXT NOT NULL,
    trigger_context TEXT NOT NULL,
    intensity FLOAT NOT NULL DEFAULT 0.5,
    episode_id UUID REFERENCES episodic_memories(id) ON DELETE SET NULL,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_emotional_embedding
    ON emotional_memories USING hnsw (embedding vector_cosine_ops);

-- Layer 4: Procedural Memory (habits, behavioral patterns)
CREATE TABLE IF NOT EXISTS procedural_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern TEXT NOT NULL UNIQUE,
    frequency TEXT,
    confidence FLOAT NOT NULL DEFAULT 0.5,
    last_observed TIMESTAMPTZ,
    observation_count INT NOT NULL DEFAULT 1,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_procedural_embedding
    ON procedural_memories USING hnsw (embedding vector_cosine_ops);

-- Conversation messages (retained for message persistence)
CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON conversation_messages (conversation_id, created_at);
