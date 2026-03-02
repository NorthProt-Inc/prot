# Compaction-Driven Memory Architecture

## Goal

Replace the per-exchange memory extraction pipeline with a compaction-triggered system.
Redesign the DB schema around a 4-layer human-inspired memory model.
Add time-decay scoring for recency-weighted retrieval.
Upgrade embedding model from voyage-context-3 to voyage-4-large.

## Current State

- Memory extraction: Claude call every 3 exchanges (`memory_extraction_interval`)
- Storage: entities + relationships + communities + community_members + conversation_messages
- Cost per 3 exchanges: 1 Claude + 1 Voyage embed + 2 Voyage rerank
- Cost per 15 exchanges: +1 Claude (community summary) + 1 Voyage embed
- Embedding model: voyage-context-3 ($0.18/M tokens, 1024-dim)
- Reranker: rerank-2.5 (instruction-following capable, not currently using instructions)
- Compaction: configured at 50K tokens, no detection or utilization of compaction events

## Design

### Trigger Model

No real-time extraction. Memory pipeline fires only at two points:

1. **Compaction event** — `pause_after_compaction: true` intercepts the compaction summary
2. **Shutdown** — forced summarization using the known default compaction prompt

```
Normal operation:
  User ↔ LLM (zero extraction cost)

Compaction fires (~50K tokens):
  API returns stop_reason: "compaction"
  → Extract compaction summary text
  → Send to Haiku/Flash with structured extraction prompt
  → Parse into 4 memory layers
  → Embed with voyage-4-large + store in DB
  → Resume conversation

Shutdown (no compaction occurred):
  → Collect current messages
  → Send to Haiku/Flash with compaction-equivalent summary prompt
  → Same extraction pipeline as above
  → Store before exit
```

### Why Compaction-Only Works

- Before compaction, the full conversation is in the LLM context window — no need to extract
- Compaction = natural "save point" where context would otherwise be lost
- The compaction summary is already a Claude-distilled view — higher quality than raw extraction
- Eliminates all per-exchange Claude API calls for memory

### Compaction Summary Extraction (2-Step)

**Step 1: Get the summary.** Either from `pause_after_compaction` (compaction event) or
from a manual Haiku/Flash call with the default compaction prompt (shutdown).

The default compaction prompt (publicly known):

> You have written a partial transcript for the initial task above. Please write a
> summary of the transcript. The purpose of this summary is to provide continuity so
> you can continue to make progress towards solving the task in a future context, where
> the raw history above may not be accessible and will be replaced with this summary.
> Write down anything that would be helpful, including the state, next steps, learnings
> etc. You must wrap your summary in a `<summary></summary>` block.

**Step 2: Structured extraction.** Send the summary text to a cheap model (Haiku 4.5 or
Gemini 3 Flash Preview) with a prompt that parses it into the 4 memory layers.

The extraction prompt outputs JSON with sections for semantic, episodic, emotional, and
procedural memories. Each section has defined fields matching the DB schema.

### DB Schema — 4-Layer Memory

Drop all existing tables (entities, relationships, communities, community_members,
conversation_messages). Fresh start.

#### Layer 1: Semantic Memory (facts, knowledge, preferences)

```sql
CREATE TABLE IF NOT EXISTS semantic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category TEXT NOT NULL,           -- person, preference, fact, skill, relationship
    subject TEXT NOT NULL,            -- "user", "Axel", "Python"
    predicate TEXT NOT NULL,          -- "likes", "uses", "works as"
    object TEXT NOT NULL,             -- "coffee", "VS Code", "developer"
    confidence FLOAT NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'compaction',
    mention_count INT NOT NULL DEFAULT 1,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_semantic_embedding
    ON semantic_memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_semantic_subject
    ON semantic_memories USING gin (subject gin_trgm_ops);
```

SPO triples unify entities + relationships into one table. Deduplication by
(subject, predicate, object) — on conflict, increment mention_count and update confidence.

#### Layer 2: Episodic Memory (experiences, conversation episodes)

```sql
CREATE TABLE IF NOT EXISTS episodic_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    summary TEXT NOT NULL,
    topics TEXT[] NOT NULL DEFAULT '{}',
    emotional_tone TEXT,              -- warm, tense, playful, curious, etc.
    significance FLOAT NOT NULL DEFAULT 0.5,
    duration_turns INT NOT NULL DEFAULT 0,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_episodic_embedding
    ON episodic_memories USING hnsw (embedding vector_cosine_ops);
```

Stores the compaction summary itself as an episode. The summary is the richest
representation of what happened in a conversation.

#### Layer 3: Emotional Memory (emotional context, bonding)

```sql
CREATE TABLE IF NOT EXISTS emotional_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    emotion TEXT NOT NULL,            -- joy, frustration, curiosity, gratitude, etc.
    trigger_context TEXT NOT NULL,    -- what triggered this emotion
    intensity FLOAT NOT NULL DEFAULT 0.5,
    episode_id UUID REFERENCES episodic_memories(id) ON DELETE SET NULL,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_emotional_embedding
    ON emotional_memories USING hnsw (embedding vector_cosine_ops);
```

Captures emotional context from conversations. Essential for Jarvis-like bonding.
Linked to episodes for context.

#### Layer 4: Procedural Memory (habits, behavioral patterns)

```sql
CREATE TABLE IF NOT EXISTS procedural_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern TEXT NOT NULL,            -- "asks about weather in the morning"
    frequency TEXT,                   -- daily, weekly, occasional
    confidence FLOAT NOT NULL DEFAULT 0.5,
    last_observed TIMESTAMPTZ,
    observation_count INT NOT NULL DEFAULT 1,
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_procedural_embedding
    ON procedural_memories USING hnsw (embedding vector_cosine_ops);
```

Patterns build up over multiple compaction events. Low confidence initially,
increasing with repeated observations.

### Embedding Model Upgrade

**voyage-context-3 → voyage-4-large**

| Aspect | voyage-context-3 (old) | voyage-4-large (new) |
|--------|----------------------|---------------------|
| Price  | $0.18/M tokens       | $0.12/M tokens      |
| Architecture | Standard        | MoE (40% lower serving cost) |
| Shared space | No             | Yes (v4 family intercompat) |
| Dimensions | 1024 (configurable) | 1024 (configurable) |

Rationale: New schema stores short SPO triples and episode summaries. The contextual
embedding feature of voyage-context-3 (cross-chunk document awareness) provides limited
benefit for self-contained memory entries. voyage-4-large offers better general retrieval
quality at 33% lower cost.

Future option: If retrieval volume grows, store with voyage-4-lite ($0.02/M) and query
with voyage-4-large ($0.12/M) — they share the same embedding space.

**Reranker stays at rerank-2.5.** Instruction-following capability available for future
optimization (e.g., "prioritize recent emotional context") but not used in initial release.

### Time-Decay Scoring (AdaptiveDecayCalculator)

Ported from genuine-axel. Applied at **query time**, not storage.

Core formula:
```
effective_score = cosine_similarity * decay_factor * importance_weight
```

Where:
```python
decay_factor = importance * exp(-effective_rate * hours_since_creation)
effective_rate = BASE_DECAY_RATE * type_multiplier / stability * (1 - resistance)
```

Modulating factors:
- **stability**: `1 + 0.3 * log(1 + mention_count)` — more mentions = slower decay
- **resistance**: `min(1.0, connection_count * 0.1)` — more connections = slower decay
- **type_multiplier**: fact(0.3) < preference(0.5) < insight(0.7) < conversation(1.0)
- **recency boost**: old but recently accessed → 1.3x
- **min retention floor**: never decays below `importance * MIN_RETENTION`

Simplified for prot (no channel_mentions, no dynamic_decay):
- `mention_count` maps to `access_count`
- `category` maps to `memory_type`
- Applied in `pre_load_context()` after vector search, before reranking

### RAG Retrieval Flow

```
User query
  → embed query (voyage-4-large)
  → search all 4 memory tables (cosine similarity)
  → apply time-decay to scores
  → merge and rerank top-k results (rerank-2.5)
  → format into Block 2 context
  → inject into system prompt
```

Block 2 format includes sections: semantic facts, relevant episodes, emotional context,
known patterns. Token budget: ~4096 tokens (configurable).

### Code Changes

#### Delete
- `src/prot/community.py` — Louvain detection no longer needed
- `tests/test_community.py`

#### Rewrite
- `src/prot/memory.py` — new extraction logic (compaction summary → 4-layer parsing)
- `src/prot/graphrag.py` — new storage layer (4 tables instead of entity/relationship)
- `src/prot/schema.sql` — new 4-table schema

#### Modify
- `src/prot/llm.py` — add `pause_after_compaction: true`, detect compaction events in stream
- `src/prot/pipeline.py` — replace `_extract_memories_bg` with compaction handler,
  add shutdown summarization
- `src/prot/context.py` — update `pre_load_context` for 4-table retrieval
- `src/prot/config.py` — remove `memory_extraction_interval`, `community_*` settings;
  update voyage model to `voyage-4-large`; add decay settings
- `src/prot/embeddings.py` — switch from `contextualized_embed()` to standard `embed()` API

#### New
- `src/prot/decay.py` — AdaptiveDecayCalculator (ported from genuine-axel, simplified)

### Configuration

```python
# Compaction
compaction_trigger: int = 50000        # existing
pause_after_compaction: bool = True    # new

# Memory extraction
memory_extraction_model: str = "claude-haiku-4-5-20251001"  # existing, repurposed
# Alternative: gemini-3-flash-preview for Korean extraction quality

# Embeddings
voyage_model: str = "voyage-4-large"   # upgraded from voyage-context-3
voyage_rerank_model: str = "rerank-2.5"  # unchanged

# Decay
decay_base_rate: float = 0.002
decay_min_retention: float = 0.1

# RAG
rag_token_budget: int = 4096          # existing
rag_top_k: int = 10                   # existing
```

### Migration

- Drop all existing tables on startup if schema version changes
- No data migration — fresh start
- Old data was already backed up as CSV (data/db/, since deleted)

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Compaction summary language (English vs Korean) | Extraction prompt handles both; persona is Korean-first |
| Structured extraction quality from cheap model | Validate JSON output; retry once on parse failure |
| Shutdown before compaction = short-lived memories | Shutdown handler runs manual summarization |
| App crash = memory loss | Acceptable — voice conversations are ephemeral by nature |
| SPO triples produce short embeddings | Concatenate subject+predicate+object for embedding input |
| voyage-4-large not available in region | Fallback to voyage-4 (same shared space, $0.06/M) |

## Out of Scope

- Restart context loading from previous compaction summary (separate task)
- Dynamic decay (per-user adaptive behavior) — single-user system
- Real-time extraction fallback — explicitly rejected in favor of compaction-only
- Reranker instruction-following — future optimization after baseline is stable
