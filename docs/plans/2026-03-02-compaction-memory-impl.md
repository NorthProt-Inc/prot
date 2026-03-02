# Compaction-Driven Memory Architecture — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use execute-flow to implement this plan task-by-task.

**Goal:** Replace per-exchange memory extraction with compaction-triggered 4-layer memory system, upgrade Voyage embeddings, add time-decay scoring.

**Architecture:** Compaction events (~50K tokens) trigger a 2-step pipeline: (1) capture compaction summary, (2) send to Haiku/Flash for structured extraction into 4 memory layers (semantic, episodic, emotional, procedural). Shutdown performs forced summarization using the known default compaction prompt. Time-decay scoring (ported from genuine-axel) applies at query time.

**Tech Stack:** Anthropic Claude API (compaction beta), Voyage AI (voyage-4-large, rerank-2.5), asyncpg/pgvector, Python 3.12+

**Design doc:** `docs/plans/2026-03-02-compaction-memory-design.md`

---

### Task 1: Delete community.py, db_gc.py, and all references

Remove the Louvain community detection module, the old DB GC script (operates on entities/relationships tables that no longer exist), and their dependencies.

**Files:**
- Delete: `src/prot/community.py`
- Delete: `tests/test_community.py`
- Delete: `scripts/db_gc.py` — operates on old entity/relationship schema, no longer meaningful
- Delete: `tests/test_db_gc.py` — tests pure helpers from db_gc.py
- Modify: `src/prot/logging/constants.py:25` — remove `"community"` entry, add `"decay"` entry
- Modify: `pyproject.toml:26` — remove `networkx>=3.2` dependency

**Step 1: Delete files**

```bash
rm src/prot/community.py tests/test_community.py scripts/db_gc.py tests/test_db_gc.py
```

**Step 2: Update logging constants**

In `src/prot/logging/constants.py`, remove `"community"` and add `"decay"`:
```python
# Remove:
    "community":        ("COM", "\033[38;5;183m"),
# Add:
    "decay":            ("DCY", "\033[38;5;183m"),
```

**Step 3: Remove `networkx` dependency**

In `pyproject.toml`, remove:
```
    "networkx>=3.2",
```

**Step 4: Run tests**

```bash
uv run pytest -x -q
```
Expected: All pass (community.py is only imported lazily inside pipeline.py `startup()`, so no module-level import failures).

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: delete community.py, db_gc.py, remove networkx dependency"
```

---

### Task 2: Update config.py — remove old settings, add new ones

**Files:**
- Modify: `src/prot/config.py`

**Step 1: Apply config changes**

Remove these settings:
- `memory_extraction_interval: int = 3`
- `community_rebuild_interval: int = 5`
- `community_min_entities: int = 5`

Rename:
- `voyage_context_model` → `voyage_model` with new default `"voyage-4-large"`

Add new settings:
- `pause_after_compaction: bool = True`
- `decay_base_rate: float = 0.002`
- `decay_min_retention: float = 0.1`

Final config.py memory/embeddings section:
```python
    # Embeddings
    voyage_api_key: str = ""
    voyage_model: str = "voyage-4-large"

    # Reranker
    rerank_model: str = "rerank-2.5"
    rerank_top_k: int = 5

    # Memory
    memory_extraction_model: str = "claude-haiku-4-5-20251001"
    rag_context_target_tokens: int = 4096
    rag_top_k: int = 10

    # Compaction
    pause_after_compaction: bool = True

    # Decay
    decay_base_rate: float = 0.002
    decay_min_retention: float = 0.1
```

**Step 2: Run tests**

```bash
uv run pytest -x -q
```
Expected: Some failures in tests referencing `voyage_context_model` or `memory_extraction_interval`.

**Step 3: Fix test references**

Update any test that patches `settings.voyage_context_model` to use `settings.voyage_model`.
Update any test that patches `settings.memory_extraction_interval` — remove those patches.

**Step 4: Run tests again**

```bash
uv run pytest -x -q
```
Expected: All pass.

**Step 5: Commit**

```bash
git add src/prot/config.py tests/
git commit -m "refactor: update config for compaction memory architecture"
```

---

### Task 3: Rewrite schema.sql — 4-layer memory tables

**Files:**
- Rewrite: `src/prot/schema.sql`

**Step 1: Write new schema**

Replace entire `src/prot/schema.sql` with:

```sql
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
```

**Step 2: Commit**

```bash
git add src/prot/schema.sql
git commit -m "refactor: rewrite schema for 4-layer memory model"
```

---

### Task 4: Write AdaptiveDecayCalculator

Port the decay calculator from genuine-axel, simplified for prot (no channel_mentions, no dynamic_decay, no native module).

**Files:**
- Create: `src/prot/decay.py`
- Create: `tests/test_decay.py`

**Step 1: Write test file `tests/test_decay.py`**

```python
"""Tests for AdaptiveDecayCalculator — time-decay scoring for memory retrieval."""

import math
import pytest
from prot.decay import AdaptiveDecayCalculator, MEMORY_TYPE_MULTIPLIERS


class TestAdaptiveDecayCalculator:
    def setup_method(self):
        self.calc = AdaptiveDecayCalculator()

    def test_no_decay_at_zero_hours(self):
        result = self.calc.calculate(importance=0.8, hours_passed=0.0)
        assert abs(result - 0.8) < 1e-6

    def test_basic_decay(self):
        result = self.calc.calculate(importance=1.0, hours_passed=100.0)
        expected = 1.0 * math.exp(-0.002 * 100.0)
        assert abs(result - expected) < 1e-6

    def test_minimum_retention_floor(self):
        result = self.calc.calculate(importance=1.0, hours_passed=100000.0)
        assert result >= 1.0 * 0.1 - 1e-6

    def test_access_count_slows_decay(self):
        no_access = self.calc.calculate(importance=1.0, hours_passed=500.0, access_count=0)
        high_access = self.calc.calculate(importance=1.0, hours_passed=500.0, access_count=100)
        assert high_access > no_access

    def test_connection_count_slows_decay(self):
        no_conn = self.calc.calculate(importance=1.0, hours_passed=500.0, connection_count=0)
        high_conn = self.calc.calculate(importance=1.0, hours_passed=500.0, connection_count=10)
        assert high_conn > no_conn

    def test_fact_decays_slower_than_conversation(self):
        conv = self.calc.calculate(importance=1.0, hours_passed=500.0, memory_type="conversation")
        fact = self.calc.calculate(importance=1.0, hours_passed=500.0, memory_type="fact")
        assert fact > conv

    def test_recency_paradox_boost(self):
        # Old (>168h) but recently accessed (<24h)
        with_boost = self.calc.calculate(
            importance=1.0, hours_passed=200.0, last_access_hours=10.0
        )
        # Old (>168h) but NOT recently accessed (>24h)
        without_boost = self.calc.calculate(
            importance=1.0, hours_passed=200.0, last_access_hours=50.0
        )
        assert with_boost > without_boost

    def test_unknown_memory_type_uses_default(self):
        result = self.calc.calculate(importance=1.0, hours_passed=100.0, memory_type="unknown")
        default = self.calc.calculate(importance=1.0, hours_passed=100.0, memory_type="conversation")
        assert abs(result - default) < 1e-6

    def test_memory_type_multipliers_defined(self):
        assert MEMORY_TYPE_MULTIPLIERS["fact"] == 0.3
        assert MEMORY_TYPE_MULTIPLIERS["preference"] == 0.5
        assert MEMORY_TYPE_MULTIPLIERS["insight"] == 0.7
        assert MEMORY_TYPE_MULTIPLIERS["conversation"] == 1.0

    def test_calculate_batch(self):
        memories = [
            {"importance": 0.8, "hours_passed": 50.0},
            {"importance": 1.0, "hours_passed": 200.0, "access_count": 10},
            {"importance": 0.5, "hours_passed": 0.0},
        ]
        results = self.calc.calculate_batch(memories)
        assert len(results) == 3
        assert abs(results[2] - 0.5) < 1e-6  # zero hours = no decay

    def test_calculate_batch_empty(self):
        assert self.calc.calculate_batch([]) == []

    def test_custom_config(self):
        calc = AdaptiveDecayCalculator(base_rate=0.01, min_retention=0.2)
        result = calc.calculate(importance=1.0, hours_passed=100000.0)
        assert result >= 1.0 * 0.2 - 1e-6
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_decay.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'prot.decay'`

**Step 3: Implement `src/prot/decay.py`**

```python
"""Adaptive decay calculator for memory importance scoring.

Ported from genuine-axel's AdaptiveDecayCalculator, simplified:
- No channel_mentions / channel_diversity
- No circadian stability / dynamic_decay
- No native C++ module
- Pure Python only
"""

from __future__ import annotations

import math

# Memory type-specific decay multipliers (lower = slower decay)
MEMORY_TYPE_MULTIPLIERS: dict[str, float] = {
    "fact": 0.3,
    "preference": 0.5,
    "insight": 0.7,
    "conversation": 1.0,
}

# Recency paradox thresholds
_RECENCY_AGE_HOURS = 168       # 1 week
_RECENCY_ACCESS_HOURS = 24     # accessed within 24h
_RECENCY_BOOST = 1.3

# Stability/resistance constants
_ACCESS_STABILITY_K = 0.3
_RELATION_RESISTANCE_K = 0.1


class AdaptiveDecayCalculator:
    """Time-decay scoring for memory retrieval.

    Core formula:
        effective_rate = base_rate * type_mult / stability * (1 - resistance)
        decayed = importance * exp(-effective_rate * hours_passed)

    Modulating factors:
        - stability: 1 + 0.3 * log(1 + access_count) — more access = slower decay
        - resistance: min(1.0, connection_count * 0.1) — more connections = slower decay
        - type_multiplier: fact(0.3) < preference(0.5) < insight(0.7) < conversation(1.0)
        - recency boost: old but recently accessed → 1.3x
        - min retention floor: never below importance * min_retention
    """

    def __init__(
        self,
        base_rate: float = 0.002,
        min_retention: float = 0.1,
    ) -> None:
        self.base_rate = base_rate
        self.min_retention = min_retention

    def calculate(
        self,
        importance: float,
        hours_passed: float,
        access_count: int = 0,
        connection_count: int = 0,
        last_access_hours: float = -1.0,
        memory_type: str = "conversation",
    ) -> float:
        """Calculate decayed importance score."""
        stability = 1 + _ACCESS_STABILITY_K * math.log(1 + access_count)
        resistance = min(1.0, connection_count * _RELATION_RESISTANCE_K)
        type_mult = MEMORY_TYPE_MULTIPLIERS.get(memory_type, 1.0)

        effective_rate = self.base_rate * type_mult / stability * (1 - resistance)
        decayed = importance * math.exp(-effective_rate * hours_passed)

        # Recency paradox: old memory recently accessed gets boost
        if (
            last_access_hours >= 0
            and hours_passed > _RECENCY_AGE_HOURS
            and last_access_hours < _RECENCY_ACCESS_HOURS
        ):
            decayed *= _RECENCY_BOOST

        return max(decayed, importance * self.min_retention)

    def calculate_batch(self, memories: list[dict]) -> list[float]:
        """Calculate decayed importance for a batch of memories."""
        return [
            self.calculate(
                importance=m.get("importance", 0.5),
                hours_passed=m.get("hours_passed", 0.0),
                access_count=m.get("access_count", 0),
                connection_count=m.get("connection_count", 0),
                last_access_hours=m.get("last_access_hours", -1.0),
                memory_type=m.get("memory_type", "conversation"),
            )
            for m in memories
        ]
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_decay.py -v
```
Expected: All pass.

**Step 5: Commit**

```bash
git add src/prot/decay.py tests/test_decay.py
git commit -m "feat: add AdaptiveDecayCalculator for time-decay memory scoring"
```

---

### Task 5: Rewrite embeddings.py — switch to voyage-4-large

Switch from `contextualized_embed()` to standard `embed()` API. Rename methods to remove `_contextual` suffix.

**Files:**
- Rewrite: `src/prot/embeddings.py`
- Rewrite: `tests/test_embeddings.py`

**Step 1: Write new test file `tests/test_embeddings.py`**

```python
"""Tests for AsyncVoyageEmbedder — Voyage AI voyage-4-large embeddings."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prot.embeddings import AsyncVoyageEmbedder


@pytest.mark.asyncio
class TestAsyncVoyageEmbedder:
    async def test_embed_query(self):
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vector = await embedder.embed_query("search term")
            assert len(vector) == 1024
            mock_client.embed.assert_called_once_with(
                texts=["search term"],
                model="voyage-4-large",
                input_type="query",
            )

    async def test_embed_texts(self):
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_client.embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vectors = await embedder.embed_texts(["desc A", "desc B", "desc C"])
            assert len(vectors) == 3
            mock_client.embed.assert_called_once_with(
                texts=["desc A", "desc B", "desc C"],
                model="voyage-4-large",
                input_type="document",
            )

    async def test_close_without_close_method(self):
        mock_client = MagicMock(spec=[])
        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            await embedder.close()  # should not raise
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/test_embeddings.py -v
```
Expected: FAIL — old methods still exist, new ones don't.

**Step 3: Rewrite `src/prot/embeddings.py`**

```python
import voyageai

from prot.config import settings
from prot.logging import logged


async def _close_voyage_client(client) -> None:
    """Close a Voyage AI client (duck-typed for SDK version compat)."""
    if hasattr(client, "close"):
        await client.close()
    elif hasattr(client, "aclose"):
        await client.aclose()


class AsyncVoyageEmbedder:
    """Async embedding client using Voyage AI voyage-4-large."""

    def __init__(self, api_key: str | None = None):
        self._client = voyageai.AsyncClient(
            api_key=api_key or settings.voyage_api_key,
        )

    async def close(self) -> None:
        await _close_voyage_client(self._client)

    @logged(slow_ms=1000)
    async def embed_query(self, text: str) -> list[float]:
        """Embed single query text (input_type='query')."""
        result = await self._client.embed(
            texts=[text],
            model=settings.voyage_model,
            input_type="query",
        )
        return result.embeddings[0]

    @logged(slow_ms=2000)
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts independently (input_type='document')."""
        result = await self._client.embed(
            texts=texts,
            model=settings.voyage_model,
            input_type="document",
        )
        return result.embeddings
```

**Step 4: Run embeddings tests**

```bash
uv run pytest tests/test_embeddings.py -v
```
Expected: All pass.

**Step 5: Run full test suite — expect some failures**

```bash
uv run pytest -x -q
```
Expected: Failures in `test_memory.py` (references `embed_texts_contextual`, `embed_query_contextual`). These will be fixed in Task 7 when memory.py is rewritten.

**Step 6: Commit**

```bash
git add src/prot/embeddings.py tests/test_embeddings.py
git commit -m "refactor: switch embeddings from voyage-context-3 to voyage-4-large"
```

---

### Task 6: Rewrite graphrag.py — 4-layer memory store

Replace entity/relationship/community CRUD with 4-layer memory table operations.

**Files:**
- Rewrite: `src/prot/graphrag.py`
- Rewrite: `tests/test_graphrag.py`

**Step 1: Write `tests/test_graphrag.py`**

```python
"""Tests for MemoryStore — 4-layer memory storage with pgvector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from prot.graphrag import MemoryStore


def make_mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


def mock_record(**kwargs):
    record = MagicMock()
    record.keys.return_value = kwargs.keys()
    record.__getitem__ = MagicMock(side_effect=kwargs.__getitem__)
    record.__iter__ = MagicMock(return_value=iter(kwargs))
    record.items.return_value = kwargs.items()
    record.__len__ = MagicMock(return_value=len(kwargs))
    return record


class TestUpsertSemantic:
    async def test_inserts_spo_triple(self):
        pool, conn = make_mock_pool()
        store = MemoryStore(pool)
        new_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=new_id))

        result = await store.upsert_semantic(
            category="person", subject="user", predicate="likes",
            object_="coffee", embedding=[0.1] * 1024,
        )
        assert result == new_id
        call = conn.fetchrow.call_args
        assert "semantic_memories" in call.args[0]
        assert "ON CONFLICT" in call.args[0]

    async def test_uses_provided_conn(self):
        pool, _ = make_mock_pool()
        store = MemoryStore(pool)
        ext_conn = AsyncMock()
        ext_conn.fetchrow = AsyncMock(return_value=mock_record(id=uuid4()))

        await store.upsert_semantic(
            category="fact", subject="sky", predicate="is", object_="blue",
            conn=ext_conn,
        )
        ext_conn.fetchrow.assert_awaited_once()
        pool.acquire.assert_not_called()


class TestInsertEpisodic:
    async def test_inserts_episode(self):
        pool, conn = make_mock_pool()
        store = MemoryStore(pool)
        new_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=new_id))

        result = await store.insert_episodic(
            summary="Discussed Python debugging",
            topics=["python", "debugging"],
            emotional_tone="curious",
            significance=0.7,
            duration_turns=10,
            embedding=[0.1] * 1024,
        )
        assert result == new_id
        call = conn.fetchrow.call_args
        assert "episodic_memories" in call.args[0]


class TestInsertEmotional:
    async def test_inserts_emotion(self):
        pool, conn = make_mock_pool()
        store = MemoryStore(pool)
        new_id = uuid4()
        episode_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=new_id))

        result = await store.insert_emotional(
            emotion="joy", trigger_context="solved a hard bug",
            intensity=0.8, episode_id=episode_id, embedding=[0.1] * 1024,
        )
        assert result == new_id
        call = conn.fetchrow.call_args
        assert "emotional_memories" in call.args[0]


class TestUpsertProcedural:
    async def test_inserts_pattern(self):
        pool, conn = make_mock_pool()
        store = MemoryStore(pool)
        new_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=new_id))

        result = await store.upsert_procedural(
            pattern="asks about weather in the morning",
            frequency="daily", confidence=0.6, embedding=[0.1] * 1024,
        )
        assert result == new_id
        call = conn.fetchrow.call_args
        assert "procedural_memories" in call.args[0]
        assert "ON CONFLICT" in call.args[0]


class TestSearchAll:
    async def test_search_all_returns_merged_results(self):
        pool, conn = make_mock_pool()
        store = MemoryStore(pool)

        sem_row = mock_record(
            id=uuid4(), table_name="semantic", category="person",
            subject="user", predicate="likes", object="coffee",
            text="user likes coffee", similarity=0.9,
            confidence=1.0, mention_count=3, created_at="2026-01-01T00:00:00Z",
        )
        conn.fetch = AsyncMock(side_effect=[
            [sem_row],  # semantic
            [],         # episodic
            [],         # emotional
            [],         # procedural
        ])

        results = await store.search_all(query_embedding=[0.1] * 1024, top_k=10)
        assert len(results) >= 1
        assert conn.fetch.await_count == 4  # one query per table


class TestSaveMessage:
    async def test_save_message(self):
        pool, conn = make_mock_pool()
        store = MemoryStore(pool)
        msg_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=msg_id))

        result = await store.save_message(uuid4(), "user", "Hello")
        assert result == msg_id
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/test_graphrag.py -v
```
Expected: FAIL — `MemoryStore` does not exist.

**Step 3: Rewrite `src/prot/graphrag.py`**

```python
"""4-layer memory storage with pgvector semantic search."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg

from prot.logging import logged


class MemoryStore:
    """pgvector-backed 4-layer memory storage."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    def acquire(self):
        return self._pool.acquire()

    @asynccontextmanager
    async def _conn(
        self, conn: asyncpg.Connection | None = None,
    ) -> AsyncIterator[asyncpg.Connection]:
        if conn is not None:
            yield conn
        else:
            async with self._pool.acquire() as c:
                yield c

    # -- Semantic memories (SPO triples) --

    @logged(slow_ms=500)
    async def upsert_semantic(
        self,
        category: str,
        subject: str,
        predicate: str,
        object_: str,
        confidence: float = 1.0,
        embedding: list[float] | None = None,
        source: str = "compaction",
        conn: asyncpg.Connection | None = None,
    ) -> UUID:
        query = """INSERT INTO semantic_memories
                   (category, subject, predicate, object, confidence, source, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (subject, predicate, object)
                DO UPDATE SET mention_count = semantic_memories.mention_count + 1,
                             confidence = GREATEST(semantic_memories.confidence, EXCLUDED.confidence),
                             embedding = COALESCE(EXCLUDED.embedding, semantic_memories.embedding),
                             updated_at = now()
                RETURNING id"""
        async with self._conn(conn) as c:
            row = await c.fetchrow(
                query, category, subject, predicate, object_, confidence, source, embedding,
            )
            return row["id"]

    # -- Episodic memories --

    @logged(slow_ms=500)
    async def insert_episodic(
        self,
        summary: str,
        topics: list[str] | None = None,
        emotional_tone: str | None = None,
        significance: float = 0.5,
        duration_turns: int = 0,
        embedding: list[float] | None = None,
        conn: asyncpg.Connection | None = None,
    ) -> UUID:
        query = """INSERT INTO episodic_memories
                   (summary, topics, emotional_tone, significance, duration_turns, embedding)
                VALUES ($1, $2, $3, $4, $5, $6) RETURNING id"""
        async with self._conn(conn) as c:
            row = await c.fetchrow(
                query, summary, topics or [], emotional_tone,
                significance, duration_turns, embedding,
            )
            return row["id"]

    # -- Emotional memories --

    @logged(slow_ms=500)
    async def insert_emotional(
        self,
        emotion: str,
        trigger_context: str,
        intensity: float = 0.5,
        episode_id: UUID | None = None,
        embedding: list[float] | None = None,
        conn: asyncpg.Connection | None = None,
    ) -> UUID:
        query = """INSERT INTO emotional_memories
                   (emotion, trigger_context, intensity, episode_id, embedding)
                VALUES ($1, $2, $3, $4, $5) RETURNING id"""
        async with self._conn(conn) as c:
            row = await c.fetchrow(
                query, emotion, trigger_context, intensity, episode_id, embedding,
            )
            return row["id"]

    # -- Procedural memories --

    @logged(slow_ms=500)
    async def upsert_procedural(
        self,
        pattern: str,
        frequency: str | None = None,
        confidence: float = 0.5,
        embedding: list[float] | None = None,
        conn: asyncpg.Connection | None = None,
    ) -> UUID:
        query = """INSERT INTO procedural_memories
                   (pattern, frequency, confidence, embedding, last_observed)
                VALUES ($1, $2, $3, $4, now())
                ON CONFLICT (pattern)
                DO UPDATE SET observation_count = procedural_memories.observation_count + 1,
                             confidence = GREATEST(procedural_memories.confidence, EXCLUDED.confidence),
                             frequency = COALESCE(EXCLUDED.frequency, procedural_memories.frequency),
                             embedding = COALESCE(EXCLUDED.embedding, procedural_memories.embedding),
                             last_observed = now(),
                             updated_at = now()
                RETURNING id"""
        async with self._conn(conn) as c:
            row = await c.fetchrow(query, pattern, frequency, confidence, embedding)
            return row["id"]

    # -- Search across all layers --

    @logged(slow_ms=500)
    async def search_all(
        self, query_embedding: list[float], top_k: int = 10,
    ) -> list[dict]:
        """Search all 4 memory tables by cosine similarity. Returns merged list."""
        async with self._pool.acquire() as conn:
            sem = await conn.fetch(
                """SELECT id, 'semantic' AS table_name, category,
                          subject, predicate, object,
                          subject || ' ' || predicate || ' ' || object AS text,
                          confidence, mention_count,
                          1 - (embedding <=> $1::vector) AS similarity,
                          created_at
                FROM semantic_memories WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector LIMIT $2""",
                query_embedding, top_k,
            )
            epi = await conn.fetch(
                """SELECT id, 'episodic' AS table_name, summary AS text,
                          topics, emotional_tone, significance,
                          1 - (embedding <=> $1::vector) AS similarity,
                          created_at
                FROM episodic_memories WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector LIMIT $2""",
                query_embedding, top_k,
            )
            emo = await conn.fetch(
                """SELECT id, 'emotional' AS table_name,
                          emotion || ': ' || trigger_context AS text,
                          emotion, trigger_context, intensity,
                          1 - (embedding <=> $1::vector) AS similarity,
                          created_at
                FROM emotional_memories WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector LIMIT $2""",
                query_embedding, top_k,
            )
            proc = await conn.fetch(
                """SELECT id, 'procedural' AS table_name, pattern AS text,
                          frequency, confidence, observation_count,
                          1 - (embedding <=> $1::vector) AS similarity,
                          created_at
                FROM procedural_memories WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector LIMIT $2""",
                query_embedding, top_k,
            )

        return [dict(r) for r in [*sem, *epi, *emo, *proc]]

    # -- Conversation messages (unchanged) --

    async def save_message(
        self, conversation_id: UUID, role: str, content: str,
    ) -> UUID:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO conversation_messages
                   (conversation_id, role, content)
                   VALUES ($1, $2, $3) RETURNING id""",
                conversation_id, role, content,
            )
            return row["id"]
```

**Step 4: Run graphrag tests**

```bash
uv run pytest tests/test_graphrag.py -v
```
Expected: All pass.

**Step 5: Update `tests/test_db.py` integration tests**

Update the `test_schema_execution` integration test to assert new table names:
```python
                assert "semantic_memories" in table_names
                assert "episodic_memories" in table_names
                assert "emotional_memories" in table_names
                assert "procedural_memories" in table_names
                assert "conversation_messages" in table_names
```

Rewrite `test_basic_entity_crud` to test `semantic_memories` instead of `entities`:
```python
    async def test_basic_semantic_crud(self) -> None:
        pool = await init_pool()
        try:
            schema_sql = self.SCHEMA_PATH.read_text(encoding="utf-8")
            async with pool.acquire() as conn:
                await conn.execute(schema_sql)
                row = await conn.fetchrow(
                    """INSERT INTO semantic_memories
                       (category, subject, predicate, object)
                    VALUES ($1, $2, $3, $4) RETURNING id, subject""",
                    "person", "user", "likes", "coffee",
                )
                assert row is not None
                assert row["subject"] == "user"
                await conn.execute(
                    "DELETE FROM semantic_memories WHERE id = $1", row["id"]
                )
        finally:
            await pool.close()
            db_module._pool = None
```

**Step 6: Commit**

```bash
git add src/prot/graphrag.py tests/test_graphrag.py tests/test_db.py src/prot/schema.sql
git commit -m "refactor: rewrite graphrag as MemoryStore for 4-layer schema"
```

---

### Task 7: Rewrite memory.py — compaction-driven extraction

Replace per-exchange extraction with compaction summary processing. The new `MemoryExtractor` receives a compaction summary, sends it to Haiku/Flash for structured 4-layer parsing, embeds, and stores.

**Files:**
- Rewrite: `src/prot/memory.py`
- Rewrite: `tests/test_memory.py`

**Step 1: Write `tests/test_memory.py`**

```python
"""Tests for MemoryExtractor — compaction-driven 4-layer memory extraction."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prot.memory import MemoryExtractor, DEFAULT_COMPACTION_PROMPT


def _make_store_with_conn():
    mock_store = AsyncMock()
    mock_conn = AsyncMock()
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx.__aexit__ = AsyncMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=mock_tx)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_store.acquire = MagicMock(return_value=mock_ctx)
    return mock_store, mock_conn


SAMPLE_EXTRACTION = {
    "semantic": [
        {"category": "preference", "subject": "user", "predicate": "likes", "object": "coffee", "confidence": 0.9}
    ],
    "episodic": {
        "summary": "User discussed morning routines",
        "topics": ["morning", "coffee"],
        "emotional_tone": "warm",
        "significance": 0.6,
        "duration_turns": 8,
    },
    "emotional": [
        {"emotion": "joy", "trigger_context": "talking about coffee", "intensity": 0.7}
    ],
    "procedural": [
        {"pattern": "asks about weather in the morning", "frequency": "daily", "confidence": 0.4}
    ],
}


@pytest.mark.asyncio
class TestMemoryExtractor:
    async def test_extract_from_summary_calls_llm(self):
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(SAMPLE_EXTRACTION))]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            ext = MemoryExtractor(store=AsyncMock(), embedder=AsyncMock())
            result = await ext.extract_from_summary("User likes coffee.")
            assert "semantic" in result
            assert len(result["semantic"]) == 1

    async def test_extract_handles_malformed_json(self):
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not json")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            ext = MemoryExtractor(store=AsyncMock(), embedder=AsyncMock())
            result = await ext.extract_from_summary("test")
            assert result["semantic"] == []

    async def test_save_extraction_stores_all_layers(self):
        store, conn = _make_store_with_conn()
        store.upsert_semantic.return_value = MagicMock()
        store.insert_episodic.return_value = MagicMock()
        store.insert_emotional.return_value = MagicMock()
        store.upsert_procedural.return_value = MagicMock()

        embedder = AsyncMock()
        embedder.embed_texts.return_value = [[0.1] * 1024] * 5  # enough for all texts

        ext = MemoryExtractor(store=store, embedder=embedder)
        await ext.save_extraction(SAMPLE_EXTRACTION)

        store.upsert_semantic.assert_called_once()
        store.insert_episodic.assert_called_once()
        store.insert_emotional.assert_called_once()
        store.upsert_procedural.assert_called_once()

    async def test_save_extraction_empty_data(self):
        store = AsyncMock()
        embedder = AsyncMock()
        ext = MemoryExtractor(store=store, embedder=embedder)
        await ext.save_extraction({"semantic": [], "episodic": None, "emotional": [], "procedural": []})
        store.upsert_semantic.assert_not_called()

    async def test_generate_shutdown_summary_calls_llm(self):
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="<summary>User discussed coding.</summary>")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            ext = MemoryExtractor(store=AsyncMock(), embedder=AsyncMock())
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            summary = await ext.generate_shutdown_summary(messages)
            assert "coding" in summary

    async def test_pre_load_context_returns_formatted_text(self):
        store = AsyncMock()
        store.search_all.return_value = [
            {"table_name": "semantic", "text": "user likes coffee",
             "similarity": 0.9, "created_at": "2026-01-01T00:00:00Z"},
        ]
        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024

        ext = MemoryExtractor(store=store, embedder=embedder)
        text = await ext.pre_load_context("Tell me about preferences")
        assert "coffee" in text

    async def test_pre_load_context_no_results(self):
        store = AsyncMock()
        store.search_all.return_value = []
        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024

        ext = MemoryExtractor(store=store, embedder=embedder)
        text = await ext.pre_load_context("unknown topic")
        assert text == "(no memory context)"

    async def test_default_compaction_prompt_exists(self):
        assert "summary" in DEFAULT_COMPACTION_PROMPT.lower()

    async def test_close_closes_clients(self):
        mock_anthropic = AsyncMock()
        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            ext = MemoryExtractor(store=AsyncMock(), embedder=AsyncMock())
            mock_reranker = AsyncMock()
            ext._reranker = mock_reranker
            await ext.close()
            mock_anthropic.close.assert_awaited_once()
            mock_reranker.close.assert_awaited_once()
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/test_memory.py -v
```
Expected: FAIL.

**Step 3: Rewrite `src/prot/memory.py`**

```python
"""Compaction-driven memory extraction and RAG context retrieval.

Memory pipeline fires at two points:
1. Compaction event — pause_after_compaction intercepts the summary
2. Shutdown — forced summarization using the default compaction prompt

Extraction is 2-step:
1. Get compaction summary (from API event or manual Haiku/Flash call)
2. Send to Haiku/Flash for structured 4-layer parsing
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from anthropic import AsyncAnthropic

from prot.config import settings
from prot.decay import AdaptiveDecayCalculator
from prot.embeddings import AsyncVoyageEmbedder
from prot.graphrag import MemoryStore
from prot.logging import get_logger, logged
from prot.processing import content_to_text, strip_markdown_fences

logger = get_logger(__name__)

DEFAULT_COMPACTION_PROMPT = (
    "You have written a partial transcript for the initial task above. "
    "Please write a summary of the transcript. The purpose of this summary is "
    "to provide continuity so you can continue to make progress towards solving "
    "the task in a future context, where the raw history above may not be accessible "
    "and will be replaced with this summary. Write down anything that would be helpful, "
    "including the state, next steps, learnings etc. "
    "You must wrap your summary in a `<summary></summary>` block."
)

_EXTRACTION_PROMPT = """You are a memory extraction system. Given a conversation summary,
extract structured memories into 4 layers. The summary may be in Korean or English.
Keep names and terms in their original language.

Return JSON with this exact structure:
{
  "semantic": [
    {"category": "person|preference|fact|skill|relationship", "subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0}
  ],
  "episodic": {
    "summary": "...",
    "topics": ["..."],
    "emotional_tone": "warm|tense|playful|curious|neutral|...",
    "significance": 0.0-1.0,
    "duration_turns": 0
  },
  "emotional": [
    {"emotion": "joy|frustration|curiosity|gratitude|...", "trigger_context": "...", "intensity": 0.0-1.0}
  ],
  "procedural": [
    {"pattern": "...", "frequency": "daily|weekly|occasional|null", "confidence": 0.0-1.0}
  ]
}

Rules:
- semantic: SPO triples for facts, preferences, knowledge. Be specific and concise.
- episodic: ONE episode summarizing this conversation segment. Always include.
- emotional: Capture emotional highlights. Empty array if no notable emotions.
- procedural: Behavioral patterns observed. Empty array if none.
- If the summary has no meaningful content, return empty arrays and null episodic."""


class MemoryExtractor:
    """Compaction-driven memory extraction and retrieval."""

    def __init__(
        self,
        anthropic_key: str | None = None,
        store: MemoryStore | None = None,
        embedder: AsyncVoyageEmbedder | None = None,
        reranker=None,
    ):
        self._llm = AsyncAnthropic(api_key=anthropic_key or settings.anthropic_api_key)
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._decay = AdaptiveDecayCalculator(
            base_rate=settings.decay_base_rate,
            min_retention=settings.decay_min_retention,
        )

    async def close(self) -> None:
        await self._llm.close()
        if self._reranker:
            await self._reranker.close()

    @logged(slow_ms=3000)
    async def extract_from_summary(self, summary_text: str) -> dict:
        """Send compaction summary to Haiku/Flash for structured 4-layer extraction."""
        logger.info("Extracting from summary", chars=len(summary_text))
        response = await self._llm.messages.create(
            model=settings.memory_extraction_model,
            max_tokens=4000,
            system=[{
                "type": "text",
                "text": _EXTRACTION_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": summary_text}],
        )
        try:
            raw = response.content[0].text
        except (IndexError, AttributeError):
            logger.warning("Empty extraction response")
            return {"semantic": [], "episodic": None, "emotional": [], "procedural": []}

        raw = strip_markdown_fences(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Extraction JSON parse failed", raw=raw[:200])
            return {"semantic": [], "episodic": None, "emotional": [], "procedural": []}

    @logged(slow_ms=5000)
    async def save_extraction(self, extraction: dict) -> None:
        """Embed and save extracted memories across all 4 layers."""
        semantics = extraction.get("semantic") or []
        episodic = extraction.get("episodic")
        emotionals = extraction.get("emotional") or []
        procedurals = extraction.get("procedural") or []

        if not semantics and not episodic and not emotionals and not procedurals:
            logger.debug("Extraction empty, skipping save")
            return

        # Collect all texts to embed in one batch
        texts_to_embed = []
        # Semantic: concatenate SPO for embedding
        for s in semantics:
            texts_to_embed.append(f"{s['subject']} {s['predicate']} {s['object']}")
        sem_count = len(semantics)

        # Episodic
        if episodic and episodic.get("summary"):
            texts_to_embed.append(episodic["summary"])
        epi_count = 1 if (episodic and episodic.get("summary")) else 0

        # Emotional
        for e in emotionals:
            texts_to_embed.append(f"{e['emotion']}: {e['trigger_context']}")
        emo_count = len(emotionals)

        # Procedural
        for p in procedurals:
            texts_to_embed.append(p["pattern"])

        if not texts_to_embed:
            return

        embeddings = await self._embedder.embed_texts(texts_to_embed)
        idx = 0

        async with self._store.acquire() as conn:
            async with conn.transaction():
                # Semantic
                for s, emb in zip(semantics, embeddings[idx:idx + sem_count]):
                    await self._store.upsert_semantic(
                        category=s["category"],
                        subject=s["subject"],
                        predicate=s["predicate"],
                        object_=s["object"],
                        confidence=s.get("confidence", 1.0),
                        embedding=emb,
                        conn=conn,
                    )
                idx += sem_count

                # Episodic
                episode_id = None
                if epi_count:
                    episode_id = await self._store.insert_episodic(
                        summary=episodic["summary"],
                        topics=episodic.get("topics", []),
                        emotional_tone=episodic.get("emotional_tone"),
                        significance=episodic.get("significance", 0.5),
                        duration_turns=episodic.get("duration_turns", 0),
                        embedding=embeddings[idx],
                        conn=conn,
                    )
                    idx += 1

                # Emotional (linked to episode)
                for e, emb in zip(emotionals, embeddings[idx:idx + emo_count]):
                    await self._store.insert_emotional(
                        emotion=e["emotion"],
                        trigger_context=e["trigger_context"],
                        intensity=e.get("intensity", 0.5),
                        episode_id=episode_id,
                        embedding=emb,
                        conn=conn,
                    )
                idx += emo_count

                # Procedural
                for p, emb in zip(procedurals, embeddings[idx:]):
                    await self._store.upsert_procedural(
                        pattern=p["pattern"],
                        frequency=p.get("frequency"),
                        confidence=p.get("confidence", 0.5),
                        embedding=emb,
                        conn=conn,
                    )

        logger.info(
            "Saved memories",
            semantic=len(semantics), episodic=bool(episodic),
            emotional=len(emotionals), procedural=len(procedurals),
        )

    @logged(slow_ms=5000)
    async def generate_shutdown_summary(self, messages: list[dict]) -> str:
        """Generate a compaction-equivalent summary at shutdown using Haiku/Flash."""
        conversation_text = "\n".join(
            f"{m['role']}: {content_to_text(m['content'])}" for m in messages
        )
        response = await self._llm.messages.create(
            model=settings.memory_extraction_model,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": f"{conversation_text}\n\n{DEFAULT_COMPACTION_PROMPT}"},
            ],
        )
        try:
            text = response.content[0].text
        except (IndexError, AttributeError):
            return ""

        # Extract content from <summary> tags if present
        if "<summary>" in text and "</summary>" in text:
            text = text.split("<summary>", 1)[1].split("</summary>", 1)[0]
        return text.strip()

    async def pre_load_context(self, query: str) -> str:
        """Search all memory layers, apply time-decay, optionally rerank, format for Block 2."""
        query_embedding = await self._embedder.embed_query(query)
        results = await self._store.search_all(
            query_embedding=query_embedding, top_k=settings.rag_top_k,
        )

        if not results:
            return "(no memory context)"

        # Apply time-decay scoring
        now = datetime.now(timezone.utc)
        for r in results:
            created = r.get("created_at")
            if created:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                hours = (now - created).total_seconds() / 3600
            else:
                hours = 0.0

            memory_type = _table_to_memory_type(r.get("table_name", ""))
            decay_score = self._decay.calculate(
                importance=r.get("significance", r.get("confidence", 0.5)),
                hours_passed=hours,
                access_count=r.get("mention_count", r.get("observation_count", 0)),
                memory_type=memory_type,
            )
            r["effective_score"] = r["similarity"] * decay_score

        # Sort by effective score
        results.sort(key=lambda r: r["effective_score"], reverse=True)

        # Optional reranking
        if self._reranker and len(results) > 1:
            results = await self._reranker.rerank(
                query=query, items=results, text_key="text",
                top_k=settings.rerank_top_k,
            )

        # Format into Block 2 context with token budget
        parts: list[str] = []
        token_estimate = 0
        for r in results:
            text = r.get("text", "")
            token_estimate += len(text) // 4
            if token_estimate > settings.rag_context_target_tokens:
                break
            table = r.get("table_name", "unknown")
            parts.append(f"[{table}] {text}")

        return "\n".join(parts) if parts else "(no memory context)"


def _table_to_memory_type(table_name: str) -> str:
    """Map table_name to decay memory_type."""
    return {
        "semantic": "fact",
        "episodic": "conversation",
        "emotional": "insight",
        "procedural": "preference",
    }.get(table_name, "conversation")
```

**Step 4: Run memory tests**

```bash
uv run pytest tests/test_memory.py -v
```
Expected: All pass.

**Step 5: Commit**

```bash
git add src/prot/memory.py tests/test_memory.py
git commit -m "refactor: rewrite memory.py for compaction-driven 4-layer extraction"
```

---

### Task 8: Update llm.py — compaction event detection

Add `pause_after_compaction` to the compaction edit config. Add compaction summary extraction from stream events.

**Files:**
- Modify: `src/prot/llm.py`
- Modify: `tests/test_llm.py`

**Step 1: Write new tests in `tests/test_llm.py`**

Add to existing test file:

```python
class TestCompactionDetection:
    async def test_compaction_edit_includes_pause(self):
        """Compaction edit should include pause_after_compaction when enabled."""
        with patch("prot.llm.settings") as ms:
            ms.thinking_keep_turns = 2
            ms.tool_clear_trigger = 30000
            ms.tool_clear_keep = 3
            ms.compaction_trigger = 50000
            ms.pause_after_compaction = True

            from prot.llm import _build_context_management
            cm = _build_context_management()
            compact_edit = cm["edits"][2]
            assert compact_edit["type"] == "compact_20260112"
            assert compact_edit["pause"] is True

    async def test_last_compaction_summary_initially_none(self):
        client = LLMClient.__new__(LLMClient)
        client._last_compaction_summary = None
        assert client.last_compaction_summary is None

    async def test_stop_reason_compaction_detected(self):
        """When stop_reason is 'compaction', last_compaction_summary should be populated."""
        mock_stream = AsyncMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration)

        # Simulate compaction stop reason with compaction content block
        compaction_block = MagicMock()
        compaction_block.type = "compaction"
        compaction_block.summary = "User discussed Python debugging techniques."
        final_msg = MagicMock()
        final_msg.content = [compaction_block]
        final_msg.stop_reason = "compaction"
        final_msg.usage = MagicMock()
        mock_stream.get_final_message = AsyncMock(return_value=final_msg)

        with patch("prot.llm.AsyncAnthropic") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.beta.messages.stream.return_value = mock_stream

            client = LLMClient(api_key="test")
            async for _ in client.stream_response([], None, []):
                pass

            assert client.last_compaction_summary == "User discussed Python debugging techniques."
```

**Step 2: Run to verify failure**

```bash
uv run pytest tests/test_llm.py::TestCompactionDetection -v
```
Expected: FAIL — `last_compaction_summary` doesn't exist.

**Step 3: Modify `src/prot/llm.py`**

Changes:
1. In `_build_context_management()`, add `"pause": True` to compact edit when `settings.pause_after_compaction` is True.
2. Add `_last_compaction_summary` attribute.
3. In `stream_response()`, after `get_final_message()`, check `stop_reason == "compaction"` and extract summary from compaction content block.

Key code additions:

In `_build_context_management()`:
```python
compact_edit = {
    "type": "compact_20260112",
    "trigger": {"type": "input_tokens", "value": settings.compaction_trigger},
}
if settings.pause_after_compaction:
    compact_edit["pause"] = True
```

In `LLMClient.__init__`:
```python
self._last_compaction_summary = None
```

After `get_final_message()`:
```python
# Detect compaction events
self._last_compaction_summary = None
if getattr(final, "stop_reason", None) == "compaction":
    for block in final.content:
        if getattr(block, "type", None) == "compaction":
            self._last_compaction_summary = getattr(block, "summary", None)
            break
```

Add property:
```python
@property
def last_compaction_summary(self) -> str | None:
    return self._last_compaction_summary
```

**Step 4: Run llm tests**

```bash
uv run pytest tests/test_llm.py -v
```
Expected: All pass.

**Step 5: Commit**

```bash
git add src/prot/llm.py tests/test_llm.py
git commit -m "feat: add compaction event detection with pause_after_compaction"
```

---

### Task 9: Rewrite pipeline.py memory integration

Replace `_extract_memories_bg()` with compaction handler. Add shutdown summarization. Remove community detector. Update imports.

**Files:**
- Modify: `src/prot/pipeline.py`
- Modify: `tests/test_pipeline.py`

**Step 1: Update `pipeline.py` — startup() and imports**

Replace the memory subsystem initialization in `startup()` (lines 96-124):

```python
        try:
            from prot.graphrag import MemoryStore
            from prot.embeddings import AsyncVoyageEmbedder
            from prot.memory import MemoryExtractor
            from prot.reranker import VoyageReranker

            self._graphrag = MemoryStore(pool=self._pool)
            self._embedder = AsyncVoyageEmbedder()
            self._reranker = VoyageReranker()
            self._memory = MemoryExtractor(
                store=self._graphrag,
                embedder=self._embedder,
                reranker=self._reranker,
            )
        except Exception:
            logger.warning("Memory subsystem not available")
```

Remove entirely (lines 119-124 in original):
```python
        # Seed known entities from DB
        try:
            if self._memory:
                await self._memory.seed_known_entities()
        except Exception:
            logger.debug("Entity seed failed", exc_info=True)
```

Remove `self._exchange_count` from `Pipeline.__init__` (no longer needed — no interval gating).

**Step 2: Replace `_extract_memories_bg()` with `_handle_compaction_bg()`**

```python
def _handle_compaction_bg(self) -> None:
    """Process compaction summary for memory extraction."""
    if not self._memory:
        return
    summary = self._llm.last_compaction_summary
    if not summary:
        return
    query = self._current_transcript

    async def _process():
        try:
            extraction = await self._memory.extract_from_summary(summary)
            await self._memory.save_extraction(extraction)
            if query:
                rag = await self._memory.pre_load_context(query)
                self._ctx.update_rag_context(rag)
        except Exception:
            logger.warning("Compaction memory extraction failed", exc_info=True)

    self._bg(_process())
```

**Step 3: Replace `_extract_memories_bg()` call site**

In `_process_response()` (around line 360), replace `self._extract_memories_bg()` with `self._handle_compaction_bg()`.
Remove the `self._exchange_count` increment and `memory_extraction_interval` check that gated the old call.

**Step 4: Replace shutdown memory logic**

Replace the existing shutdown extraction block with:
```python
# Best-effort shutdown summarization
if self._memory:
    try:
        messages = self._ctx.get_messages()
        if messages:
            summary = await self._memory.generate_shutdown_summary(messages)
            if summary:
                extraction = await self._memory.extract_from_summary(summary)
                await self._memory.save_extraction(extraction)
    except Exception:
        logger.debug("Shutdown memory extraction failed", exc_info=True)
```

**Step 5: Update `tests/test_pipeline.py`**

Key changes to `_make_pipeline()`:
- Remove `p._exchange_count = 0` (no longer exists)

Update/replace `TestExtractMemoriesBg` with `TestCompactionHandler`:
```python
class TestCompactionHandler:
    async def test_compaction_triggers_memory_extraction(self):
        p = _make_pipeline()
        mock_memory = AsyncMock()
        mock_memory.extract_from_summary = AsyncMock(
            return_value={"semantic": [], "episodic": None, "emotional": [], "procedural": []}
        )
        mock_memory.save_extraction = AsyncMock()
        p._memory = mock_memory
        p._llm.last_compaction_summary = "User discussed coding."

        p._handle_compaction_bg()
        assert len(p._background_tasks) == 1
        await asyncio.sleep(0.05)

        mock_memory.extract_from_summary.assert_awaited_once_with("User discussed coding.")

    async def test_no_compaction_no_extraction(self):
        p = _make_pipeline()
        mock_memory = AsyncMock()
        p._memory = mock_memory
        p._llm.last_compaction_summary = None

        p._handle_compaction_bg()
        assert len(p._background_tasks) == 0
```

Replace `TestExtractionInterval` → remove entirely (no interval gating anymore).

Update `TestShutdownFinalExtraction`:
```python
class TestShutdownFinalExtraction:
    async def test_shutdown_runs_summarization(self):
        p = _make_pipeline()
        mock_memory = AsyncMock()
        mock_memory.generate_shutdown_summary = AsyncMock(return_value="Summary text")
        mock_memory.extract_from_summary = AsyncMock(
            return_value={"semantic": [], "episodic": None, "emotional": [], "procedural": []}
        )
        mock_memory.save_extraction = AsyncMock()
        mock_memory.close = AsyncMock()
        p._memory = mock_memory
        p._ctx.get_messages.return_value = [{"role": "user", "content": "hello"}]

        await p.shutdown()

        mock_memory.generate_shutdown_summary.assert_awaited_once()
        mock_memory.extract_from_summary.assert_awaited_once_with("Summary text")
        mock_memory.save_extraction.assert_awaited_once()

    async def test_shutdown_skips_empty_summary(self):
        p = _make_pipeline()
        mock_memory = AsyncMock()
        mock_memory.generate_shutdown_summary = AsyncMock(return_value="")
        mock_memory.close = AsyncMock()
        p._memory = mock_memory
        p._ctx.get_messages.return_value = [{"role": "user", "content": "hello"}]

        await p.shutdown()

        mock_memory.extract_from_summary.assert_not_awaited()
```

**Step 6: Run pipeline tests**

```bash
uv run pytest tests/test_pipeline.py -v
```
Expected: All pass.

**Step 7: Run full test suite**

```bash
uv run pytest -x -q
```
Expected: All pass.

**Step 8: Commit**

```bash
git add src/prot/pipeline.py tests/test_pipeline.py
git commit -m "refactor: replace per-exchange extraction with compaction handler"
```

---

### Task 10: Update CLAUDE.md

Reflect the new architecture in project documentation.

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update architecture section**

Key changes:
- Remove `community.py` from file listing
- Update `memory.py` description: "Compaction-driven 4-layer memory extraction + RAG context retrieval"
- Update `graphrag.py` description: "pgvector-backed 4-layer memory storage (semantic, episodic, emotional, procedural)"
- Add `decay.py` to listing: "AdaptiveDecayCalculator for time-decay memory scoring"
- Update `embeddings.py` description: "Voyage AI embeddings (voyage-4-large)"
- Remove `conversation_log.py` if still listed
- Update Memory extraction pattern description
- Update DB optional description

**Step 2: Update Code Patterns section**

Replace memory extraction pattern:
```
- **Memory extraction**: Compaction-driven. No per-exchange extraction.
  Fires on compaction events (pause_after_compaction) and shutdown (forced summarization).
  Haiku/Flash extracts 4-layer structured memories from compaction summary.
  Time-decay scoring (AdaptiveDecayCalculator) at query time.
  Embeddings: voyage-4-large. Reranker: rerank-2.5.
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for compaction-driven memory architecture"
```

---

### Task 11: Final integration verification

**Step 1: Run full test suite**

```bash
uv run pytest --cov=prot --cov-report=term-missing -q
```
Expected: All pass, coverage reasonable.

**Step 2: Verify no stale imports**

```bash
uv run python -c "from prot.pipeline import Pipeline; print('OK')"
uv run python -c "from prot.memory import MemoryExtractor; print('OK')"
uv run python -c "from prot.graphrag import MemoryStore; print('OK')"
uv run python -c "from prot.decay import AdaptiveDecayCalculator; print('OK')"
uv run python -c "from prot.embeddings import AsyncVoyageEmbedder; print('OK')"
```

**Step 3: Verify community.py is fully gone**

```bash
grep -r "community" src/prot/ --include="*.py" | grep -v "__pycache__"
```
Expected: No results (or only comments/strings, not imports).

**Step 4: Verify networkx is removed**

```bash
grep -r "networkx" . --include="*.toml" --include="*.py" | grep -v __pycache__ | grep -v ".venv"
```
Expected: No results.

**Step 5: Sync dependencies**

```bash
uv sync
```
Expected: networkx no longer installed.

---

## Summary of Changes

| Action | File | Description |
|--------|------|-------------|
| DELETE | `src/prot/community.py` | Louvain community detection |
| DELETE | `tests/test_community.py` | Community tests |
| DELETE | `scripts/db_gc.py` | Old entity/relationship GC script |
| DELETE | `tests/test_db_gc.py` | GC script tests |
| CREATE | `src/prot/decay.py` | AdaptiveDecayCalculator |
| CREATE | `tests/test_decay.py` | Decay calculator tests |
| REWRITE | `src/prot/schema.sql` | 4-layer memory schema |
| REWRITE | `src/prot/graphrag.py` | MemoryStore (4 tables) |
| REWRITE | `tests/test_graphrag.py` | MemoryStore tests |
| REWRITE | `src/prot/memory.py` | Compaction-driven extraction |
| REWRITE | `tests/test_memory.py` | Memory extractor tests |
| REWRITE | `src/prot/embeddings.py` | voyage-4-large (standard embed API) |
| REWRITE | `tests/test_embeddings.py` | Embeddings tests |
| MODIFY | `src/prot/llm.py` | Compaction detection + pause |
| MODIFY | `tests/test_llm.py` | Compaction detection tests |
| MODIFY | `src/prot/pipeline.py` | Compaction handler + shutdown summary |
| MODIFY | `tests/test_pipeline.py` | Updated pipeline tests |
| MODIFY | `src/prot/config.py` | Remove old settings, add new |
| MODIFY | `src/prot/logging/constants.py` | Remove community, add decay entry |
| MODIFY | `tests/test_db.py` | Update integration tests for new tables |
| MODIFY | `pyproject.toml` | Remove networkx |
| MODIFY | `CLAUDE.md` | Updated architecture docs |
