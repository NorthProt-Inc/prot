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
