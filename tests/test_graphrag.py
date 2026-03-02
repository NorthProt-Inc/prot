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
