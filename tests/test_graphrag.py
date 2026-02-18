"""Tests for prot.graphrag — GraphRAG store with pgvector semantic search."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from prot.graphrag import GraphRAGStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_pool():
    """Create a mock asyncpg pool with async context manager for acquire()."""
    pool = MagicMock()
    conn = AsyncMock()
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = ctx
    return pool, conn


def mock_record(**kwargs):
    """Create a mock that behaves like asyncpg.Record for dict() conversion."""
    record = MagicMock()
    record.keys.return_value = kwargs.keys()
    record.__getitem__ = MagicMock(side_effect=kwargs.__getitem__)
    record.__iter__ = MagicMock(return_value=iter(kwargs))
    record.items.return_value = kwargs.items()
    record.__len__ = MagicMock(return_value=len(kwargs))
    return record


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestUpsertEntity:
    """upsert_entity should insert new or update existing entities."""

    async def test_upsert_entity_uses_on_conflict(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        new_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=new_id))

        result = await store.upsert_entity(
            name="TestEntity",
            entity_type="person",
            description="A test entity",
            embedding=[0.1, 0.2, 0.3],
            namespace="default",
        )

        assert result == new_id

        # Verify single INSERT ... ON CONFLICT query
        call = conn.fetchrow.call_args
        assert "INSERT" in call.args[0]
        assert "ON CONFLICT" in call.args[0]
        assert "mention_count" in call.args[0]
        assert conn.fetchrow.await_count == 1


    async def test_upsert_entity_with_provided_conn(self) -> None:
        pool, _ = make_mock_pool()
        store = GraphRAGStore(pool)

        new_id = uuid4()
        ext_conn = AsyncMock()
        ext_conn.fetchrow = AsyncMock(return_value=mock_record(id=new_id))

        result = await store.upsert_entity(
            name="TestEntity",
            entity_type="person",
            description="A test entity",
            conn=ext_conn,
        )

        assert result == new_id
        ext_conn.fetchrow.assert_awaited_once()
        # Pool should NOT have been used
        pool.acquire.assert_not_called()


class TestGetEntityByName:
    """get_entity_by_name should return dict or None."""

    async def test_get_entity_by_name_returns_dict(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        entity_id = uuid4()
        row = mock_record(
            id=entity_id,
            name="TestEntity",
            entity_type="person",
            description="A test entity",
            namespace="default",
            mention_count=1,
        )
        conn.fetchrow = AsyncMock(return_value=row)

        result = await store.get_entity_by_name("TestEntity")

        assert result is not None
        assert isinstance(result, dict)
        assert result["name"] == "TestEntity"
        assert result["id"] == entity_id

    async def test_get_entity_by_name_returns_none(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        conn.fetchrow = AsyncMock(return_value=None)

        result = await store.get_entity_by_name("NonExistent")

        assert result is None


class TestUpsertRelationship:
    """upsert_relationship should insert a new relationship."""

    async def test_upsert_relationship(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        rel_id = uuid4()
        source_id = uuid4()
        target_id = uuid4()

        conn.fetchrow = AsyncMock(return_value=mock_record(id=rel_id))

        result = await store.upsert_relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type="works_with",
            description="Collaborative relationship",
            weight=0.8,
        )

        assert result == rel_id

        # Verify INSERT ... ON CONFLICT was called on relationships table
        call = conn.fetchrow.call_args
        assert "INSERT" in call.args[0]
        assert "ON CONFLICT" in call.args[0]
        assert "relationships" in call.args[0]


    async def test_upsert_relationship_with_provided_conn(self) -> None:
        pool, _ = make_mock_pool()
        store = GraphRAGStore(pool)

        rel_id = uuid4()
        ext_conn = AsyncMock()
        ext_conn.fetchrow = AsyncMock(return_value=mock_record(id=rel_id))

        result = await store.upsert_relationship(
            source_id=uuid4(),
            target_id=uuid4(),
            relation_type="works_with",
            description="Collaborative relationship",
            conn=ext_conn,
        )

        assert result == rel_id
        ext_conn.fetchrow.assert_awaited_once()
        pool.acquire.assert_not_called()


class TestSearchEntitiesSemantic:
    """search_entities_semantic should return list of dicts with similarity."""

    async def test_search_entities_semantic(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        entity1_id = uuid4()
        entity2_id = uuid4()

        rows = [
            mock_record(
                id=entity1_id,
                name="Entity1",
                entity_type="person",
                description="First entity",
                mention_count=5,
                similarity=0.95,
            ),
            mock_record(
                id=entity2_id,
                name="Entity2",
                entity_type="organization",
                description="Second entity",
                mention_count=2,
                similarity=0.87,
            ),
        ]
        conn.fetch = AsyncMock(return_value=rows)

        results = await store.search_entities_semantic(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
        )

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        assert results[0]["name"] == "Entity1"
        assert results[1]["name"] == "Entity2"

        # Verify the query uses cosine distance operator
        call = conn.fetch.call_args
        assert "<=>" in call.args[0]
        assert "LIMIT" in call.args[0]


class TestUpsertCommunity:
    """upsert_community should insert a new community."""

    async def test_upsert_community(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        community_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=community_id))

        result = await store.upsert_community(
            level=1,
            summary="A community of researchers",
            summary_embedding=[0.1, 0.2, 0.3],
            entity_count=10,
        )

        assert result == community_id

        # Verify INSERT on communities table
        call = conn.fetchrow.call_args
        assert "INSERT" in call.args[0]
        assert "communities" in call.args[0]


class TestSearchCommunities:
    """search_communities should return list of dicts with similarity."""

    async def test_search_communities(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        community_id = uuid4()
        rows = [
            mock_record(
                id=community_id,
                level=1,
                summary="Research community",
                entity_count=10,
                similarity=0.92,
            ),
        ]
        conn.fetch = AsyncMock(return_value=rows)

        results = await store.search_communities(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=5,
        )

        assert len(results) == 1
        assert isinstance(results[0], dict)
        assert results[0]["summary"] == "Research community"
        assert results[0]["similarity"] == 0.92


class TestSaveMessage:
    """save_message should persist a conversation message."""

    async def test_save_message(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        msg_id = uuid4()
        conv_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=msg_id))

        result = await store.save_message(
            conversation_id=conv_id,
            role="user",
            content="Hello world",
        )

        assert result == msg_id
        call = conn.fetchrow.call_args
        assert "conversation_messages" in call.args[0]
        assert "INSERT" in call.args[0]

    async def test_save_message_with_embedding(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        msg_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=msg_id))

        result = await store.save_message(
            conversation_id=uuid4(),
            role="assistant",
            content="Hi there",
            embedding=[0.1] * 1024,
        )

        assert result == msg_id
        # Verify embedding was passed as 4th positional arg
        call = conn.fetchrow.call_args
        assert call.args[4] == [0.1] * 1024


class TestLoadGraphForCommunityDetection:
    """load_graph_for_community_detection should return entities and relationships."""

    async def test_returns_entities_and_relationships(self):
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        entity_record = mock_record(
            id=uuid4(), name="Bob", entity_type="person", description="A friend"
        )
        rel_record = mock_record(
            source_id=uuid4(), target_id=uuid4(), weight=1.0
        )
        conn.fetch = AsyncMock(side_effect=[[entity_record], [rel_record]])

        entities, relationships = await store.load_graph_for_community_detection()
        assert len(entities) == 1
        assert entities[0]["name"] == "Bob"
        assert len(relationships) == 1
        assert relationships[0]["weight"] == 1.0

    async def test_returns_empty_lists_when_no_data(self):
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)
        conn.fetch = AsyncMock(return_value=[])

        entities, relationships = await store.load_graph_for_community_detection()
        assert entities == []
        assert relationships == []


class TestRebuildCommunities:
    """rebuild_communities should delete and re-insert atomically."""

    async def test_deletes_and_inserts_in_transaction(self):
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=tx)
        tx.__aexit__ = AsyncMock(return_value=False)
        conn.transaction = MagicMock(return_value=tx)

        community_id = uuid4()
        entity_id = uuid4()
        conn.fetchrow = AsyncMock(return_value=mock_record(id=community_id))

        await store.rebuild_communities([{
            "summary": "Test group",
            "summary_embedding": [0.1] * 1024,
            "entity_ids": [entity_id],
        }])

        conn.execute.assert_called_once()
        assert "DELETE" in str(conn.execute.call_args)
        conn.fetchrow.assert_called_once()
        conn.executemany.assert_called_once()

    async def test_empty_communities_just_deletes(self):
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)
        tx = MagicMock()
        tx.__aenter__ = AsyncMock(return_value=tx)
        tx.__aexit__ = AsyncMock(return_value=False)
        conn.transaction = MagicMock(return_value=tx)

        await store.rebuild_communities([])

        conn.execute.assert_called_once()
        assert "DELETE" in str(conn.execute.call_args)
        conn.fetchrow.assert_not_called()


class TestGetEntityCount:
    """get_entity_count should return total entity count."""

    async def test_returns_count(self):
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)
        conn.fetchrow = AsyncMock(return_value=mock_record(cnt=42))

        count = await store.get_entity_count()
        assert count == 42


class TestGetEntityNeighbors:
    """get_entity_neighbors should return neighboring entities."""

    async def test_get_entity_neighbors(self) -> None:
        pool, conn = make_mock_pool()
        store = GraphRAGStore(pool)

        entity_id = uuid4()
        neighbor1_id = uuid4()
        neighbor2_id = uuid4()

        rows = [
            mock_record(
                id=neighbor1_id,
                name="Neighbor1",
                entity_type="person",
                description="First neighbor",
            ),
            mock_record(
                id=neighbor2_id,
                name="Neighbor2",
                entity_type="organization",
                description="Second neighbor",
            ),
        ]
        conn.fetch = AsyncMock(return_value=rows)

        results = await store.get_entity_neighbors(entity_id, max_depth=2)

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)
        assert results[0]["name"] == "Neighbor1"
        assert results[1]["name"] == "Neighbor2"

        # Verify recursive CTE is used
        call = conn.fetch.call_args
        assert "RECURSIVE" in call.args[0]
