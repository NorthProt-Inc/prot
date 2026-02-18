"""Tests for prot.db — database pool management and schema execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pgvector.asyncpg import register_vector

import prot.db as db_module
from prot.db import init_pool


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_pool():
    """Ensure _pool is None before and after every test."""
    db_module._pool = None
    yield
    db_module._pool = None


# ---------------------------------------------------------------------------
# A) Unit tests — mock-based, always run
# ---------------------------------------------------------------------------


class TestInitPool:
    """init_pool should create and store an asyncpg pool."""

    async def test_init_pool_creates_pool(self) -> None:
        mock_pool = MagicMock()
        mock_create = AsyncMock(return_value=mock_pool)
        with patch("prot.db.asyncpg.create_pool", mock_create):
            result = await init_pool(dsn="postgresql://test:test@localhost/test")

        mock_create.assert_awaited_once_with(
            dsn="postgresql://test:test@localhost/test",
            min_size=2,
            max_size=10,
            init=register_vector,
        )
        assert result is mock_pool
        assert db_module._pool is mock_pool

    async def test_init_pool_registers_vector_codec(self) -> None:
        mock_pool = MagicMock()
        mock_create = AsyncMock(return_value=mock_pool)
        with patch("prot.db.asyncpg.create_pool", mock_create):
            await init_pool(dsn="postgresql://test:test@localhost/test")

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["init"] is register_vector

    async def test_init_pool_raises_when_already_initialized(self) -> None:
        db_module._pool = MagicMock()
        with pytest.raises(RuntimeError, match="Pool already initialized"):
            await init_pool(dsn="postgresql://test:test@localhost/test")


    async def test_init_pool_uses_settings_dsn_when_none(self) -> None:
        mock_pool = MagicMock()
        mock_create = AsyncMock(return_value=mock_pool)
        with patch("prot.db.asyncpg.create_pool", mock_create):
            await init_pool()

        # Should fall back to settings.database_url
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["dsn"] is not None


# ---------------------------------------------------------------------------
# B) Integration tests — require real PostgreSQL, skipped by default
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegrationPool:
    """Integration tests requiring a running PostgreSQL instance."""

    SCHEMA_PATH = Path(__file__).parent.parent / "src" / "prot" / "schema.sql"

    async def test_pool_creation(self) -> None:
        pool = await init_pool()
        try:
            assert pool is not None
            async with pool.acquire() as conn:
                row = await conn.fetchval("SELECT 1")
                assert row == 1
        finally:
            await pool.close()
            db_module._pool = None

    async def test_schema_execution(self) -> None:
        pool = await init_pool()
        try:
            schema_sql = self.SCHEMA_PATH.read_text(encoding="utf-8")
            async with pool.acquire() as conn:
                await conn.execute(schema_sql)
                tables = await conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
                table_names = {r["tablename"] for r in tables}
                assert "entities" in table_names
                assert "relationships" in table_names
                assert "communities" in table_names
                assert "community_members" in table_names
                assert "conversation_messages" in table_names
        finally:
            await pool.close()
            db_module._pool = None

    async def test_basic_entity_crud(self) -> None:
        pool = await init_pool()
        try:
            schema_sql = self.SCHEMA_PATH.read_text(encoding="utf-8")
            async with pool.acquire() as conn:
                await conn.execute(schema_sql)
                # Insert
                row = await conn.fetchrow(
                    """
                    INSERT INTO entities (namespace, name, entity_type, description)
                    VALUES ($1, $2, $3, $4)
                    RETURNING id, name
                    """,
                    "test",
                    "TestEntity",
                    "person",
                    "A test entity",
                )
                assert row is not None
                entity_id = row["id"]
                assert row["name"] == "TestEntity"

                # Delete
                await conn.execute(
                    "DELETE FROM entities WHERE id = $1", entity_id
                )
                deleted = await conn.fetchrow(
                    "SELECT id FROM entities WHERE id = $1", entity_id
                )
                assert deleted is None
        finally:
            await pool.close()
            db_module._pool = None
