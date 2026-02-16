"""Database pool management for asyncpg with pgvector schema."""

from __future__ import annotations

from pathlib import Path

import asyncpg

from prot.config import settings
from prot.log import get_logger

logger = get_logger(__name__)

_pool: asyncpg.Pool | None = None

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def init_pool(dsn: str | None = None) -> asyncpg.Pool:
    """Create asyncpg connection pool."""
    global _pool
    if _pool is not None:
        raise RuntimeError("Pool already initialized. Call close_pool() first.")
    _pool = await asyncpg.create_pool(
        dsn=dsn or settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
    )
    logger.info("Pool created", min=settings.db_pool_min, max=settings.db_pool_max)
    return _pool


async def get_pool() -> asyncpg.Pool:
    """Get existing pool or raise."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_pool() first.")
    return _pool


async def close_pool() -> None:
    """Close connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def execute_schema(pool: asyncpg.Pool) -> None:
    """Execute schema.sql to create tables and indexes."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
