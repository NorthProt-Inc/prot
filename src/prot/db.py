"""Database pool management for asyncpg with pgvector schema."""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from prot.config import settings
from prot.logging import get_logger, logged

logger = get_logger(__name__)

_pool: asyncpg.Pool | None = None


@logged(slow_ms=5000)
async def init_pool(dsn: str | None = None) -> asyncpg.Pool:
    """Create asyncpg connection pool."""
    global _pool
    if _pool is not None:
        raise RuntimeError("Pool already initialized.")
    _pool = await asyncpg.create_pool(
        dsn=dsn or settings.database_url,
        min_size=settings.db_pool_min,
        max_size=settings.db_pool_max,
        init=register_vector,
    )
    logger.info("Pool created", min=settings.db_pool_min, max=settings.db_pool_max)
    return _pool


