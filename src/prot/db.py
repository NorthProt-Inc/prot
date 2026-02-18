"""Database pool management for asyncpg with pgvector schema."""

from __future__ import annotations

from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from prot.config import settings
from prot.log import get_logger, logged

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


EXPORT_TABLES = [
    "entities",
    "relationships",
    "communities",
    "community_members",
    "conversation_messages",
]


async def export_tables(pool: asyncpg.Pool, output_dir: str | None = None) -> None:
    """Export all tables to CSV using asyncpg COPY TO.

    Vector columns are exported as their PostgreSQL text representation.
    Each table is exported independently so one failure does not block others.
    """
    out = Path(output_dir or settings.db_export_dir)
    out.mkdir(parents=True, exist_ok=True)
    async with pool.acquire() as conn:
        for table in EXPORT_TABLES:
            dest = str(out / f"{table}.csv")
            try:
                result = await conn.copy_from_query(
                    f"SELECT * FROM {table}",
                    output=dest,
                    format="csv",
                    header=True,
                )
                logger.info("Exported", table=table, result=result)
            except Exception:
                logger.warning("Export failed", table=table, exc_info=True)
