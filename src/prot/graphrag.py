"""pgvector-backed GraphRAG storage for entity, relationship, and community management."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

import asyncpg


class GraphRAGStore:
    """pgvector-backed GraphRAG storage."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    def acquire(self):
        """Acquire a connection from the pool (async context manager)."""
        return self._pool.acquire()

    @asynccontextmanager
    async def _conn(
        self, conn: asyncpg.Connection | None = None,
    ) -> AsyncIterator[asyncpg.Connection]:
        """Yield *conn* if provided, otherwise acquire one from the pool."""
        if conn is not None:
            yield conn
        else:
            async with self._pool.acquire() as c:
                yield c

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        embedding: list[float] | None = None,
        namespace: str = "default",
        conn: asyncpg.Connection | None = None,
    ) -> UUID:
        """Insert or update entity. Increments mention_count on conflict."""
        query = """INSERT INTO entities (namespace, name, entity_type, description, name_embedding)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (namespace, name)
                DO UPDATE SET description = CASE
                    WHEN entities.description = '' THEN EXCLUDED.description
                    WHEN POSITION(EXCLUDED.description IN entities.description) > 0 THEN entities.description
                    ELSE LEFT(entities.description || E'\n' || EXCLUDED.description, 500)
                    END,
                             mention_count = entities.mention_count + 1,
                             name_embedding = COALESCE(EXCLUDED.name_embedding, entities.name_embedding),
                             updated_at = now()
                RETURNING id"""
        async with self._conn(conn) as c:
            row = await c.fetchrow(query, namespace, name, entity_type, description, embedding)
            return row["id"]

    async def upsert_relationship(
        self,
        source_id: UUID,
        target_id: UUID,
        relation_type: str,
        description: str,
        weight: float = 1.0,
        conn: asyncpg.Connection | None = None,
    ) -> UUID:
        """Insert or update a relationship between two entities."""
        query = """INSERT INTO relationships (source_id, target_id, relation_type, description, weight)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (source_id, target_id, relation_type)
                DO UPDATE SET description = EXCLUDED.description,
                             weight = EXCLUDED.weight,
                             updated_at = now()
                RETURNING id"""
        async with self._conn(conn) as c:
            row = await c.fetchrow(query, source_id, target_id, relation_type, description, weight)
            return row["id"]

    async def get_entity_id_by_name(
        self, name: str, namespace: str = "default",
        conn: asyncpg.Connection | None = None,
    ) -> UUID | None:
        """Look up entity ID by name. Returns None if not found."""
        query = "SELECT id FROM entities WHERE namespace = $1 AND name = $2"
        async with self._conn(conn) as c:
            row = await c.fetchrow(query, namespace, name)
        return row["id"] if row else None

    async def get_entity_names(self, namespace: str = "default") -> list[str]:
        """Return all entity names ordered by mention count."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT name FROM entities WHERE namespace = $1 ORDER BY mention_count DESC",
                namespace,
            )
            return [r["name"] for r in rows]

    async def search_entities_semantic(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[dict]:
        """Search entities by semantic similarity using pgvector cosine distance."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, name, entity_type, description, mention_count,
                          1 - (name_embedding <=> $1::vector) AS similarity
                FROM entities WHERE name_embedding IS NOT NULL
                ORDER BY name_embedding <=> $1::vector LIMIT $2""",
                query_embedding,
                top_k,
            )
            return [dict(r) for r in rows]

    async def search_communities(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[dict]:
        """Search communities by semantic similarity using pgvector cosine distance."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, level, summary, entity_count,
                          1 - (summary_embedding <=> $1::vector) AS similarity
                FROM communities WHERE summary_embedding IS NOT NULL
                ORDER BY summary_embedding <=> $1::vector LIMIT $2""",
                query_embedding,
                top_k,
            )
            return [dict(r) for r in rows]

    async def save_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        embedding: list[float] | None = None,
    ) -> UUID:
        """Persist a conversation message to the database."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO conversation_messages
                   (conversation_id, role, content, content_embedding)
                   VALUES ($1, $2, $3, $4) RETURNING id""",
                conversation_id,
                role,
                content,
                embedding,
            )
            return row["id"]

    async def load_graph_for_community_detection(
        self,
    ) -> tuple[list[dict], list[dict]]:
        """Load all entities and relationships for community detection."""
        async with self._pool.acquire() as conn:
            entity_rows = await conn.fetch(
                "SELECT id, name, entity_type, description FROM entities"
            )
            rel_rows = await conn.fetch(
                "SELECT source_id, target_id, weight FROM relationships"
            )
            return [dict(r) for r in entity_rows], [dict(r) for r in rel_rows]

    async def rebuild_communities(self, communities: list[dict]) -> None:
        """Clear all communities and insert new ones atomically.

        Each community dict: {summary, summary_embedding, entity_ids}
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM communities")
                for comm in communities:
                    row = await conn.fetchrow(
                        """INSERT INTO communities (level, summary, summary_embedding, entity_count)
                        VALUES (0, $1, $2, $3) RETURNING id""",
                        comm["summary"],
                        comm["summary_embedding"],
                        len(comm["entity_ids"]),
                    )
                    if comm["entity_ids"]:
                        await conn.executemany(
                            """INSERT INTO community_members (community_id, entity_id)
                            VALUES ($1, $2) ON CONFLICT DO NOTHING""",
                            [(row["id"], eid) for eid in comm["entity_ids"]],
                        )

    async def get_entity_count(self) -> int:
        """Return total number of entities."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT count(*) AS cnt FROM entities")
            return row["cnt"]

    async def get_entity_neighbors(
        self, entity_id: UUID,
    ) -> list[dict]:
        """Get neighboring entities with relationship info (depth-1 only)."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT e.id, e.name, e.entity_type, e.description,
                       r.relation_type, r.description AS rel_description
                FROM relationships r
                JOIN entities e ON e.id = CASE
                    WHEN r.source_id = $1 THEN r.target_id
                    ELSE r.source_id
                END
                WHERE (r.source_id = $1 OR r.target_id = $1)
                  AND e.id != $1
                ORDER BY r.weight DESC
                """,
                entity_id,
            )
            return [dict(r) for r in rows]
