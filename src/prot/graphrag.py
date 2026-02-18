"""pgvector-backed GraphRAG storage for entity, relationship, and community management."""

from __future__ import annotations

from uuid import UUID

import asyncpg


class GraphRAGStore:
    """pgvector-backed GraphRAG storage."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    def acquire(self):
        """Acquire a connection from the pool (async context manager)."""
        return self._pool.acquire()

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
                DO UPDATE SET description = EXCLUDED.description,
                             mention_count = entities.mention_count + 1,
                             name_embedding = COALESCE(EXCLUDED.name_embedding, entities.name_embedding),
                             updated_at = now()
                RETURNING id"""
        args = (namespace, name, entity_type, description, embedding)
        if conn is not None:
            row = await conn.fetchrow(query, *args)
            return row["id"]
        async with self._pool.acquire() as c:
            row = await c.fetchrow(query, *args)
            return row["id"]

    async def get_entity_by_name(
        self, name: str, namespace: str = "default"
    ) -> dict | None:
        """Get entity by name within a namespace."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM entities WHERE namespace = $1 AND name = $2",
                namespace,
                name,
            )
            return dict(row) if row else None

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
        args = (source_id, target_id, relation_type, description, weight)
        if conn is not None:
            row = await conn.fetchrow(query, *args)
            return row["id"]
        async with self._pool.acquire() as c:
            row = await c.fetchrow(query, *args)
            return row["id"]

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

    async def upsert_community(
        self,
        level: int,
        summary: str,
        summary_embedding: list[float] | None = None,
        entity_count: int = 0,
    ) -> UUID:
        """Insert a community record."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO communities (level, summary, summary_embedding, entity_count)
                VALUES ($1, $2, $3, $4) RETURNING id""",
                level,
                summary,
                summary_embedding,
                entity_count,
            )
            return row["id"]

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
        self, entity_id: UUID, max_depth: int = 1
    ) -> list[dict]:
        """Get neighboring entities via relationships using recursive CTE.

        The recursive step uses two UNION ALL branches instead of OR so that
        each branch can use its respective index (source_id or target_id).
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH RECURSIVE neighbors AS (
                    SELECT target_id AS id, 1 AS depth
                    FROM relationships WHERE source_id = $1
                    UNION
                    SELECT source_id AS id, 1 AS depth
                    FROM relationships WHERE target_id = $1
                    UNION ALL
                    SELECT r.target_id, n.depth + 1
                    FROM neighbors n
                    JOIN relationships r ON r.source_id = n.id
                    WHERE n.depth < $2
                    UNION ALL
                    SELECT r.source_id, n.depth + 1
                    FROM neighbors n
                    JOIN relationships r ON r.target_id = n.id
                    WHERE n.depth < $2
                )
                SELECT DISTINCT e.id, e.name, e.entity_type, e.description
                FROM neighbors n
                JOIN entities e ON e.id = n.id
                WHERE e.id != $1
                """,
                entity_id,
                max_depth,
            )
            return [dict(r) for r in rows]
