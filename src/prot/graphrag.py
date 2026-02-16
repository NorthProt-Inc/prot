"""pgvector-backed GraphRAG storage for entity, relationship, and community management."""

from __future__ import annotations

from uuid import UUID

import asyncpg


class GraphRAGStore:
    """pgvector-backed GraphRAG storage."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        embedding: list[float] | None = None,
        namespace: str = "default",
    ) -> UUID:
        """Insert or update entity. Increments mention_count on conflict."""
        async with self._pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT id, mention_count FROM entities WHERE namespace = $1 AND name = $2",
                namespace,
                name,
            )
            if existing:
                await conn.execute(
                    """UPDATE entities
                    SET description = $1, mention_count = mention_count + 1,
                        name_embedding = COALESCE($2, name_embedding), updated_at = now()
                    WHERE id = $3""",
                    description,
                    embedding,
                    existing["id"],
                )
                return existing["id"]
            else:
                row = await conn.fetchrow(
                    """INSERT INTO entities (namespace, name, entity_type, description, name_embedding)
                    VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                    namespace,
                    name,
                    entity_type,
                    description,
                    embedding,
                )
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
    ) -> UUID:
        """Insert a relationship between two entities."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO relationships (source_id, target_id, relation_type, description, weight)
                VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                source_id,
                target_id,
                relation_type,
                description,
                weight,
            )
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

    async def get_entity_neighbors(
        self, entity_id: UUID, max_depth: int = 1
    ) -> list[dict]:
        """Get neighboring entities via relationships using recursive CTE."""
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
                    SELECT CASE WHEN r.source_id = n.id THEN r.target_id ELSE r.source_id END,
                           n.depth + 1
                    FROM neighbors n
                    JOIN relationships r ON r.source_id = n.id OR r.target_id = n.id
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
