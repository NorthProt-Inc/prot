"""Community detection and summarization for GraphRAG."""

from __future__ import annotations

import networkx as nx
from anthropic import AsyncAnthropic

from prot.config import settings
from prot.embeddings import AsyncVoyageEmbedder
from prot.graphrag import GraphRAGStore
from prot.logging import get_logger, logged

logger = get_logger(__name__)

COMMUNITY_SUMMARY_PROMPT = """You are summarizing a group of related entities from a personal assistant's memory.
These entities were found to form a community (cluster) in the user's knowledge graph.

Given the entities and their descriptions below, write a concise summary (2-3 sentences)
that captures what this group represents and why these entities are related.
The summary should be useful as context for future conversations.

Entities may be in Korean or English. Write the summary in the same language as the majority of entities.
If mixed, prefer Korean.

Entities:
{entity_list}

Write ONLY the summary, no preamble or formatting."""


class CommunityDetector:
    """Detects communities in the entity graph and generates summaries."""

    def __init__(
        self,
        store: GraphRAGStore,
        embedder: AsyncVoyageEmbedder,
        anthropic_key: str | None = None,
    ):
        self._store = store
        self._embedder = embedder
        self._llm = AsyncAnthropic(api_key=anthropic_key or settings.anthropic_api_key)

    async def close(self) -> None:
        """Close the underlying Anthropic client."""
        await self._llm.close()

    @logged(slow_ms=5000)
    async def rebuild(self) -> int:
        """Full community rebuild: detect, summarize, embed, save.

        Returns number of communities created.
        """
        entity_count = await self._store.get_entity_count()
        if entity_count < settings.community_min_entities:
            logger.debug(
                "Too few entities for community detection",
                count=entity_count,
                min=settings.community_min_entities,
            )
            return 0

        entities, relationships = (
            await self._store.load_graph_for_community_detection()
        )
        if not entities:
            return 0

        G = self._build_graph(entities, relationships)
        if G.number_of_nodes() < 2:
            await self._store.rebuild_communities([])
            logger.info("No communities detected, cleared stale data")
            return 0

        partitions = self._detect_communities(G)

        entity_lookup = {str(e["id"]): e for e in entities}
        communities = []
        for partition in partitions:
            member_entities = [
                entity_lookup[nid]
                for nid in partition
                if nid in entity_lookup
            ]
            if len(member_entities) < 2:
                continue

            summary = await self._summarize_community(member_entities)
            embedding = await self._embedder.embed_chunks_contextual([summary])
            communities.append({
                "summary": summary,
                "summary_embedding": embedding[0],
                "entity_ids": [e["id"] for e in member_entities],
            })

        if communities:
            await self._store.rebuild_communities(communities)
            logger.info(
                "Communities rebuilt",
                count=len(communities),
                total_entities=sum(len(c["entity_ids"]) for c in communities),
            )
        else:
            await self._store.rebuild_communities([])
            logger.info("No communities detected, cleared stale data")

        return len(communities)

    @staticmethod
    def _build_graph(
        entities: list[dict], relationships: list[dict]
    ) -> nx.Graph:
        """Build undirected weighted NetworkX graph."""
        G = nx.Graph()
        for entity in entities:
            G.add_node(
                str(entity["id"]),
                **{k: v for k, v in entity.items() if k != "id"},
            )
        for rel in relationships:
            src, tgt = str(rel["source_id"]), str(rel["target_id"])
            if src in G and tgt in G:
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] += rel.get("weight", 1.0)
                else:
                    G.add_edge(src, tgt, weight=rel.get("weight", 1.0))
        return G

    @staticmethod
    def _detect_communities(G: nx.Graph) -> list[set[str]]:
        """Run Louvain community detection. Returns list of node-ID sets."""
        try:
            partitions = nx.community.louvain_communities(
                G, weight="weight", resolution=1.0, seed=42
            )
            return [p for p in partitions if len(p) >= 2]
        except Exception:
            logger.warning("Louvain detection failed", exc_info=True)
            return []

    async def _summarize_community(self, entities: list[dict]) -> str:
        """Generate a concise community summary via LLM."""
        entity_list = "\n".join(
            f"- {e['name']} ({e['entity_type']}): {e['description']}"
            for e in entities
        )
        prompt = COMMUNITY_SUMMARY_PROMPT.format(entity_list=entity_list)
        try:
            response = await self._llm.messages.create(
                model=settings.memory_extraction_model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception:
            logger.warning("Community summarization failed", exc_info=True)
            names = ", ".join(e["name"] for e in entities[:5])
            return f"Group related to: {names}"
