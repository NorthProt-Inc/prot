"""Memory extraction and pre-loading via GraphRAG."""

from __future__ import annotations

import asyncio
import json

from anthropic import AsyncAnthropic

from prot.config import settings
from prot.embeddings import AsyncVoyageEmbedder
from prot.graphrag import GraphRAGStore
from prot.log import get_logger, logged
from prot.processing import content_to_text

logger = get_logger(__name__)


EXTRACTION_PROMPT = """Extract entities and relationships from this conversation segment.
The conversation may be in Korean or English. Keep entity names in their original language.

{known_entities_block}

Return JSON with this exact structure:
{{
  "entities": [{{"name": "...", "type": "person|place|concept|event|preference", "description": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "type": "...", "description": "..."}}]
}}

Extract names, places, preferences, plans, opinions, and technical topics.
When you encounter pronouns or references, resolve them to known entities where possible.
Skip generic greetings and filler. If nothing meaningful, return empty arrays."""

_KNOWN_ENTITIES_TEMPLATE = "Previously known entities: {names}. Link new information to these when relevant."


class MemoryExtractor:
    """Extract and manage long-term memory from conversations."""

    def __init__(
        self,
        anthropic_key: str | None = None,
        store: GraphRAGStore | None = None,
        embedder: AsyncVoyageEmbedder | None = None,
        community_detector=None,
        reranker=None,
    ):
        self._llm = AsyncAnthropic(api_key=anthropic_key or settings.anthropic_api_key)
        self._store = store
        self._embedder = embedder
        self._community_detector = community_detector
        self._reranker = reranker
        self._extraction_count: int = 0
        self._last_extracted_index: int = 0
        self._known_entities: set[str] = set()

    async def close(self) -> None:
        """Close the underlying Anthropic client."""
        await self._llm.close()
        if self._community_detector:
            await self._community_detector.close()
        if self._reranker:
            await self._reranker.close()

    @logged(slow_ms=3000)
    async def extract_from_conversation(
        self, messages: list[dict], known_entity_names: list[str] | None = None,
    ) -> dict:
        """Extract entities and relationships from conversation segment."""
        logger.info("Extracting", messages=len(messages))
        conversation_text = "\n".join(
            f"{m['role']}: {content_to_text(m['content'])}" for m in messages
        )
        if known_entity_names:
            block = _KNOWN_ENTITIES_TEMPLATE.format(names=", ".join(known_entity_names))
        else:
            block = ""
        system = EXTRACTION_PROMPT.format(known_entities_block=block)
        response = await self._llm.messages.create(
            model=settings.memory_extraction_model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": conversation_text}],
        )
        try:
            raw = response.content[0].text
        except (IndexError, AttributeError):
            logger.warning("Empty LLM response for extraction")
            return {"entities": [], "relationships": []}
        # Strip markdown fencing if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Extraction JSON parse failed", raw=raw[:200])
            return {"entities": [], "relationships": []}

    @logged(slow_ms=2000)
    async def save_extraction(self, extraction: dict) -> None:
        """Embed and save extracted entities and relationships in a single transaction."""
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        if not entities:
            logger.debug("Extraction empty, skipping save")
            return

        descriptions = [e["description"] for e in entities]
        embeddings = await self._embedder.embed_texts_contextual(descriptions)

        async with self._store.acquire() as conn:
            async with conn.transaction():
                entity_ids = {}
                for entity, embedding in zip(entities, embeddings):
                    eid = await self._store.upsert_entity(
                        name=entity["name"],
                        entity_type=entity["type"],
                        description=entity["description"],
                        embedding=embedding,
                        conn=conn,
                    )
                    entity_ids[entity["name"]] = eid
                    self._known_entities.add(entity["name"])

                for rel in relationships:
                    src_id = entity_ids.get(rel["source"])
                    tgt_id = entity_ids.get(rel["target"])
                    if not src_id:
                        src_id = await self._store.get_entity_id_by_name(
                            rel["source"], conn=conn,
                        )
                    if not tgt_id:
                        tgt_id = await self._store.get_entity_id_by_name(
                            rel["target"], conn=conn,
                        )
                    if src_id and tgt_id:
                        await self._store.upsert_relationship(
                            source_id=src_id,
                            target_id=tgt_id,
                            relation_type=rel["type"],
                            description=rel["description"],
                            conn=conn,
                        )
        logger.info("Saved", entities=len(entities), rels=len(relationships))

        self._extraction_count += 1
        if (
            self._community_detector
            and self._extraction_count % settings.community_rebuild_interval == 0
        ):
            await self._maybe_rebuild_communities()

    async def _maybe_rebuild_communities(self) -> None:
        """Trigger community detection rebuild."""
        try:
            count = await self._community_detector.rebuild()
            logger.info("Community rebuild complete", communities=count)
        except Exception:
            logger.warning("Community rebuild failed", exc_info=True)

    async def extract_incremental(self, all_messages: list[dict]) -> dict:
        """Extract from only the most recent conversation window."""
        window_size = settings.memory_extraction_window_turns * 2  # user+assistant pairs
        start = max(self._last_extracted_index, len(all_messages) - window_size)
        segment = all_messages[start:]
        if not segment:
            return {"entities": [], "relationships": []}
        known_names = sorted(self._known_entities)
        extraction = await self.extract_from_conversation(segment, known_entity_names=known_names)
        self._last_extracted_index = len(all_messages)
        return extraction

    async def seed_known_entities(self) -> None:
        """Seed known entities from DB on startup."""
        if not self._store:
            return
        names = await self._store.get_entity_names()
        self._known_entities.update(names)

    async def pre_load_context(self, query: str) -> str:
        """Search GraphRAG (entities + neighbors + communities) and assemble Block 2 context."""
        query_embedding = await self._embedder.embed_query_contextual(query)

        parts: list[str] = []
        token_estimate = 0

        def _add(text: str) -> bool:
            nonlocal token_estimate
            token_estimate += len(text) // 4
            if token_estimate > settings.rag_context_target_tokens:
                return False
            parts.append(text)
            return True

        # 1. Entity semantic search + neighbor traversal (concurrent)
        entities = await self._store.search_entities_semantic(
            query_embedding=query_embedding, top_k=settings.rag_top_k,
        )

        # Rerank entities if reranker available
        if self._reranker and len(entities) > 1:
            entities = await self._reranker.rerank(
                query=query, items=entities, text_key="description",
                top_k=settings.rerank_top_k,
            )

        neighbor_lists = await asyncio.gather(*(
            self._store.get_entity_neighbors(entity["id"], max_depth=1)
            for entity in entities
        ))
        for entity, neighbors in zip(entities, neighbor_lists):
            line = f"- {entity['name']} ({entity['entity_type']}): {entity['description']}"
            if not _add(line):
                break
            for n in neighbors[:3]:
                rel = f" ({n['relation_type']})" if n.get("relation_type") else ""
                desc = n.get("rel_description") or n["description"]
                nline = f"  > {n['name']}{rel}: {desc}"
                if not _add(nline):
                    break

        # 2. Community summaries
        communities = await self._store.search_communities(
            query_embedding=query_embedding,
            top_k=settings.rag_top_k,
        )

        # Rerank communities if reranker available
        if self._reranker and len(communities) > 1:
            communities = await self._reranker.rerank(
                query=query, items=communities, text_key="summary",
                top_k=settings.rerank_top_k,
            )

        for community in communities:
            if not _add(community["summary"]):
                break

        return "\n".join(parts) if parts else "(no memory context)"
