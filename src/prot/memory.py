"""Memory extraction and pre-loading using Haiku 4.5 and GraphRAG."""

from __future__ import annotations

import json

from anthropic import AsyncAnthropic

from prot.config import settings
from prot.embeddings import AsyncVoyageEmbedder
from prot.graphrag import GraphRAGStore

EXTRACTION_PROMPT = """Extract entities and relationships from this conversation.

Return JSON with this exact structure:
{
  "entities": [{"name": "...", "type": "person|place|concept|event", "description": "..."}],
  "relationships": [{"source": "...", "target": "...", "type": "...", "description": "..."}]
}

Only extract factual information. Skip greetings and filler."""


class MemoryExtractor:
    """Extract and manage long-term memory from conversations."""

    def __init__(
        self,
        anthropic_key: str | None = None,
        store: GraphRAGStore | None = None,
        embedder: AsyncVoyageEmbedder | None = None,
    ):
        self._llm = AsyncAnthropic(api_key=anthropic_key or settings.anthropic_api_key)
        self._store = store
        self._embedder = embedder

    async def extract_from_conversation(self, messages: list[dict]) -> dict:
        """Use Haiku 4.5 to extract entities and relationships from conversation."""
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        response = await self._llm.messages.create(
            model=settings.memory_extraction_model,
            max_tokens=2000,
            system=EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": conversation_text}],
        )
        try:
            raw = response.content[0].text
        except (IndexError, AttributeError):
            return {"entities": [], "relationships": []}
        # Strip markdown fencing if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"entities": [], "relationships": []}

    async def save_extraction(self, extraction: dict) -> None:
        """Embed and save extracted entities and relationships."""
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        if not entities:
            return

        descriptions = [e["description"] for e in entities]
        embeddings = await self._embedder.embed_texts(descriptions)

        entity_ids = {}
        for entity, embedding in zip(entities, embeddings):
            eid = await self._store.upsert_entity(
                name=entity["name"],
                entity_type=entity["type"],
                description=entity["description"],
                embedding=embedding,
            )
            entity_ids[entity["name"]] = eid

        for rel in relationships:
            src_id = entity_ids.get(rel["source"])
            tgt_id = entity_ids.get(rel["target"])
            if src_id and tgt_id:
                await self._store.upsert_relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation_type=rel["type"],
                    description=rel["description"],
                )

    async def pre_load_context(self, query: str) -> str:
        """Search GraphRAG and assemble Block 2 context text."""
        query_embedding = await self._embedder.embed_query(query)
        communities = await self._store.search_communities(
            query_embedding=query_embedding,
            top_k=settings.rag_top_k,
        )

        parts = []
        token_estimate = 0
        for community in communities:
            summary = community["summary"]
            parts.append(summary)
            token_estimate += len(summary) // 4
            if token_estimate >= settings.rag_context_target_tokens:
                break

        return "\n\n".join(parts) if parts else "(no memory context)"
