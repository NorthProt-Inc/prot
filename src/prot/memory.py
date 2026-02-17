"""Memory extraction and pre-loading using Haiku 4.5 and GraphRAG."""

from __future__ import annotations

import json

from anthropic import AsyncAnthropic

from prot.config import settings
from prot.embeddings import AsyncVoyageEmbedder
from prot.graphrag import GraphRAGStore
from prot.log import get_logger

logger = get_logger(__name__)


def _content_to_text(content) -> str:
    """Extract plain text from str or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.text if hasattr(block, "text") else
            str(block.get("text", "")) if isinstance(block, dict) else ""
            for block in content
        )
    return str(content)


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
        logger.info("Extracting", messages=len(messages))
        conversation_text = "\n".join(
            f"{m['role']}: {_content_to_text(m['content'])}" for m in messages
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
        """Embed and save extracted entities and relationships in a single transaction."""
        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        if not entities:
            return

        logger.info("Saved", entities=len(entities), rels=len(relationships))

        descriptions = [e["description"] for e in entities]
        embeddings = await self._embedder.embed_texts(descriptions)

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

                for rel in relationships:
                    src_id = entity_ids.get(rel["source"])
                    tgt_id = entity_ids.get(rel["target"])
                    if src_id and tgt_id:
                        await self._store.upsert_relationship(
                            source_id=src_id,
                            target_id=tgt_id,
                            relation_type=rel["type"],
                            description=rel["description"],
                            conn=conn,
                        )

    async def pre_load_context(self, query: str) -> str:
        """Search GraphRAG (entities + neighbors + communities) and assemble Block 2 context."""
        query_embedding = await self._embedder.embed_query(query)

        parts: list[str] = []
        token_estimate = 0

        def _add(text: str) -> bool:
            nonlocal token_estimate
            token_estimate += len(text) // 4
            if token_estimate > settings.rag_context_target_tokens:
                return False
            parts.append(text)
            return True

        # 1. Entity semantic search + neighbor traversal
        entities = await self._store.search_entities_semantic(
            query_embedding=query_embedding, top_k=5,
        )
        for entity in entities:
            line = f"- {entity['name']} ({entity['entity_type']}): {entity['description']}"
            if not _add(line):
                break
            neighbors = await self._store.get_entity_neighbors(entity["id"], max_depth=1)
            for n in neighbors[:3]:
                nline = f"  > {n['name']}: {n['description']}"
                if not _add(nline):
                    break

        # 2. Community summaries
        communities = await self._store.search_communities(
            query_embedding=query_embedding,
            top_k=settings.rag_top_k,
        )
        for community in communities:
            if not _add(community["summary"]):
                break

        return "\n".join(parts) if parts else "(no memory context)"
