"""Compaction-driven memory extraction and RAG context retrieval.

Memory pipeline fires at two points:
1. Compaction event — pause_after_compaction intercepts the summary
2. Shutdown — forced summarization using the default compaction prompt

Extraction is 2-step:
1. Get compaction summary (from API event or manual Haiku/Flash call)
2. Send to Haiku/Flash for structured 4-layer parsing
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from anthropic import AsyncAnthropic

from prot.config import settings
from prot.decay import AdaptiveDecayCalculator
from prot.embeddings import AsyncVoyageEmbedder
from prot.graphrag import MemoryStore
from prot.logging import get_logger, logged
from prot.processing import content_to_text, strip_markdown_fences

logger = get_logger(__name__)

DEFAULT_COMPACTION_PROMPT = (
    "You have written a partial transcript for the initial task above. "
    "Please write a summary of the transcript. The purpose of this summary is "
    "to provide continuity so you can continue to make progress towards solving "
    "the task in a future context, where the raw history above may not be accessible "
    "and will be replaced with this summary. Write down anything that would be helpful, "
    "including the state, next steps, learnings etc. "
    "You must wrap your summary in a `<summary></summary>` block."
)

_EXTRACTION_PROMPT = """You are a memory extraction system. Given a conversation summary,
extract structured memories into 4 layers. The summary may be in Korean or English.
Keep names and terms in their original language.

Return JSON with this exact structure:
{
  "semantic": [
    {"category": "person|preference|fact|skill|relationship", "subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0}
  ],
  "episodic": {
    "summary": "...",
    "topics": ["..."],
    "emotional_tone": "warm|tense|playful|curious|neutral|...",
    "significance": 0.0-1.0,
    "duration_turns": 0
  },
  "emotional": [
    {"emotion": "joy|frustration|curiosity|gratitude|...", "trigger_context": "...", "intensity": 0.0-1.0}
  ],
  "procedural": [
    {"pattern": "...", "frequency": "daily|weekly|occasional|null", "confidence": 0.0-1.0}
  ]
}

Rules:
- semantic: SPO triples for facts, preferences, knowledge. Be specific and concise.
- episodic: ONE episode summarizing this conversation segment. Always include.
- emotional: Capture emotional highlights. Empty array if no notable emotions.
- procedural: Behavioral patterns observed. Empty array if none.
- If the summary has no meaningful content, return empty arrays and null episodic."""


class MemoryExtractor:
    """Compaction-driven memory extraction and retrieval."""

    def __init__(
        self,
        anthropic_key: str | None = None,
        store: MemoryStore | None = None,
        embedder: AsyncVoyageEmbedder | None = None,
        reranker=None,
    ):
        self._llm = AsyncAnthropic(api_key=anthropic_key or settings.anthropic_api_key)
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._decay = AdaptiveDecayCalculator(
            base_rate=settings.decay_base_rate,
            min_retention=settings.decay_min_retention,
        )

    async def close(self) -> None:
        await self._llm.close()
        if self._reranker:
            await self._reranker.close()

    @logged(slow_ms=3000)
    async def extract_from_summary(self, summary_text: str) -> dict:
        """Send compaction summary to Haiku/Flash for structured 4-layer extraction."""
        logger.info("Extracting from summary", chars=len(summary_text))
        response = await self._llm.messages.create(
            model=settings.memory_extraction_model,
            max_tokens=4000,
            system=[{
                "type": "text",
                "text": _EXTRACTION_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": summary_text}],
        )
        try:
            raw = response.content[0].text
        except (IndexError, AttributeError):
            logger.warning("Empty extraction response")
            return {"semantic": [], "episodic": None, "emotional": [], "procedural": []}

        raw = strip_markdown_fences(raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Extraction JSON parse failed", raw=raw[:200])
            return {"semantic": [], "episodic": None, "emotional": [], "procedural": []}

    @logged(slow_ms=5000)
    async def save_extraction(self, extraction: dict) -> None:
        """Embed and save extracted memories across all 4 layers."""
        semantics = extraction.get("semantic") or []
        episodic = extraction.get("episodic")
        emotionals = extraction.get("emotional") or []
        procedurals = extraction.get("procedural") or []

        if not semantics and not episodic and not emotionals and not procedurals:
            logger.debug("Extraction empty, skipping save")
            return

        # Collect all texts to embed in one batch
        texts_to_embed = []
        # Semantic: concatenate SPO for embedding
        for s in semantics:
            texts_to_embed.append(f"{s['subject']} {s['predicate']} {s['object']}")
        sem_count = len(semantics)

        # Episodic
        if episodic and episodic.get("summary"):
            texts_to_embed.append(episodic["summary"])
        epi_count = 1 if (episodic and episodic.get("summary")) else 0

        # Emotional
        for e in emotionals:
            texts_to_embed.append(f"{e['emotion']}: {e['trigger_context']}")
        emo_count = len(emotionals)

        # Procedural
        for p in procedurals:
            texts_to_embed.append(p["pattern"])

        if not texts_to_embed:
            return

        embeddings = await self._embedder.embed_texts(texts_to_embed)
        idx = 0

        async with self._store.acquire() as conn:
            async with conn.transaction():
                # Semantic
                for s, emb in zip(semantics, embeddings[idx:idx + sem_count]):
                    await self._store.upsert_semantic(
                        category=s["category"],
                        subject=s["subject"],
                        predicate=s["predicate"],
                        object_=s["object"],
                        confidence=s.get("confidence", 1.0),
                        embedding=emb,
                        conn=conn,
                    )
                idx += sem_count

                # Episodic
                episode_id = None
                if epi_count:
                    episode_id = await self._store.insert_episodic(
                        summary=episodic["summary"],
                        topics=episodic.get("topics", []),
                        emotional_tone=episodic.get("emotional_tone"),
                        significance=episodic.get("significance", 0.5),
                        duration_turns=episodic.get("duration_turns", 0),
                        embedding=embeddings[idx],
                        conn=conn,
                    )
                    idx += 1

                # Emotional (linked to episode)
                for e, emb in zip(emotionals, embeddings[idx:idx + emo_count]):
                    await self._store.insert_emotional(
                        emotion=e["emotion"],
                        trigger_context=e["trigger_context"],
                        intensity=e.get("intensity", 0.5),
                        episode_id=episode_id,
                        embedding=emb,
                        conn=conn,
                    )
                idx += emo_count

                # Procedural
                for p, emb in zip(procedurals, embeddings[idx:]):
                    await self._store.upsert_procedural(
                        pattern=p["pattern"],
                        frequency=p.get("frequency"),
                        confidence=p.get("confidence", 0.5),
                        embedding=emb,
                        conn=conn,
                    )

        logger.info(
            "Saved memories",
            semantic=len(semantics), episodic=bool(episodic),
            emotional=len(emotionals), procedural=len(procedurals),
        )

    @logged(slow_ms=5000)
    async def generate_shutdown_summary(self, messages: list[dict]) -> str:
        """Generate a compaction-equivalent summary at shutdown using Haiku/Flash."""
        conversation_text = "\n".join(
            f"{m['role']}: {content_to_text(m['content'])}" for m in messages
        )
        response = await self._llm.messages.create(
            model=settings.memory_extraction_model,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": f"{conversation_text}\n\n{DEFAULT_COMPACTION_PROMPT}"},
            ],
        )
        try:
            text = response.content[0].text
        except (IndexError, AttributeError):
            return ""

        # Extract content from <summary> tags if present
        if "<summary>" in text and "</summary>" in text:
            text = text.split("<summary>", 1)[1].split("</summary>", 1)[0]
        return text.strip()

    @logged(slow_ms=2000)
    async def pre_load_context(self, query: str) -> str:
        """Search all memory layers, apply time-decay, optionally rerank, format for Block 2."""
        query_embedding = await self._embedder.embed_query(query)
        results = await self._store.search_all(
            query_embedding=query_embedding, top_k=settings.rag_top_k,
        )

        if not results:
            return "(no memory context)"

        # Apply time-decay scoring
        now = datetime.now(timezone.utc)
        for r in results:
            created = r.get("created_at")
            if created:
                if isinstance(created, str):
                    created = datetime.fromisoformat(created.replace("Z", "+00:00"))
                hours = (now - created).total_seconds() / 3600
            else:
                hours = 0.0

            memory_type = _table_to_memory_type(r.get("table_name", ""))
            decay_score = self._decay.calculate(
                importance=r.get("significance", r.get("confidence", 0.5)),
                hours_passed=hours,
                access_count=r.get("mention_count", r.get("observation_count", 0)),
                memory_type=memory_type,
            )
            r["effective_score"] = r["similarity"] * decay_score

        # Sort by effective score
        results.sort(key=lambda r: r["effective_score"], reverse=True)

        # Optional reranking
        if self._reranker and len(results) > 1:
            results = await self._reranker.rerank(
                query=query, items=results, text_key="text",
                top_k=settings.rerank_top_k,
            )

        # Format into Block 2 context with token budget
        parts: list[str] = []
        token_estimate = 0
        for r in results:
            text = r.get("text", "")
            token_estimate += len(text) // 4
            if token_estimate > settings.rag_context_target_tokens:
                break
            table = r.get("table_name", "unknown")
            parts.append(f"[{table}] {text}")

        return "\n".join(parts) if parts else "(no memory context)"


def _table_to_memory_type(table_name: str) -> str:
    """Map table_name to decay memory_type."""
    return {
        "semantic": "fact",
        "episodic": "conversation",
        "emotional": "insight",
        "procedural": "preference",
    }.get(table_name, "conversation")
