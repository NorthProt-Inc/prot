"""Tests for MemoryExtractor — compaction-driven 4-layer memory extraction."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prot.memory import MemoryExtractor, DEFAULT_COMPACTION_PROMPT


def _make_store_with_conn():
    mock_store = AsyncMock()
    mock_conn = AsyncMock()
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx.__aexit__ = AsyncMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=mock_tx)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_store.acquire = MagicMock(return_value=mock_ctx)
    return mock_store, mock_conn


SAMPLE_EXTRACTION = {
    "semantic": [
        {"category": "preference", "subject": "user", "predicate": "likes", "object": "coffee", "confidence": 0.9}
    ],
    "episodic": {
        "summary": "User discussed morning routines",
        "topics": ["morning", "coffee"],
        "emotional_tone": "warm",
        "significance": 0.6,
        "duration_turns": 8,
    },
    "emotional": [
        {"emotion": "joy", "trigger_context": "talking about coffee", "intensity": 0.7}
    ],
    "procedural": [
        {"pattern": "asks about weather in the morning", "frequency": "daily", "confidence": 0.4}
    ],
}


@pytest.mark.asyncio
class TestMemoryExtractor:
    async def test_extract_from_summary_calls_llm(self):
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps(SAMPLE_EXTRACTION))]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            ext = MemoryExtractor(store=AsyncMock(), embedder=AsyncMock())
            result = await ext.extract_from_summary("User likes coffee.")
            assert "semantic" in result
            assert len(result["semantic"]) == 1

    async def test_extract_handles_malformed_json(self):
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not json")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            ext = MemoryExtractor(store=AsyncMock(), embedder=AsyncMock())
            result = await ext.extract_from_summary("test")
            assert result["semantic"] == []

    async def test_save_extraction_stores_all_layers(self):
        store, conn = _make_store_with_conn()
        store.upsert_semantic.return_value = MagicMock()
        store.insert_episodic.return_value = MagicMock()
        store.insert_emotional.return_value = MagicMock()
        store.upsert_procedural.return_value = MagicMock()

        embedder = AsyncMock()
        embedder.embed_texts.return_value = [[0.1] * 1024] * 5  # enough for all texts

        ext = MemoryExtractor(store=store, embedder=embedder)
        await ext.save_extraction(SAMPLE_EXTRACTION)

        store.upsert_semantic.assert_called_once()
        store.insert_episodic.assert_called_once()
        store.insert_emotional.assert_called_once()
        store.upsert_procedural.assert_called_once()

    async def test_save_extraction_empty_data(self):
        store = AsyncMock()
        embedder = AsyncMock()
        ext = MemoryExtractor(store=store, embedder=embedder)
        await ext.save_extraction({"semantic": [], "episodic": None, "emotional": [], "procedural": []})
        store.upsert_semantic.assert_not_called()

    async def test_generate_shutdown_summary_calls_llm(self):
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="<summary>User discussed coding.</summary>")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            ext = MemoryExtractor(store=AsyncMock(), embedder=AsyncMock())
            messages = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            summary = await ext.generate_shutdown_summary(messages)
            assert "coding" in summary

    async def test_pre_load_context_returns_formatted_text(self):
        store = AsyncMock()
        store.search_all.return_value = [
            {"table_name": "semantic", "text": "user likes coffee",
             "similarity": 0.9, "created_at": "2026-01-01T00:00:00Z"},
        ]
        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024

        ext = MemoryExtractor(store=store, embedder=embedder)
        text = await ext.pre_load_context("Tell me about preferences")
        assert "coffee" in text

    async def test_pre_load_context_no_results(self):
        store = AsyncMock()
        store.search_all.return_value = []
        embedder = AsyncMock()
        embedder.embed_query.return_value = [0.1] * 1024

        ext = MemoryExtractor(store=store, embedder=embedder)
        text = await ext.pre_load_context("unknown topic")
        assert text == "(no memory context)"

    async def test_default_compaction_prompt_exists(self):
        assert "summary" in DEFAULT_COMPACTION_PROMPT.lower()

    async def test_close_closes_clients(self):
        mock_anthropic = AsyncMock()
        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            ext = MemoryExtractor(store=AsyncMock(), embedder=AsyncMock())
            mock_reranker = AsyncMock()
            ext._reranker = mock_reranker
            await ext.close()
            mock_anthropic.close.assert_awaited_once()
            mock_reranker.close.assert_awaited_once()
