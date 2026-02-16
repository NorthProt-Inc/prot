import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from prot.memory import MemoryExtractor


def _make_store_with_conn():
    """Create mock store with acquire() returning an async context manager."""
    mock_store = AsyncMock()
    mock_conn = AsyncMock()
    # conn.transaction() returns an async context manager (sync call)
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_tx)
    mock_tx.__aexit__ = AsyncMock(return_value=False)
    mock_conn.transaction = MagicMock(return_value=mock_tx)
    # acquire() returns an async context manager (sync call)
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_store.acquire = MagicMock(return_value=mock_ctx)
    return mock_store, mock_conn


@pytest.mark.asyncio
class TestMemoryExtractor:
    async def test_extract_from_conversation_returns_entities(self):
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"entities": [{"name": "Bob", "type": "person", "description": "A friend"}], "relationships": []}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            result = await extractor.extract_from_conversation([
                {"role": "user", "content": "I met Bob today"},
                {"role": "assistant", "content": "How was it?"},
            ])
            assert len(result["entities"]) == 1
            assert result["entities"][0]["name"] == "Bob"

    async def test_save_extraction_calls_store(self):
        mock_store, mock_conn = _make_store_with_conn()
        mock_store.upsert_entity.return_value = "fake-uuid"
        mock_embedder = AsyncMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 1024]

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        await extractor.save_extraction({
            "entities": [{"name": "Bob", "type": "person", "description": "A friend"}],
            "relationships": [],
        })
        mock_store.upsert_entity.assert_called_once()
        mock_embedder.embed_texts.assert_called_once()
        # Verify conn was passed through
        call_kwargs = mock_store.upsert_entity.call_args.kwargs
        assert call_kwargs["conn"] is mock_conn

    async def test_pre_load_context_returns_text(self):
        mock_store = AsyncMock()
        mock_store.search_communities.return_value = [
            {"summary": "Community about friends", "similarity": 0.9},
            {"summary": "Community about work", "similarity": 0.8},
        ]
        mock_embedder = AsyncMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        text = await extractor.pre_load_context("Tell me about friends")
        assert "friends" in text
        assert "work" in text

    async def test_save_extraction_with_relationships(self):
        """Verify upsert_relationship is called when extraction has relationships."""
        mock_store, mock_conn = _make_store_with_conn()
        mock_store.upsert_entity.side_effect = ["uuid-bob", "uuid-alice"]
        mock_embedder = AsyncMock()
        mock_embedder.embed_texts.return_value = [[0.1] * 1024, [0.2] * 1024]

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        await extractor.save_extraction({
            "entities": [
                {"name": "Bob", "type": "person", "description": "A friend"},
                {"name": "Alice", "type": "person", "description": "A colleague"},
            ],
            "relationships": [
                {
                    "source": "Bob",
                    "target": "Alice",
                    "type": "knows",
                    "description": "Bob knows Alice from work",
                },
            ],
        })
        mock_store.upsert_relationship.assert_called_once_with(
            source_id="uuid-bob",
            target_id="uuid-alice",
            relation_type="knows",
            description="Bob knows Alice from work",
            conn=mock_conn,
        )

    async def test_save_extraction_empty_entities_returns_early(self):
        """Verify early return when entities list is empty."""
        mock_store = AsyncMock()
        mock_embedder = AsyncMock()

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        await extractor.save_extraction({
            "entities": [],
            "relationships": [],
        })
        mock_store.upsert_entity.assert_not_called()
        mock_embedder.embed_texts.assert_not_called()

    async def test_extract_from_conversation_handles_malformed_json(self):
        """Verify graceful fallback when LLM returns non-JSON."""
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Sorry, I cannot extract entities")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            result = await extractor.extract_from_conversation([
                {"role": "user", "content": "Hello"},
            ])
            assert result == {"entities": [], "relationships": []}

    async def test_extract_from_conversation_handles_markdown_fenced_json(self):
        """Verify markdown code fencing is stripped before parsing."""
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"entities": [{"name": "X", "type": "person", "description": "test"}], "relationships": []}\n```')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            result = await extractor.extract_from_conversation([
                {"role": "user", "content": "I know X"},
            ])
            assert len(result["entities"]) == 1
            assert result["entities"][0]["name"] == "X"

    async def test_pre_load_context_no_results(self):
        """Verify returns '(no memory context)' when no communities found."""
        mock_store = AsyncMock()
        mock_store.search_communities.return_value = []
        mock_embedder = AsyncMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        text = await extractor.pre_load_context("Tell me about something unknown")
        assert text == "(no memory context)"
