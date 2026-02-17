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
        mock_store.search_entities_semantic.return_value = [
            {"id": "e1", "name": "Bob", "entity_type": "person", "description": "A friend"},
        ]
        mock_store.get_entity_neighbors.return_value = [
            {"name": "Alice", "entity_type": "person", "description": "Bob's colleague"},
        ]
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
        assert "Bob" in text
        assert "Alice" in text
        assert "friends" in text

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
        """Verify returns '(no memory context)' when no entities or communities found."""
        mock_store = AsyncMock()
        mock_store.search_entities_semantic.return_value = []
        mock_store.search_communities.return_value = []
        mock_embedder = AsyncMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        text = await extractor.pre_load_context("Tell me about something unknown")
        assert text == "(no memory context)"

    async def test_extract_from_conversation_with_list_content(self):
        """Verify extraction handles list content blocks (compact API compaction)."""
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"entities": [], "relationships": []}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            result = await extractor.extract_from_conversation([
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "hi there"},
                    {"type": "text", "text": "how are you"},
                ]},
            ])
            # Should not raise — list content handled by _content_to_text
            assert "entities" in result
            # Verify the conversation text sent to LLM contains the extracted text
            call_args = mock_anthropic.messages.create.call_args
            sent_text = call_args.kwargs["messages"][0]["content"]
            assert "hi there" in sent_text
            assert "how are you" in sent_text

    async def test_extract_from_conversation_with_object_content(self):
        """Verify extraction handles object content blocks with .text attribute."""
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"entities": [], "relationships": []}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            # Simulate Anthropic SDK content block objects (with .text attribute)
            block = MagicMock()
            block.text = "compacted response"
            result = await extractor.extract_from_conversation([
                {"role": "assistant", "content": [block]},
            ])
            assert "entities" in result
            call_args = mock_anthropic.messages.create.call_args
            sent_text = call_args.kwargs["messages"][0]["content"]
            assert "compacted response" in sent_text

    async def test_pre_load_context_respects_token_budget(self):
        """Verify token budget stops accumulation."""
        mock_store = AsyncMock()
        # Each entity desc ~1000 chars = ~250 tokens. With target_tokens=3000, should stop early.
        mock_store.search_entities_semantic.return_value = [
            {"id": f"e{i}", "name": f"Entity{i}", "entity_type": "concept",
             "description": "x" * 4000}
            for i in range(20)
        ]
        mock_store.get_entity_neighbors.return_value = []
        mock_store.search_communities.return_value = []
        mock_embedder = AsyncMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        text = await extractor.pre_load_context("test")
        # Should not include all 20 entities due to token budget
        lines = [l for l in text.split("\n") if l.strip()]
        assert len(lines) < 20
