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
        mock_embedder.embed_texts_contextual.return_value = [[0.1] * 1024]

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        await extractor.save_extraction({
            "entities": [{"name": "Bob", "type": "person", "description": "A friend"}],
            "relationships": [],
        })
        mock_store.upsert_entity.assert_called_once()
        mock_embedder.embed_texts_contextual.assert_called_once()
        # Verify conn was passed through
        call_kwargs = mock_store.upsert_entity.call_args.kwargs
        assert call_kwargs["conn"] is mock_conn

    async def test_save_extraction_uses_independent_contextual_embeddings(self):
        """Verify save_extraction uses embed_texts_contextual (independent), not embed_chunks_contextual."""
        mock_store, mock_conn = _make_store_with_conn()
        mock_store.upsert_entity.return_value = "fake-uuid"
        mock_embedder = AsyncMock()
        mock_embedder.embed_texts_contextual.return_value = [[0.1] * 1024]

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        await extractor.save_extraction({
            "entities": [{"name": "Bob", "type": "person", "description": "A friend"}],
            "relationships": [],
        })
        mock_embedder.embed_texts_contextual.assert_called_once_with(["A friend"])
        mock_embedder.embed_chunks_contextual.assert_not_called()

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
        mock_embedder.embed_query_contextual.return_value = [0.1] * 1024

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
        mock_embedder.embed_texts_contextual.return_value = [[0.1] * 1024, [0.2] * 1024]

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
        mock_embedder.embed_texts_contextual.assert_not_called()

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
        mock_embedder.embed_query_contextual.return_value = [0.1] * 1024

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
        mock_embedder.embed_query_contextual.return_value = [0.1] * 1024

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        text = await extractor.pre_load_context("test")
        # Should not include all 20 entities due to token budget
        lines = [l for l in text.split("\n") if l.strip()]
        assert len(lines) < 20

    async def test_save_extraction_logs_on_empty_entities(self):
        """Empty extraction should log, not silently return."""
        mock_store = AsyncMock()
        mock_embedder = AsyncMock()
        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        with patch("prot.memory.logger") as mock_logger:
            await extractor.save_extraction({"entities": [], "relationships": []})
            mock_logger.debug.assert_called_once()
        mock_store.upsert_entity.assert_not_called()

    async def test_extract_logs_warning_on_json_parse_failure(self):
        """JSON parse failure should log warning with raw text."""
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Not valid JSON")]
        mock_anthropic.messages.create.return_value = mock_response
        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic), \
             patch("prot.memory.logger") as mock_logger:
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            result = await extractor.extract_from_conversation([
                {"role": "user", "content": "Hello"},
            ])
            assert result == {"entities": [], "relationships": []}
            mock_logger.warning.assert_called_once()

    async def test_pre_load_context_includes_relationship_info(self):
        """Neighbor formatting should include relation_type."""
        mock_store = AsyncMock()
        mock_store.search_entities_semantic.return_value = [
            {"id": "e1", "name": "Bob", "entity_type": "person", "description": "A dev"},
        ]
        mock_store.get_entity_neighbors.return_value = [
            {"name": "Alice", "entity_type": "person", "description": "A designer",
             "relation_type": "works_with", "rel_description": "Collaborates on prot"},
        ]
        mock_store.search_communities.return_value = []
        mock_embedder = AsyncMock()
        mock_embedder.embed_query_contextual.return_value = [0.1] * 1024

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder,
        )
        text = await extractor.pre_load_context("Bob's team")
        assert "works_with" in text
        assert "Alice" in text

    async def test_pre_load_context_uses_reranker(self):
        """Verify reranker.rerank is called after semantic search with correct args."""
        mock_store = AsyncMock()
        entities = [
            {"id": "e1", "name": "Bob", "entity_type": "person", "description": "A friend"},
            {"id": "e2", "name": "Alice", "entity_type": "person", "description": "A colleague"},
            {"id": "e3", "name": "Carol", "entity_type": "person", "description": "A neighbor"},
        ]
        mock_store.search_entities_semantic.return_value = entities
        mock_store.get_entity_neighbors.return_value = []
        mock_store.search_communities.return_value = []
        mock_embedder = AsyncMock()
        mock_embedder.embed_query_contextual.return_value = [0.1] * 1024

        mock_reranker = AsyncMock()
        # Reranker returns a subset (top 2)
        reranked = [entities[1], entities[0]]
        mock_reranker.rerank.return_value = reranked

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder,
            reranker=mock_reranker,
        )
        text = await extractor.pre_load_context("Tell me about colleagues")

        mock_reranker.rerank.assert_called_once_with(
            query="Tell me about colleagues",
            items=entities,
            text_key="description",
            top_k=5,  # settings.rerank_top_k default
        )
        # Output should reflect reranked order (Alice first)
        assert "Alice" in text

    async def test_pre_load_context_works_without_reranker(self):
        """Verify backward compatibility when no reranker is passed."""
        mock_store = AsyncMock()
        mock_store.search_entities_semantic.return_value = [
            {"id": "e1", "name": "Bob", "entity_type": "person", "description": "A friend"},
            {"id": "e2", "name": "Alice", "entity_type": "person", "description": "A colleague"},
        ]
        mock_store.get_entity_neighbors.return_value = []
        mock_store.search_communities.return_value = []
        mock_embedder = AsyncMock()
        mock_embedder.embed_query_contextual.return_value = [0.1] * 1024

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder,
        )
        text = await extractor.pre_load_context("Tell me about friends")

        # Should work without reranker, returning semantic search results directly
        assert "Bob" in text
        assert "Alice" in text

    async def test_pre_load_context_reranks_communities(self):
        """Verify reranker.rerank is called on community results."""
        mock_store = AsyncMock()
        mock_store.search_entities_semantic.return_value = []
        mock_store.search_communities.return_value = [
            {"summary": "Community A", "similarity": 0.9},
            {"summary": "Community B", "similarity": 0.8},
        ]
        mock_embedder = AsyncMock()
        mock_embedder.embed_query_contextual.return_value = [0.1] * 1024
        mock_reranker = AsyncMock()
        mock_reranker.rerank.return_value = [
            {"summary": "Community B", "relevance_score": 0.95},
        ]

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder,
            reranker=mock_reranker,
        )
        text = await extractor.pre_load_context("work projects")
        mock_reranker.rerank.assert_called_once()
        assert mock_reranker.rerank.call_args.kwargs["text_key"] == "summary"

    async def test_save_extraction_cross_extraction_relationship(self):
        """Verify relationships referencing prior-extraction entities are saved via DB lookup."""
        from uuid import uuid4

        mock_store, mock_conn = _make_store_with_conn()
        prior_entity_id = uuid4()
        mock_store.upsert_entity.return_value = uuid4()  # current extraction entity
        mock_store.get_entity_id_by_name = AsyncMock(return_value=prior_entity_id)
        mock_embedder = AsyncMock()
        mock_embedder.embed_texts_contextual.return_value = [[0.1] * 1024]

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        await extractor.save_extraction({
            "entities": [
                {"name": "NewEntity", "type": "concept", "description": "Just extracted"},
            ],
            "relationships": [
                {
                    "source": "PriorEntity",
                    "target": "NewEntity",
                    "type": "related_to",
                    "description": "Cross-extraction link",
                },
            ],
        })

        # PriorEntity not in current extraction → should fall back to DB lookup
        mock_store.get_entity_id_by_name.assert_called_once_with(
            "PriorEntity", conn=mock_conn,
        )
        mock_store.upsert_relationship.assert_called_once()
        call_kwargs = mock_store.upsert_relationship.call_args.kwargs
        assert call_kwargs["source_id"] == prior_entity_id
        assert call_kwargs["relation_type"] == "related_to"


class TestCommunityRebuildTrigger:
    async def test_rebuild_triggered_on_interval(self):
        mock_store, mock_conn = _make_store_with_conn()
        mock_store.upsert_entity.return_value = "fake-uuid"
        mock_embedder = AsyncMock()
        mock_embedder.embed_texts_contextual.return_value = [[0.1] * 1024]
        mock_detector = AsyncMock()
        mock_detector.rebuild.return_value = 2

        extractor = MemoryExtractor(
            anthropic_key="test",
            store=mock_store,
            embedder=mock_embedder,
            community_detector=mock_detector,
        )
        extraction = {
            "entities": [{"name": "Bob", "type": "person", "description": "A friend"}],
            "relationships": [],
        }

        # Extractions 1-4: no rebuild
        for _ in range(4):
            await extractor.save_extraction(extraction)
        mock_detector.rebuild.assert_not_called()

        # Extraction 5: triggers rebuild
        await extractor.save_extraction(extraction)
        mock_detector.rebuild.assert_called_once()

    async def test_rebuild_skipped_without_detector(self):
        mock_store, mock_conn = _make_store_with_conn()
        mock_store.upsert_entity.return_value = "fake-uuid"
        mock_embedder = AsyncMock()
        mock_embedder.embed_texts_contextual.return_value = [[0.1] * 1024]

        extractor = MemoryExtractor(
            anthropic_key="test",
            store=mock_store,
            embedder=mock_embedder,
        )
        extraction = {
            "entities": [{"name": "Bob", "type": "person", "description": "A friend"}],
            "relationships": [],
        }

        # Should not raise even after 5+ extractions
        for _ in range(6):
            await extractor.save_extraction(extraction)

    async def test_rebuild_failure_does_not_break_extraction(self):
        mock_store, mock_conn = _make_store_with_conn()
        mock_store.upsert_entity.return_value = "fake-uuid"
        mock_embedder = AsyncMock()
        mock_embedder.embed_texts_contextual.return_value = [[0.1] * 1024]
        mock_detector = AsyncMock()
        mock_detector.rebuild.side_effect = Exception("boom")

        extractor = MemoryExtractor(
            anthropic_key="test",
            store=mock_store,
            embedder=mock_embedder,
            community_detector=mock_detector,
        )
        extraction = {
            "entities": [{"name": "Bob", "type": "person", "description": "A friend"}],
            "relationships": [],
        }

        # Should not raise despite rebuild failure
        for _ in range(5):
            await extractor.save_extraction(extraction)

        mock_detector.rebuild.assert_called_once()


@pytest.mark.asyncio
class TestMemoryExtractorConcurrency:
    async def test_pre_load_context_fetches_neighbors_concurrently(self):
        """Verify neighbor queries run concurrently, not sequentially."""
        import asyncio

        call_order = []

        async def mock_get_neighbors(entity_id):
            call_order.append(("start", entity_id))
            await asyncio.sleep(0.01)  # simulate DB latency
            call_order.append(("end", entity_id))
            return [{"name": f"Neighbor-of-{entity_id}", "description": "neighbor"}]

        mock_store = AsyncMock()
        mock_store.search_entities_semantic.return_value = [
            {"id": f"e{i}", "name": f"Entity{i}", "entity_type": "person",
             "description": f"Desc {i}"}
            for i in range(3)
        ]
        mock_store.get_entity_neighbors = mock_get_neighbors
        mock_store.search_communities.return_value = []
        mock_embedder = AsyncMock()
        mock_embedder.embed_query_contextual.return_value = [0.1] * 1024

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        text = await extractor.pre_load_context("test query")

        # If concurrent: all "start" before any "end"
        # If sequential: start/end/start/end/start/end
        starts = [i for i, (action, _) in enumerate(call_order) if action == "start"]
        ends = [i for i, (action, _) in enumerate(call_order) if action == "end"]
        # All starts should come before all ends
        assert max(starts) < min(ends), (
            f"Neighbor queries appear sequential: {call_order}"
        )

        # Results should still be in output
        assert "Entity0" in text
        assert "Neighbor-of-e0" in text


@pytest.mark.asyncio
class TestIncrementalExtraction:
    async def test_extract_incremental_sends_only_recent_messages(self):
        """Verify only the last window_size messages are extracted."""
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"entities": [], "relationships": []}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic), \
             patch("prot.memory.settings") as mock_settings:
            mock_settings.memory_extraction_model = "test-model"
            mock_settings.memory_extraction_window_turns = 2  # 4 messages
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            messages = [
                {"role": "user", "content": f"msg{i}"}
                for i in range(10)
            ]
            await extractor.extract_incremental(messages)

            # Should only send last 4 messages (2 turns * 2)
            call_args = mock_anthropic.messages.create.call_args
            sent_text = call_args.kwargs["messages"][0]["content"]
            assert "msg6" in sent_text
            assert "msg9" in sent_text
            assert "msg0" not in sent_text

    async def test_extract_incremental_includes_known_entities(self):
        """Verify known entities are passed to extraction prompt."""
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"entities": [], "relationships": []}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic), \
             patch("prot.memory.settings") as mock_settings:
            mock_settings.memory_extraction_model = "test-model"
            mock_settings.memory_extraction_window_turns = 3
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            extractor._known_entities = {"Alice", "Bob"}
            messages = [{"role": "user", "content": "hello"}]
            await extractor.extract_incremental(messages)

            call_args = mock_anthropic.messages.create.call_args
            system = call_args.kwargs["system"]
            assert "Alice" in system
            assert "Bob" in system

    async def test_extract_incremental_advances_pointer(self):
        """Verify _last_extracted_index advances after extraction."""
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"entities": [], "relationships": []}')]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic), \
             patch("prot.memory.settings") as mock_settings:
            mock_settings.memory_extraction_model = "test-model"
            mock_settings.memory_extraction_window_turns = 3
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            assert extractor._last_extracted_index == 0
            messages = [{"role": "user", "content": "hello"}] * 5
            await extractor.extract_incremental(messages)
            assert extractor._last_extracted_index == 5

    async def test_extract_incremental_empty_when_no_new_messages(self):
        """Verify empty result when all messages already extracted."""
        mock_anthropic = AsyncMock()
        with patch("prot.memory.AsyncAnthropic", return_value=mock_anthropic):
            extractor = MemoryExtractor(
                anthropic_key="test", store=AsyncMock(), embedder=AsyncMock()
            )
            extractor._last_extracted_index = 10
            messages = [{"role": "user", "content": "hello"}] * 10
            result = await extractor.extract_incremental(messages)
            assert result == {"entities": [], "relationships": []}
            mock_anthropic.messages.create.assert_not_called()

    async def test_save_extraction_tracks_known_entities(self):
        """Verify entity names are added to _known_entities after save."""
        mock_store, mock_conn = _make_store_with_conn()
        mock_store.upsert_entity.return_value = "fake-uuid"
        mock_embedder = AsyncMock()
        mock_embedder.embed_texts_contextual.return_value = [[0.1] * 1024, [0.2] * 1024]

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=mock_embedder
        )
        assert len(extractor._known_entities) == 0
        await extractor.save_extraction({
            "entities": [
                {"name": "Bob", "type": "person", "description": "A friend"},
                {"name": "Alice", "type": "person", "description": "A colleague"},
            ],
            "relationships": [],
        })
        assert extractor._known_entities == {"Bob", "Alice"}

    async def test_seed_known_entities_from_db(self):
        """Verify seed_known_entities loads names from store."""
        mock_store = AsyncMock()
        mock_store.get_entity_names.return_value = ["Bob", "Alice", "Carol"]

        extractor = MemoryExtractor(
            anthropic_key="test", store=mock_store, embedder=AsyncMock()
        )
        await extractor.seed_known_entities()
        assert extractor._known_entities == {"Bob", "Alice", "Carol"}

    async def test_seed_known_entities_skips_without_store(self):
        """Verify seed_known_entities does nothing without a store."""
        extractor = MemoryExtractor(
            anthropic_key="test", store=None, embedder=AsyncMock()
        )
        await extractor.seed_known_entities()
        assert extractor._known_entities == set()
