"""Tests for prot.community — community detection and summarization."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import networkx as nx

from prot.community import CommunityDetector


class TestBuildGraph:
    """_build_graph — constructs NetworkX graph from DB data."""

    def test_builds_nodes_and_edges(self):
        e1, e2 = uuid4(), uuid4()
        entities = [
            {"id": e1, "name": "A", "entity_type": "person", "description": "desc A"},
            {"id": e2, "name": "B", "entity_type": "person", "description": "desc B"},
        ]
        relationships = [
            {"source_id": e1, "target_id": e2, "weight": 1.0},
        ]
        G = CommunityDetector._build_graph(entities, relationships)
        assert G.number_of_nodes() == 2
        assert G.number_of_edges() == 1

    def test_aggregates_weights_for_multi_relationships(self):
        e1, e2 = uuid4(), uuid4()
        entities = [
            {"id": e1, "name": "A", "entity_type": "person", "description": ""},
            {"id": e2, "name": "B", "entity_type": "person", "description": ""},
        ]
        relationships = [
            {"source_id": e1, "target_id": e2, "weight": 1.0},
            {"source_id": e1, "target_id": e2, "weight": 0.5},
        ]
        G = CommunityDetector._build_graph(entities, relationships)
        assert G[str(e1)][str(e2)]["weight"] == 1.5

    def test_skips_edges_with_missing_nodes(self):
        e1 = uuid4()
        entities = [{"id": e1, "name": "A", "entity_type": "person", "description": ""}]
        relationships = [{"source_id": e1, "target_id": uuid4(), "weight": 1.0}]
        G = CommunityDetector._build_graph(entities, relationships)
        assert G.number_of_edges() == 0

    def test_empty_inputs(self):
        G = CommunityDetector._build_graph([], [])
        assert G.number_of_nodes() == 0


class TestDetectCommunities:
    """_detect_communities — Louvain community detection."""

    def test_finds_two_disconnected_cliques(self):
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")], weight=1.0)
        G.add_edges_from([("D", "E"), ("E", "F"), ("D", "F")], weight=1.0)
        communities = CommunityDetector._detect_communities(G)
        assert len(communities) == 2

    def test_filters_singletons(self):
        G = nx.Graph()
        G.add_node("lonely")
        G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")], weight=1.0)
        communities = CommunityDetector._detect_communities(G)
        for comm in communities:
            assert len(comm) >= 2

    def test_deterministic_with_seed(self):
        G = nx.Graph()
        G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C"),
                          ("D", "E"), ("E", "F"), ("D", "F")], weight=1.0)
        c1 = CommunityDetector._detect_communities(G)
        c2 = CommunityDetector._detect_communities(G)
        assert [sorted(c) for c in sorted(c1, key=sorted)] == \
               [sorted(c) for c in sorted(c2, key=sorted)]

    def test_returns_empty_for_single_node(self):
        G = nx.Graph()
        G.add_node("alone")
        communities = CommunityDetector._detect_communities(G)
        assert communities == []


class TestSummarizeCommunity:
    """_summarize_community — LLM summarization."""

    async def test_calls_haiku_with_entity_list(self):
        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Bob and Alice are friends.")]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.community.AsyncAnthropic", return_value=mock_anthropic):
            detector = CommunityDetector(
                store=AsyncMock(), embedder=AsyncMock(), anthropic_key="test"
            )
            result = await detector._summarize_community([
                {"name": "Bob", "entity_type": "person", "description": "A friend"},
                {"name": "Alice", "entity_type": "person", "description": "Another friend"},
            ])
            assert result == "Bob and Alice are friends."
            call_args = mock_anthropic.messages.create.call_args
            prompt = call_args.kwargs["messages"][0]["content"]
            assert "Bob" in prompt
            assert "Alice" in prompt

    async def test_fallback_on_llm_error(self):
        mock_anthropic = AsyncMock()
        mock_anthropic.messages.create.side_effect = Exception("API error")

        with patch("prot.community.AsyncAnthropic", return_value=mock_anthropic):
            detector = CommunityDetector(
                store=AsyncMock(), embedder=AsyncMock(), anthropic_key="test"
            )
            result = await detector._summarize_community([
                {"name": "Bob", "entity_type": "person", "description": "A friend"},
                {"name": "Alice", "entity_type": "person", "description": "Another friend"},
            ])
            assert "Bob" in result
            assert "Alice" in result


class TestRebuild:
    """rebuild — full detect + summarize + embed + save cycle."""

    async def test_rebuild_creates_communities(self):
        mock_store = AsyncMock()
        mock_store.get_entity_count.return_value = 6
        e1, e2, e3 = uuid4(), uuid4(), uuid4()
        e4, e5, e6 = uuid4(), uuid4(), uuid4()
        mock_store.load_graph_for_community_detection.return_value = (
            [
                {"id": e1, "name": "A", "entity_type": "person", "description": "desc A"},
                {"id": e2, "name": "B", "entity_type": "person", "description": "desc B"},
                {"id": e3, "name": "C", "entity_type": "person", "description": "desc C"},
                {"id": e4, "name": "D", "entity_type": "place", "description": "desc D"},
                {"id": e5, "name": "E", "entity_type": "place", "description": "desc E"},
                {"id": e6, "name": "F", "entity_type": "place", "description": "desc F"},
            ],
            [
                {"source_id": e1, "target_id": e2, "weight": 1.0},
                {"source_id": e2, "target_id": e3, "weight": 1.0},
                {"source_id": e1, "target_id": e3, "weight": 1.0},
                {"source_id": e4, "target_id": e5, "weight": 1.0},
                {"source_id": e5, "target_id": e6, "weight": 1.0},
                {"source_id": e4, "target_id": e6, "weight": 1.0},
            ],
        )
        mock_embedder = AsyncMock()
        mock_embedder.embed_chunks_contextual.return_value = [[0.1] * 1024, [0.2] * 1024]

        mock_anthropic = AsyncMock()
        # Batch summarization returns JSON
        batch_json = '{"1": "Group of people A B C", "2": "Group of places D E F"}'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=batch_json)]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.community.AsyncAnthropic", return_value=mock_anthropic):
            detector = CommunityDetector(
                store=mock_store, embedder=mock_embedder, anthropic_key="test"
            )
            count = await detector.rebuild()

        assert count == 2
        mock_store.rebuild_communities.assert_called_once()
        communities = mock_store.rebuild_communities.call_args[0][0]
        assert len(communities) == 2
        # Single batch LLM call instead of N individual calls
        assert mock_anthropic.messages.create.call_count == 1

    async def test_rebuild_skips_when_too_few_entities(self):
        mock_store = AsyncMock()
        mock_store.get_entity_count.return_value = 3

        with patch("prot.community.AsyncAnthropic", return_value=AsyncMock()):
            detector = CommunityDetector(
                store=mock_store, embedder=AsyncMock(), anthropic_key="test"
            )
            count = await detector.rebuild()

        assert count == 0
        mock_store.load_graph_for_community_detection.assert_not_called()

    async def test_rebuild_clears_stale_when_no_communities_found(self):
        mock_store = AsyncMock()
        mock_store.get_entity_count.return_value = 5
        e1 = uuid4()
        mock_store.load_graph_for_community_detection.return_value = (
            [{"id": e1, "name": "Lonely", "entity_type": "person", "description": "alone"}],
            [],
        )

        with patch("prot.community.AsyncAnthropic", return_value=AsyncMock()):
            detector = CommunityDetector(
                store=mock_store, embedder=AsyncMock(), anthropic_key="test"
            )
            count = await detector.rebuild()

        assert count == 0
        mock_store.rebuild_communities.assert_called_once_with([])


class TestBatchSummarization:
    """_summarize_communities_batch — batch LLM summarization."""

    async def test_batch_summarizes_multiple_communities(self):
        """Single LLM call summarizes all communities, returns parsed summaries."""
        mock_anthropic = AsyncMock()
        batch_json = '{"1": "People group summary", "2": "Places group summary"}'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=batch_json)]
        mock_anthropic.messages.create.return_value = mock_response

        with patch("prot.community.AsyncAnthropic", return_value=mock_anthropic):
            detector = CommunityDetector(
                store=AsyncMock(), embedder=AsyncMock(), anthropic_key="test"
            )
            groups = [
                [{"name": "A", "entity_type": "person", "description": "desc A"},
                 {"name": "B", "entity_type": "person", "description": "desc B"}],
                [{"name": "C", "entity_type": "place", "description": "desc C"},
                 {"name": "D", "entity_type": "place", "description": "desc D"}],
            ]
            result = await detector._summarize_communities_batch(groups)

        assert result == ["People group summary", "Places group summary"]
        mock_anthropic.messages.create.assert_called_once()

    async def test_batch_falls_back_on_parse_failure(self):
        """Malformed JSON falls back to individual summarization."""
        mock_anthropic = AsyncMock()
        # First call: batch (returns bad JSON)
        # Subsequent calls: individual fallback
        responses = [
            MagicMock(content=[MagicMock(text="not valid json")]),
            MagicMock(content=[MagicMock(text="Summary A")]),
            MagicMock(content=[MagicMock(text="Summary B")]),
        ]
        mock_anthropic.messages.create.side_effect = responses

        with patch("prot.community.AsyncAnthropic", return_value=mock_anthropic):
            detector = CommunityDetector(
                store=AsyncMock(), embedder=AsyncMock(), anthropic_key="test"
            )
            groups = [
                [{"name": "A", "entity_type": "person", "description": "desc A"},
                 {"name": "B", "entity_type": "person", "description": "desc B"}],
                [{"name": "C", "entity_type": "place", "description": "desc C"},
                 {"name": "D", "entity_type": "place", "description": "desc D"}],
            ]
            result = await detector._summarize_communities_batch(groups)

        assert result == ["Summary A", "Summary B"]
        # 1 batch attempt + 2 individual fallbacks
        assert mock_anthropic.messages.create.call_count == 3

    def test_parse_batch_summaries_valid(self):
        """Valid JSON parses correctly."""
        raw = '{"1": "Summary one", "2": "Summary two"}'
        result = CommunityDetector._parse_batch_summaries(raw, 2)
        assert result == ["Summary one", "Summary two"]

    def test_parse_batch_summaries_incomplete(self):
        """Incomplete JSON (missing keys) returns None."""
        raw = '{"1": "Summary one"}'
        result = CommunityDetector._parse_batch_summaries(raw, 2)
        assert result is None

    def test_parse_batch_summaries_invalid_json(self):
        """Non-JSON returns None."""
        result = CommunityDetector._parse_batch_summaries("not json", 2)
        assert result is None

    def test_parse_batch_summaries_strips_markdown_fencing(self):
        """Markdown code fencing is stripped before parsing."""
        raw = '```json\n{"1": "A", "2": "B"}\n```'
        result = CommunityDetector._parse_batch_summaries(raw, 2)
        assert result == ["A", "B"]
