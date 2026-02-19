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
        mock_embedder.embed_chunks_contextual.return_value = [[0.1] * 1024]

        mock_anthropic = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="A summary")]
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
