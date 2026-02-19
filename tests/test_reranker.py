import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from prot.reranker import VoyageReranker


@pytest.mark.asyncio
class TestVoyageReranker:
    async def test_rerank_returns_reordered_items(self):
        mock_client = AsyncMock()
        mock_result = MagicMock()
        # Simulate reranking: original items at index 2, 0, 1 reordered by relevance
        r0 = MagicMock()
        r0.index = 2
        r0.relevance_score = 0.95
        r1 = MagicMock()
        r1.index = 0
        r1.relevance_score = 0.80
        r2 = MagicMock()
        r2.index = 1
        r2.relevance_score = 0.60
        mock_result.results = [r0, r1, r2]
        mock_client.rerank.return_value = mock_result

        items = [
            {"text": "first doc", "id": 1},
            {"text": "second doc", "id": 2},
            {"text": "third doc", "id": 3},
        ]

        with patch("prot.reranker.voyageai.AsyncClient", return_value=mock_client):
            reranker = VoyageReranker(api_key="test")
            result = await reranker.rerank("query", items)

            assert len(result) == 3
            # Reordered: third doc (0.95), first doc (0.80), second doc (0.60)
            assert result[0]["id"] == 3
            assert result[0]["relevance_score"] == 0.95
            assert result[1]["id"] == 1
            assert result[1]["relevance_score"] == 0.80
            assert result[2]["id"] == 2
            assert result[2]["relevance_score"] == 0.60

            mock_client.rerank.assert_called_once_with(
                query="query",
                documents=["first doc", "second doc", "third doc"],
                model="rerank-2.5",
                top_k=None,
            )

    async def test_rerank_empty_items_returns_empty(self):
        mock_client = AsyncMock()

        with patch("prot.reranker.voyageai.AsyncClient", return_value=mock_client):
            reranker = VoyageReranker(api_key="test")
            result = await reranker.rerank("query", [])

            assert result == []
            mock_client.rerank.assert_not_called()

    async def test_rerank_single_item_skips_api(self):
        mock_client = AsyncMock()
        items = [{"text": "only doc", "id": 1}]

        with patch("prot.reranker.voyageai.AsyncClient", return_value=mock_client):
            reranker = VoyageReranker(api_key="test")
            result = await reranker.rerank("query", items)

            assert result == items
            mock_client.rerank.assert_not_called()

    async def test_close_closes_client(self):
        mock_client = AsyncMock()

        with patch("prot.reranker.voyageai.AsyncClient", return_value=mock_client):
            reranker = VoyageReranker(api_key="test")
            await reranker.close()

            mock_client.close.assert_awaited_once()
