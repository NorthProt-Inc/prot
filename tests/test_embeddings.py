"""Tests for AsyncVoyageEmbedder — Voyage AI voyage-4-large embeddings."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prot.embeddings import AsyncVoyageEmbedder


@pytest.mark.asyncio
class TestAsyncVoyageEmbedder:
    async def test_embed_query(self):
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024]
        mock_client.embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vector = await embedder.embed_query("search term")
            assert len(vector) == 1024
            mock_client.embed.assert_called_once_with(
                texts=["search term"],
                model="voyage-4-large",
                input_type="query",
            )

    async def test_embed_texts(self):
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        mock_client.embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vectors = await embedder.embed_texts(["desc A", "desc B", "desc C"])
            assert len(vectors) == 3
            mock_client.embed.assert_called_once_with(
                texts=["desc A", "desc B", "desc C"],
                model="voyage-4-large",
                input_type="document",
            )

    async def test_close_without_close_method(self):
        mock_client = MagicMock(spec=[])
        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            await embedder.close()  # should not raise
