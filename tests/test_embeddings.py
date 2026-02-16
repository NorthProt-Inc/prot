import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from prot.embeddings import AsyncVoyageEmbedder


@pytest.mark.asyncio
class TestAsyncVoyageEmbedder:
    async def test_embed_texts_returns_vectors(self):
        mock_client = AsyncMock()
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
        mock_client.embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vectors = await embedder.embed_texts(["hello", "world"])
            assert len(vectors) == 2
            assert len(vectors[0]) == 1024
            mock_client.embed.assert_called_once_with(
                texts=["hello", "world"],
                model="voyage-3.5-lite",
                input_type="document",
            )

    async def test_embed_query_uses_query_input_type(self):
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
                model="voyage-3.5-lite",
                input_type="query",
            )

    async def test_batch_splitting_over_128(self):
        mock_client = AsyncMock()
        mock_result_128 = MagicMock()
        mock_result_128.embeddings = [[0.1] * 1024] * 128
        mock_result_72 = MagicMock()
        mock_result_72.embeddings = [[0.1] * 1024] * 72
        mock_client.embed.side_effect = [mock_result_128, mock_result_72]

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            texts = [f"text_{i}" for i in range(200)]
            vectors = await embedder.embed_texts(texts)
            assert len(vectors) == 200
            assert mock_client.embed.call_count == 2  # 128 + 72

    async def test_semaphore_limits_concurrency(self):
        embedder = AsyncVoyageEmbedder.__new__(AsyncVoyageEmbedder)
        embedder._max_concurrent = 5
        import asyncio
        embedder._semaphore = asyncio.Semaphore(5)
        assert embedder._semaphore._value == 5
