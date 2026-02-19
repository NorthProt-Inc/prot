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
                model="voyage-4",
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
                model="voyage-4",
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

    async def test_embed_chunks_contextual_calls_contextualized_embed(self):
        mock_client = AsyncMock()
        mock_inner = MagicMock()
        mock_inner.embeddings = [[0.1] * 1024, [0.2] * 1024]
        mock_result = MagicMock()
        mock_result.results = [mock_inner]
        mock_client.contextualized_embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vectors = await embedder.embed_chunks_contextual(["chunk1", "chunk2"])
            assert len(vectors) == 2
            assert len(vectors[0]) == 1024
            mock_client.contextualized_embed.assert_called_once_with(
                inputs=[["chunk1", "chunk2"]],
                model="voyage-context-3",
                input_type="document",
            )

    async def test_embed_chunks_contextual_returns_flat_embeddings(self):
        mock_client = AsyncMock()
        expected_embeddings = [[0.3] * 1024, [0.4] * 1024, [0.5] * 1024]
        mock_inner = MagicMock()
        mock_inner.embeddings = expected_embeddings
        mock_result = MagicMock()
        mock_result.results = [mock_inner]
        mock_client.contextualized_embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vectors = await embedder.embed_chunks_contextual(["a", "b", "c"])
            # Return is a flat list of embeddings, not nested
            assert vectors == expected_embeddings
            assert isinstance(vectors, list)
            assert isinstance(vectors[0], list)

    async def test_embed_query_contextual_uses_contextualized_embed(self):
        mock_client = AsyncMock()
        mock_inner = MagicMock()
        mock_inner.embeddings = [[0.1] * 1024]
        mock_result = MagicMock()
        mock_result.results = [mock_inner]
        mock_client.contextualized_embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vector = await embedder.embed_query_contextual("search term")
            assert len(vector) == 1024
            mock_client.contextualized_embed.assert_called_once_with(
                inputs=[["search term"]],
                model="voyage-context-3",
                input_type="query",
            )

    async def test_embed_texts_contextual_wraps_each_independently(self):
        """Each text = its own document: inputs=[[t1], [t2], [t3]]."""
        mock_client = AsyncMock()
        mock_r0 = MagicMock(); mock_r0.embeddings = [[0.1] * 1024]
        mock_r1 = MagicMock(); mock_r1.embeddings = [[0.2] * 1024]
        mock_r2 = MagicMock(); mock_r2.embeddings = [[0.3] * 1024]
        mock_result = MagicMock(); mock_result.results = [mock_r0, mock_r1, mock_r2]
        mock_client.contextualized_embed.return_value = mock_result

        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            vectors = await embedder.embed_texts_contextual(["desc A", "desc B", "desc C"])
            assert len(vectors) == 3
            assert vectors[1] == [0.2] * 1024
            mock_client.contextualized_embed.assert_called_once_with(
                inputs=[["desc A"], ["desc B"], ["desc C"]],
                model="voyage-context-3",
                input_type="document",
            )

    async def test_close_without_close_method(self):
        """close() handles missing close() gracefully."""
        mock_client = MagicMock(spec=[])  # no close attr
        with patch("prot.embeddings.voyageai.AsyncClient", return_value=mock_client):
            embedder = AsyncVoyageEmbedder(api_key="test")
            await embedder.close()  # should not raise

    async def test_semaphore_limits_concurrency(self):
        embedder = AsyncVoyageEmbedder.__new__(AsyncVoyageEmbedder)
        embedder._max_concurrent = 5
        import asyncio
        embedder._semaphore = asyncio.Semaphore(5)
        assert embedder._semaphore._value == 5
