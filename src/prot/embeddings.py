import asyncio

import voyageai

from prot.config import settings


class AsyncVoyageEmbedder:
    """Async embedding client using Voyage AI."""

    MAX_BATCH = 128

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_concurrent: int = 5,
    ):
        self._client = voyageai.AsyncClient(
            api_key=api_key or settings.voyage_api_key,
        )
        self._model = model or settings.voyage_model
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (input_type='document'). Auto-batches over MAX_BATCH."""
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.MAX_BATCH):
            batch = texts[i : i + self.MAX_BATCH]
            async with self._semaphore:
                result = await self._client.embed(
                    texts=batch,
                    model=self._model,
                    input_type="document",
                )
            all_vectors.extend(result.embeddings)
        return all_vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed single query text (input_type='query')."""
        async with self._semaphore:
            result = await self._client.embed(
                texts=[text],
                model=self._model,
                input_type="query",
            )
        return result.embeddings[0]

    async def embed_query_contextual(self, text: str) -> list[float]:
        """Embed single query using voyage-context-3 (input_type='query')."""
        async with self._semaphore:
            result = await self._client.contextualized_embed(
                inputs=[[text]],
                model=settings.voyage_context_model,
                input_type="query",
            )
        return result.results[0].embeddings[0]

    async def embed_chunks_contextual(self, chunks: list[str]) -> list[list[float]]:
        """Embed chunks using voyage-context-3. All chunks treated as one document's segments."""
        async with self._semaphore:
            result = await self._client.contextualized_embed(
                inputs=[chunks],  # single document, multiple chunks
                model=settings.voyage_context_model,
                input_type="document",
            )
        return result.results[0].embeddings
