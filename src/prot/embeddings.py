import voyageai

from prot.config import settings


class AsyncVoyageEmbedder:
    """Async embedding client using Voyage AI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self._client = voyageai.AsyncClient(
            api_key=api_key or settings.voyage_api_key,
        )
        self._model = model or settings.voyage_model

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if hasattr(self._client, "close"):
            await self._client.close()
        elif hasattr(self._client, "aclose"):
            await self._client.aclose()

    async def embed_query_contextual(self, text: str) -> list[float]:
        """Embed single query using voyage-context-3 (input_type='query')."""
        result = await self._client.contextualized_embed(
            inputs=[[text]],
            model=settings.voyage_context_model,
            input_type="query",
        )
        return result.results[0].embeddings[0]

    async def embed_chunks_contextual(self, chunks: list[str]) -> list[list[float]]:
        """Embed chunks using voyage-context-3. All chunks treated as one document's segments."""
        result = await self._client.contextualized_embed(
            inputs=[chunks],  # single document, multiple chunks
            model=settings.voyage_context_model,
            input_type="document",
        )
        return result.results[0].embeddings

    async def embed_texts_contextual(self, texts: list[str]) -> list[list[float]]:
        """Embed independent texts using voyage-context-3. Each text is its own document."""
        result = await self._client.contextualized_embed(
            inputs=[[text] for text in texts],
            model=settings.voyage_context_model,
            input_type="document",
        )
        return [r.embeddings[0] for r in result.results]
