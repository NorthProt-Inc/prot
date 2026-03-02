import voyageai

from prot.config import settings
from prot.logging import logged


async def _close_voyage_client(client) -> None:
    """Close a Voyage AI client (duck-typed for SDK version compat)."""
    if hasattr(client, "close"):
        await client.close()
    elif hasattr(client, "aclose"):
        await client.aclose()


class AsyncVoyageEmbedder:
    """Async embedding client using Voyage AI."""

    def __init__(
        self,
        api_key: str | None = None,
    ):
        self._client = voyageai.AsyncClient(
            api_key=api_key or settings.voyage_api_key,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await _close_voyage_client(self._client)

    @logged(slow_ms=1000)
    async def embed_query_contextual(self, text: str) -> list[float]:
        """Embed single query using voyage-context-3 (input_type='query')."""
        result = await self._client.contextualized_embed(
            inputs=[[text]],
            model=settings.voyage_model,
            input_type="query",
        )
        return result.results[0].embeddings[0]

    @logged(slow_ms=2000)
    async def embed_chunks_contextual(self, chunks: list[str]) -> list[list[float]]:
        """Embed chunks using voyage-context-3. All chunks treated as one document's segments."""
        result = await self._client.contextualized_embed(
            inputs=[chunks],  # single document, multiple chunks
            model=settings.voyage_model,
            input_type="document",
        )
        return result.results[0].embeddings

    async def embed_texts_contextual(self, texts: list[str]) -> list[list[float]]:
        """Embed independent texts using voyage-context-3. Each text is its own document."""
        result = await self._client.contextualized_embed(
            inputs=[[text] for text in texts],
            model=settings.voyage_model,
            input_type="document",
        )
        return [r.embeddings[0] for r in result.results]
