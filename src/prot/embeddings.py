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
    """Async embedding client using Voyage AI voyage-4-large."""

    def __init__(self, api_key: str | None = None):
        self._client = voyageai.AsyncClient(
            api_key=api_key or settings.voyage_api_key,
        )

    async def close(self) -> None:
        await _close_voyage_client(self._client)

    @logged(slow_ms=1000)
    async def embed_query(self, text: str) -> list[float]:
        """Embed single query text (input_type='query')."""
        result = await self._client.embed(
            texts=[text],
            model=settings.voyage_model,
            input_type="query",
        )
        return result.embeddings[0]

    @logged(slow_ms=2000)
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts independently (input_type='document')."""
        result = await self._client.embed(
            texts=texts,
            model=settings.voyage_model,
            input_type="document",
        )
        return result.embeddings
