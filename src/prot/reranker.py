import voyageai

from prot.config import settings


class VoyageReranker:
    """Async reranker client using Voyage AI."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self._client = voyageai.AsyncClient(
            api_key=api_key or settings.voyage_api_key,
        )
        self._model = model or settings.rerank_model

    async def rerank(
        self,
        query: str,
        items: list[dict],
        text_key: str = "text",
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank items by relevance to query.

        Returns items reordered by relevance score, each augmented
        with a 'relevance_score' field. Skips API call for 0 or 1 items.
        """
        if len(items) <= 1:
            return items

        documents = [item[text_key] for item in items]
        result = await self._client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_k=top_k,
        )
        return [
            dict(items[r.index]) | {"relevance_score": r.relevance_score}
            for r in result.results
        ]

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()
