from __future__ import annotations

from collections.abc import Sequence

from app.services.base import ProviderHealth
from app.services.retrieval.reranker.base import RerankerProvider, RerankResult


class NoOpReranker(RerankerProvider):
    """Explicitly disabled reranker.

    Calling :meth:`rerank` raises rather than returning fabricated scores, so a
    misconfiguration can never be reported as a successful rerank.
    """

    name = "none"
    enabled = False

    def __init__(self) -> None:
        super().__init__(model="disabled")

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        raise RuntimeError("The reranker is disabled (RERANKER_PROVIDER=none)")

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            ok=True,
            provider=self.name,
            model=self.model,
            detail="Reranking disabled by configuration; fused order is returned unchanged",
        )
