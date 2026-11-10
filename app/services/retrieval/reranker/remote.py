from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx

from app.core.exceptions import ProviderUnavailableError
from app.services.base import ProviderHealth
from app.services.retrieval.reranker.base import RerankerProvider, RerankResult


class RemoteReranker(RerankerProvider):
    """Cohere / Jina / OpenAI-compatible ``POST /rerank`` client.

    The endpoint contract is intentionally narrow::

        request  {"model": ..., "query": ..., "documents": [...], "top_n": n}
        response {"results": [{"index": 0, "relevance_score": 0.91}, ...]}

    Both ``relevance_score`` and ``score`` are accepted, matching Cohere v2 and
    Jina respectively. Documents the provider omits from its response receive
    the lowest possible score instead of being dropped silently.
    """

    name = "remote"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(model=model)
        self.base_url = base_url.rstrip("/")
        self._url = self.base_url if self.base_url.endswith("/rerank") else f"{self.base_url}/rerank"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            headers=headers,
        )

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        if not documents:
            return RerankResult(scores=[], provider=self.name, model=self.model)

        started = time.perf_counter()
        payload = {
            "model": self.model,
            "query": query,
            "documents": list(documents),
            "top_n": len(documents),
        }
        try:
            response = await self._client.post(self._url, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"The reranker endpoint returned HTTP {exc.response.status_code}",
                details={"provider": self.name, "model": self.model, "url": self._url},
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderUnavailableError(
                "The reranker endpoint is unreachable or returned invalid JSON",
                details={"provider": self.name, "model": self.model, "url": self._url},
            ) from exc

        scores = self._parse_scores(body, expected=len(documents))
        return RerankResult(
            scores=scores,
            provider=self.name,
            model=self.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def _parse_scores(self, body: Any, *, expected: int) -> list[float]:
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise ProviderUnavailableError(
                "The reranker endpoint response has no 'results' array",
                details={"provider": self.name, "model": self.model},
            )

        unreturned_score = float("-1e30")
        scores = [unreturned_score] * expected
        for item in results:
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if not isinstance(index, int) or not isinstance(score, (int, float)):
                continue
            if not 0 <= index < expected:
                raise ProviderUnavailableError(
                    "The reranker endpoint returned an out-of-range document index",
                    details={"provider": self.name, "model": self.model, "index": index},
                )
            scores[index] = float(score)
        return scores

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            ok=True,
            provider=self.name,
            model=self.model,
            detail=f"Remote reranker configured at {self._url} (not probed to avoid spending quota)",
        )

    async def close(self) -> None:
        await self._client.aclose()
