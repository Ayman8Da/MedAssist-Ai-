from __future__ import annotations

import time
from collections.abc import Sequence

import httpx

from app.core.exceptions import ProviderUnavailableError
from app.services.base import ProviderHealth
from app.services.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embeddings through any OpenAI-compatible /embeddings endpoint."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        dim: int,
        batch_size: int = 16,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(model=model, dim=dim)
        self.batch_size = max(1, batch_size)
        self._api_key = api_key
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._api_key:
            raise ProviderUnavailableError(
                "OPENAI_API_KEY is not configured for the OpenAI embedding provider"
            )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(await self._embed_batch(batch))
        return self._validate(vectors, len(texts))

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/embeddings", json={"model": self.model, "input": batch}
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"Embedding provider request failed with HTTP {exc.response.status_code}",
                details={"model": self.model, "provider": self.name},
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "Embedding provider is unreachable",
                details={"model": self.model, "provider": self.name},
            ) from exc

        payload = response.json()
        items = payload.get("data")
        if not isinstance(items, list) or len(items) != len(batch):
            raise ProviderUnavailableError(
                "Embedding provider returned an unexpected payload",
                details={"model": self.model},
            )
        items = sorted(items, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        if not self._api_key:
            return ProviderHealth(
                ok=False,
                provider=self.name,
                model=self.model,
                detail="OPENAI_API_KEY is not configured",
            )
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
            return ProviderHealth(
                ok=True,
                provider=self.name,
                model=self.model,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except httpx.HTTPError:
            return ProviderHealth(
                ok=False,
                provider=self.name,
                model=self.model,
                detail="Embedding provider health check failed",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

    async def close(self) -> None:
        await self._client.aclose()
