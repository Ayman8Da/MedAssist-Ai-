from __future__ import annotations

import time
from collections.abc import Sequence

import httpx

from app.core.exceptions import ProviderUnavailableError
from app.services.base import ProviderHealth
from app.services.embeddings.base import EmbeddingProvider


class OllamaEmbeddingProvider(EmbeddingProvider):
    """BGE-M3 (or any Ollama embedding model) via the local Ollama HTTP API."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dim: int,
        batch_size: int = 16,
        timeout_seconds: float = 120.0,
    ) -> None:
        super().__init__(model=model, dim=dim)
        self.batch_size = max(1, batch_size)
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors.extend(await self._embed_batch(batch))
        return self._validate(vectors, len(texts))

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                "/api/embed", json={"model": self.model, "input": batch}
            )
            if response.status_code == 404:
                # Older Ollama versions only expose /api/embeddings (single input).
                return [await self._embed_legacy(text) for text in batch]
            response.raise_for_status()
            payload = response.json()
            embeddings = payload.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise ProviderUnavailableError(
                    "Ollama returned an unexpected embeddings payload",
                    details={"model": self.model},
                )
            return embeddings
        except ProviderUnavailableError:
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"Ollama embeddings request failed with HTTP {exc.response.status_code}",
                details={"model": self.model, "provider": self.name},
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "Ollama embeddings endpoint is unreachable",
                details={"model": self.model, "provider": self.name},
            ) from exc

    async def _embed_legacy(self, text: str) -> list[float]:
        response = await self._client.post(
            "/api/embeddings", json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        payload = response.json()
        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise ProviderUnavailableError(
                "Ollama returned an unexpected embedding payload",
                details={"model": self.model},
            )
        return embedding

    async def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            response = await self._client.get("/api/tags")
            response.raise_for_status()
            models = [item.get("name", "") for item in response.json().get("models", [])]
            base_model = self.model.split(":")[0]
            found = any(name == self.model or name.split(":")[0] == base_model for name in models)
            detail = None
            if not found:
                detail = f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
            return ProviderHealth(
                ok=found,
                provider=self.name,
                model=self.model,
                detail=detail,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except httpx.HTTPError:
            return ProviderHealth(
                ok=False,
                provider=self.name,
                model=self.model,
                detail="Ollama is unreachable. Is `ollama serve` running?",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

    async def close(self) -> None:
        await self._client.aclose()
