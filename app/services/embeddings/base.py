from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.core.exceptions import ProviderUnavailableError
from app.services.base import ProviderHealth


class EmbeddingProvider(ABC):
    """Interface implemented by every embedding backend."""

    name: str = "base"

    def __init__(self, model: str, dim: int) -> None:
        self.model = model
        self.dim = dim

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]

    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError

    async def close(self) -> None:
        return None

    def _validate(self, vectors: Sequence[Sequence[float]], expected: int) -> list[list[float]]:
        if len(vectors) != expected:
            raise ProviderUnavailableError(
                "Embedding provider returned the wrong number of vectors",
                details={"expected": expected, "received": len(vectors), "model": self.model},
            )

        cleaned: list[list[float]] = []
        for vector in vectors:
            if len(vector) != self.dim:
                raise ProviderUnavailableError(
                    "Embedding dimension mismatch. Check EMBEDDING_DIM against the model.",
                    details={
                        "expected_dim": self.dim,
                        "received_dim": len(vector),
                        "model": self.model,
                    },
                )
            values = [float(x) for x in vector]
            if not all(math.isfinite(x) for x in values):
                raise ProviderUnavailableError(
                    "Embedding provider returned non-finite values", details={"model": self.model}
                )
            cleaned.append(values)
        return cleaned
