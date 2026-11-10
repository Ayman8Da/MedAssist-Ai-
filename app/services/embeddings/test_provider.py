from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from app.services.base import ProviderHealth
from app.services.embeddings.base import EmbeddingProvider

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicTestEmbeddingProvider(EmbeddingProvider):
    """Deterministic lexical hash embedding used ONLY in automated tests.

    It is fast and offline, but it is not a semantic model. The production
    default is Ollama BGE-M3. This provider exists so CI can exercise the full
    ingestion/retrieval path without downloading a model.
    """

    name = "test"

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = [self._embed_one(text) for text in texts]
        return self._validate(vectors, len(texts))

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.dim
            vector[index] += 1.0
            digest2 = hashlib.sha256(b"bigram:" + token.encode("utf-8")).digest()
            index2 = int.from_bytes(digest2[:8], "big") % self.dim
            vector[index2] += 0.5
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    async def health(self) -> ProviderHealth:
        return ProviderHealth(
            ok=True,
            provider=self.name,
            model=self.model,
            detail="Deterministic test embeddings (not for production)",
        )
