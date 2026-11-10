"""Reranker provider contract.

A reranker is an optional precision stage that runs after rank fusion. It is
never allowed to *pretend* to run: when it is disabled or fails, the caller
keeps the fused order and records ``reranker_used=false`` plus a reason.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from app.services.base import ProviderHealth


@dataclass(slots=True)
class RerankResult:
    scores: list[float]
    provider: str
    model: str
    latency_ms: float = 0.0


class RerankerProvider(ABC):
    name: str = "base"
    enabled: bool = True

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError

    async def close(self) -> None:
        return None
