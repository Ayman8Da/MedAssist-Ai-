from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.services.base import ProviderHealth


@dataclass(slots=True)
class LLMResult:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    async def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResult:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError

    async def close(self) -> None:
        return None
