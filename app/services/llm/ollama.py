from __future__ import annotations

import time

import httpx

from app.core.exceptions import ProviderUnavailableError
from app.services.base import ProviderHealth
from app.services.llm.base import LLMProvider, LLMResult


class OllamaLLMProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_seconds: float = 180.0,
    ) -> None:
        super().__init__(model=model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResult:
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }
        try:
            response = await self._client.post("/api/chat", json=payload)
            if response.status_code == 404:
                raise ProviderUnavailableError(
                    f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}",
                    details={"model": self.model},
                )
            response.raise_for_status()
        except ProviderUnavailableError:
            raise
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"Ollama chat request failed with HTTP {exc.response.status_code}",
                details={"model": self.model, "provider": self.name},
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "Ollama chat endpoint is unreachable",
                details={"model": self.model, "provider": self.name},
            ) from exc

        data = response.json()
        content = (data.get("message") or {}).get("content", "").strip()
        if not content:
            raise ProviderUnavailableError(
                "Ollama returned an empty completion", details={"model": self.model}
            )
        return LLMResult(
            content=content,
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

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
