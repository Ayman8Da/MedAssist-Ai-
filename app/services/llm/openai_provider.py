from __future__ import annotations

import time

import httpx

from app.core.exceptions import ProviderUnavailableError
from app.services.base import ProviderHealth
from app.services.llm.base import LLMProvider, LLMResult


class OpenAILLMProvider(LLMProvider):
    """Chat completions through an OpenAI-compatible endpoint."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_seconds: float = 180.0,
    ) -> None:
        super().__init__(model=model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResult:
        if not self._api_key:
            raise ProviderUnavailableError(
                "OPENAI_API_KEY is not configured for the OpenAI LLM provider"
            )
        started = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderUnavailableError(
                f"LLM request failed with HTTP {exc.response.status_code}",
                details={"model": self.model, "provider": self.name},
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "LLM provider is unreachable",
                details={"model": self.model, "provider": self.name},
            ) from exc

        data = response.json()
        choices = data.get("choices") or []
        content = ""
        if choices:
            content = (choices[0].get("message") or {}).get("content", "").strip()
        if not content:
            raise ProviderUnavailableError(
                "LLM provider returned an empty completion", details={"model": self.model}
            )
        usage = data.get("usage") or {}
        return LLMResult(
            content=content,
            model=self.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

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
                detail="LLM provider health check failed",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

    async def close(self) -> None:
        await self._client.aclose()
