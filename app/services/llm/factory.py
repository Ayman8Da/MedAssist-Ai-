from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import ValidationAppError
from app.services.llm.base import LLMProvider
from app.services.llm.ollama import OllamaLLMProvider
from app.services.llm.openai_provider import OpenAILLMProvider


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaLLMProvider(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "openai":
        return OpenAILLMProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    raise ValidationAppError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
