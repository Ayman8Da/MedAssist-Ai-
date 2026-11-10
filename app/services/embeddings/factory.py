from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import ValidationAppError
from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.ollama import OllamaEmbeddingProvider
from app.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.services.embeddings.test_provider import DeterministicTestEmbeddingProvider


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
    if settings.embedding_provider == "test":
        if settings.app_env != "test":
            raise ValidationAppError(
                "The deterministic test embedding provider is only allowed when APP_ENV=test"
            )
        return DeterministicTestEmbeddingProvider(model=settings.embedding_model, dim=settings.embedding_dim)
    raise ValidationAppError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider}")
