from __future__ import annotations

from app.core.config import Settings
from app.core.exceptions import ValidationAppError
from app.services.retrieval.reranker.base import RerankerProvider
from app.services.retrieval.reranker.local import LocalCrossEncoderReranker
from app.services.retrieval.reranker.none import NoOpReranker
from app.services.retrieval.reranker.remote import RemoteReranker


def build_reranker(settings: Settings) -> RerankerProvider:
    provider = settings.reranker_provider

    if provider == "none":
        return NoOpReranker()

    if provider == "local":
        return LocalCrossEncoderReranker(
            model=settings.reranker_model,
            cache_dir=settings.reranker_cache_dir,
            max_length=settings.reranker_max_length,
            batch_size=settings.reranker_batch_size,
        )

    if provider == "remote":
        if not settings.reranker_base_url:
            raise ValidationAppError(
                "RERANKER_BASE_URL is required when RERANKER_PROVIDER=remote",
                details={"provider": provider},
            )
        return RemoteReranker(
            base_url=settings.reranker_base_url,
            model=settings.reranker_remote_model or settings.reranker_model,
            api_key=settings.reranker_api_key,
            timeout_seconds=settings.reranker_timeout_seconds,
        )

    raise ValidationAppError(
        f"Unsupported reranker provider '{provider}'",
        details={"supported": ["none", "local", "remote"]},
    )
