from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.core.exceptions import ProviderUnavailableError
from app.services.base import ProviderHealth
from app.services.retrieval.reranker.base import RerankerProvider, RerankResult


class LocalCrossEncoderReranker(RerankerProvider):
    """Real ONNX cross-encoder reranker served by fastembed.

    The default model is ``BAAI/bge-reranker-v2-m3-int8`` (0.57 GB). The model
    is lazily downloaded and loaded on the first rerank call, in a worker
    thread, so application startup never blocks on a multi-hundred-megabyte
    download.
    """

    name = "local"

    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path | None = None,
        max_length: int | None = 512,
        batch_size: int = 8,
        threads: int | None = None,
    ) -> None:
        super().__init__(model=model)
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.batch_size = max(1, batch_size)
        self.threads = threads
        self._encoder: Any = None
        self._load_error: str | None = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()

    @staticmethod
    def supported_models() -> list[str]:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return [item["model"] for item in TextCrossEncoder.list_supported_models()]

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        if not documents:
            return RerankResult(scores=[], provider=self.name, model=self.model)
        if not query.strip():
            raise ProviderUnavailableError("Cannot rerank with an empty query")

        started = time.perf_counter()
        async with self._inference_lock:
            encoder = await self._ensure_encoder()
            try:
                scores = await asyncio.to_thread(self._score, encoder, query, list(documents))
            except Exception as exc:  # noqa: BLE001 - translated for the caller
                raise ProviderUnavailableError(
                    "The local reranker failed during inference",
                    details={"provider": self.name, "model": self.model, "reason": str(exc)},
                ) from exc

        if len(scores) != len(documents):
            raise ProviderUnavailableError(
                "The local reranker returned an unexpected number of scores",
                details={
                    "provider": self.name,
                    "model": self.model,
                    "expected": len(documents),
                    "received": len(scores),
                },
            )
        return RerankResult(
            scores=scores,
            provider=self.name,
            model=self.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def _ensure_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        async with self._load_lock:
            if self._encoder is not None:
                return self._encoder
            if self._load_error is not None:
                raise ProviderUnavailableError(
                    "The local reranker model could not be loaded earlier",
                    details={"provider": self.name, "model": self.model, "reason": self._load_error},
                )
            try:
                self._encoder = await asyncio.to_thread(self._build_encoder)
            except Exception as exc:  # noqa: BLE001
                self._load_error = str(exc)
                raise ProviderUnavailableError(
                    "The local reranker model could not be loaded or downloaded",
                    details={"provider": self.name, "model": self.model, "reason": str(exc)},
                ) from exc
            return self._encoder

    def _build_encoder(self) -> Any:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        supported = self.supported_models()
        if self.model not in supported:
            raise ValueError(
                f"Unsupported reranker model '{self.model}'. Supported models: {supported}"
            )

        kwargs: dict[str, Any] = {
            "model_name": self.model,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "threads": self.threads,
        }
        encoder = TextCrossEncoder(**kwargs)
        if self.max_length and getattr(encoder, "tokenizer", None) is not None:
            try:
                encoder.tokenizer.enable_truncation(max_length=self.max_length)
            except Exception:  # noqa: BLE001 - tokenizers API differences are non-fatal
                pass
        return encoder

    def _score(self, encoder: Any, query: str, documents: list[str]) -> list[float]:
        return [
            float(score)
            for score in encoder.rerank(query, documents, batch_size=self.batch_size)
        ]

    async def health(self) -> ProviderHealth:
        if self._load_error is not None:
            return ProviderHealth(
                ok=False,
                provider=self.name,
                model=self.model,
                detail=f"Model failed to load: {self._load_error}",
            )
        if self._encoder is None:
            return ProviderHealth(
                ok=True,
                provider=self.name,
                model=self.model,
                detail="ONNX reranker configured; model loads on first use (lazy)",
            )
        return ProviderHealth(
            ok=True,
            provider=self.name,
            model=self.model,
            detail="ONNX reranker loaded and ready",
        )

    async def close(self) -> None:
        self._encoder = None
