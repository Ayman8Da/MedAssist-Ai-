from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError, ValidationAppError
from app.services.retrieval.reranker.factory import build_reranker
from app.services.retrieval.reranker.local import LocalCrossEncoderReranker
from app.services.retrieval.reranker.none import NoOpReranker
from app.services.retrieval.reranker.remote import RemoteReranker


async def test_none_reranker_never_returns_fake_scores() -> None:
    reranker = NoOpReranker()
    assert reranker.enabled is False
    health = await reranker.health()
    assert health.ok is True
    assert "disabled" in (health.detail or "")

    with pytest.raises(RuntimeError):
        await reranker.rerank("metformin", ["metformin is renally cleared"])


def test_factory_builds_noop_reranker() -> None:
    reranker = build_reranker(Settings(_env_file=None, reranker_provider="none"))
    assert isinstance(reranker, NoOpReranker)
    assert reranker.enabled is False


def test_factory_requires_base_url_for_remote() -> None:
    with pytest.raises(ValidationAppError):
        build_reranker(
            Settings(
                _env_file=None,
                reranker_provider="remote",
                reranker_base_url=None,
            )
        )


def test_local_supported_models_include_bge_reranker() -> None:
    models = LocalCrossEncoderReranker.supported_models()
    assert "BAAI/bge-reranker-v2-m3-int8" in models


async def test_local_reranker_rejects_unknown_model_before_download() -> None:
    reranker = LocalCrossEncoderReranker(model="not/a-real-reranker")
    with pytest.raises(ProviderUnavailableError) as excinfo:
        await reranker._ensure_encoder()
    assert "Unsupported reranker model" in (excinfo.value.details.get("reason") or "")


async def test_remote_reranker_parses_cohere_style_scores() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode()
        assert '"top_n":3' in body
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
        )

    reranker = RemoteReranker(
        base_url="https://reranker.invalid/v1", model="rerank-model"
    )
    reranker._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await reranker.rerank("metformin", ["a", "b", "c"])
    finally:
        await reranker.close()

    assert result.scores[0] == pytest.approx(0.4)
    assert result.scores[2] == pytest.approx(0.9)
    # Documents omitted by the provider are ranked last, not silently dropped.
    assert result.scores[1] < result.scores[0]


async def test_remote_reranker_converts_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    reranker = RemoteReranker(
        base_url="https://reranker.invalid/v1", model="rerank-model"
    )
    reranker._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ProviderUnavailableError):
            await reranker.rerank("metformin", ["a"])
    finally:
        await reranker.close()
