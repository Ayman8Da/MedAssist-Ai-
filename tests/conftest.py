from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.base import ProviderHealth
from app.services.llm.base import LLMResult


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        app_env="test",
        data_dir=tmp_path / "data",
        upload_dir=tmp_path / "uploads",
        qdrant_path=tmp_path / "qdrant",
        database_url=f"sqlite+aiosqlite:///{(tmp_path / 'test.db').as_posix()}",
        embedding_provider="test",
        embedding_model="test-embed",
        embedding_dim=64,
        llm_provider="ollama",
        llm_model="test-llm",
        top_k=4,
        chunk_size=400,
        chunk_overlap=60,
        max_context_chars=4000,
        reranker_provider="none",
        log_level="WARNING",
    )


@pytest.fixture
def app(test_settings: Settings):
    return create_app(test_settings)


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


class FakeLLM:
    """Deterministic test double. Never used by the running application."""

    name = "fake"
    model = "fake-llm"

    def __init__(
        self,
        content: str | None = None,
        error: Exception | None = None,
    ) -> None:
        self.content = content or (
            "## Summary\nGrounded clinical answer based on the evidence [1].\n\n"
            "## Sources\n[1] Retrieved evidence.\n"
        )
        self.error = error

    async def generate(self, *, system_prompt: str, user_prompt: str) -> LLMResult:
        if self.error is not None:
            raise self.error
        return LLMResult(
            content=self.content,
            model=self.model,
            prompt_tokens=120,
            completion_tokens=30,
            latency_ms=1.0,
        )

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider="fake", model=self.model)

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_llm(client: TestClient, app) -> FakeLLM:
    llm = FakeLLM()
    app.state.rag.llm = llm
    app.state.hybrid_rag.llm = llm
    app.state.agentic_rag.llm = llm
    return llm
