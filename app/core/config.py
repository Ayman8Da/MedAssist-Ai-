from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application configuration loaded from environment variables / .env.

    All AI providers are configurable. No credentials are hardcoded.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "MedAssist AI"
    app_version: str = "0.1.0"
    app_env: Literal["development", "test", "production"] = "development"
    api_prefix: str = "/api"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"]
    )

    # Storage
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    upload_dir: Path | None = None
    max_upload_mb: int = 25

    # Application database
    database_url: str | None = None

    # Vector database
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: Path | None = None
    qdrant_collection: str = "medassist_chunks"

    # Embeddings
    embedding_provider: Literal["ollama", "openai", "test"] = "ollama"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 16
    embedding_timeout_seconds: float = 120.0

    # Providers
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"

    # LLM
    llm_provider: Literal["ollama", "openai"] = "ollama"
    llm_model: str = "phi3:latest"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 180.0

    # Retrieval / chunking
    top_k: int = 6
    score_threshold: float | None = None
    max_context_chars: int = 12000
    retrieval_timeout_seconds: float = 30.0
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Hybrid retrieval (Phase 2)
    bm25_enabled: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_stemming: bool = True
    rrf_k: int = 60
    dense_candidate_k: int = 30
    keyword_candidate_k: int = 30
    rerank_candidates: int = 20
    latest_versions_only: bool = False

    # Reranker (Phase 2). "none" = explicitly disabled, never simulated.
    reranker_provider: Literal["none", "local", "remote"] = "local"
    reranker_model: str = "BAAI/bge-reranker-v2-m3-int8"
    reranker_max_length: int = 512
    reranker_batch_size: int = 8
    reranker_cache_dir: Path | None = None
    reranker_timeout_seconds: float = 120.0
    reranker_base_url: str | None = None
    reranker_remote_model: str | None = None
    reranker_api_key: str | None = None

    @field_validator(
        "database_url",
        "qdrant_url",
        "qdrant_api_key",
        "openai_api_key",
        "openai_base_url",
        "upload_dir",
        "qdrant_path",
        "score_threshold",
        "reranker_cache_dir",
        "reranker_base_url",
        "reranker_remote_model",
        "reranker_api_key",
        mode="before",
    )
    @classmethod
    def _blank_strings_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_must_be_smaller_than_chunk(cls, value: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 900)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = (self.data_dir / "medassist.db").as_posix()
        return f"sqlite+aiosqlite:///{db_path}"

    @property
    def uploads_dir(self) -> Path:
        return self.upload_dir or (self.data_dir / "uploads")

    @property
    def resolved_qdrant_path(self) -> Path:
        return self.qdrant_path or (self.data_dir / "qdrant")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        if not self.qdrant_url:
            self.resolved_qdrant_path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
