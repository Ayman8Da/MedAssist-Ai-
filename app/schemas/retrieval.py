from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatFilters(BaseModel):
    source: str | None = None
    category: str | None = None
    version: str | None = None
    document_ids: list[str] | None = None
    published_after: date | None = None
    published_before: date | None = None


class RetrievalStageOut(BaseModel):
    name: str
    latency_ms: float
    candidates: int | None = None


class RetrievalDiagnostics(BaseModel):
    """Honest, per-stage retrieval report.

    ``reranker_used`` is false whenever the reranker was disabled, timed out or
    failed; ``reranker_skipped_reason`` then explains why. The answer is still
    returned in fused order, never relabelled as reranked.
    """

    strategy: Literal["traditional_dense", "hybrid"]
    bm25_enabled: bool = False
    dense_candidates: int = 0
    keyword_candidates: int = 0
    fused_candidates: int = 0
    duplicates_removed: int = 0
    reranked_candidates: int = 0
    returned: int = 0
    rrf_k: int | None = None
    reranker_provider: str
    reranker_model: str
    reranker_enabled: bool
    reranker_used: bool = False
    reranker_skipped_reason: str | None = None
    latest_versions_only: bool = False
    stale_versions_excluded: int = 0
    allowed_document_ids: int | None = None
    stages: list[RetrievalStageOut] = Field(default_factory=list)


class RetrievalSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: ChatFilters | None = None

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("query must contain at least 3 non-space characters")
        return value


class RetrievalSearchHit(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    title: str
    source: str
    publication_date: date | None = None
    version: str | None = None
    category: str | None = None
    section: str | None = None
    page_number: int | None = None
    text: str
    score: float
    score_kind: str
    dense_score: float | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    keyword_rank: int | None = None
    matched: list[str] = Field(default_factory=list)


class RetrievalSearchResponse(BaseModel):
    query: str
    hits: list[RetrievalSearchHit]
    total_ms: float
    diagnostics: RetrievalDiagnostics
