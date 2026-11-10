from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.retrieval import ChatFilters, RetrievalDiagnostics


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: ChatFilters | None = None

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("question must contain at least 3 non-space characters")
        return value


class Citation(BaseModel):
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
    score: float
    score_kind: str | None = None
    used: bool = False


class RetrievedChunk(Citation):
    text: str


class TimingBreakdown(BaseModel):
    embedding_ms: float
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    request_id: str
    system: Literal["traditional", "hybrid", "agentic"]
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    insufficient_evidence: bool
    timings: TimingBreakdown
    usage: TokenUsage
    retrieval_attempts: int
    model: str | None = None
    warnings: list[str] = Field(default_factory=list)
    diagnostics: RetrievalDiagnostics | None = None


class BenchmarkNotImplementedResponse(BaseModel):
    code: str = "not_implemented"
    message: str
    phase: int = 4
