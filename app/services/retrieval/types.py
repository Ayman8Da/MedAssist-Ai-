"""Unified retrieval result types.

Both pipelines (dense/Qdrant and lexical/BM25) are normalised into
:class:`RetrievalHit` so fusion, deduplication, reranking and citation
building never need provider-specific branches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any, Self

from app.services.retrieval.records import IndexedChunk
from app.services.vectorstore.qdrant_store import SearchHit


@dataclass(slots=True, frozen=True)
class RetrievalFilters:
    source: str | None = None
    category: str | None = None
    version: str | None = None
    document_ids: tuple[str, ...] | None = None
    published_after: date | None = None
    published_before: date | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> RetrievalFilters:
        if not value:
            return cls()

        document_ids = value.get("document_ids")
        published_after = value.get("published_after")
        published_before = value.get("published_before")

        return cls(
            source=value.get("source") or None,
            category=value.get("category") or None,
            version=value.get("version") or None,
            document_ids=tuple(document_ids) if document_ids else None,
            published_after=_coerce_date(published_after),
            published_before=_coerce_date(published_before),
        )

    def as_qdrant_mapping(self) -> dict[str, Any] | None:
        payload: dict[str, Any] = {
            "source": self.source,
            "category": self.category,
            "version": self.version,
            "document_ids": list(self.document_ids) if self.document_ids else None,
            "published_after": self.published_after,
            "published_before": self.published_before,
        }
        payload = {key: value for key, value in payload.items() if value is not None}
        return payload or None


def _coerce_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    document_id: str
    text: str
    payload: dict[str, Any]
    score: float
    score_kind: str = "rrf"
    dense_score: float | None = None
    keyword_score: float | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
    dense_rank: int | None = None
    keyword_rank: int | None = None
    final_rank: int | None = None
    matched: set[str] = field(default_factory=set)

    @classmethod
    def from_dense(cls, hit: SearchHit, *, rank: int) -> Self:
        payload = dict(hit.payload)
        return cls(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            text=hit.text,
            payload=payload,
            score=hit.score,
            score_kind="cosine",
            dense_score=hit.score,
            dense_rank=rank,
            matched={"dense"},
        )

    @classmethod
    def from_keyword(cls, record: IndexedChunk, *, score: float, rank: int) -> Self:
        return cls(
            chunk_id=record.chunk_id,
            document_id=record.document_id,
            text=record.text,
            payload=record.payload(),
            score=score,
            score_kind="bm25",
            keyword_score=score,
            keyword_rank=rank,
            matched={"keyword"},
        )

    def with_score(self, score: float, *, kind: str) -> RetrievalHit:
        updated = replace(self, score=score, score_kind=kind)
        updated.matched = set(self.matched)
        return updated

    @property
    def title(self) -> str:
        return str(self.payload.get("title") or "Untitled document")

    @property
    def source(self) -> str:
        return str(self.payload.get("source") or "unknown source")
