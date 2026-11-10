"""In-memory BM25 index with explicit, thread-safe lifecycle.

The index is a mirror of the ``chunks`` table restricted to indexed documents.
It is rebuilt from the database at startup and kept in sync by the ingestion
service (upsert after indexing, remove after deletion). If a mutation happens
while a search is running the index is rebuilt lazily under a lock; it never
serves stale results silently.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from datetime import UTC, datetime

from rank_bm25 import BM25Okapi

from app.services.base import ProviderHealth
from app.services.retrieval.records import IndexedChunk, to_publication_ts
from app.services.retrieval.tokenize import Tokenizer
from app.services.retrieval.types import RetrievalFilters


class BM25Index:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75, stemming: bool = True) -> None:
        self.k1 = k1
        self.b = b
        self._tokenizer = Tokenizer(stemming=stemming)
        self._lock = threading.RLock()
        self._records: dict[str, IndexedChunk] = {}
        self._tokens: dict[str, list[str]] = {}
        self._order: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._dirty = True
        self._built_at: datetime | None = None
        self._build_ms: float = 0.0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def ready(self) -> bool:
        with self._lock:
            return not self._dirty

    @property
    def last_built_at(self) -> datetime | None:
        with self._lock:
            return self._built_at

    @property
    def build_ms(self) -> float:
        with self._lock:
            return self._build_ms

    def rebuild(self, records: Iterable[IndexedChunk]) -> int:
        """Replace the whole index. Used at startup and after bulk changes."""

        with self._lock:
            self._records = {record.chunk_id: record for record in records}
            self._tokens = {
                chunk_id: self._tokenizer.tokenize(record.text)
                for chunk_id, record in self._records.items()
            }
            self._dirty = True
            self._ensure_built_locked()
            return len(self._records)

    def upsert_document(self, records: Iterable[IndexedChunk]) -> int:
        """Replace all chunks belonging to the documents in ``records``."""

        incoming = list(records)
        with self._lock:
            document_ids = {record.document_id for record in incoming}
            if document_ids:
                self._remove_documents_locked(document_ids)
            for record in incoming:
                self._records[record.chunk_id] = record
                self._tokens[record.chunk_id] = self._tokenizer.tokenize(record.text)
            self._dirty = True
            return len(self._records)

    def remove_document(self, document_id: str) -> int:
        with self._lock:
            removed = self._remove_documents_locked({document_id})
            if removed:
                self._dirty = True
            return len(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._tokens.clear()
            self._order = []
            self._bm25 = None
            self._dirty = True

    def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: RetrievalFilters | None = None,
        allowed_document_ids: set[str] | None = None,
    ) -> list[tuple[IndexedChunk, float]]:
        """Return ``(chunk, bm25_score)`` ordered best-first.

        ``allowed_document_ids`` restricts results to a caller-provided set
        (used for latest-version filtering). An empty set means "no
        documents", which is different from ``None`` ("no restriction").
        """

        if top_k <= 0:
            return []

        query_tokens = self._tokenizer.tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            self._ensure_built_locked()
            if self._bm25 is None or not self._order:
                return []

            scores = self._bm25.get_scores(query_tokens)
            filters = filters or RetrievalFilters()

            ranked: list[tuple[IndexedChunk, float]] = []
            for chunk_id, score in zip(self._order, scores, strict=True):
                numeric = float(score)
                if numeric <= 0.0:
                    continue
                record = self._records[chunk_id]
                if not self._matches(record, filters, allowed_document_ids):
                    continue
                ranked.append((record, numeric))

        ranked.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return ranked[:top_k]

    def health(self) -> ProviderHealth:
        started = time.perf_counter()
        with self._lock:
            state = "ready" if not self._dirty else "rebuild pending"
            detail = f"{len(self._records)} chunks indexed ({state})"
            if self._built_at is not None:
                detail += f", last build {self._built_at.isoformat(timespec='seconds')}"
            return ProviderHealth(
                ok=True,
                provider="bm25",
                model=f"k1={self.k1},b={self.b}",
                detail=detail,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

    def _remove_documents_locked(self, document_ids: set[str]) -> int:
        removed = [
            chunk_id
            for chunk_id, record in self._records.items()
            if record.document_id in document_ids
        ]
        for chunk_id in removed:
            self._records.pop(chunk_id, None)
            self._tokens.pop(chunk_id, None)
        if removed:
            self._dirty = True
        return len(removed)

    def _ensure_built_locked(self) -> None:
        if not self._dirty:
            return

        started = time.perf_counter()
        self._order = list(self._records.keys())
        corpus = [self._tokens.get(chunk_id, []) for chunk_id in self._order]
        if corpus and any(corpus):
            self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        else:
            self._bm25 = None
        self._dirty = False
        self._built_at = datetime.now(UTC)
        self._build_ms = round((time.perf_counter() - started) * 1000, 2)

    @staticmethod
    def _matches(
        record: IndexedChunk,
        filters: RetrievalFilters,
        allowed_document_ids: set[str] | None,
    ) -> bool:
        if allowed_document_ids is not None and record.document_id not in allowed_document_ids:
            return False
        if filters.source and record.source != filters.source:
            return False
        if filters.category and record.category != filters.category:
            return False
        if filters.version and record.version != filters.version:
            return False
        if filters.document_ids and record.document_id not in filters.document_ids:
            return False

        if filters.published_after or filters.published_before:
            if record.publication_ts is None:
                return False
            if filters.published_after:
                after_ts = to_publication_ts(filters.published_after)
                if after_ts is not None and record.publication_ts < after_ts:
                    return False
            if filters.published_before:
                before_ts = to_publication_ts(filters.published_before)
                if before_ts is not None and record.publication_ts > before_ts:
                    return False
        return True
