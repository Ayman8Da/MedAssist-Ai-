"""Typed records shared by the BM25 index and the ingestion pipeline.

A single :class:`IndexedChunk` representation is used everywhere so the
keyword index cannot drift from what was written to Qdrant/SQL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any

from app.utils.text import normalize_whitespace, sha256_text


def to_publication_ts(value: date | None) -> int | None:
    if value is None:
        return None
    return int(datetime.combine(value, time.min, tzinfo=UTC).timestamp())


@dataclass(slots=True, frozen=True)
class IndexedChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int | None = None
    page_number: int | None = None
    section: str | None = None
    title: str | None = None
    source: str | None = None
    category: str | None = None
    version: str | None = None
    publication_date: str | None = None
    publication_ts: int | None = None
    document_key: str | None = None

    @classmethod
    def from_chunk(cls, document: Any, chunk: Any) -> IndexedChunk:
        """Build from ORM rows without leaking SQLAlchemy types into retrieval."""

        published = document.publication_date
        return cls(
            chunk_id=chunk.id,
            document_id=document.id,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            section=chunk.section,
            title=document.title,
            source=document.source,
            category=document.category,
            version=document.version,
            publication_date=published.isoformat() if published else None,
            publication_ts=to_publication_ts(published),
            document_key=document.document_key,
        )

    @property
    def normalized_text_hash(self) -> str:
        """Hash used for exact-duplicate chunk removal after fusion."""

        return sha256_text(normalize_whitespace(self.text).casefold())

    def payload(self) -> dict[str, Any]:
        """Qdrant-compatible payload (``None`` values omitted)."""

        payload: dict[str, Any] = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "source": self.source,
            "publication_date": self.publication_date,
            "publication_ts": self.publication_ts,
            "version": self.version,
            "category": self.category,
            "document_key": self.document_key,
            "section": self.section,
            "page_number": self.page_number,
            "text": self.text,
        }
        return {key: value for key, value in payload.items() if value is not None}
