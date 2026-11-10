from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from datetime import time as dt_time
from pathlib import Path

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import AppError, IngestionError, NotFoundError
from app.db.models import Chunk, Document, utcnow
from app.db.repositories import DocumentRepository
from app.services.embeddings.base import EmbeddingProvider
from app.services.ingestion.chunker import chunk_pages
from app.services.ingestion.extractors import (
    TextPage,
    clean_page,
    extract_pages,
    validate_upload,
)
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.records import IndexedChunk
from app.services.vectorstore.qdrant_store import QdrantVectorStore, VectorRecord
from app.utils.ids import new_chunk_id, new_document_id
from app.utils.text import sha256_bytes, truncate

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentMeta:
    title: str
    source: str
    publication_date: date | None = None
    version: str | None = None
    category: str | None = None
    document_key: str | None = None


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, AppError):
        return exc.message
    return truncate(f"{type(exc).__name__}: {exc}", 400)


class IngestionService:
    def __init__(
        self,
        *,
        settings: Settings,
        embeddings: EmbeddingProvider,
        vector_store: QdrantVectorStore,
        bm25: BM25Index | None = None,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.bm25 = bm25

    async def ingest_bytes(
        self,
        session: AsyncSession,
        *,
        data: bytes,
        filename: str,
        content_type: str | None,
        meta: DocumentMeta,
    ) -> tuple[Document, bool]:
        """Validate, store and index an uploaded document.

        Returns ``(document, duplicate)``. Duplicate detection uses the SHA-256
        of the raw bytes, so re-uploading identical content is idempotent.
        """

        suffix = validate_upload(
            filename=filename,
            content_type=content_type,
            size_bytes=len(data),
            max_upload_mb=self.settings.max_upload_mb,
        )
        content_hash = sha256_bytes(data)
        repo = DocumentRepository(session)

        existing = await repo.find_by_hash(content_hash)
        if existing is not None:
            return existing, True

        document_id = new_document_id()
        stored_path = self.settings.uploads_dir / f"{document_id}{suffix}"
        stored_path.write_bytes(data)

        document = Document(
            id=document_id,
            title=(meta.title or Path(filename).stem).strip() or "Untitled medical document",
            source=(meta.source or "unknown").strip() or "unknown",
            publication_date=meta.publication_date,
            version=meta.version,
            category=meta.category,
            document_key=meta.document_key,
            filename=filename,
            content_type=(content_type or "application/octet-stream").split(";")[0].strip(),
            file_path=str(stored_path),
            content_hash=content_hash,
            status="processing",
        )
        session.add(document)
        try:
            await session.commit()
        except IntegrityError:
            # Concurrent upload of identical content: return the winner.
            await session.rollback()
            stored_path.unlink(missing_ok=True)
            existing = await repo.find_by_hash(content_hash)
            if existing is not None:
                return existing, True
            raise

        await session.refresh(document)
        logger.info("Ingestion started document_id=%s", document.id)

        try:
            await self._process_document(session, document)
        except Exception as exc:
            await session.rollback()
            await self._mark_failed(session, document_id, exc)
            await self._safe_delete_vectors(document_id)
            self._remove_from_keyword_index(document_id)
            logger.warning(
                "Ingestion failed document_id=%s error=%s", document_id, _safe_error(exc)
            )
            if isinstance(exc, AppError):
                raise
            raise IngestionError(
                "Document ingestion failed", details={"reason": _safe_error(exc)}
            ) from exc

        logger.info(
            "Ingestion complete document_id=%s chunks=%s",
            document.id,
            document.chunk_count,
        )
        return document, False

    async def reindex(self, session: AsyncSession, document_id: str) -> Document:
        repo = DocumentRepository(session)
        document = await repo.get(document_id)
        if document is None:
            raise NotFoundError("Document not found", details={"document_id": document_id})

        stored_path = Path(document.file_path)
        if not stored_path.exists():
            raise IngestionError(
                "The stored file is missing; re-upload the document before reindexing",
                details={"document_id": document_id},
            )

        await asyncio.to_thread(self.vector_store.delete_document, document_id)
        await repo.delete_chunks(document_id)

        document.status = "processing"
        document.error_message = None
        document.chunk_count = 0
        document.indexed_at = None
        await session.commit()

        logger.info("Reindex started document_id=%s", document_id)
        try:
            await self._process_document(session, document)
        except Exception as exc:
            await session.rollback()
            await self._mark_failed(session, document_id, exc)
            self._remove_from_keyword_index(document_id)
            logger.warning(
                "Reindex failed document_id=%s error=%s", document_id, _safe_error(exc)
            )
            document = await repo.get(document_id)
        return document

    async def delete(self, session: AsyncSession, document_id: str) -> None:
        repo = DocumentRepository(session)
        document = await repo.get(document_id)
        if document is None:
            raise NotFoundError("Document not found", details={"document_id": document_id})

        stored_path = Path(document.file_path)
        await asyncio.to_thread(self.vector_store.delete_document, document_id)
        await repo.delete_chunks(document_id)
        await session.delete(document)
        await session.commit()
        self._remove_from_keyword_index(document_id)

        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not remove stored file document_id=%s", document_id)

    async def _process_document(self, session: AsyncSession, document: Document) -> None:
        stored_path = Path(document.file_path)
        if not stored_path.exists():
            raise IngestionError("The stored file is missing")

        pages = await asyncio.to_thread(extract_pages, stored_path, stored_path.suffix.lower())
        cleaned_pages: list[TextPage] = []
        for page in pages:
            text = clean_page(page.text)
            if text:
                cleaned_pages.append(TextPage(text=text, page_number=page.page_number))
        if not cleaned_pages:
            raise IngestionError("No usable text remained after cleaning")

        drafts = chunk_pages(
            cleaned_pages,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not drafts:
            raise IngestionError("The document produced no chunks")

        vectors = await self.embeddings.embed_documents([draft.text for draft in drafts])

        publication_ts: int | None = None
        if document.publication_date:
            publication_ts = int(
                datetime.combine(
                    document.publication_date, dt_time.min, tzinfo=UTC
                ).timestamp()
            )

        records: list[VectorRecord] = []
        rows: list[Chunk] = []
        for draft, vector in zip(drafts, vectors, strict=True):
            chunk_id = new_chunk_id()
            payload = {
                "document_id": document.id,
                "chunk_id": chunk_id,
                "chunk_index": draft.index,
                "title": document.title,
                "source": document.source,
                "publication_date": document.publication_date.isoformat()
                if document.publication_date
                else None,
                "publication_ts": publication_ts,
                "version": document.version,
                "category": document.category,
                "document_key": document.document_key,
                "section": draft.section,
                "page_number": draft.page_number,
                "text": draft.text,
            }
            payload = {key: value for key, value in payload.items() if value is not None}
            records.append(VectorRecord(id=chunk_id, vector=vector, payload=payload))
            rows.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    chunk_index=draft.index,
                    text=draft.text,
                    page_number=draft.page_number,
                    section=draft.section,
                    char_count=draft.char_count,
                )
            )

        await asyncio.to_thread(self.vector_store.upsert_records, records)
        repo = DocumentRepository(session)
        await repo.add_chunks(rows)

        document.status = "indexed"
        document.chunk_count = len(rows)
        document.indexed_at = utcnow()
        document.error_message = None
        await session.commit()
        self._update_keyword_index(document, rows)

    async def _mark_failed(
        self, session: AsyncSession, document_id: str, exc: Exception
    ) -> None:
        repo = DocumentRepository(session)
        document = await repo.get(document_id)
        if document is None:
            return
        document.status = "failed"
        document.error_message = _safe_error(exc)
        document.chunk_count = 0
        document.indexed_at = None
        try:
            await session.commit()
        except Exception:  # noqa: BLE001
            await session.rollback()
            logger.exception("Could not persist ingestion failure document_id=%s", document_id)

    async def _safe_delete_vectors(self, document_id: str) -> None:
        try:
            await asyncio.to_thread(self.vector_store.delete_document, document_id)
        except Exception:  # noqa: BLE001
            logger.warning("Could not clean up vectors document_id=%s", document_id)

    def _update_keyword_index(self, document: Document, rows: list[Chunk]) -> None:
        if self.bm25 is None:
            return
        try:
            self.bm25.upsert_document(
                [IndexedChunk.from_chunk(document, chunk) for chunk in rows]
            )
        except Exception:  # noqa: BLE001 - the index rebuilds from the DB on next start
            logger.exception("Could not update the BM25 index document_id=%s", document.id)

    def _remove_from_keyword_index(self, document_id: str) -> None:
        if self.bm25 is None:
            return
        try:
            self.bm25.remove_document(document_id)
        except Exception:  # noqa: BLE001
            logger.exception("Could not remove document from the BM25 index document_id=%s", document_id)
