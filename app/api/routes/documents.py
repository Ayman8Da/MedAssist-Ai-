from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Response, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_ingestion, get_session
from app.core.exceptions import AppError, IngestionError, NotFoundError, ValidationAppError
from app.db.repositories import DocumentRepository
from app.schemas.documents import (
    DocumentListOut,
    DocumentOut,
    ReindexItem,
    ReindexRequest,
    ReindexResponse,
    UploadResponse,
)
from app.services.ingestion.pipeline import DocumentMeta, IngestionService

router = APIRouter(tags=["documents"])


@router.post("/documents/upload", response_model=UploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    source: str = Form(..., min_length=1, max_length=256),
    title: str | None = Form(default=None, max_length=512),
    publication_date: str | None = Form(default=None),
    version: str | None = Form(default=None, max_length=64),
    category: str | None = Form(default=None, max_length=128),
    document_key: str | None = Form(default=None, max_length=128),
    session: AsyncSession = Depends(get_session),
    ingestion: IngestionService = Depends(get_ingestion),
) -> UploadResponse:
    parsed_date: date | None = None
    if publication_date and publication_date.strip():
        try:
            parsed_date = date.fromisoformat(publication_date.strip())
        except ValueError as exc:
            raise ValidationAppError(
                "publication_date must use ISO format YYYY-MM-DD"
            ) from exc

    max_bytes = ingestion.settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise IngestionError(
            f"The uploaded file exceeds the {ingestion.settings.max_upload_mb} MB limit"
        )

    meta = DocumentMeta(
        title=title or Path(file.filename or "document").stem,
        source=source,
        publication_date=parsed_date,
        version=version or None,
        category=category or None,
        document_key=document_key or None,
    )
    document, duplicate = await ingestion.ingest_bytes(
        session,
        data=data,
        filename=file.filename or "document.txt",
        content_type=file.content_type,
        meta=meta,
    )
    message = "Document already exists (identical content)." if duplicate else "Document indexed."
    return UploadResponse(
        document=DocumentOut.model_validate(document),
        duplicate=duplicate,
        message=message,
    )


@router.get("/documents", response_model=DocumentListOut)
async def list_documents(
    status: str | None = Query(default=None),
    search: str | None = Query(default=None, max_length=256),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> DocumentListOut:
    items, total = await DocumentRepository(session).list(
        status=status, search=search, limit=limit, offset=offset
    )
    return DocumentListOut(
        items=[DocumentOut.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: str, session: AsyncSession = Depends(get_session)
) -> DocumentOut:
    document = await DocumentRepository(session).get(document_id)
    if document is None:
        raise NotFoundError("Document not found", details={"document_id": document_id})
    return DocumentOut.model_validate(document)


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),
    ingestion: IngestionService = Depends(get_ingestion),
) -> Response:
    await ingestion.delete(session, document_id)
    return Response(status_code=204)


@router.post("/documents/reindex", response_model=ReindexResponse)
async def reindex_documents(
    payload: ReindexRequest,
    session: AsyncSession = Depends(get_session),
    ingestion: IngestionService = Depends(get_ingestion),
) -> ReindexResponse:
    repo = DocumentRepository(session)
    document_ids = payload.document_ids or await repo.list_ids(["indexed", "failed"])

    items: list[ReindexItem] = []
    for document_id in document_ids:
        try:
            document = await ingestion.reindex(session, document_id)
            message = "Reindexed" if document.status == "indexed" else (document.error_message or "Failed")
            items.append(
                ReindexItem(
                    document_id=document_id,
                    status=document.status,
                    chunk_count=document.chunk_count,
                    message=message,
                )
            )
        except AppError as exc:
            items.append(
                ReindexItem(
                    document_id=document_id,
                    status="failed",
                    chunk_count=0,
                    message=exc.message,
                )
            )
    return ReindexResponse(items=items)
