from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    source: str
    publication_date: date | None = None
    version: str | None = None
    category: str | None = None
    document_key: str | None = None
    filename: str
    content_type: str
    status: str
    error_message: str | None = None
    chunk_count: int
    created_at: datetime
    updated_at: datetime
    indexed_at: datetime | None = None


class DocumentListOut(BaseModel):
    items: list[DocumentOut]
    total: int
    limit: int
    offset: int


class UploadResponse(BaseModel):
    document: DocumentOut
    duplicate: bool
    message: str


class ReindexRequest(BaseModel):
    document_ids: list[str] | None = Field(
        default=None,
        description="Specific documents to reindex. Omit to reindex all indexed/failed documents.",
    )


class ReindexItem(BaseModel):
    document_id: str
    status: str
    chunk_count: int
    message: str


class ReindexResponse(BaseModel):
    items: list[ReindexItem]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
