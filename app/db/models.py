from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import Date, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512))
    source: Mapped[str] = mapped_column(String(256), index=True)
    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    document_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128))
    file_path: Mapped[str] = mapped_column(String(1024))
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section: Mapped[str | None] = mapped_column(String(256), nullable=True)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")


class BenchmarkRun(Base):
    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    completed_questions: Mapped[int] = mapped_column(Integer, default=0)
    results_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class AgentTrace(Base):
    __tablename__ = "agent_traces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    mode: Mapped[str] = mapped_column(String(32), default="traditional", index=True)
    query_sha256: Mapped[str] = mapped_column(String(64))
    query_preview: Mapped[str] = mapped_column(String(240))
    steps_json: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(String(32), default="completed", index=True)
    retrieval_attempts: Mapped[int] = mapped_column(Integer, default=1)
    latency_ms: Mapped[float] = mapped_column(default=0.0)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
