from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

from sqlalchemy import delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AgentTrace, Chunk, Document
from app.services.retrieval.records import IndexedChunk


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, document_id: str) -> Document | None:
        return await self.session.get(Document, document_id)

    async def find_by_hash(self, content_hash: str) -> Document | None:
        result = await self.session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
        return result.scalar_one_or_none()

    async def list(
        self,
        *,
        status: str | None = None,
        search: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        stmt = select(Document)
        count_stmt = select(func.count()).select_from(Document)

        if status:
            stmt = stmt.where(Document.status == status)
            count_stmt = count_stmt.where(Document.status == status)
        if search:
            pattern = f"%{search.strip()}%"
            condition = or_(Document.title.ilike(pattern), Document.source.ilike(pattern))
            stmt = stmt.where(condition)
            count_stmt = count_stmt.where(condition)

        stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
        items = (await self.session.execute(stmt)).scalars().all()
        total = (await self.session.execute(count_stmt)).scalar_one()
        return list(items), int(total)

    async def list_ids(self, statuses: Sequence[str]) -> list[str]:
        result = await self.session.execute(
            select(Document.id).where(Document.status.in_(list(statuses)))
        )
        return [row[0] for row in result.all()]

    async def delete_chunks(self, document_id: str) -> None:
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))

    async def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        self.session.add_all(list(chunks))

    async def count_chunks(self, document_id: str) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        return int(result.scalar_one())

    async def list_for_versioning(self) -> list[Document]:
        result = await self.session.execute(
            select(Document).where(Document.document_key.is_not(None))
        )
        return list(result.scalars().all())

    async def latest_version_ids(self) -> set[str]:
        """Return the newest document id for every document_key.

        Grouping is explicit: documents carry a ``document_key`` (for example a
        stable guideline identifier) so versions can be compared. A document
        without a key is never treated as a version of anything.
        """

        return {document.id for document in (await self._latest_by_key()).values()}

    async def stale_version_ids(self) -> set[str]:
        """Return ids of keyed documents superseded by a newer version."""

        documents = await self.list_for_versioning()
        latest_ids = {
            document.id for document in (await self._latest_by_key(documents)).values()
        }
        return {document.id for document in documents if document.id not in latest_ids}

    async def _latest_by_key(
        self, documents: Sequence[Document] | None = None
    ) -> dict[str, Document]:
        if documents is None:
            documents = await self.list_for_versioning()
        newest: dict[str, Document] = {}
        for document in documents:
            key = document.document_key or document.id
            current = newest.get(key)
            if current is None or self._version_sort_key(document) > self._version_sort_key(current):
                newest[key] = document
        return newest

    @staticmethod
    def _version_sort_key(document: Document) -> tuple:
        published = document.publication_date or date.min
        return (published, document.created_at)


class ChunkRepository:
    """Read model used to build the in-memory BM25 index."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list_for_index(self) -> list[IndexedChunk]:
        stmt = (
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.status == "indexed")
        )
        rows = (await self.session.execute(stmt)).all()
        return [
            IndexedChunk.from_chunk(document, chunk)
            for chunk, document in rows
        ]


class TraceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, trace: AgentTrace) -> AgentTrace:
        self.session.add(trace)
        return trace

    async def list(
        self, *, mode: str | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[AgentTrace], int]:
        stmt = select(AgentTrace)
        count_stmt = select(func.count()).select_from(AgentTrace)
        if mode:
            stmt = stmt.where(AgentTrace.mode == mode)
            count_stmt = count_stmt.where(AgentTrace.mode == mode)
        stmt = stmt.order_by(AgentTrace.created_at.desc()).limit(limit).offset(offset)
        items = (await self.session.execute(stmt)).scalars().all()
        total = (await self.session.execute(count_stmt)).scalar_one()
        return list(items), int(total)

    async def count_since(self, since: datetime) -> int:
        result = await self.session.execute(
            select(func.count()).select_from(AgentTrace).where(AgentTrace.created_at >= since)
        )
        return int(result.scalar_one())
