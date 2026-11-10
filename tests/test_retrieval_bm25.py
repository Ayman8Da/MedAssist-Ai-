from __future__ import annotations

from datetime import date

from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.records import IndexedChunk, to_publication_ts
from app.services.retrieval.types import RetrievalFilters


def _chunk(
    chunk_id: str,
    document_id: str,
    text: str,
    *,
    source: str = "guideline",
    category: str | None = "nephrology",
    version: str | None = "1.0",
    publication_date: str | None = None,
    document_key: str | None = None,
) -> IndexedChunk:
    published = date.fromisoformat(publication_date) if publication_date else None
    return IndexedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=0,
        title="Synthetic guideline",
        source=source,
        category=category,
        version=version,
        publication_date=publication_date,
        publication_ts=to_publication_ts(published),
        document_key=document_key,
    )


RECORDS = [
    _chunk(
        "c1",
        "d1",
        "Metformin is contraindicated when eGFR is below 30 because of lactic acidosis risk.",
        source="renal-guideline",
        publication_date="2023-01-01",
        document_key="metformin-renal",
    ),
    _chunk(
        "c2",
        "d2",
        "Metformin dose review is required in renal impairment between 30 and 45.",
        source="endocrine-guideline",
        publication_date="2024-01-01",
        document_key="metformin-renal",
    ),
    _chunk(
        "c3",
        "d3",
        "Insulin therapy is first line in diabetic ketoacidosis.",
        source="endocrine-guideline",
        category="endocrinology",
        publication_date="2022-01-01",
    ),
]


def _index() -> BM25Index:
    index = BM25Index(k1=1.5, b=0.75)
    index.rebuild(RECORDS)
    return index


def test_bm25_ranks_most_relevant_chunk_first() -> None:
    results = _index().search("metformin renal impairment", top_k=3)
    assert results
    assert results[0][0].chunk_id in {"c1", "c2"}
    assert results[0][1] > 0


def test_bm25_applies_metadata_filters() -> None:
    results = _index().search(
        "metformin renal",
        top_k=3,
        filters=RetrievalFilters(source="renal-guideline"),
    )
    assert [chunk.chunk_id for chunk, _ in results] == ["c1"]


def test_bm25_applies_publication_date_filter() -> None:
    results = _index().search(
        "metformin renal",
        top_k=3,
        filters=RetrievalFilters(published_after=date(2023, 6, 1)),
    )
    assert [chunk.chunk_id for chunk, _ in results] == ["c2"]


def test_bm25_respects_allowed_document_ids() -> None:
    results = _index().search(
        "metformin renal", top_k=3, allowed_document_ids={"d2"}
    )
    assert [chunk.chunk_id for chunk, _ in results] == ["c2"]


def test_bm25_empty_allowed_set_returns_nothing() -> None:
    assert _index().search("metformin", top_k=3, allowed_document_ids=set()) == []


def test_bm25_upsert_replaces_document_chunks() -> None:
    index = _index()
    replacement = _chunk("c9", "d1", "Metformin contraindicated in severe renal failure.")
    index.upsert_document([replacement])

    assert index.size == 3
    results = index.search("severe renal failure", top_k=3)
    assert results[0][0].chunk_id == "c9"


def test_bm25_remove_document_rebuilds_on_next_search() -> None:
    index = _index()
    index.remove_document("d1")
    assert index.size == 2

    results = index.search("contraindicated", top_k=3)
    assert all(chunk.document_id != "d1" for chunk, _ in results)
    assert index.ready is True


def test_bm25_health_reports_index_state() -> None:
    health = _index().health()
    assert health.ok is True
    assert health.provider == "bm25"
    assert "3 chunks" in (health.detail or "")
