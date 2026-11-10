from __future__ import annotations

from app.schemas.retrieval import (
    RetrievalDiagnostics,
    RetrievalSearchHit,
    RetrievalStageOut,
)
from app.services.rag.generator import parse_date
from app.services.retrieval.hybrid import HybridRetrievalResult
from app.services.retrieval.reranker.base import RerankerProvider
from app.services.retrieval.types import RetrievalHit


def build_diagnostics(
    retrieval: HybridRetrievalResult,
    *,
    rrf_k: int,
    reranker: RerankerProvider,
) -> RetrievalDiagnostics:
    return RetrievalDiagnostics(
        strategy="hybrid",
        bm25_enabled=retrieval.bm25_enabled,
        dense_candidates=retrieval.dense_candidates,
        keyword_candidates=retrieval.keyword_candidates,
        fused_candidates=retrieval.fused_candidates,
        duplicates_removed=retrieval.duplicates_removed,
        reranked_candidates=retrieval.reranked_candidates,
        returned=retrieval.returned,
        rrf_k=rrf_k,
        reranker_provider=reranker.name,
        reranker_model=reranker.model,
        reranker_enabled=reranker.enabled,
        reranker_used=retrieval.reranker_used,
        reranker_skipped_reason=retrieval.reranker_skipped_reason,
        latest_versions_only=retrieval.latest_versions_only,
        stale_versions_excluded=retrieval.stale_versions_excluded,
        allowed_document_ids=retrieval.allowed_document_ids,
        stages=[
            RetrievalStageOut(
                name=stage.name,
                latency_ms=stage.latency_ms,
                candidates=stage.candidates,
            )
            for stage in retrieval.stages
        ],
    )


def to_search_hit(hit: RetrievalHit, *, index: int) -> RetrievalSearchHit:
    payload = hit.payload
    return RetrievalSearchHit(
        index=index,
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        title=str(payload.get("title") or "Untitled document"),
        source=str(payload.get("source") or "unknown source"),
        publication_date=parse_date(payload.get("publication_date")),
        version=str(payload["version"]) if payload.get("version") else None,
        category=str(payload["category"]) if payload.get("category") else None,
        section=str(payload["section"]) if payload.get("section") else None,
        page_number=(
            int(payload["page_number"]) if payload.get("page_number") is not None else None
        ),
        text=hit.text,
        score=round(float(hit.score), 6),
        score_kind=hit.score_kind,
        dense_score=hit.dense_score,
        keyword_score=hit.keyword_score,
        rrf_score=hit.rrf_score,
        rerank_score=hit.rerank_score,
        dense_rank=hit.dense_rank,
        keyword_rank=hit.keyword_rank,
        matched=sorted(hit.matched),
    )
