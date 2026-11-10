"""Hybrid retriever: dense + BM25 -> RRF -> dedupe -> rerank.

The pipeline is explicit and observable. Every stage records its latency and
candidate count, and reranker failures are reported instead of being hidden.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError
from app.db.repositories import DocumentRepository
from app.services.embeddings.base import EmbeddingProvider
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.fusion import deduplicate_hits, reciprocal_rank_fusion
from app.services.retrieval.reranker.base import RerankerProvider
from app.services.retrieval.types import RetrievalFilters, RetrievalHit
from app.services.vectorstore.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalStage:
    name: str
    latency_ms: float
    candidates: int | None = None


@dataclass(slots=True)
class HybridRetrievalResult:
    hits: list[RetrievalHit]
    stages: list[RetrievalStage]
    dense_candidates: int
    keyword_candidates: int
    fused_candidates: int
    duplicates_removed: int
    reranked_candidates: int
    returned: int
    reranker_used: bool
    reranker_skipped_reason: str | None
    bm25_enabled: bool
    latest_versions_only: bool
    allowed_document_ids: int | None
    stale_versions_excluded: int


class HybridRetriever:
    def __init__(
        self,
        *,
        settings: Settings,
        embeddings: EmbeddingProvider,
        vector_store: QdrantVectorStore,
        bm25: BM25Index | None,
        reranker: RerankerProvider,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker

    async def retrieve(
        self,
        session: AsyncSession,
        *,
        query: str,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> HybridRetrievalResult:
        settings = self.settings
        filters = filters or RetrievalFilters()
        effective_top_k = top_k or settings.top_k
        dense_k = max(settings.dense_candidate_k, effective_top_k)
        keyword_k = max(settings.keyword_candidate_k, effective_top_k)
        stages: list[RetrievalStage] = []

        allowed_document_ids, stale_count = await self._version_scope(session)

        # --- dense -------------------------------------------------------
        started = time.perf_counter()
        try:
            query_vector = await asyncio.wait_for(
                self.embeddings.embed_query(query),
                timeout=settings.retrieval_timeout_seconds,
            )
        except TimeoutError as exc:
            raise ProviderUnavailableError("Embedding request timed out") from exc
        stages.append(
            RetrievalStage(
                name="embed_query",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )

        dense_hits: list[RetrievalHit] = []
        search_dense, dense_filters = self._dense_filters(filters, allowed_document_ids)
        started = time.perf_counter()
        if search_dense:
            try:
                raw_dense = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.vector_store.search,
                        query_vector,
                        top_k=dense_k,
                        filters=dense_filters,
                        score_threshold=settings.score_threshold,
                    ),
                    timeout=settings.retrieval_timeout_seconds,
                )
            except TimeoutError as exc:
                raise ProviderUnavailableError("Vector retrieval timed out") from exc
            dense_hits = [
                RetrievalHit.from_dense(hit, rank=rank)
                for rank, hit in enumerate(raw_dense, start=1)
            ]
        stages.append(
            RetrievalStage(
                name="dense_search",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                candidates=len(dense_hits),
            )
        )

        # --- BM25 --------------------------------------------------------
        keyword_hits: list[RetrievalHit] = []
        started = time.perf_counter()
        if self.bm25 is not None:
            raw_keyword = await asyncio.to_thread(
                self.bm25.search,
                query,
                top_k=keyword_k,
                filters=filters,
                allowed_document_ids=allowed_document_ids,
            )
            keyword_hits = [
                RetrievalHit.from_keyword(record, score=score, rank=rank)
                for rank, (record, score) in enumerate(raw_keyword, start=1)
            ]
        stages.append(
            RetrievalStage(
                name="bm25_search",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                candidates=len(keyword_hits),
            )
        )

        # --- fusion + duplicate removal ----------------------------------
        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"dense": dense_hits, "keyword": keyword_hits},
            k=settings.rrf_k,
        )
        deduped, duplicates_removed = deduplicate_hits(fused)
        stages.append(
            RetrievalStage(
                name="rrf_fusion",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                candidates=len(deduped),
            )
        )

        # --- rerank ------------------------------------------------------
        reranker_used = False
        skipped_reason: str | None = None
        reranked_count = 0
        started = time.perf_counter()

        rerank_window = deduped[: max(1, settings.rerank_candidates)]
        if not self.reranker.enabled:
            skipped_reason = "disabled_by_configuration"
        elif not rerank_window:
            skipped_reason = "no_candidates"
        else:
            scores: list[float] | None = None
            try:
                outcome = await asyncio.wait_for(
                    self.reranker.rerank(
                        query, [hit.text[:4000] for hit in rerank_window]
                    ),
                    timeout=settings.reranker_timeout_seconds,
                )
                scores = outcome.scores
            except TimeoutError:
                skipped_reason = "timeout"
                logger.warning(
                    "Reranker timed out after %.1fs; returning RRF order",
                    settings.reranker_timeout_seconds,
                )
            except Exception as exc:  # noqa: BLE001 - degrade explicitly
                skipped_reason = str(exc)
                logger.warning("Reranker failed; returning RRF order: %s", exc)

            if scores is not None and len(scores) == len(rerank_window):
                reranked = [
                    replace(
                        hit,
                        rerank_score=float(score),
                        score=float(score),
                        score_kind="rerank",
                    )
                    for hit, score in zip(rerank_window, scores, strict=True)
                ]
                reranked.sort(key=lambda hit: (-float(hit.rerank_score or 0.0), hit.chunk_id))
                deduped = reranked + deduped[len(rerank_window) :]
                reranker_used = True
                reranked_count = len(reranked)
            elif scores is not None:
                skipped_reason = "score_count_mismatch"

        stages.append(
            RetrievalStage(
                name="rerank",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                candidates=reranked_count,
            )
        )

        final_hits = deduped[:effective_top_k]
        for rank, hit in enumerate(final_hits, start=1):
            hit.final_rank = rank

        return HybridRetrievalResult(
            hits=final_hits,
            stages=stages,
            dense_candidates=len(dense_hits),
            keyword_candidates=len(keyword_hits),
            fused_candidates=len(fused),
            duplicates_removed=duplicates_removed,
            reranked_candidates=reranked_count,
            returned=len(final_hits),
            reranker_used=reranker_used,
            reranker_skipped_reason=skipped_reason,
            bm25_enabled=self.bm25 is not None,
            latest_versions_only=settings.latest_versions_only,
            allowed_document_ids=(
                None if allowed_document_ids is None else len(allowed_document_ids)
            ),
            stale_versions_excluded=stale_count,
        )

    async def _version_scope(self, session: AsyncSession) -> tuple[set[str] | None, int]:
        if not self.settings.latest_versions_only:
            return None, 0

        repo = DocumentRepository(session)
        stale = await repo.stale_version_ids()
        if not stale:
            return None, 0

        indexed = set(await repo.list_ids(["indexed"]))
        return indexed - stale, len(stale)

    def _dense_filters(
        self, filters: RetrievalFilters, allowed_document_ids: set[str] | None
    ) -> tuple[bool, dict[str, object] | None]:
        """Return ``(should_search, qdrant_filters)``.

        ``None`` filters mean "no restriction"; the boolean exists because an
        empty allowed-document set must skip the search entirely rather than
        run an unrestricted query.
        """

        if allowed_document_ids is None:
            return True, filters.as_qdrant_mapping()

        requested = set(filters.document_ids) if filters.document_ids else None
        effective = [
            document_id
            for document_id in allowed_document_ids
            if requested is None or document_id in requested
        ]
        if not effective:
            return False, None

        payload = dict(filters.as_qdrant_mapping() or {})
        payload["document_ids"] = effective
        return True, payload