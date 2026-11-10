from __future__ import annotations

import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import AppError
from app.schemas.chat import ChatRequest, ChatResponse, TimingBreakdown
from app.services.llm.base import LLMProvider
from app.services.rag.diagnostics import build_diagnostics
from app.services.rag.generator import (
    generate_answer,
    insufficient_response,
    save_trace,
    token_usage,
)
from app.services.retrieval.hybrid import HybridRetrievalResult, HybridRetriever
from app.services.retrieval.types import RetrievalFilters
from app.utils.ids import new_trace_id

logger = logging.getLogger(__name__)


class HybridRAGService:
    """BM25 + dense retrieval fused with RRF, then optionally reranked."""

    def __init__(
        self,
        *,
        settings: Settings,
        retriever: HybridRetriever,
        llm: LLMProvider,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    async def answer(
        self, session: AsyncSession, payload: ChatRequest, *, request_id: str
    ) -> ChatResponse:
        started = time.perf_counter()
        trace_id = new_trace_id()
        steps: list[dict[str, object]] = []

        try:
            filters = (
                RetrievalFilters.from_mapping(payload.filters.model_dump(exclude_none=True))
                if payload.filters
                else RetrievalFilters()
            )
            retrieval = await self.retriever.retrieve(
                session,
                query=payload.question,
                top_k=payload.top_k,
                filters=filters,
            )
            for stage in retrieval.stages:
                steps.append(
                    {
                        "step": stage.name,
                        "agent": "retrieval",
                        "status": "completed",
                        "latency_ms": stage.latency_ms,
                        "retrieved": stage.candidates,
                    }
                )

            outcome = await generate_answer(
                settings=self.settings,
                llm=self.llm,
                question=payload.question,
                hits=retrieval.hits,
            )
            if not outcome.insufficient:
                steps.append(
                    {
                        "step": "generate_answer",
                        "agent": "generator",
                        "status": "completed",
                        "latency_ms": outcome.generation_ms,
                    }
                )

            embedding_ms, retrieval_ms = _split_timings(retrieval)
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            timings = TimingBreakdown(
                embedding_ms=embedding_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=outcome.generation_ms,
                total_ms=total_ms,
            )
            diagnostics = build_diagnostics(
                retrieval, rrf_k=self.settings.rrf_k, reranker=self.retriever.reranker
            )

            if outcome.insufficient:
                response = insufficient_response(
                    request_id=request_id,
                    system="hybrid",
                    timings=timings,
                    warnings=outcome.warnings,
                )
                response.diagnostics = diagnostics
                await save_trace(
                    session,
                    trace_id=trace_id,
                    mode="hybrid",
                    request_id=request_id,
                    question=payload.question,
                    steps=steps,
                    status="completed",
                    latency_ms=total_ms,
                    retrieval_attempts=1,
                )
                return response

            warnings = list(outcome.warnings)
            if not retrieval.reranker_used:
                warnings.append(
                    "reranker_skipped"
                    if self.retriever.reranker.enabled
                    else "reranker_disabled"
                )

            await save_trace(
                session,
                trace_id=trace_id,
                mode="hybrid",
                request_id=request_id,
                question=payload.question,
                steps=steps,
                status="completed",
                latency_ms=total_ms,
                retrieval_attempts=1,
                prompt_tokens=outcome.prompt_tokens,
                completion_tokens=outcome.completion_tokens,
            )

            return ChatResponse(
                request_id=request_id,
                system="hybrid",
                answer=outcome.content or "",
                citations=outcome.citations,
                retrieved_chunks=outcome.retrieved_chunks,
                insufficient_evidence=False,
                timings=timings,
                usage=token_usage(outcome),
                retrieval_attempts=1,
                model=outcome.model,
                warnings=warnings,
                diagnostics=diagnostics,
            )

        except AppError as exc:
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            await save_trace(
                session,
                trace_id=trace_id,
                mode="hybrid",
                request_id=request_id,
                question=payload.question,
                steps=steps,
                status="failed",
                latency_ms=total_ms,
                retrieval_attempts=1,
                error=f"{exc.code}: {exc.message}",
            )
            raise


def _split_timings(retrieval: HybridRetrievalResult) -> tuple[float, float]:
    embedding_ms = 0.0
    retrieval_ms = 0.0
    for stage in retrieval.stages:
        if stage.name == "embed_query":
            embedding_ms += stage.latency_ms
        else:
            retrieval_ms += stage.latency_ms
    return round(embedding_ms, 2), round(retrieval_ms, 2)
