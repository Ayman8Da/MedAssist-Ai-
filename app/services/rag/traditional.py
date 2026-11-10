from __future__ import annotations

import asyncio
import logging
import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import AppError, ProviderUnavailableError
from app.schemas.chat import ChatRequest, ChatResponse, TimingBreakdown
from app.services.embeddings.base import EmbeddingProvider
from app.services.llm.base import LLMProvider
from app.services.rag.generator import (
    generate_answer,
    insufficient_response,
    save_trace,
    token_usage,
)
from app.services.vectorstore.qdrant_store import QdrantVectorStore
from app.utils.ids import new_trace_id

logger = logging.getLogger(__name__)


class TraditionalRAGService:
    """System A baseline: embed query, vector search, LLM generation.

    This path is intentionally dense-only. The Phase 4 benchmark compares it
    against the agentic system, so hybrid fusion must not leak into it.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        embeddings: EmbeddingProvider,
        vector_store: QdrantVectorStore,
        llm: LLMProvider,
    ) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.llm = llm

    async def answer(
        self, session: AsyncSession, payload: ChatRequest, *, request_id: str
    ) -> ChatResponse:
        started = time.perf_counter()
        trace_id = new_trace_id()
        steps: list[dict[str, object]] = []

        try:
            embedding_started = time.perf_counter()
            try:
                query_vector = await asyncio.wait_for(
                    self.embeddings.embed_query(payload.question),
                    timeout=self.settings.retrieval_timeout_seconds,
                )
            except TimeoutError as exc:
                raise ProviderUnavailableError("Embedding request timed out") from exc
            embedding_ms = round((time.perf_counter() - embedding_started) * 1000, 2)
            steps.append(
                {
                    "step": "embed_query",
                    "agent": "retrieval",
                    "status": "completed",
                    "latency_ms": embedding_ms,
                }
            )

            retrieval_started = time.perf_counter()
            top_k = payload.top_k or self.settings.top_k
            filters = (
                payload.filters.model_dump(exclude_none=True) if payload.filters else None
            )
            try:
                hits = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.vector_store.search,
                        query_vector,
                        top_k=top_k,
                        filters=filters,
                        score_threshold=self.settings.score_threshold,
                    ),
                    timeout=self.settings.retrieval_timeout_seconds,
                )
            except TimeoutError as exc:
                raise ProviderUnavailableError("Vector retrieval timed out") from exc
            retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 2)
            steps.append(
                {
                    "step": "vector_search",
                    "agent": "retrieval",
                    "status": "completed",
                    "latency_ms": retrieval_ms,
                    "retrieved": len(hits),
                    "top_k": top_k,
                }
            )

            outcome = await generate_answer(
                settings=self.settings,
                llm=self.llm,
                question=payload.question,
                hits=hits,
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

            total_ms = round((time.perf_counter() - started) * 1000, 2)
            timings = TimingBreakdown(
                embedding_ms=embedding_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=outcome.generation_ms,
                total_ms=total_ms,
            )

            if outcome.insufficient:
                response = insufficient_response(
                    request_id=request_id,
                    system="traditional",
                    timings=timings,
                    warnings=outcome.warnings,
                )
                await save_trace(
                    session,
                    trace_id=trace_id,
                    mode="traditional",
                    request_id=request_id,
                    question=payload.question,
                    steps=steps,
                    status="completed",
                    latency_ms=total_ms,
                    retrieval_attempts=1,
                )
                return response

            await save_trace(
                session,
                trace_id=trace_id,
                mode="traditional",
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
                system="traditional",
                answer=outcome.content or "",
                citations=outcome.citations,
                retrieved_chunks=outcome.retrieved_chunks,
                insufficient_evidence=False,
                timings=timings,
                usage=token_usage(outcome),
                retrieval_attempts=1,
                model=outcome.model,
                warnings=outcome.warnings,
            )

        except AppError as exc:
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            await save_trace(
                session,
                trace_id=trace_id,
                mode="traditional",
                request_id=request_id,
                question=payload.question,
                steps=steps,
                status="failed",
                latency_ms=total_ms,
                retrieval_attempts=1,
                error=f"{exc.code}: {exc.message}",
            )
            raise
