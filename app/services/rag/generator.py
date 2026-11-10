"""Shared grounded-generation helpers.

Traditional and hybrid RAG differ only in how evidence is retrieved. Prompting,
citation assembly, insufficient-evidence handling and trace persistence live
here so both systems stay behaviourally comparable for the Phase 4 benchmark.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import ProviderUnavailableError
from app.db.models import AgentTrace
from app.schemas.chat import (
    ChatResponse,
    Citation,
    RetrievedChunk,
    TimingBreakdown,
    TokenUsage,
)
from app.services.llm.base import LLMProvider
from app.services.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.services.rag.context import ContextBlock, build_context, render_context
from app.utils.text import sha256_text, truncate

logger = logging.getLogger(__name__)

INSUFFICIENT_ANSWER = (
    "Insufficient evidence: the indexed knowledge base did not return any "
    "relevant document for this question. No clinical recommendation is "
    "provided. Add the relevant guideline or report to the knowledge base, "
    "or rephrase the question with more clinical detail. This output is "
    "decision support only and requires review by a qualified healthcare "
    "professional."
)


@dataclass(slots=True)
class GenerationOutcome:
    blocks: list[ContextBlock] = field(default_factory=list)
    content: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    generation_ms: float = 0.0
    citations: list[Citation] = field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def insufficient(self) -> bool:
        return self.content is None


def parse_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


def to_citation(block: ContextBlock, *, used: bool) -> Citation:
    payload = block.hit.payload
    return Citation(
        index=block.index,
        chunk_id=block.hit.chunk_id,
        document_id=block.hit.document_id,
        title=str(payload.get("title") or "Untitled document"),
        source=str(payload.get("source") or "unknown source"),
        publication_date=parse_date(payload.get("publication_date")),
        version=str(payload["version"]) if payload.get("version") else None,
        category=str(payload["category"]) if payload.get("category") else None,
        section=str(payload["section"]) if payload.get("section") else None,
        page_number=(
            int(payload["page_number"]) if payload.get("page_number") is not None else None
        ),
        score=round(float(block.hit.score), 6),
        score_kind=getattr(block.hit, "score_kind", None),
        used=used,
    )


async def generate_answer(
    *,
    settings: Settings,
    llm: LLMProvider,
    question: str,
    hits: Sequence[Any],
) -> GenerationOutcome:
    blocks = build_context(hits, settings.max_context_chars)
    if not blocks:
        return GenerationOutcome(warnings=["no_relevant_documents"])

    started = time.perf_counter()
    try:
        llm_result = await asyncio.wait_for(
            llm.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(question, render_context(blocks)),
            ),
            timeout=settings.llm_timeout_seconds,
        )
    except TimeoutError as exc:
        raise ProviderUnavailableError("LLM generation timed out") from exc

    generation_ms = round((time.perf_counter() - started) * 1000, 2)
    citations = [
        to_citation(block, used=f"[{block.index}]" in llm_result.content) for block in blocks
    ]
    retrieved_chunks = [
        RetrievedChunk(**citation.model_dump(), text=block.hit.text)
        for citation, block in zip(citations, blocks, strict=True)
    ]

    warnings = ["for_professional_review_only"]
    if not any(citation.used for citation in citations):
        warnings.append("answer_does_not_cite_retrieved_evidence")

    return GenerationOutcome(
        blocks=blocks,
        content=llm_result.content,
        model=llm_result.model,
        prompt_tokens=llm_result.prompt_tokens,
        completion_tokens=llm_result.completion_tokens,
        generation_ms=generation_ms,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        warnings=warnings,
    )


def token_usage(outcome: GenerationOutcome) -> TokenUsage:
    prompt_tokens = outcome.prompt_tokens
    completion_tokens = outcome.completion_tokens
    total = (
        (prompt_tokens or 0) + (completion_tokens or 0)
        if prompt_tokens is not None or completion_tokens is not None
        else None
    )
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total,
    )


def insufficient_response(
    *,
    request_id: str,
    system: Literal["traditional", "hybrid", "agentic"],
    timings: TimingBreakdown,
    warnings: list[str],
) -> ChatResponse:
    return ChatResponse(
        request_id=request_id,
        system=system,
        answer=INSUFFICIENT_ANSWER,
        citations=[],
        retrieved_chunks=[],
        insufficient_evidence=True,
        timings=timings,
        usage=TokenUsage(),
        retrieval_attempts=1,
        model=None,
        warnings=["for_professional_review_only", *warnings],
    )


async def save_trace(
    session: AsyncSession,
    *,
    trace_id: str,
    mode: str,
    request_id: str,
    question: str,
    steps: list[dict[str, object]],
    status: str,
    latency_ms: float,
    retrieval_attempts: int,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    error: str | None = None,
) -> None:
    trace = AgentTrace(
        id=trace_id,
        request_id=request_id,
        mode=mode,
        query_sha256=sha256_text(question),
        query_preview=truncate(question, 200),
        steps_json=json.dumps(steps),
        status=status,
        retrieval_attempts=retrieval_attempts,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        error=truncate(error, 500) if error else None,
    )
    try:
        session.add(trace)
        await session.commit()
    except Exception:  # noqa: BLE001 - observability must not break the request
        await session.rollback()
        logger.warning("Could not persist agent trace trace_id=%s", trace_id)
