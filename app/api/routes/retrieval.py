from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_retriever, get_session
from app.schemas.retrieval import RetrievalSearchRequest, RetrievalSearchResponse
from app.services.rag.diagnostics import build_diagnostics, to_search_hit
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.types import RetrievalFilters

router = APIRouter(tags=["retrieval"])


@router.post("/retrieval/search", response_model=RetrievalSearchResponse)
async def retrieval_search(
    payload: RetrievalSearchRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    retriever: HybridRetriever = Depends(get_retriever),
) -> RetrievalSearchResponse:
    """Run the Phase 2 retrieval pipeline without calling the LLM.

    Useful for inspecting fusion and reranking behaviour, and for evaluating
    retrieval quality independently of generation.
    """

    started = time.perf_counter()
    filters = (
        RetrievalFilters.from_mapping(payload.filters.model_dump(exclude_none=True))
        if payload.filters
        else RetrievalFilters()
    )
    result = await retriever.retrieve(
        session,
        query=payload.query,
        top_k=payload.top_k,
        filters=filters,
    )
    return RetrievalSearchResponse(
        query=payload.query,
        hits=[
            to_search_hit(hit, index=index)
            for index, hit in enumerate(result.hits, start=1)
        ],
        total_ms=round((time.perf_counter() - started) * 1000, 2),
        diagnostics=build_diagnostics(
            result,
            rrf_k=request.app.state.settings.rrf_k,
            reranker=request.app.state.reranker,
        ),
    )
