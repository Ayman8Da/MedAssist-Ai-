from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_agentic_rag, get_hybrid_rag, get_rag, get_request_id, get_session
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag.agentic import AgenticRAGService
from app.services.rag.hybrid import HybridRAGService
from app.services.rag.traditional import TraditionalRAGService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    rag: TraditionalRAGService = Depends(get_rag),
) -> ChatResponse:
    """Default chat endpoint. Routes to the dense-only Traditional RAG baseline."""
    return await rag.answer(session, payload, request_id=get_request_id(request))


@router.post("/chat/traditional", response_model=ChatResponse)
async def chat_traditional(
    payload: ChatRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    rag: TraditionalRAGService = Depends(get_rag),
) -> ChatResponse:
    """System A: dense vector retrieval + single LLM generation."""
    return await rag.answer(session, payload, request_id=get_request_id(request))


@router.post("/chat/hybrid", response_model=ChatResponse)
async def chat_hybrid(
    payload: ChatRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    rag: HybridRAGService = Depends(get_hybrid_rag),
) -> ChatResponse:
    """Phase 2 retrieval: BM25 + dense -> RRF -> dedupe -> reranker."""
    return await rag.answer(session, payload, request_id=get_request_id(request))


@router.post("/chat/agentic", response_model=ChatResponse)
async def chat_agentic(
    payload: ChatRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    rag: AgenticRAGService = Depends(get_agentic_rag),
) -> ChatResponse:
    """Phase 3: LangGraph agentic pipeline (planner, evidence, verifier, safety, correction)."""
    return await rag.answer(session, payload, request_id=get_request_id(request))
