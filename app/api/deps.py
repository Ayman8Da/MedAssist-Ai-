from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.services.benchmark import BenchmarkRunner
from app.services.embeddings.base import EmbeddingProvider
from app.services.ingestion.pipeline import IngestionService
from app.services.llm.base import LLMProvider
from app.services.rag.agentic import AgenticRAGService
from app.services.rag.hybrid import HybridRAGService
from app.services.rag.traditional import TraditionalRAGService
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.reranker.base import RerankerProvider
from app.services.vectorstore.qdrant_store import QdrantVectorStore


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.db.session_factory() as session:
        yield session


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ingestion(request: Request) -> IngestionService:
    return request.app.state.ingestion


def get_rag(request: Request) -> TraditionalRAGService:
    return request.app.state.rag


def get_hybrid_rag(request: Request) -> HybridRAGService:
    return request.app.state.hybrid_rag

def get_agentic_rag(request: Request) -> AgenticRAGService:
    return request.app.state.agentic_rag


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_bm25(request: Request) -> BM25Index | None:
    return request.app.state.bm25


def get_reranker(request: Request) -> RerankerProvider:
    return request.app.state.reranker


def get_vector_store(request: Request) -> QdrantVectorStore:
    return request.app.state.vector_store


def get_embeddings(request: Request) -> EmbeddingProvider:
    return request.app.state.embeddings


def get_llm(request: Request) -> LLMProvider:
    return request.app.state.llm


def get_benchmark_runner(request: Request) -> BenchmarkRunner:
    return request.app.state.benchmark


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")
