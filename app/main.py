from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import benchmark, chat, documents, health, retrieval, traces
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging, request_id_ctx
from app.db.migrate import apply_additive_migrations
from app.db.repositories import ChunkRepository
from app.db.session import Database
from app.services.benchmark import BenchmarkRunner
from app.services.embeddings.factory import build_embedding_provider
from app.services.ingestion.pipeline import IngestionService
from app.services.llm.factory import build_llm_provider
from app.services.rag.agentic import AgenticRAGService
from app.services.rag.hybrid import HybridRAGService
from app.services.rag.traditional import TraditionalRAGService
from app.services.retrieval.bm25 import BM25Index
from app.services.retrieval.hybrid import HybridRetriever
from app.services.retrieval.reranker.factory import build_reranker
from app.services.vectorstore.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)

PHASE = 4


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_settings.ensure_directories()

        db = Database(app_settings.resolved_database_url)
        await db.init()
        applied = await apply_additive_migrations(db.engine)
        if applied:
            logger.info("Schema migrations applied: %s", ", ".join(applied))

        embeddings = build_embedding_provider(app_settings)
        llm = build_llm_provider(app_settings)
        vector_store = QdrantVectorStore(app_settings)
        reranker = build_reranker(app_settings)

        bm25: BM25Index | None = None
        bm25_error: str | None = None
        if app_settings.bm25_enabled:
            bm25 = BM25Index(
                k1=app_settings.bm25_k1,
                b=app_settings.bm25_b,
                stemming=app_settings.bm25_stemming,
            )
            try:
                async with db.session_factory() as session:
                    records = await ChunkRepository(session).list_for_index()
                indexed = await asyncio.to_thread(bm25.rebuild, records)
                logger.info("BM25 index built with %s chunks", indexed)
            except Exception as exc:  # noqa: BLE001 - health reports the failure
                bm25_error = str(exc)
                logger.error("BM25 index initialisation failed: %s", exc)

        vector_store_error: str | None = None
        try:
            await asyncio.to_thread(vector_store.ensure_collection)
        except Exception as exc:  # noqa: BLE001 - app stays up; health reports degraded
            vector_store_error = str(exc)
            logger.error("Vector store initialisation failed: %s", exc)

        app.state.settings = app_settings
        app.state.db = db
        app.state.embeddings = embeddings
        app.state.llm = llm
        app.state.vector_store = vector_store
        app.state.vector_store_error = vector_store_error
        app.state.bm25 = bm25
        app.state.bm25_error = bm25_error
        app.state.reranker = reranker
        app.state.ingestion = IngestionService(
            settings=app_settings,
            embeddings=embeddings,
            vector_store=vector_store,
            bm25=bm25,
        )
        app.state.rag = TraditionalRAGService(
            settings=app_settings,
            embeddings=embeddings,
            vector_store=vector_store,
            llm=llm,
        )
        app.state.retriever = HybridRetriever(
            settings=app_settings,
            embeddings=embeddings,
            vector_store=vector_store,
            bm25=bm25,
            reranker=reranker,
        )
        app.state.hybrid_rag = HybridRAGService(
            settings=app_settings,
            retriever=app.state.retriever,
            llm=llm,
        )
        app.state.agentic_rag = AgenticRAGService(
            settings=app_settings,
            retriever=app.state.retriever,
            hybrid_service=app.state.hybrid_rag,
            llm=llm,
        )
        app.state.benchmark = BenchmarkRunner(
            settings=app_settings,
            db=db,
            traditional=app.state.rag,
            agentic=app.state.agentic_rag,
            llm=llm,
        )

        logger.info(
            "MedAssist AI started env=%s phase=%s embeddings=%s/%s llm=%s/%s "
            "qdrant=%s bm25=%s reranker=%s",
            app_settings.app_env,
            PHASE,
            app_settings.embedding_provider,
            app_settings.embedding_model,
            app_settings.llm_provider,
            app_settings.llm_model,
            vector_store.mode,
            f"{bm25.size} chunks" if bm25 is not None else "disabled",
            f"{reranker.name}/{reranker.model}" if reranker.enabled else "disabled",
        )

        try:
            yield
        finally:
            await embeddings.close()
            await llm.close()
            await reranker.close()
            try:
                await asyncio.to_thread(vector_store.close)
            except Exception:  # noqa: BLE001
                logger.warning("Could not close the vector store cleanly")
            await db.dispose()

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        description=(
            "MedAssist AI - clinical knowledge and decision-support prototype. "
            "Phase 2: BM25 + dense retrieval fused with RRF, optional BGE reranker. "
            "Not a diagnostic system and not a substitute for clinical judgement."
        ),
        lifespan=lifespan,
    )
    app.state.settings = app_settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid4().hex[:16]}"
        request.state.request_id = request_id
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %s (%.2f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.warning("application_error code=%s path=%s", exc.code, request.url.path)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "details": jsonable_encoder(exc.details),
                }
            },
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed",
                    "details": {"errors": jsonable_encoder(exc.errors())},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error path=%s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected internal error occurred",
                    "details": {},
                }
            },
        )

    prefix = app_settings.api_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(documents.router, prefix=prefix)
    app.include_router(chat.router, prefix=prefix)
    app.include_router(retrieval.router, prefix=prefix)
    app.include_router(traces.router, prefix=prefix)
    app.include_router(benchmark.router, prefix=prefix)

    @app.get("/", tags=["meta"])
    async def root() -> dict[str, object]:
        return {
            "name": app_settings.app_name,
            "version": app_settings.app_version,
            "phase": PHASE,
            "docs": "/docs",
            "health": f"{prefix}/health",
        }

    return app


app = create_app()
