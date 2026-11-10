from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.schemas.common import HealthCheck, HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, session: AsyncSession = Depends(get_session)) -> HealthResponse:
    checks: list[HealthCheck] = []

    started = time.perf_counter()
    try:
        await session.execute(text("SELECT 1"))
        checks.append(
            HealthCheck(
                name="database",
                status="ok",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        )
    except Exception:  # noqa: BLE001
        checks.append(
            HealthCheck(name="database", status="error", detail="Database query failed")
        )

    state = request.app.state
    if getattr(state, "vector_store_error", None):
        checks.append(
            HealthCheck(
                name="vector_store",
                status="error",
                detail="Qdrant initialisation failed; check server logs",
            )
        )
    else:
        vector_health = await asyncio.to_thread(state.vector_store.health)
        checks.append(
            HealthCheck(
                name="vector_store",
                status="ok" if vector_health.ok else "degraded",
                detail=vector_health.detail,
                latency_ms=vector_health.latency_ms,
            )
        )

    for name, provider in (("embeddings", state.embeddings), ("llm", state.llm)):
        try:
            provider_health = await provider.health()
            checks.append(
                HealthCheck(
                    name=name,
                    status="ok" if provider_health.ok else "degraded",
                    detail=provider_health.detail,
                    latency_ms=provider_health.latency_ms,
                )
            )
        except Exception:  # noqa: BLE001
            checks.append(
                HealthCheck(name=name, status="error", detail="Health check failed")
            )

    if getattr(state, "bm25_error", None):
        checks.append(
            HealthCheck(
                name="bm25",
                status="error",
                detail="BM25 index initialisation failed; keyword retrieval is empty",
            )
        )
    elif getattr(state, "bm25", None) is None:
        checks.append(
            HealthCheck(
                name="bm25",
                status="degraded",
                detail="Keyword retrieval disabled (BM25_ENABLED=false)",
            )
        )
    else:
        bm25_health = await asyncio.to_thread(state.bm25.health)
        checks.append(
            HealthCheck(
                name="bm25",
                status="ok" if bm25_health.ok else "degraded",
                detail=bm25_health.detail,
                latency_ms=bm25_health.latency_ms,
            )
        )

    try:
        reranker_health = await state.reranker.health()
        checks.append(
            HealthCheck(
                name="reranker",
                status=(
                    "ok"
                    if reranker_health.ok and state.reranker.enabled
                    else "degraded"
                ),
                detail=(
                    reranker_health.detail
                    if state.reranker.enabled
                    else "Reranking disabled; fused RRF order is returned"
                ),
                latency_ms=reranker_health.latency_ms,
            )
        )
    except Exception:  # noqa: BLE001
        checks.append(
            HealthCheck(name="reranker", status="error", detail="Health check failed")
        )

    status = "ok" if all(check.status == "ok" for check in checks) else "degraded"
    return HealthResponse(
        status=status,
        app=state.settings.app_name,
        version=state.settings.app_version,
        phase=4,
        checks=checks,
    )
