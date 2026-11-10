from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_session
from app.db.repositories import TraceRepository
from app.schemas.traces import TraceListOut, TraceOut

router = APIRouter(tags=["agents"])


@router.get("/agents/traces", response_model=TraceListOut)
async def list_agent_traces(
    mode: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> TraceListOut:
    items, total = await TraceRepository(session).list(mode=mode, limit=limit, offset=offset)
    traces: list[TraceOut] = []
    for item in items:
        try:
            steps = json.loads(item.steps_json or "[]")
        except json.JSONDecodeError:
            steps = []
        traces.append(
            TraceOut(
                id=item.id,
                request_id=item.request_id,
                mode=item.mode,
                query_sha256=item.query_sha256,
                query_preview=item.query_preview,
                steps=steps,
                status=item.status,
                retrieval_attempts=item.retrieval_attempts,
                latency_ms=item.latency_ms,
                prompt_tokens=item.prompt_tokens,
                completion_tokens=item.completion_tokens,
                error=item.error,
                created_at=item.created_at,
            )
        )
    return TraceListOut(items=traces, total=total, limit=limit, offset=offset)
