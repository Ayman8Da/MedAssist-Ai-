from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_benchmark_runner, get_session
from app.schemas.benchmark import BenchmarkResultsOut, BenchmarkRunResponse
from app.services.benchmark import BenchmarkRunner

router = APIRouter(tags=["benchmark"])


@router.post("/benchmark/run", response_model=BenchmarkRunResponse)
async def run_benchmark(
    request: Request,
    session: AsyncSession = Depends(get_session),
    runner: BenchmarkRunner = Depends(get_benchmark_runner),
) -> BenchmarkRunResponse:
    run_id = await runner.start(session)
    return BenchmarkRunResponse(run_id=run_id, status="pending")


@router.get("/benchmark/results", response_model=BenchmarkResultsOut | None)
async def get_benchmark_results(
    request: Request,
    run_id: str | None = None,
    session: AsyncSession = Depends(get_session),
    runner: BenchmarkRunner = Depends(get_benchmark_runner),
) -> BenchmarkResultsOut | None:
    import json as _json


    if run_id:
        run = await runner.get(session, run_id)
    else:
        run = await runner.get_latest(session)
    if run is None:
        return None
    results = _json.loads(run.results_json) if run.results_json else None
    summary = _json.loads(run.summary_json) if run.summary_json else None
    from app.schemas.benchmark import BenchmarkRunOut

    return BenchmarkResultsOut(
        run=BenchmarkRunOut.model_validate(run),
        summary=summary,
        results=results,
    )
