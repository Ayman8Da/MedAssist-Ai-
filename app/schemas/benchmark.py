from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class BenchmarkRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    created_at: datetime
    updated_at: datetime
    total_questions: int
    completed_questions: int
    error: str | None = None


class BenchmarkResultsOut(BaseModel):
    run: BenchmarkRunOut
    summary: dict[str, Any] | None = None
    results: list[dict[str, Any]] | None = None


class BenchmarkRunResponse(BaseModel):
    run_id: str
    status: str
