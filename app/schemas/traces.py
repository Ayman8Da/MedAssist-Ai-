from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class TraceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    request_id: str | None = None
    mode: str
    query_sha256: str
    query_preview: str
    steps: list[dict[str, Any]]
    status: str
    retrieval_attempts: int
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    created_at: datetime


class TraceListOut(BaseModel):
    items: list[TraceOut]
    total: int
    limit: int
    offset: int
