from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthCheck(BaseModel):
    name: str
    status: Literal["ok", "degraded", "error"]
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    version: str
    phase: int
    checks: list[HealthCheck]
