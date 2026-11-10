from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProviderHealth:
    ok: bool
    provider: str
    model: str
    detail: str | None = None
    latency_ms: float | None = None
