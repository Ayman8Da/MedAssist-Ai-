from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base application error rendered as a structured JSON response."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class ConflictError(AppError):
    status_code = 409
    code = "conflict"


class ValidationAppError(AppError):
    status_code = 422
    code = "validation_error"


class IngestionError(AppError):
    status_code = 422
    code = "ingestion_error"


class ProviderUnavailableError(AppError):
    status_code = 503
    code = "provider_unavailable"


class ServiceUnavailableError(AppError):
    status_code = 503
    code = "service_unavailable"


class NotImplementedAppError(AppError):
    status_code = 501
    code = "not_implemented"
