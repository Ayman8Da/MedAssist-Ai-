from __future__ import annotations

import contextvars
import logging
import sys

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | req=%(request_id)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RequestIdFilter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Avoid noisy third-party debug logging.
    for noisy in ("httpx", "httpcore", "qdrant_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
