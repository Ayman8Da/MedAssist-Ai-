from __future__ import annotations

from uuid import uuid4


def new_document_id() -> str:
    return f"doc_{uuid4().hex}"


def new_chunk_id() -> str:
    """Qdrant point IDs must be UUIDs, so chunk IDs are plain UUID strings."""
    return str(uuid4())


def new_trace_id() -> str:
    return f"trc_{uuid4().hex}"
