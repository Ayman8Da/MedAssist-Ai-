from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from datetime import time as dt_time
from typing import Any

from qdrant_client import QdrantClient, models

from app.core.config import Settings
from app.core.exceptions import ServiceUnavailableError
from app.services.base import ProviderHealth


@dataclass(slots=True)
class VectorRecord:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    document_id: str
    score: float
    payload: dict[str, Any]

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))


class QdrantVectorStore:
    """Qdrant adapter. Uses on-disk local mode when QDRANT_URL is empty."""

    def __init__(self, settings: Settings) -> None:
        self.collection = settings.qdrant_collection
        self.dim = settings.embedding_dim
        if settings.qdrant_url:
            self.mode = "remote"
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=30,
            )
        else:
            self.mode = "local"
            settings.resolved_qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(settings.resolved_qdrant_path))

    def ensure_collection(self) -> None:
        try:
            exists = self.client.collection_exists(self.collection)
        except Exception as exc:  # noqa: BLE001 - normalise provider failure
            raise ServiceUnavailableError(
                "Qdrant is unavailable",
                details={"mode": self.mode, "collection": self.collection},
            ) from exc

        if not exists:
            try:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=self.dim, distance=models.Distance.COSINE
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                raise ServiceUnavailableError(
                    "Failed to create the Qdrant collection",
                    details={"collection": self.collection},
                ) from exc
            return

        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            size = next(iter(vectors.values())).size
        else:
            size = getattr(vectors, "size", None)
        if size is not None and int(size) != self.dim:
            raise ServiceUnavailableError(
                "Qdrant collection dimension does not match EMBEDDING_DIM. "
                "Delete the collection or change EMBEDDING_DIM.",
                details={
                    "collection": self.collection,
                    "collection_dim": int(size),
                    "configured_dim": self.dim,
                },
            )

    def upsert_records(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        points = [
            models.PointStruct(id=record.id, vector=record.vector, payload=record.payload)
            for record in records
        ]
        try:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailableError(
                "Failed to write vectors to Qdrant",
                details={"collection": self.collection},
            ) from exc

    def search(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchHit]:
        try:
            response = self.client.query_points(
                collection_name=self.collection,
                query=list(vector),
                query_filter=self._build_filter(filters),
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailableError(
                "Vector search failed",
                details={"collection": self.collection},
            ) from exc

        hits: list[SearchHit] = []
        for point in response.points:
            payload = dict(point.payload or {})
            hits.append(
                SearchHit(
                    chunk_id=str(point.id),
                    document_id=str(payload.get("document_id", "")),
                    score=float(point.score),
                    payload=payload,
                )
            )
        return hits

    def count(self) -> int:
        result = self.client.count(collection_name=self.collection, exact=True)
        return int(result.count)

    def delete_document(self, document_id: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        )
        self.client.delete(collection_name=self.collection, points_selector=selector, wait=True)

    def health(self) -> ProviderHealth:
        started = time.perf_counter()
        try:
            count = self.count()
            return ProviderHealth(
                ok=True,
                provider="qdrant",
                model=self.collection,
                detail=f"{count} indexed chunks ({self.mode} mode)",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except Exception:  # noqa: BLE001
            return ProviderHealth(
                ok=False,
                provider="qdrant",
                model=self.collection,
                detail="Qdrant collection is unavailable",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def _build_filter(self, filters: Mapping[str, Any] | None) -> models.Filter | None:
        if not filters:
            return None

        must: list[models.FieldCondition] = []
        for key in ("source", "category", "version"):
            value = filters.get(key)
            if value:
                must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))

        document_ids = filters.get("document_ids")
        if document_ids:
            must.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=list(document_ids))
                )
            )

        published_after = filters.get("published_after")
        published_before = filters.get("published_before")
        if published_after:
            must.append(
                models.FieldCondition(
                    key="publication_ts",
                    range=models.Range(gte=self._to_timestamp(published_after)),
                )
            )
        if published_before:
            must.append(
                models.FieldCondition(
                    key="publication_ts",
                    range=models.Range(lte=self._to_timestamp(published_before)),
                )
            )

        return models.Filter(must=must) if must else None

    @staticmethod
    def _to_timestamp(value: date | datetime | str) -> int:
        if isinstance(value, str):
            value = date.fromisoformat(value)
        if isinstance(value, datetime):
            moment = value
            if moment.tzinfo is None:
                moment = moment.replace(tzinfo=UTC)
        else:
            moment = datetime.combine(value, dt_time.min, tzinfo=UTC)
        return int(moment.timestamp())
