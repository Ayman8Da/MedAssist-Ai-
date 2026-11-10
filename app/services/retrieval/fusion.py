"""Rank fusion and duplicate removal.

Reciprocal Rank Fusion is deliberately rank-based: BM25 and cosine scores live
on different, non-comparable scales, so only their order is used. The formula
is the standard ``sum(weight / (k + rank))`` with 1-based ranks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.services.retrieval.types import RetrievalHit
from app.utils.text import normalize_whitespace, sha256_text


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalHit]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[RetrievalHit]:
    """Fuse ranked lists into one deterministically ordered list.

    ``rankings`` maps a source name (``dense``, ``keyword``) to hits ordered
    best-first. A chunk that appears in both lists accumulates both
    contributions. Ties are broken by chunk id so results are reproducible.
    """

    if k < 0:
        raise ValueError("RRF k must be non-negative")

    weights = weights or {}
    fused: dict[str, RetrievalHit] = {}
    scores: dict[str, float] = {}
    matched: dict[str, set[str]] = {}

    for source, hits in rankings.items():
        weight = weights.get(source, 1.0)
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (k + rank)
            matched.setdefault(hit.chunk_id, set()).add(source)
            existing = fused.get(hit.chunk_id)
            # Prefer a dense payload (it carries the canonical indexed fields)
            # over a lexical reconstruction of the same chunk.
            if existing is None or (source == "dense" and "dense" not in existing.matched):
                fused[hit.chunk_id] = hit

    ordered = sorted(fused.values(), key=lambda hit: (-scores[hit.chunk_id], hit.chunk_id))
    result: list[RetrievalHit] = []
    for hit in ordered:
        updated = hit.with_score(scores[hit.chunk_id], kind="rrf")
        updated.rrf_score = scores[hit.chunk_id]
        updated.matched = set(matched[hit.chunk_id])
        result.append(updated)
    return result


def deduplicate_hits(
    hits: Sequence[RetrievalHit],
) -> tuple[list[RetrievalHit], int]:
    """Drop repeated chunks and exact duplicate text, keeping the best rank.

    ``hits`` must already be ordered best-first. Identity is ``chunk_id``;
    content identity is the SHA-256 of whitespace-normalised, case-folded text.
    Near-duplicate detection is intentionally *not* fuzzy: a false merge in a
    clinical knowledge base hides evidence.
    """

    seen_chunks: set[str] = set()
    seen_text: set[str] = set()
    kept: list[RetrievalHit] = []
    removed = 0

    for hit in hits:
        if hit.chunk_id in seen_chunks:
            removed += 1
            continue
        text_hash = sha256_text(normalize_whitespace(hit.text).casefold())
        if text_hash in seen_text:
            removed += 1
            continue
        seen_chunks.add(hit.chunk_id)
        seen_text.add(text_hash)
        kept.append(hit)

    return kept, removed
