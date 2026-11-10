from __future__ import annotations

import pytest

from app.services.retrieval.fusion import deduplicate_hits, reciprocal_rank_fusion
from app.services.retrieval.types import RetrievalHit


def _hit(chunk_id: str, text: str = "evidence") -> RetrievalHit:
    return RetrievalHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        text=text,
        payload={"title": "t", "source": "s"},
        score=0.0,
    )


def test_rrf_scores_follow_formula_and_order() -> None:
    fused = reciprocal_rank_fusion(
        {
            "dense": [_hit("a"), _hit("b"), _hit("c")],
            "keyword": [_hit("c"), _hit("a")],
        },
        k=60,
    )

    assert [hit.chunk_id for hit in fused] == ["a", "c", "b"]
    scores = {hit.chunk_id: hit.rrf_score for hit in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)


def test_rrf_weights_change_ranking() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": [_hit("a")], "keyword": [_hit("b")]},
        k=0,
        weights={"dense": 1.0, "keyword": 2.0},
    )
    assert [hit.chunk_id for hit in fused] == ["b", "a"]


def test_rrf_marks_sources_and_tie_breaks_deterministically() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": [_hit("b"), _hit("a")], "keyword": [_hit("a"), _hit("b")]},
        k=60,
    )
    # Equal fused scores -> chunk id decides.
    assert [hit.chunk_id for hit in fused] == ["a", "b"]
    assert fused[0].matched == {"dense", "keyword"}


def test_deduplicate_removes_identical_text_and_keeps_first() -> None:
    hits = [
        _hit("a", "Metformin renal  impairment."),
        _hit("b", "metformin renal impairment."),
        _hit("c", "different evidence"),
    ]
    kept, removed = deduplicate_hits(hits)
    assert [hit.chunk_id for hit in kept] == ["a", "c"]
    assert removed == 1


def test_deduplicate_removes_repeated_chunk_id() -> None:
    kept, removed = deduplicate_hits([_hit("a", "one"), _hit("a", "two")])
    assert len(kept) == 1
    assert removed == 1
