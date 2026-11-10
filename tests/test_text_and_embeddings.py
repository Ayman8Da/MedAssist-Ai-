from __future__ import annotations

import pytest

from app.core.exceptions import IngestionError, ProviderUnavailableError
from app.services.embeddings.test_provider import DeterministicTestEmbeddingProvider
from app.services.ingestion.extractors import validate_upload
from app.utils.text import normalize_whitespace, sha256_text, truncate


def test_normalize_whitespace_and_hyphenation() -> None:
    raw = "Metformin is\r\ncleared by the kid-\nneys.\n\n\n\nNext   line."
    cleaned = normalize_whitespace(raw)
    assert "kidneys" in cleaned
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned


def test_sha256_and_truncate() -> None:
    assert sha256_text("abc") == sha256_text("abc")
    assert sha256_text("abc") != sha256_text("abd")
    assert truncate("abcdef", 4) == "abc…"


def test_validate_upload_rejects_unsupported_types() -> None:
    with pytest.raises(IngestionError):
        validate_upload(
            filename="scan.png", content_type="image/png", size_bytes=10, max_upload_mb=5
        )


def test_validate_upload_rejects_empty_and_oversized_files() -> None:
    with pytest.raises(IngestionError):
        validate_upload(filename="a.txt", content_type="text/plain", size_bytes=0, max_upload_mb=5)
    with pytest.raises(IngestionError):
        validate_upload(
            filename="a.txt",
            content_type="text/plain",
            size_bytes=6 * 1024 * 1024,
            max_upload_mb=5,
        )


async def test_deterministic_embeddings_are_stable_and_normalized() -> None:
    provider = DeterministicTestEmbeddingProvider(model="test", dim=64)
    vectors = await provider.embed_documents(
        ["metformin renal impairment", "metformin renal impairment", "sunshine cooking"]
    )
    assert len(vectors[0]) == 64
    assert vectors[0] == vectors[1]
    norm = sum(value * value for value in vectors[0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6
    similarity = sum(a * b for a, b in zip(vectors[0], vectors[2], strict=True))
    assert similarity < 0.99


async def test_embedding_dimension_mismatch_is_rejected() -> None:
    provider = DeterministicTestEmbeddingProvider(model="test", dim=8)
    with pytest.raises(ProviderUnavailableError):
        provider._validate([[0.1, 0.2, 0.3]], expected=1)
