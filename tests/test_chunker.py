from __future__ import annotations

from app.services.ingestion.chunker import chunk_pages
from app.services.ingestion.extractors import TextPage


def test_chunking_respects_size_pages_and_indices() -> None:
    body = ("Clinical assessment paragraph with several words. " * 30 + "\n\n") * 4
    chunks = chunk_pages(
        [TextPage(text="SECTION HEADING\n\n" + body, page_number=3)],
        chunk_size=300,
        overlap=50,
    )

    assert chunks
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert chunks[0].section == "SECTION HEADING"
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.char_count == len(chunk.text) for chunk in chunks)


def test_chunking_produces_overlap() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 20
    chunks = chunk_pages([TextPage(text=text, page_number=1)], chunk_size=200, overlap=40)

    assert len(chunks) >= 2
    first_words = chunks[0].text.split()[-4:]
    second_text = chunks[1].text
    assert any(word in second_text for word in first_words)


def test_chunking_handles_empty_input() -> None:
    assert chunk_pages([TextPage(text="   \n\n  ", page_number=1)]) == []


def test_chunking_splits_long_unbroken_paragraph() -> None:
    text = "x" * 2500
    chunks = chunk_pages([TextPage(text=text, page_number=2)], chunk_size=500, overlap=50)
    assert len(chunks) >= 5
    assert all(len(chunk.text) <= 500 for chunk in chunks)
