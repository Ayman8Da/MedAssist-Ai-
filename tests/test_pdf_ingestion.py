from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import IngestionError
from app.services.ingestion.extractors import extract_pages


def _build_pdf(objects: list[bytes]) -> bytes:
    """Serialise PDF objects with a valid xref table.

    Written by hand so the tests need no PDF-writing dependency.
    """
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_position = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode()
    return bytes(out)


def make_pdf_bytes(text: str) -> bytes:
    """Build a minimal, valid single-page PDF with one line of text."""
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1")
    return _build_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
            ),
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
    )


def make_image_only_pdf_bytes() -> bytes:
    """Build a valid PDF whose single page has no text content stream."""
    return _build_pdf(
        [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        ]
    )


def test_pdf_text_extraction(tmp_path) -> None:
    path = tmp_path / "sample.pdf"
    path.write_bytes(make_pdf_bytes("Metformin is contraindicated when eGFR is below 30."))

    pages = extract_pages(path, ".pdf")

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "contraindicated" in pages[0].text


def test_pdf_without_text_layer_is_rejected(tmp_path) -> None:
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_image_only_pdf_bytes())

    with pytest.raises(IngestionError) as excinfo:
        extract_pages(path, ".pdf")

    message = excinfo.value.message
    assert "OCR" in message or "No extractable text" in message


def test_pdf_upload_indexes_through_api(client: TestClient, fake_llm) -> None:
    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "renal.pdf",
                make_pdf_bytes("Metformin contraindicated eGFR below 30 renal impairment."),
                "application/pdf",
            )
        },
        data={"source": "synthetic-pdf", "title": "Synthetic renal PDF"},
    )
    assert response.status_code == 201, response.text
    document = response.json()["document"]
    assert document["status"] == "indexed"
    assert document["content_type"] == "application/pdf"
    assert document["chunk_count"] >= 1

    answer = client.post(
        "/api/chat/traditional", json={"question": "metformin eGFR renal contraindication"}
    )
    assert answer.status_code == 200, answer.text
    chunks = answer.json()["retrieved_chunks"]
    assert chunks
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["source"] == "synthetic-pdf"
