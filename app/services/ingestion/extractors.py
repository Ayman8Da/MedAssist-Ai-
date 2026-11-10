from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import IngestionError

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/octet-stream",
}


class TextPage:
    __slots__ = ("text", "page_number")

    def __init__(self, text: str, page_number: int | None = None) -> None:
        self.text = text
        self.page_number = page_number


def validate_upload(
    *, filename: str, content_type: str | None, size_bytes: int, max_upload_mb: int
) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise IngestionError(
            "Unsupported file type. Allowed types: PDF, TXT, MD.",
            details={"filename": filename, "allowed": sorted(ALLOWED_SUFFIXES)},
        )
    if size_bytes <= 0:
        raise IngestionError("The uploaded file is empty")
    max_bytes = max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise IngestionError(
            f"The uploaded file exceeds the {max_upload_mb} MB limit",
            details={"size_bytes": size_bytes, "max_bytes": max_bytes},
        )
    if content_type and content_type.split(";")[0].strip() not in ALLOWED_CONTENT_TYPES:
        raise IngestionError(
            "Unsupported content type",
            details={"content_type": content_type},
        )
    return suffix


def extract_pages(path: Path, suffix: str) -> list[TextPage]:
    if suffix == ".pdf":
        return _extract_pdf(path)
    return [TextPage(text=_read_text(path), page_number=None)]


def _extract_pdf(path: Path) -> list[TextPage]:
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise IngestionError("Encrypted PDFs are not supported") from exc

        pages: list[TextPage] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - keep processing other pages
                text = ""
            pages.append(TextPage(text=text, page_number=page_number))
    except IngestionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise IngestionError("Failed to parse the PDF document") from exc

    if not any(page.text.strip() for page in pages):
        raise IngestionError(
            "No extractable text found. Scanned PDFs require OCR, which is not supported in Phase 1."
        )
    return pages


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise IngestionError("Failed to read the uploaded file") from exc

    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 always decodes
        raise IngestionError("Unable to decode the text file")

    if not text.strip():
        raise IngestionError("The text file contains no content")
    return text


def clean_page(text: str) -> str:
    from app.utils.text import normalize_whitespace

    text = normalize_whitespace(text)
    # Drop page-number-only lines that add noise to chunks.
    lines = [line for line in text.split("\n") if not re.fullmatch(r"\s*(page\s*)?\d{1,4}\s*", line, re.I)]
    return "\n".join(lines).strip()
