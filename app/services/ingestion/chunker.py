from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.ingestion.extractors import TextPage

_HEADING_NUMBERED = re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S")
_UPPER_HEADING = re.compile(r"^[A-Z][A-Z0-9 ,:/\-()]{7,}$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class ChunkDraft:
    index: int
    text: str
    page_number: int | None
    section: str | None
    char_count: int


@dataclass(slots=True)
class _Segment:
    text: str
    page_number: int | None
    section: str | None


def _is_heading(paragraph: str) -> bool:
    text = paragraph.strip()
    if not text or len(text) > 120:
        return False
    if text.endswith(":") and len(text) <= 80:
        return True
    if _HEADING_NUMBERED.match(text):
        return True
    if _UPPER_HEADING.match(text) and text.upper() == text:
        return True
    return False


def _split_long_text(text: str, limit: int) -> list[str]:
    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > limit:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            current = ""
            for word in sentence.split():
                candidate = word if not current else f"{current} {word}"
                if len(candidate) <= limit:
                    current = candidate
                else:
                    if current:
                        pieces.append(current)
                    while len(word) > limit:
                        pieces.append(word[:limit])
                        word = word[limit:]
                    current = word
            if current:
                pieces.append(current)
            continue
        candidate = sentence if not buffer else f"{buffer} {sentence}"
        if len(candidate) <= limit:
            buffer = candidate
        else:
            if buffer:
                pieces.append(buffer)
            buffer = sentence
    if buffer:
        pieces.append(buffer)
    return pieces


def _iter_segments(pages: list[TextPage], chunk_size: int):
    current_section: str | None = None
    for page in pages:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", page.text) if part.strip()]
        for paragraph in paragraphs:
            if _is_heading(paragraph):
                current_section = paragraph.strip().rstrip(":").strip() or current_section
                yield _Segment(text=paragraph.strip(), page_number=page.page_number, section=current_section)
                continue
            if len(paragraph) <= chunk_size:
                yield _Segment(text=paragraph, page_number=page.page_number, section=current_section)
            else:
                for piece in _split_long_text(paragraph, chunk_size):
                    yield _Segment(text=piece, page_number=page.page_number, section=current_section)


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap:
        first_space = tail.find(" ")
        if first_space != -1:
            tail = tail[first_space + 1 :]
    return tail.strip()


def chunk_pages(
    pages: list[TextPage],
    *,
    chunk_size: int = 900,
    overlap: int = 150,
    min_chars: int = 40,
) -> list[ChunkDraft]:
    """Split cleaned pages into overlapping, paragraph-aware chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[ChunkDraft] = []
    buffer = ""
    buffer_page: int | None = None
    buffer_section: str | None = None

    def flush() -> None:
        nonlocal buffer, buffer_page, buffer_section
        text = buffer.strip()
        if text:
            chunks.append(
                ChunkDraft(
                    index=len(chunks),
                    text=text,
                    page_number=buffer_page,
                    section=buffer_section,
                    char_count=len(text),
                )
            )
        buffer = ""
        buffer_page = None
        buffer_section = None

    for segment in _iter_segments(pages, chunk_size):
        if not buffer:
            buffer = segment.text
            buffer_page = segment.page_number
            buffer_section = segment.section
            continue

        candidate = f"{buffer}\n\n{segment.text}"
        if len(candidate) <= chunk_size:
            buffer = candidate
            if buffer_section is None:
                buffer_section = segment.section
            continue

        flush()
        carry = _overlap_tail(chunks[-1].text, overlap) if chunks else ""
        if carry and len(carry) + 2 + len(segment.text) <= chunk_size:
            buffer = f"{carry}\n\n{segment.text}"
            buffer_page = segment.page_number
            buffer_section = segment.section or chunks[-1].section
        else:
            buffer = segment.text
            buffer_page = segment.page_number
            buffer_section = segment.section

    flush()

    if len(chunks) >= 2 and len(chunks[-1].text) < min_chars:
        last = chunks.pop()
        previous = chunks[-1]
        if len(previous.text) + 2 + len(last.text) <= chunk_size:
            previous.text = f"{previous.text}\n\n{last.text}"
            previous.char_count = len(previous.text)
        else:
            chunks.append(last)

    for position, chunk in enumerate(chunks):
        chunk.index = position
    return chunks
