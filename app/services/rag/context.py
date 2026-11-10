from __future__ import annotations

from dataclasses import dataclass

from app.services.vectorstore.qdrant_store import SearchHit


@dataclass(slots=True)
class ContextBlock:
    index: int
    hit: SearchHit
    rendered: str


def build_context(hits: list[SearchHit], max_chars: int) -> list[ContextBlock]:
    blocks: list[ContextBlock] = []
    total = 0
    for hit in hits:
        payload = hit.payload
        title = payload.get("title") or "Untitled document"
        source = payload.get("source") or "unknown source"
        publication_date = payload.get("publication_date") or "n/a"
        version = payload.get("version") or "n/a"
        section = payload.get("section") or "n/a"
        page_number = payload.get("page_number")
        page = f"p.{page_number}" if page_number is not None else "n/a"
        header = (
            f"[{len(blocks) + 1}] {title} | source: {source} | date: {publication_date} | "
            f"version: {version} | section: {section} | page: {page}"
        )
        text = hit.text.strip()
        if not text:
            continue
        block = f"{header}\n{text}"
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(block) > remaining:
            if remaining < 200:
                break
            block = block[:remaining].rstrip()
        blocks.append(ContextBlock(index=len(blocks) + 1, hit=hit, rendered=block))
        total += len(block) + 2
    return blocks


def render_context(blocks: list[ContextBlock]) -> str:
    return "\n\n".join(block.rendered for block in blocks)
