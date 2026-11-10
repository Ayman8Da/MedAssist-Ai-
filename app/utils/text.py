from __future__ import annotations

import hashlib
import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\f\v]+")
_HYPHENATED_LINEBREAK = re.compile(r"(\w)-\n(\w)")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub(" ", text)
    text = _HYPHENATED_LINEBREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = _MULTI_NEWLINES.sub("\n\n", text)
    return text.strip()


def truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def word_count(text: str) -> int:
    return len(text.split())
