"""Lexical tokenisation tuned for clinical text.

Design goals, in order:

1. Never lose a term a clinician typed. The tokenizer only lower-cases,
   normalises Unicode, splits on non-alphanumeric characters, removes a small
   English stop-word list and applies a deterministic Snowball stem so
   ``contraindication``/``contraindicated`` match.
2. Keep it transparent. No external NLP service, no hidden synonyms.
3. Be identical for indexing and querying. The same :class:`Tokenizer`
   instance is used by both code paths.

Numbers are kept (``30``, ``45``, ``eGFR`` values matter clinically); single
characters and pure stop-words are dropped.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence

_WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "after",
        "again",
        "against",
        "all",
        "also",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


class Tokenizer:
    """Deterministic medical-text tokenizer.

    ``stemming`` is on by default and uses the Snowball English stemmer from
    ``py-rust-stemmers`` (a fastembed dependency). If that optional package is
    missing the tokenizer still works, it simply falls back to unstemmed
    tokens rather than failing at runtime.
    """

    def __init__(self, *, stopwords: Iterable[str] | None = None, stemming: bool = True) -> None:
        self._stopwords = frozenset(stopwords) if stopwords is not None else STOPWORDS
        self._stemmer = self._load_stemmer() if stemming else None

    @staticmethod
    def _load_stemmer() -> object | None:
        try:
            from py_rust_stemmers import SnowballStemmer

            return SnowballStemmer("english")
        except Exception:  # noqa: BLE001 - optional acceleration only
            return None

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        normalized = unicodedata.normalize("NFKC", text).lower()
        tokens = [
            token
            for token in _WORD_RE.findall(normalized)
            if len(token) > 1 and token not in self._stopwords
        ]
        if self._stemmer is None or not tokens:
            return tokens
        return list(self._stemmer.stem_words(tokens))

    def tokenize_many(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.tokenize(text) for text in texts]


_default_tokenizer = Tokenizer()


def tokenize(text: str) -> list[str]:
    """Tokenize using the shared default instance."""

    return _default_tokenizer.tokenize(text)
