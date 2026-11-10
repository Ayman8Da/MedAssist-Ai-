from __future__ import annotations

from app.services.retrieval.tokenize import Tokenizer, tokenize


def test_tokenizer_normalises_case_and_punctuation() -> None:
    tokens = tokenize("eGFR < 30 mL/min/1.73m2; BETA-blockers")
    assert "egfr" in tokens
    assert "beta" in tokens
    assert "30" in tokens
    assert all(token == token.lower() for token in tokens)


def test_tokenizer_removes_stopwords_and_single_characters() -> None:
    tokens = tokenize("the patient is on a drug and has no fever")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "a" not in tokens
    assert {"patient", "drug", "fever"} <= set(tokens)


def test_stemming_matches_word_forms() -> None:
    tokenizer = Tokenizer(stemming=True)
    assert tokenizer.tokenize("contraindications") == tokenizer.tokenize("contraindicated")
    assert tokenizer.tokenize("guidelines") == tokenizer.tokenize("guideline")


def test_tokenizer_handles_empty_and_numeric_text() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("123") == ["123"]
