from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.exceptions import ProviderUnavailableError
from tests.conftest import FakeLLM
from tests.helpers import DIABETES_DOC, upload_document


def test_traditional_rag_returns_grounded_answer(client: TestClient, fake_llm) -> None:
    upload_document(client, source="renal-guideline")
    upload_document(
        client, text=DIABETES_DOC, filename="diabetes.txt", source="endocrine-guideline"
    )

    response = client.post(
        "/api/chat",
        json={"question": "What are the contraindications of metformin in renal impairment?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["system"] == "traditional"
    assert body["insufficient_evidence"] is False
    assert body["retrieved_chunks"]
    assert body["citations"]
    assert body["citations"][0]["source"] == "renal-guideline"
    assert body["citations"][0]["used"] is True
    assert body["retrieval_attempts"] == 1
    assert body["timings"]["total_ms"] >= 0
    assert body["usage"]["prompt_tokens"] == 120
    assert "for_professional_review_only" in body["warnings"]


def test_traditional_rag_insufficient_evidence_without_documents(client: TestClient) -> None:
    response = client.post(
        "/api/chat/traditional",
        json={"question": "metformin renal impairment"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert body["citations"] == []
    assert body["retrieved_chunks"] == []
    assert "no_relevant_documents" in body["warnings"]


def test_chat_trace_is_persisted(client: TestClient, fake_llm) -> None:
    upload_document(client)
    assert (
        client.post("/api/chat", json={"question": "metformin eGFR contraindication"}).status_code
        == 200
    )

    traces = client.get("/api/agents/traces")
    assert traces.status_code == 200
    body = traces.json()
    assert body["total"] >= 1
    trace = body["items"][0]
    assert trace["mode"] == "traditional"
    assert trace["status"] == "completed"
    assert trace["steps"]
    assert trace["query_preview"]
    # The raw question is not persisted in full; only a short preview and hash.
    assert trace["query_sha256"]


def test_llm_unavailable_returns_503_and_failed_trace(client: TestClient, app) -> None:
    app.state.rag.llm = FakeLLM(error=ProviderUnavailableError("LLM is down"))
    upload_document(client)

    response = client.post("/api/chat", json={"question": "metformin eGFR contraindication"})
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "provider_unavailable"

    traces = client.get("/api/agents/traces").json()
    assert any(trace["status"] == "failed" for trace in traces["items"])


def test_question_validation(client: TestClient) -> None:
    response = client.post("/api/chat", json={"question": "hi"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
