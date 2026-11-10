from __future__ import annotations

from collections.abc import Sequence

from fastapi.testclient import TestClient

from app.core.exceptions import ProviderUnavailableError
from app.main import create_app
from app.services.base import ProviderHealth
from app.services.retrieval.reranker.base import RerankerProvider, RerankResult
from tests.helpers import upload_document


class FixedReranker(RerankerProvider):
    name = "fixed-test"
    enabled = True

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(model="fixed-reranker")
        self.fail = fail
        self.calls = 0

    async def rerank(self, query: str, documents: Sequence[str]) -> RerankResult:
        self.calls += 1
        if self.fail:
            raise ProviderUnavailableError("reranker exploded")
        return RerankResult(
            scores=[float(len(documents) - index) for index in range(len(documents))],
            provider=self.name,
            model=self.model,
        )

    async def health(self) -> ProviderHealth:
        return ProviderHealth(ok=True, provider=self.name, model=self.model)


def test_hybrid_chat_reports_fusion_and_disabled_reranker(
    client: TestClient, fake_llm
) -> None:
    upload_document(client)

    response = client.post(
        "/api/chat/hybrid",
        json={"question": "What are the contraindications of metformin in renal impairment?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["system"] == "hybrid"
    assert body["insufficient_evidence"] is False
    assert body["citations"]
    assert "reranker_disabled" in body["warnings"]

    diagnostics = body["diagnostics"]
    assert diagnostics["strategy"] == "hybrid"
    assert diagnostics["bm25_enabled"] is True
    assert diagnostics["keyword_candidates"] >= 1
    assert diagnostics["dense_candidates"] >= 1
    assert diagnostics["fused_candidates"] >= 1
    assert diagnostics["reranker_provider"] == "none"
    assert diagnostics["reranker_enabled"] is False
    assert diagnostics["reranker_used"] is False
    assert diagnostics["reranker_skipped_reason"] == "disabled_by_configuration"
    assert diagnostics["rrf_k"] == 60
    assert [stage["name"] for stage in diagnostics["stages"]] == [
        "embed_query",
        "dense_search",
        "bm25_search",
        "rrf_fusion",
        "rerank",
    ]
    assert body["citations"][0]["score_kind"] == "rrf"


def test_traditional_chat_has_no_hybrid_diagnostics(client: TestClient, fake_llm) -> None:
    upload_document(client)
    response = client.post("/api/chat/traditional", json={"question": "metformin renal"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["system"] == "traditional"
    assert body["diagnostics"] is None


def test_retrieval_search_returns_hits_without_generation(client: TestClient) -> None:
    upload_document(client)
    response = client.post(
        "/api/retrieval/search",
        json={"query": "metformin eGFR contraindication", "top_k": 5},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["hits"]
    first = body["hits"][0]
    assert first["text"]
    assert first["score_kind"] == "rrf"
    assert set(first["matched"]) <= {"dense", "keyword"}
    assert body["diagnostics"]["reranker_used"] is False
    assert body["total_ms"] >= 0


def test_hybrid_chat_uses_reranker_when_available(client: TestClient, app, fake_llm) -> None:
    upload_document(client)
    app.state.retriever.reranker = FixedReranker()

    response = client.post("/api/chat/hybrid", json={"question": "metformin renal"})
    assert response.status_code == 200, response.text
    diagnostics = response.json()["diagnostics"]

    assert diagnostics["reranker_used"] is True
    assert diagnostics["reranker_skipped_reason"] is None
    assert diagnostics["reranked_candidates"] >= 1
    assert diagnostics["reranker_provider"] == "fixed-test"
    assert all(chunk["score_kind"] == "rerank" for chunk in response.json()["citations"])


def test_hybrid_chat_degrades_when_reranker_fails(client: TestClient, app, fake_llm) -> None:
    upload_document(client)
    reranker = FixedReranker(fail=True)
    app.state.retriever.reranker = reranker

    response = client.post("/api/chat/hybrid", json={"question": "metformin renal"})
    assert response.status_code == 200, response.text
    body = response.json()
    diagnostics = body["diagnostics"]

    assert reranker.calls == 1
    assert diagnostics["reranker_enabled"] is True
    assert diagnostics["reranker_used"] is False
    assert "reranker exploded" in diagnostics["reranker_skipped_reason"]
    assert "reranker_skipped" in body["warnings"]
    assert body["citations"]
    assert all(chunk["score_kind"] == "rrf" for chunk in body["citations"])


def test_keyword_index_tracks_upload_and_delete(client: TestClient, app) -> None:
    bm25 = app.state.bm25
    assert bm25 is not None
    before = bm25.size

    response = upload_document(client)
    assert response.status_code == 201, response.text
    document_id = response.json()["document"]["id"]
    assert bm25.size > before

    deleted = client.delete(f"/api/documents/{document_id}")
    assert deleted.status_code == 204, deleted.text
    assert bm25.size == before


def test_latest_versions_only_filters_superseded_documents(test_settings) -> None:
    settings = test_settings.model_copy(update={"latest_versions_only": True})
    app = create_app(settings)
    with TestClient(app) as client:
        v1 = upload_document(
            client,
            text=(
                "METFORMIN RENAL GUIDELINE 2020. Metformin is contraindicated when "
                "eGFR is below 30 mL/min/1.73m2."
            ),
            filename="metformin-2020.txt",
            source="renal-guideline",
            version="2020",
            publication_date="2020-01-01",
            document_key="metformin-renal-guideline",
        )
        v2 = upload_document(
            client,
            text=(
                "METFORMIN RENAL GUIDELINE 2024. Review the metformin dose when eGFR "
                "is below 45 and stop when eGFR is below 30."
            ),
            filename="metformin-2024.txt",
            source="renal-guideline",
            version="2024",
            publication_date="2024-01-01",
            document_key="metformin-renal-guideline",
        )
        assert v1.status_code == 201, v1.text
        assert v2.status_code == 201, v2.text
        latest_id = v2.json()["document"]["id"]

        response = client.post(
            "/api/retrieval/search", json={"query": "metformin eGFR", "top_k": 10}
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["hits"]
    assert {hit["document_id"] for hit in body["hits"]} == {latest_id}
    assert body["diagnostics"]["latest_versions_only"] is True
    assert body["diagnostics"]["stale_versions_excluded"] == 1


def test_retrieval_search_validation(client: TestClient) -> None:
    response = client.post("/api/retrieval/search", json={"query": "hi"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_agentic_chat_returns_grounded_answer(client: TestClient, fake_llm) -> None:
    upload_document(client)

    response = client.post(
        "/api/chat/agentic",
        json={"question": "What are the contraindications of metformin in renal impairment?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["system"] == "agentic"
    assert body["insufficient_evidence"] is False
    assert body["citations"]
    assert body["diagnostics"] is not None
    assert body["diagnostics"]["strategy"] == "hybrid"
    assert "for_professional_review_only" in body["warnings"]


def test_empty_knowledge_base_reports_insufficient_evidence(client: TestClient, fake_llm) -> None:
    response = client.post("/api/chat/hybrid", json={"question": "metformin renal"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["insufficient_evidence"] is True
    assert "no_relevant_documents" in body["warnings"]
    assert body["diagnostics"]["returned"] == 0
