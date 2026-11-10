from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers import upload_document


def test_upload_list_get_and_duplicate(client: TestClient) -> None:
    response = upload_document(
        client,
        title="Metformin renal guidance",
        publication_date="2025-01-15",
        version="2.0",
        category="nephrology",
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["duplicate"] is False
    document = body["document"]
    assert document["status"] == "indexed"
    assert document["chunk_count"] > 0
    assert document["publication_date"] == "2025-01-15"

    listing = client.get("/api/documents")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    detail = client.get(f"/api/documents/{document['id']}")
    assert detail.status_code == 200

    duplicate = upload_document(client)
    assert duplicate.status_code == 201
    assert duplicate.json()["duplicate"] is True
    assert duplicate.json()["document"]["id"] == document["id"]


def test_upload_rejects_unsupported_and_empty_files(client: TestClient) -> None:
    unsupported = upload_document(client, filename="image.png", content_type="image/png")
    assert unsupported.status_code == 422
    assert unsupported.json()["error"]["code"] == "ingestion_error"

    empty = client.post(
        "/api/documents/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
        data={"source": "test"},
    )
    assert empty.status_code == 422
    assert empty.json()["error"]["code"] == "ingestion_error"


def test_delete_document(client: TestClient) -> None:
    document_id = upload_document(client).json()["document"]["id"]

    deleted = client.delete(f"/api/documents/{document_id}")
    assert deleted.status_code == 204

    assert client.get(f"/api/documents/{document_id}").status_code == 404
    assert client.get("/api/documents").json()["total"] == 0


def test_reindex_document(client: TestClient) -> None:
    document_id = upload_document(client).json()["document"]["id"]

    response = client.post("/api/documents/reindex", json={"document_ids": [document_id]})
    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["document_id"] == document_id
    assert item["status"] == "indexed"
    assert item["chunk_count"] > 0


def test_metadata_filters_limit_retrieval(client: TestClient, fake_llm) -> None:
    upload_document(client, source="renal-guideline", category="nephrology")
    upload_document(
        client,
        text="DIABETES AND RESPIRATORY INFECTION. Hypoxia and fever require urgent review.",
        filename="diabetes.txt",
        source="endocrine-guideline",
        category="endocrinology",
    )

    response = client.post(
        "/api/chat/traditional",
        json={
            "question": "metformin eGFR renal impairment contraindication",
            "filters": {"source": "renal-guideline"},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["retrieved_chunks"]
    assert all(chunk["source"] == "renal-guideline" for chunk in body["retrieved_chunks"])


def test_health_reports_component_checks(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    names = {check["name"] for check in body["checks"]}
    assert {"database", "vector_store", "embeddings", "llm"} <= names
    assert body["status"] in {"ok", "degraded"}
    assert body["phase"] == 4


def test_benchmark_run_starts(client: TestClient) -> None:
    response = client.post("/api/benchmark/run")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "run_id" in body
    assert body["status"] in {"pending", "running", "completed"}
