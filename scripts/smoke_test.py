"""End-to-end smoke test against a RUNNING MedAssist AI API.

This exercises the real path: health check, document ingestion with the
configured embedding provider, the hybrid retrieval pipeline (BM25 + dense ->
RRF -> reranker), and a Traditional RAG query through the configured LLM. It
prints actual measured values and fails if the pipeline does not produce
grounded citations.

Nothing is stubbed or simulated. If Ollama, the reranker or Qdrant is
unavailable the script fails with a non-zero exit code (or, for a *configured but
temporarily failing* reranker, reports the explicit degradation instead of
pretending reranking happened).

Usage (from the project root, with the API running):

    .\\.venv\\Scripts\\python.exe scripts/smoke_test.py --seed
    .\\.venv\\Scripts\\python.exe scripts/smoke_test.py --question "metformin and eGFR"
    .\\.venv\\Scripts\\python.exe scripts/smoke_test.py --seed --hybrid-chat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "sample_documents"
MANIFEST = SAMPLE_DIR / "manifest.json"

DEFAULT_QUESTION = (
    "What are the contraindications and important precautions for metformin "
    "in a patient with impaired renal function?"
)


def seed(client: httpx.Client) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["documents"]:
        path = SAMPLE_DIR / entry["file"]
        data = {
            "source": entry["source"],
            "title": entry["title"],
            "publication_date": entry["publication_date"],
            "version": entry["version"],
            "category": entry["category"],
        }
        if entry.get("document_key"):
            data["document_key"] = entry["document_key"]

        response = client.post(
            "/api/documents/upload",
            files={"file": (path.name, path.read_bytes(), "text/plain")},
            data=data,
        )
        response.raise_for_status()
        body = response.json()
        state = "exists" if body["duplicate"] else "indexed"
        print(f"  {state:<8} {entry['file']} ({body['document']['chunk_count']} chunks)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--hybrid-chat",
        action="store_true",
        help="Also run /api/chat/hybrid (this calls the LLM a second time; slow on CPU).",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Upload the synthetic sample documents before querying.",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    api = args.base_url.rstrip("/")
    failures: list[str] = []

    with httpx.Client(base_url=api, timeout=args.timeout) as client:
        print(f"== Health ({api}) ==")
        try:
            health = client.get("/api/health")
            health.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"FAIL: API unreachable: {exc}", file=sys.stderr)
            return 1
        health_body = health.json()
        print(f"status={health_body['status']} phase={health_body['phase']} version={health_body['version']}")
        for check in health_body["checks"]:
            detail = f" - {check['detail']}" if check.get("detail") else ""
            print(f"  {check['name']:<12} {check['status']:<9} {check.get('latency_ms', '')}{detail}")
            if check["status"] == "error":
                failures.append(f"health:{check['name']}")

        if args.seed:
            print("\n== Seeding synthetic documents ==")
            seed(client)

        print("\n== Documents ==")
        listing = client.get("/api/documents", params={"limit": 200})
        listing.raise_for_status()
        listing_body = listing.json()
        indexed = [item for item in listing_body["items"] if item["status"] == "indexed"]
        print(f"total={listing_body['total']} indexed={len(indexed)}")
        for item in listing_body["items"][:10]:
            print(f"  {item['id']}  {item['status']:<9} chunks={item['chunk_count']:<3} {item['title'][:70]}")
        if not indexed:
            print("FAIL: the knowledge base is empty; run with --seed", file=sys.stderr)
            return 1

        print("\n== Hybrid retrieval (BM25 + dense -> RRF -> rerank) ==")
        search_payload: dict[str, object] = {"query": args.question, "top_k": 5}
        search = client.post("/api/retrieval/search", json=search_payload)
        if search.status_code != 200:
            print(f"FAIL: HTTP {search.status_code}: {search.text[:500]}", file=sys.stderr)
            return 1
        search_body = search.json()
        diagnostics = search_body["diagnostics"]
        print(
            f"returned={diagnostics['returned']} dense={diagnostics['dense_candidates']} "
            f"keyword={diagnostics['keyword_candidates']} fused={diagnostics['fused_candidates']} "
            f"duplicates_removed={diagnostics['duplicates_removed']} rrf_k={diagnostics['rrf_k']}"
        )
        print(
            f"reranker: provider={diagnostics['reranker_provider']} "
            f"model={diagnostics['reranker_model']} enabled={diagnostics['reranker_enabled']} "
            f"used={diagnostics['reranker_used']} "
            f"reranked={diagnostics['reranked_candidates']} "
            f"skipped_reason={diagnostics['reranker_skipped_reason']}"
        )
        print(
            f"versioning: latest_only={diagnostics['latest_versions_only']} "
            f"stale_excluded={diagnostics['stale_versions_excluded']} "
            f"allowed_documents={diagnostics['allowed_document_ids']}"
        )
        for stage in diagnostics["stages"]:
            print(
                f"  stage {stage['name']:<13} {stage['latency_ms']:>9.2f} ms "
                f"candidates={stage['candidates']}"
            )
        print(f"total={search_body['total_ms']} ms")

        print("\n-- Fused / reranked hits --")
        for hit in search_body["hits"]:
            print(
                f"  [{hit['index']}] kind={hit['score_kind']:<6} score={hit['score']:.6f} "
                f"dense_rank={hit['dense_rank']} keyword_rank={hit['keyword_rank']} "
                f"{hit['title'][:50]} | {hit['source']} | v{hit['version']}"
            )

        if not search_body["hits"]:
            failures.append("hybrid_retrieval_returned_no_hits")
        if (
            diagnostics["reranker_enabled"]
            and not diagnostics["reranker_used"]
        ):
            print(
                "WARNING: reranker is enabled but did not run: "
                f"{diagnostics['reranker_skipped_reason']}"
            )
        if not diagnostics["bm25_enabled"]:
            failures.append("bm25_disabled")

        print("\n== Traditional RAG query ==")
        print(f"question: {args.question}")
        payload: dict[str, object] = {"question": args.question}
        if args.top_k is not None:
            payload["top_k"] = args.top_k
        response = client.post("/api/chat/traditional", json=payload)
        if response.status_code != 200:
            print(f"FAIL: HTTP {response.status_code}: {response.text[:500]}", file=sys.stderr)
            return 1
        body = response.json()

        timings = body["timings"]
        usage = body["usage"]
        print(
            f"insufficient_evidence={body['insufficient_evidence']} "
            f"citations={len(body['citations'])} "
            f"retrieved={len(body['retrieved_chunks'])}"
        )
        print(
            "timings_ms: "
            f"embedding={timings['embedding_ms']} retrieval={timings['retrieval_ms']} "
            f"generation={timings['generation_ms']} total={timings['total_ms']}"
        )
        print(
            "tokens: "
            f"prompt={usage['prompt_tokens']} completion={usage['completion_tokens']} "
            f"total={usage['total_tokens']} model={body['model']}"
        )
        print(f"warnings: {', '.join(body['warnings']) or 'none'}")

        print("\n-- Citations --")
        for citation in body["citations"]:
            used = "used" if citation["used"] else "unused"
            print(
                f"  [{citation['index']}] {used:<6} score={citation['score']:.4f} "
                f"{citation['title'][:60]} | section={citation['section']} | page={citation['page_number']}"
            )

        print("\n-- Answer --")
        print(body["answer"])

        if body["insufficient_evidence"]:
            failures.append("insufficient_evidence")
        if not body["citations"]:
            failures.append("no_citations")
        if not any(citation["used"] for citation in body["citations"]):
            failures.append("no_citation_referenced_in_answer")
        if not body["answer"].strip():
            failures.append("empty_answer")

        if args.hybrid_chat:
            print("\n== Hybrid RAG query ==")
            hybrid_response = client.post("/api/chat/hybrid", json=payload)
            if hybrid_response.status_code != 200:
                print(
                    f"FAIL: HTTP {hybrid_response.status_code}: "
                    f"{hybrid_response.text[:500]}",
                    file=sys.stderr,
                )
                return 1
            hybrid_body = hybrid_response.json()
            hybrid_diagnostics = hybrid_body["diagnostics"]
            print(
                f"insufficient_evidence={hybrid_body['insufficient_evidence']} "
                f"citations={len(hybrid_body['citations'])} "
                f"reranker_used={hybrid_diagnostics['reranker_used']} "
                f"score_kind={hybrid_body['citations'][0]['score_kind'] if hybrid_body['citations'] else 'n/a'}"
            )
            print(
                "timings_ms: "
                f"embedding={hybrid_body['timings']['embedding_ms']} "
                f"retrieval={hybrid_body['timings']['retrieval_ms']} "
                f"generation={hybrid_body['timings']['generation_ms']} "
                f"total={hybrid_body['timings']['total_ms']}"
            )
            print("\n-- Hybrid answer --")
            print(hybrid_body["answer"])
            if hybrid_body["insufficient_evidence"]:
                failures.append("hybrid_insufficient_evidence")
            if not hybrid_body["citations"]:
                failures.append("hybrid_no_citations")

    print("\n== Result ==")
    if failures:
        print(f"FAILED: {', '.join(sorted(set(failures)))}")
        return 1
    print("PASSED: real ingestion + retrieval + generation produced a cited answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
