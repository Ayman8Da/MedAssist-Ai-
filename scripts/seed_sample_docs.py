"""Seed the MedAssist AI knowledge base with the synthetic sample documents.

The script talks to a RUNNING API over HTTP. This is deliberate: Qdrant local
mode holds an on-disk lock, so a second process must not write to the vector
store directly while the server is running.

Usage (from the project root, with the API running):

    .\\.venv\\Scripts\\python.exe scripts/seed_sample_docs.py
    .\\.venv\\Scripts\\python.exe scripts/seed_sample_docs.py --base-url http://localhost:8000
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running MedAssist AI API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (embedding can be slow on CPU).",
    )
    args = parser.parse_args()

    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST}", file=sys.stderr)
        return 1

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    api = args.base_url.rstrip("/")

    created = 0
    duplicates = 0
    failures = 0

    with httpx.Client(base_url=api, timeout=args.timeout) as client:
        try:
            health = client.get("/api/health")
            health.raise_for_status()
        except httpx.HTTPError as exc:
            print(f"ERROR: API is not reachable at {api}: {exc}", file=sys.stderr)
            print("Start it first: uvicorn app.main:app --port 8000", file=sys.stderr)
            return 1

        health_body = health.json()
        print(f"API status: {health_body['status']} (phase {health_body['phase']})")
        for check in health_body["checks"]:
            detail = f" - {check['detail']}" if check.get("detail") else ""
            print(f"  {check['name']:<12} {check['status']}{detail}")

        for entry in documents:
            path = SAMPLE_DIR / entry["file"]
            if not path.exists():
                print(f"  SKIP  missing file: {path}")
                failures += 1
                continue

            data = {
                "source": entry["source"],
                "title": entry["title"],
                "publication_date": entry["publication_date"],
                "version": entry["version"],
                "category": entry["category"],
            }
            if entry.get("document_key"):
                data["document_key"] = entry["document_key"]
            try:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": (path.name, path.read_bytes(), "text/plain")},
                    data=data,
                )
            except httpx.HTTPError as exc:
                print(f"  FAIL  {entry['file']}: {exc}")
                failures += 1
                continue

            if response.status_code != 201:
                print(f"  FAIL  {entry['file']}: HTTP {response.status_code} {response.text[:300]}")
                failures += 1
                continue

            body = response.json()
            document = body["document"]
            if body["duplicate"]:
                duplicates += 1
                print(
                    f"  EXIST {entry['file']}: already indexed as {document['id']} "
                    f"({document['chunk_count']} chunks)"
                )
            else:
                created += 1
                print(
                    f"  OK    {entry['file']}: {document['chunk_count']} chunks, "
                    f"status={document['status']}"
                )

    print(
        f"\nDone. created={created} duplicates={duplicates} failures={failures} "
        f"total_documents={len(documents)}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
