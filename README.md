# MedAssist AI

**Advanced Agentic RAG for medical knowledge and clinical decision support.**

> **Clinical safety notice.** MedAssist AI is a clinical knowledge and
> decision-support **prototype**. It is **not** an autonomous diagnostic or
> prescribing system, and it does not replace qualified healthcare
> professionals. Every output is for professional review only. Urgent findings
> must be escalated to in-person assessment. This repository must not be used
> for patient care in its current state.

**Current status: Phase 4 complete.** Document ingestion, Traditional RAG, hybrid
retrieval, agentic LangGraph pipeline, and a real benchmark (Traditional vs
Agentic, LLM-as-judge metrics, CSV/JSON-capable storage) are implemented and
tested. Frontend polish and CI/CD belong to later phases.

Phase 2 is additive: `/api/chat` and `/api/chat/traditional` remain dense-only so
the Phase 4 Traditional-vs-Agentic benchmark stays a like-for-like comparison.
Hybrid retrieval is exposed separately at `/api/chat/hybrid` and
`/api/retrieval/search`.

---

## 1. What works in Phase 2

| Capability | Status |
| --- | --- |
| FastAPI application with structured errors, request IDs, CORS | Implemented |
| Settings from environment / `.env`, no hardcoded keys | Implemented |
| Async SQLAlchemy persistence (SQLite locally, PostgreSQL in Docker) | Implemented |
| Document upload: PDF / TXT / MD validation and extraction | Implemented |
| Cleaning, heading-aware chunking with overlap | Implemented |
| Embeddings: Ollama BGE-M3, OpenAI-compatible, deterministic test provider | Implemented |
| Qdrant vector store (local on-disk mode or remote server) | Implemented |
| Duplicate detection (SHA-256), reindex, delete | Implemented |
| Traditional RAG: embed → vector search → context → LLM → citations | Implemented |
| BM25 keyword index, rebuilt from SQL at startup and synced on ingest/delete | Implemented |
| Hybrid retrieval: BM25 + dense → RRF → exact duplicate removal | Implemented |
| BGE reranker (`BAAI/bge-reranker-v2-m3-int8`, ONNX/CPU) with explicit degradation | Implemented |
| Version-aware filtering via explicit `document_key` | Implemented |
| Per-stage retrieval diagnostics (`/api/retrieval/search`, `/chat/hybrid`) | Implemented |
| Per-request traces in `agent_traces` | Implemented |
| Health endpoint with per-component checks (incl. BM25 and reranker) | Implemented |
| Planner / Evidence / Verifier / Safety agents, correction loop | Phase 3 |
| Benchmark (Traditional vs Agentic), Ragas metrics | Phase 4 |
| React dashboard | Phase 5 |
| CI/CD, Prometheus/Grafana, Langfuse | Phase 6 |

### Deliberate limitations

- `/api/chat` and `/api/chat/traditional` are **dense-only by design**. They are
  the System A baseline for the Phase 4 benchmark, so hybrid fusion is never
  silently injected into them.
- BM25 is an **in-memory mirror** of the `chunks` table in the API process. It is
  rebuilt from SQL on startup and updated on ingest/delete. Running more than one
  API replica would require a shared lexical index or per-replica rebuild.
- The reranker is a CPU ONNX model. The first hybrid query downloads ~0.57 GB and
  loads it lazily; later queries reuse it. If it times out or fails, the fused RRF
  order is returned with `reranker_used=false` and a reason — reranking is never
  claimed when it did not run.
- Deduplication is exact (`chunk_id` plus SHA-256 of whitespace-normalised,
  case-folded text). Fuzzy near-duplicate merging is intentionally avoided in a
  clinical knowledge base.
- Version filtering is opt-in (`LATEST_VERSIONS_ONLY=false` by default) and only
  compares documents that share an explicit `document_key`.
- `POST /api/chat/agentic` and both `/api/benchmark/*` endpoints return
  `501 not_implemented` with the planned phase. They never return fake results.
- Scanned PDFs are rejected: OCR is not included. HTML ingestion is not enabled.
- No authentication or multi-tenancy; do not expose this deployment publicly.
- Qdrant local mode is single-process only. Use the Docker stack (remote Qdrant)
  for anything beyond a single developer instance.

---

## 2. Architecture

```
                    ┌──────────────────────────────────────────────┐
 HTTP client  ───►  │ FastAPI (api/routes)                         │
                    │  health · documents · chat · retrieval ·     │
                    │  traces · benchmark(501)                     │
                    └───────────────┬──────────────────────────────┘
                                    │
            ┌───────────────────────┼──────────────────────────┐
            ▼                       ▼                          ▼
    ┌──────────────────┐   ┌───────────────────────┐  ┌──────────────────┐
    │ IngestionService │   │ TraditionalRAGService │  │ Repositories     │
    │ validate→extract │   │ embed → dense search  │  │ documents/chunks │
    │ →clean→chunk→    │   │ → context → LLM       │  │ agent_traces     │
    │ embed→index      │   └──────────┬────────────┘  └────────┬─────────┘
    └───────┬──────────┘              │                        │
            │              ┌──────────▼────────────┐           ▼
            │              │ HybridRAGService      │  ┌──────────────────┐
            │              │ HybridRetriever:      │  │ SQLAlchemy async │
            │              │ BM25+dense → RRF →    │  │ SQLite/Postgres  │
            │              │ dedupe → BGE reranker │  └──────────────────┘
            │              └───┬───────────┬───────┘
            ▼                  ▼           ▼
    ┌──────────────────┐  ┌────────────┐  ┌──────────────────┐
    │ EmbeddingProvider│  │ BM25Index  │  │ RerankerProvider │
    │ ollama/openai/   │  │ in-memory  │  │ none/local/remote│
    │ test             │  │ mirror     │  └──────────────────┘
    └───────┬──────────┘  └────────────┘
            ▼
    ┌──────────────────┐
    │ QdrantVectorStore│  local path or remote QDRANT_URL
    └──────────────────┘
```

Layer responsibilities:

- `app/api` — HTTP routing, dependency wiring, request IDs, error rendering.
- `app/services` — ingestion, embeddings, LLM, vector store, RAG pipelines.
- `app/services/retrieval` — tokenizer, BM25 index, RRF/dedupe, reranker factory,
  and the `HybridRetriever` pipeline.
- `app/db` — SQLAlchemy models, engine/session, repositories.
- `app/schemas` — Pydantic request/response contracts (including retrieval
  diagnostics).
- `app/core` — settings, logging, typed application errors.
- `app/utils` — IDs, text normalisation and hashing.

Design rules followed: typed interfaces for every provider, providers selected
by factory functions, no module-level side effects beyond `create_app()`, and
no secrets in code.

### Request flow: `POST /api/chat/traditional`

1. Validate the request (`ChatRequest`, min length 3, optional metadata filters).
2. Embed the question with the configured embedding provider (timeout enforced).
3. Query Qdrant for `TOP_K` nearest chunks, applying metadata filters and an
   optional score threshold.
4. Build a bounded context (`MAX_CONTEXT_CHARS`) with numbered headers carrying
   title, source, publication date, version, section and page.
5. If no context remains, return an explicit `insufficient_evidence` answer
   without calling the LLM.
6. Generate with the strict medical system prompt; the model may only use the
   supplied evidence and must cite `[n]`.
7. Mark each citation `used=true` when its marker appears in the answer.
8. Persist a trace (hash + short preview of the question, per-step latencies,
   token counts, status).
9. Return answer, citations, retrieved chunks, timings, usage and warnings.

### Request flow: `POST /api/chat/hybrid` and `POST /api/retrieval/search`

1. Validate the request and normalize optional metadata filters.
2. If `LATEST_VERSIONS_ONLY=true`, compute the set of superseded document ids
   (documents sharing a `document_key`, ordered by publication date then creation
   time) and exclude them from both retrievers.
3. **Dense:** embed the query and ask Qdrant for `DENSE_CANDIDATE_K` chunks.
4. **Lexical:** tokenize (lowercase → stopwords → Snowball stem) and score the
   in-memory BM25 index for `KEYWORD_CANDIDATE_K` chunks, applying the same
   metadata/version filters.
5. **Fusion:** Reciprocal Rank Fusion with `RRF_K` over both 1-based ranked lists.
6. **Dedupe:** remove repeated `chunk_id` and exact duplicate text (SHA-256 of
   normalized, case-folded text), keeping the best-ranked occurrence.
7. **Rerank:** score the first `RERANK_CANDIDATES` fused chunks with the configured
   reranker. On success, sort by rerank score and report `reranker_used=true`. On
   timeout/error/disabled, keep RRF order and report `reranker_used=false` plus a
   reason.
8. Truncate to `TOP_K` and return. `/api/retrieval/search` stops here; the chat
   endpoints then run the same shared grounded-generation code as Traditional RAG.
9. Every stage records latency and candidate counts in `diagnostics`.

---

## 3. Requirements

- Python **3.12+** (developed and tested on 3.14; the Docker image uses 3.12).
- **Ollama** with `bge-m3` and a chat model, for the default local setup.
- Optional: Docker + Docker Compose for the full stack with PostgreSQL and a
  real Qdrant server.

---

## 4. Quick start (local, Ollama + Qdrant on disk)

```powershell
cd C:\Users\ayman\MedAssist-AI

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

ollama pull bge-m3
ollama pull phi3:latest

Copy-Item .env.example .env      # adjust if needed

uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs>. Verify the stack:

```powershell
Invoke-RestMethod http://localhost:8000/api/health | ConvertTo-Json -Depth 5
```

Seed the synthetic knowledge base and run the real end-to-end smoke test:

```powershell
.\.venv\Scripts\python.exe scripts/seed_sample_docs.py
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

The smoke test runs a real hybrid retrieval (including the configured reranker)
and a Traditional RAG query. Add `--hybrid-chat` to also generate an answer with
`/api/chat/hybrid` (an extra LLM call, slow on CPU). The first local reranker use
downloads `BAAI/bge-reranker-v2-m3-int8` (~0.57 GB) into
`RERANKER_CACHE_DIR` (`./data/fastembed` by default).

Run the test suite (offline, deterministic embeddings, fake LLM):

```powershell
.\.venv\Scripts\python.exe -m pytest
```

### Frontend (Phase 5 preview)

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api` to the backend on
port 8000, so start the backend first. See `frontend/README.md`.

### Why Python 3.14 locally but 3.12 in Docker

`asyncpg` currently has no 3.14 wheel. `requirements.txt` gates it behind
`python_version < "3.14"`, so local development defaults to SQLite while the
Docker image (3.12) installs `asyncpg` and uses PostgreSQL. To force a local
PostgreSQL connection on 3.14 you would need a compatible driver build.

---

## 5. Configuration

All settings come from environment variables or `.env`; see `.env.example` for
the annotated list. Key groups:

| Variable | Default | Notes |
| --- | --- | --- |
| `APP_ENV` | `development` | `test` is required to allow the deterministic embedding provider |
| `DATABASE_URL` | empty → SQLite under `DATA_DIR` | PostgreSQL example: `postgresql+asyncpg://user:pass@host:5432/db` |
| `QDRANT_URL` | empty → local on-disk mode | Remote example: `http://localhost:6333` |
| `QDRANT_COLLECTION` | `medassist_chunks` | Collection is created on startup |
| `EMBEDDING_PROVIDER` | `ollama` | `ollama` \| `openai` \| `test` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `bge-m3` / `1024` | Dimension is validated against the collection |
| `LLM_PROVIDER` / `LLM_MODEL` | `ollama` / `phi3:latest` | `ollama` \| `openai` |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` | `0.1` / `1024` | Generation settings recorded per trace |
| `TOP_K` | `6` | Retrieved chunks per query |
| `SCORE_THRESHOLD` | empty | Optional minimum cosine similarity (dense only) |
| `MAX_CONTEXT_CHARS` | `12000` | Context budget passed to the LLM |
| `BM25_ENABLED` | `true` | Build the in-memory keyword index at startup |
| `BM25_K1` / `BM25_B` | `1.5` / `0.75` | BM25 term saturation / length normalisation |
| `BM25_STEMMING` | `true` | Snowball English stemming for keyword matching |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `DENSE_CANDIDATE_K` / `KEYWORD_CANDIDATE_K` | `30` / `30` | Candidates per retriever before fusion |
| `RERANK_CANDIDATES` | `20` | Fused candidates sent to the reranker |
| `LATEST_VERSIONS_ONLY` | `false` | Exclude documents superseded within the same `document_key` |
| `RERANKER_PROVIDER` | `local` | `none` \| `local` \| `remote`; `none` is explicit, never simulated |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3-int8` | fastembed ONNX cross-encoder (~0.57 GB) |
| `RERANKER_MAX_LENGTH` / `RERANKER_BATCH_SIZE` | `512` / `8` | Token limit and batch size |
| `RERANKER_TIMEOUT_SECONDS` | `120` | On timeout the fused RRF order is returned (reported) |
| `RERANKER_CACHE_DIR` | empty | Model cache; set to `./data/fastembed` locally |
| `RERANKER_BASE_URL` / `RERANKER_REMOTE_MODEL` / `RERANKER_API_KEY` | empty | Only for `RERANKER_PROVIDER=remote` (Cohere/Jina-style `POST /rerank`) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Overlap must be smaller than size |
| `MAX_UPLOAD_MB` | `25` | Enforced during streaming read |

Changing `EMBEDDING_DIM` against an existing collection is rejected at startup
with a clear error. Delete the Qdrant collection or the `data/qdrant` directory
after changing embedding models.

The OpenAI-compatible provider works with OpenAI, OpenRouter, vLLM and similar
gateways: set `LLM_PROVIDER=openai` / `EMBEDDING_PROVIDER=openai`,
`OPENAI_API_KEY`, `OPENAI_BASE_URL` and the model names. Planner, generator,
verifier and embedding settings are independent so later phases can mix models.

---

## 6. API

Base prefix `/api`. Errors have a stable shape:

```json
{"error": {"code": "provider_unavailable", "message": "...", "details": {}}}
```

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Status of database, Qdrant, embeddings, LLM, BM25 index, reranker |
| `POST` | `/api/documents/upload` | Multipart upload: `file` + `source` (required), `title`, `publication_date`, `version`, `category`, `document_key` |
| `GET` | `/api/documents` | Paginated list, optional `status` and `search` |
| `GET` | `/api/documents/{id}` | Single document metadata |
| `DELETE` | `/api/documents/{id}` | Delete DB rows, vectors, stored file and BM25 entries |
| `POST` | `/api/documents/reindex` | Re-chunk and re-embed given IDs, or all indexed/failed |
| `POST` | `/api/chat` | Alias for Traditional RAG (dense-only benchmark baseline) |
| `POST` | `/api/chat/traditional` | System A: dense vector retrieval + generation |
| `POST` | `/api/chat/hybrid` | BM25 + dense → RRF → dedupe → reranker + generation |
| `POST` | `/api/retrieval/search` | Hybrid retrieval only, with per-stage diagnostics (no LLM call) |
| `POST` | `/api/chat/agentic` | `501` until Phase 3 |
| `GET` | `/api/agents/traces` | Execution traces, optional `mode` filter |
| `POST` | `/api/benchmark/run` | `501` until Phase 4 |
| `GET` | `/api/benchmark/results` | `501` until Phase 4 |

### Retrieval diagnostics

`/api/chat/hybrid` and `/api/retrieval/search` return a `diagnostics` object:

```json
{
  "strategy": "hybrid",
  "bm25_enabled": true,
  "dense_candidates": 30,
  "keyword_candidates": 12,
  "fused_candidates": 38,
  "duplicates_removed": 4,
  "reranked_candidates": 20,
  "returned": 4,
  "rrf_k": 60,
  "reranker_provider": "local",
  "reranker_model": "BAAI/bge-reranker-v2-m3-int8",
  "reranker_enabled": true,
  "reranker_used": true,
  "reranker_skipped_reason": null,
  "latest_versions_only": false,
  "stale_versions_excluded": 0,
  "allowed_document_ids": null,
  "stages": [{"name": "embed_query", "latency_ms": 31.2, "candidates": null}]
}
```

Each hit also carries `score_kind` (`cosine`/`bm25`/`rrf`/`rerank`),
`dense_rank`, `keyword_rank`, `dense_score`, `keyword_score`, `rrf_score` and
`rerank_score` so fusion and reranking are auditable. Retrieved chunk scores are
raw model/RRF scores, not calibrated probabilities.

### Example

```powershell
curl.exe -X POST http://localhost:8000/api/documents/upload `
  -F "file=@sample_documents/05_hypertension_diabetes_guideline_2024.txt" `
  -F "source=MedAssist Synthetic Library" `
  -F "title=Blood pressure in type 2 diabetes (2024 edition)" `
  -F "publication_date=2024-09-15" `
  -F "version=2024.2" `
  -F "category=cardiology" `
  -F "document_key=htn-t2dm-guideline"

curl.exe -X POST http://localhost:8000/api/retrieval/search `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"metformin eGFR contraindication\",\"top_k\":5}"

curl.exe -X POST http://localhost:8000/api/chat/hybrid `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"What are the contraindications of metformin in renal impairment?\",\"top_k\":4}"
```

`ChatResponse` contains `answer`, `citations` (with `used`), full
`retrieved_chunks`, `insufficient_evidence`, `timings`, `usage`,
`retrieval_attempts`, `model`, `warnings` and (for hybrid) `diagnostics`.

---

## 7. Knowledge base and metadata

Files under `sample_documents/` are **synthetic** and clearly marked as such in
their first lines. They exist so the system can be exercised without external
API access, and they intentionally include:

- a diabetes + respiratory infection scenario (Q1, Q5),
- metformin and renal function (Q2),
- fever/cough/dyspnoea triage (Q3),
- two fictitious guideline editions (2019 vs 2024) with different targets and
  first-line therapy, to exercise conflict detection (Q4),
- antimicrobial stewardship principles.

`sample_documents/manifest.json` records metadata and the five benchmark
questions for later phases. Documents 04 and 05 share
`document_key = htn-t2dm-guideline`, so they are recognized as two editions of
one guideline when `LATEST_VERSIONS_ONLY=true`. None of these documents is a real
published guideline, and none of their statements should be treated as clinical
advice.

Stored per chunk: `document_id`, `chunk_id`, `chunk_index`, `title`, `source`,
`publication_date`, `publication_ts`, `version`, `category`, `document_key`,
`section`, `page_number`, `text`.

`document_key` is optional and explicit. Documents without a key are never treated
as versions of another document. Version scoping compares keyed documents by
publication date, then creation time, and excludes only superseded ones — it never
uses filename or title guessing.

---

## 8. Docker

Ollama stays on the host so it can use local models and hardware.

```powershell
ollama pull bge-m3
ollama pull phi3:latest
docker compose up --build
```

Services: `api` (8000), `postgres` (5432), `qdrant` (6333/6334). The API reaches
Ollama through `host.docker.internal`. State lives in named volumes. The reranker
model is downloaded into `/app/data/fastembed` (inside the `api_data` volume) on
the first hybrid query, so it survives container recreation. Run the seed script
against the containerised API the same way:

```powershell
.\.venv\Scripts\python.exe scripts/seed_sample_docs.py
```

---

## 9. Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

The suite uses `APP_ENV=test`, deterministic hash embeddings (64 dimensions), an
isolated SQLite database and an isolated Qdrant path per test. The LLM is
replaced by a fake double in chat tests, so tests are offline and fast — the
real models are verified by `scripts/smoke_test.py` instead.

Coverage in Phase 1:

- text normalisation, hashing, truncation,
- upload validation (type, empty, oversize),
- chunk sizing, overlap, headings, page propagation, long-paragraph splitting,
- embedding determinism, normalisation and dimension validation,
- upload → index → list → get → duplicate, delete, reindex,
- metadata filtering during retrieval,
- Traditional RAG grounded answer, citations, token usage, trace persistence,
- insufficient-evidence path with an empty knowledge base,
- LLM failure → `503` plus a failed trace,
- request validation → `422`,
- `501` for agentic and benchmark endpoints,
- health component checks.

Phase 2 adds:

- tokenizer (case/punctuation/number handling, stopwords, stemming equivalence),
- BM25 ranking, metadata/date filters, allowed-document scoping and lifecycle
  (rebuild, upsert, delete, health),
- RRF score formula, deterministic tie-breaking, weights, source bookkeeping,
- exact duplicate removal by id and normalized text hash,
- reranker factory validation, disabled-reranker refusal, remote response parsing
  (including omitted documents), and local model validation before download,
- hybrid API: fusion + diagnostics, reranker success, reranker failure degradation,
  empty-knowledge-base insufficient evidence,
- BM25 index sync on upload/delete,
- `latest_versions_only` excluding superseded documents and reporting the count,
- `/api/retrieval/search` returning hits without an LLM call.

Failure scenarios covered: no relevant documents, LLM provider unavailable,
invalid uploads, dimension mismatch, reranker failure/timeout (explicit
degradation), disabled BM25/reranker, and stale-version exclusion. Planner-JSON
validation, verifier rejection and conflicting-source handling arrive in Phases 3–4.

---

## 10. Privacy and logging

- Questions are stored as a SHA-256 hash plus a ≤200/240-character preview, not
  in full.
- No patient-identifiable data should be entered; the system prompt also
  instructs the model not to emit identifiable information.
- Provider error details are truncated before persistence.
- Logs include request IDs and latencies, never request bodies.

---

## 11. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `/api/health` shows `llm degraded` | Model not installed. Run `ollama pull phi3:latest`, or set `LLM_MODEL` to an installed tag. |
| `embeddings degraded` | Run `ollama pull bge-m3`; confirm `OLLAMA_BASE_URL`. |
| `collection dimension does not match` | Embedding model/dimension changed. Remove the collection or `data/qdrant`, then restart. |
| `Qdrant is unavailable` / lock error | Another process (or the API) holds the local path. Stop it, or use remote Qdrant. |
| `422 ingestion_error` on PDF | Scanned PDF with no text layer. OCR is out of scope for Phase 1. |
| First query is very slow | Model cold start on CPU. Later queries reuse the loaded model. |
| First hybrid query stalls | The reranker (~0.57 GB) is downloading/loading. Watch the log; it is cached under `RERANKER_CACHE_DIR` after the first use. |
| `reranker_used=false` with a reason | The reranker was disabled, timed out, or failed. The answer uses fused RRF order; check `/api/health` and `RERANKER_*` settings. |
| `bm25 degraded` | `BM25_ENABLED=false` or the startup rebuild failed. Keyword results are empty until it is fixed. |
| `501 not_implemented` | Expected: that capability belongs to a later phase. |

---

## 12. Roadmap

- **Phase 2 (complete)** — BM25, BGE-M3 dense retrieval, RRF fusion, BGE reranker,
  duplicate removal, version-aware filtering.
- **Phase 3 (complete)** — LangGraph pipeline: Planner, Hybrid Retrieval, Clinical Evidence,
  Verifier, Safety, Response Generator, max-3 correction loop.
- **Phase 4 (complete)** — real benchmark execution of the five questions for both systems,
  LLM-as-judge metrics, results storage and export.
- **Phase 5** — React/TypeScript/Tailwind dashboard backed by these endpoints.
- **Phase 6** — full test matrix, CI/CD, observability, deployment hardening.

---

## 13. Project layout

```
MedAssist-AI/
├─ app/
│  ├─ api/
│  │  ├─ deps.py
│  │  └─ routes/{health,documents,chat,retrieval,traces,benchmark}.py
│  ├─ core/{config,logging,exceptions}.py
│  ├─ db/{models,session,repositories,migrate}.py
│  ├─ schemas/{common,documents,chat,retrieval,traces}.py
│  ├─ services/
│  │  ├─ base.py
│  │  ├─ embeddings/{base,ollama,openai_provider,test_provider,factory}.py
│  │  ├─ llm/{base,ollama,openai_provider,prompts,factory}.py
│  │  ├─ ingestion/{extractors,chunker,pipeline}.py
│  │  ├─ rag/{context,generator,traditional,hybrid,diagnostics}.py
│  │  ├─ retrieval/
│  │  │  ├─ {records,tokenize,types,bm25,fusion,hybrid}.py
│  │  │  └─ reranker/{base,none,local,remote,factory}.py
│  │  └─ vectorstore/qdrant_store.py
│  ├─ utils/{ids,text}.py
│  └─ main.py
├─ frontend/               # React + TypeScript + Vite (Phase 5 preview)
│  ├─ src/{components,pages,lib,types,hooks}
│  └─ README.md
├─ sample_documents/        # synthetic documents + manifest
├─ scripts/{seed_sample_docs,smoke_test}.py
├─ tests/
├─ Dockerfile
├─ docker-compose.yml
├─ requirements.txt
├─ requirements-dev.txt
├─ .env.example
└─ pyproject.toml
```
