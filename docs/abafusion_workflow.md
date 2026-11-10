# MedAssist AI — Agentic Workflow Blueprint (for Aba Fusion)

This document describes the workflow that mirrors the MedAssist AI application,
so it can be recreated on the Aba Fusion agentic platform
(`https://stg-agentic.abafusion.ai/flows`).

> Note: the Aba Fusion login page requires authentication, so this blueprint is
> provided as a portable specification. Import/adapt it to the platform's node
> schema as needed.

## High-level flows

### Flow A — Traditional RAG (baseline)
1. Input: `question` (string), `top_k` (int, default 4).
2. Embed query (Ollama `bge-m3`, 1024-dim).
3. Vector search (Qdrant, cosine, top_k, metadata filters).
4. Build bounded context (max 12000 chars) with numbered headers.
5. If no context → return insufficient-evidence answer (no LLM call).
6. LLM generate (system prompt: strict medical, cite `[n]`).
7. Mark citations used.
8. Persist trace.
9. Output: answer, citations, retrieved_chunks, timings, usage, warnings.

### Flow B — Hybrid RAG (Phase 2)
1. Input: `question`, `top_k`, optional filters.
2. (Optional) version scoping: drop superseded docs sharing a `document_key`.
3. Parallel:
   - Dense: embed query → Qdrant search (`DENSE_CANDIDATE_K=30`).
   - Lexical: tokenize (lowercase, stopwords, Snowball stem) → BM25 search (`KEYWORD_CANDIDATE_K=30`).
4. Reciprocal Rank Fusion (`RRF_K=60`, 1-based ranks).
5. Dedupe by `chunk_id` and SHA-256 of normalized, case-folded text.
6. Rerank first `RERANK_CANDIDATES=20` with BGE cross-encoder (`BAAI/bge-reranker-v2-m3-int8`).
   - On timeout/error/disabled → keep RRF order, `reranker_used=false` + reason.
7. Truncate to `top_k`.
8. Generate (shared grounded generator).
9. Output: answer + diagnostics (per-stage latency/candidates).

### Flow C — Agentic RAG (Phase 3, LangGraph)
Nodes and edges (see JSON below):
1. `safety_input` — flag urgent symptoms (chest pain, dyspnea, etc.) → warnings.
2. `planner` — LLM emits 1–3 sub-queries (JSON). Retry-aware.
3. `retrieve` — Hybrid retrieval per sub-query, merge by chunk_id (keep best score).
4. `verify` — evidence sufficiency (≥1 unique doc). If fail and attempts < 3 → `expand` → back to `planner`.
5. `evidence` — summarize retrieved evidence (doc count, chunk count).
6. `safety_agent` — evidence-age warning (>5 years), no-evidence warning.
7. `generate` — grounded answer with citations.
8. Output: answer, citations, diagnostics, warnings, trace.

Max correction attempts: 3.

## Node specification (JSON)

```json
{
  "name": "MedAssist Agentic RAG",
  "version": "1.0",
  "entry": "safety_input",
  "nodes": [
    {"id": "safety_input", "type": "safety", "description": "Flag urgent symptoms; initialize warnings."},
    {"id": "planner", "type": "llm", "system_prompt": "Return JSON {sub_queries:[...]}. 1-3 focused medical sub-queries.", "max_queries": 3},
    {"id": "retrieve", "type": "hybrid_retrieval", "top_k": 4, "dense_k": 30, "keyword_k": 30, "rrf_k": 60, "rerank_candidates": 20},
    {"id": "verify", "type": "verifier", "min_unique_docs": 1, "max_attempts": 3},
    {"id": "expand", "type": "transform", "description": "Mark retry; planner broadens queries on next pass."},
    {"id": "evidence", "type": "transform", "description": "Summarize evidence (chunks, documents)."},
    {"id": "safety_agent", "type": "safety", "description": "Evidence age >5y warning; no-evidence warning."},
    {"id": "generate", "type": "llm", "system_prompt": "Strict medical, cite [n], only supplied evidence.", "max_tokens": 600}
  ],
  "edges": [
    {"from": "START", "to": "safety_input"},
    {"from": "safety_input", "to": "planner"},
    {"from": "planner", "to": "retrieve"},
    {"from": "retrieve", "to": "verify"},
    {"from": "verify", "condition": "passed", "to": "evidence"},
    {"from": "verify", "condition": "retry", "to": "expand"},
    {"from": "expand", "to": "planner"},
    {"from": "evidence", "to": "safety_agent"},
    {"from": "safety_agent", "to": "generate"},
    {"from": "generate", "to": "END"}
  ],
  "state": {
    "question": "string",
    "sub_queries": "string[]",
    "retrieval_attempts": "int",
    "hits": "RetrievalHit[]",
    "verification": {"passed": "bool", "score": "float", "reasons": "string[]"},
    "safety_warnings": "string[]",
    "answer": "string",
    "citations": "Citation[]"
  },
  "tools": {
    "embed": {"provider": "ollama", "model": "bge-m3", "dim": 1024},
    "llm": {"provider": "ollama", "model": "phi3:latest", "temperature": 0.1, "max_tokens": 600},
    "vector_store": {"type": "qdrant", "collection": "medassist_chunks"},
    "reranker": {"provider": "local", "model": "BAAI/bge-reranker-v2-m3-int8"}
  }
}
```

## Configuration knobs (from .env)
- `TOP_K=4`, `MAX_CONTEXT_CHARS=6000`, `LLM_MAX_TOKENS=600`, `LLM_TIMEOUT_SECONDS=600`
- `RRF_K=60`, `DENSE_CANDIDATE_K=30`, `KEYWORD_CANDIDATE_K=30`, `RERANK_CANDIDATES=20`
- `LATEST_VERSIONS_ONLY=false` (set true to drop superseded docs sharing a `document_key`)

## Safety
Every output includes: "MedAssist AI provides evidence-based medical information for
clinical decision support. It does not replace professional medical judgment,
diagnosis, or treatment." Urgent symptoms trigger an escalation warning.
