from __future__ import annotations

from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=12)
pdf.add_page()
W = pdf.w - pdf.l_margin - pdf.r_margin

def title(text: str) -> None:
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(W, 10, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0)

def heading(text: str) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(W, 7, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

def body(text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(W, 5, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

def bullet(text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(W, 5, "- " + text, new_x="LMARGIN", new_y="NEXT")

title("MedAssist AI")
pdf.set_font("Helvetica", "", 10.5)
pdf.cell(W, 6, "Advanced Agentic RAG for Medical Knowledge & Clinical Decision Support", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)

heading("Overview")
body(
    "MedAssist AI is a clinical knowledge and decision-support platform. Healthcare professionals ask medical "
    "questions and receive grounded, cited answers retrieved from an indexed knowledge base of trusted documents. "
    "It is a prototype and is not a diagnostic or prescribing system."
)

heading("Architecture")
body(
    "Backend: Python 3.12+, FastAPI, SQLAlchemy (SQLite/PostgreSQL), Qdrant (vectors), BM25, Ollama (bge-m3 "
    "embeddings, phi3 LLM). Frontend: React + TypeScript + Vite + Tailwind (frontend/). Three RAG modes share "
    "one grounded generator so results stay comparable."
)

heading("RAG Modes")
for name, desc in [
    ("Traditional RAG", "Dense vector search only. System A baseline for benchmarking."),
    ("Hybrid RAG", "BM25 + dense -> Reciprocal Rank Fusion -> dedupe -> BGE reranker (ONNX/CPU). Best precision."),
    ("Agentic RAG", "LangGraph pipeline: safety input, planner (sub-queries), hybrid retrieval, verifier, correction loop (max 3), clinical evidence, safety agent, response generator."),
]:
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(W, 5, name, new_x="LMARGIN", new_y="NEXT")
    body(desc)

heading("Key Features")
for f in [
    "Document ingestion: PDF/TXT/MD upload, validation, cleaning, heading-aware chunking, embedding, Qdrant indexing; duplicate detection, reindex, delete.",
    "Hybrid retrieval diagnostics: per-stage latency/candidate counts, reranker status, version filtering.",
    "Version-aware filtering via explicit document_key (superseded editions excluded when enabled).",
    "Execution traces persisted per request (steps, latencies, tokens, status).",
    "Health endpoint: database, Qdrant, embeddings, LLM, BM25 index, reranker.",
    "Frontend: Dashboard (live KPIs + charts), Medical Assistant (3 modes), Agent Workflow, Knowledge Base, Benchmark Center, Observability, Settings.",
]:
    bullet(f)

heading("Implementation Phases")
for p in [
    "Phase 1 - Ingestion + Traditional RAG (complete).",
    "Phase 2 - Hybrid retrieval + BGE reranker + version filtering (complete).",
    "Phase 3 - Agentic LangGraph pipeline (complete).",
    "Phase 4 - Benchmark (Traditional vs Agentic), LLM-as-judge metrics, results storage (complete).",
    "Phase 5 - Frontend (React/TS/Vite/Tailwind) (in progress).",
    "Phase 6 - CI/CD, observability hardening (planned).",
]:
    bullet(p)

heading("How to Use")
for s in [
    "1. Start backend: .venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000",
    "2. Start frontend: cd frontend && npm run dev  (http://localhost:5173)",
    "3. Seed docs: .venv\\Scripts\\python.exe scripts/seed_sample_docs.py",
    "4. Open Medical Assistant, choose a mode, ask a clinical question.",
    "5. Inspect retrieval only: POST /api/retrieval/search with {query, top_k}.",
    "6. Run benchmark: POST /api/benchmark/run; poll /api/benchmark/results.",
]:
    body(s)

heading("Safety")
body(
    "MedAssist AI provides evidence-based medical information for clinical decision support. It does not replace "
    "professional medical judgment, diagnosis, or treatment. Urgent symptoms trigger escalation warnings; "
    "insufficient evidence returns an explicit insufficient-evidence answer."
)

heading("Example Questions")
for q in [
    "What are the contraindications of metformin in renal impairment?",
    "Compare blood pressure targets for adults with type 2 diabetes (2019 vs 2024).",
    "What information is needed to assess fever, cough, and shortness of breath?",
    "Antimicrobial stewardship principles for respiratory infections.",
]:
    bullet(q)

heading("API Endpoints (base /api)")
for e in [
    "GET /health - component status",
    "POST /documents/upload - multipart (file + source, ...)",
    "GET /documents - paginated list",
    "DELETE /documents/{id}",
    "POST /documents/reindex",
    "POST /chat/traditional - dense RAG",
    "POST /chat/hybrid - BM25+dense -> RRF -> reranker",
    "POST /chat/agentic - LangGraph agentic pipeline",
    "POST /retrieval/search - retrieval only + diagnostics",
    "GET /agents/traces - execution traces",
    "POST /benchmark/run / GET /benchmark/results",
]:
    bullet(e)

heading("Configuration Highlights")
body(
    "TOP_K=4, MAX_CONTEXT_CHARS=6000, LLM_MAX_TOKENS=600, LLM_TIMEOUT_SECONDS=600. "
    "RRF_K=60, DENSE_CANDIDATE_K=30, KEYWORD_CANDIDATE_K=30, RERANK_CANDIDATES=20. "
    "LATEST_VERSIONS_ONLY=false. RERANKER_PROVIDER=local (none|local|remote). "
    "EMBEDDING_PROVIDER=ollama (bge-m3, 1024-dim). LLM_PROVIDER=ollama (phi3:latest)."
)

heading("Troubleshooting")
for t in [
    "First query is slow: model cold start on CPU; later queries reuse the loaded model.",
    "collection dimension does not match: embedding model changed; remove data/qdrant and restart.",
    "Qdrant lock error: another process holds the path; stop it or use remote Qdrant.",
    "reranker_used=false: reranker disabled/timed out/failed; answer uses fused RRF order.",
    "501 not_implemented: capability belongs to a later phase.",
]:
    bullet(t)

heading("Project Layout")
body(
    "app/ (api, core, db, schemas, services, utils), frontend/ (React/TS/Vite), "
    "sample_documents/, scripts/, tests/, docker-compose.yml, requirements*.txt, .env.example."
)

pdf.output("MedAssist-AI-Documentation.pdf")
print("written")

