// TypeScript mirrors of the backend Pydantic schemas (app/schemas/*).
// Keep these in sync with the FastAPI contracts.

export type HealthCheck = {
  name: string
  status: 'ok' | 'degraded' | 'error'
  detail?: string | null
  latency_ms?: number | null
}

export type HealthResponse = {
  status: 'ok' | 'degraded'
  app: string
  version: string
  phase: number
  checks: HealthCheck[]
}

export type ChatFilters = {
  source?: string | null
  category?: string | null
  version?: string | null
  document_ids?: string[] | null
  published_after?: string | null
  published_before?: string | null
}

export type ChatRequest = {
  question: string
  top_k?: number | null
  filters?: ChatFilters | null
}

export type Citation = {
  index: number
  chunk_id: string
  document_id: string
  title: string
  source: string
  publication_date?: string | null
  version?: string | null
  category?: string | null
  section?: string | null
  page_number?: number | null
  score: number
  score_kind?: string | null
  used: boolean
}

export type RetrievedChunk = Citation & { text: string }

export type TimingBreakdown = {
  embedding_ms: number
  retrieval_ms: number
  generation_ms: number
  total_ms: number
}

export type TokenUsage = {
  prompt_tokens?: number | null
  completion_tokens?: number | null
  total_tokens?: number | null
}

export type RetrievalStageOut = {
  name: string
  latency_ms: number
  candidates: number | null
}

export type RetrievalDiagnostics = {
  strategy: 'traditional_dense' | 'hybrid'
  bm25_enabled: boolean
  dense_candidates: number
  keyword_candidates: number
  fused_candidates: number
  duplicates_removed: number
  reranked_candidates: number
  returned: number
  rrf_k: number | null
  reranker_provider: string
  reranker_model: string
  reranker_enabled: boolean
  reranker_used: boolean
  reranker_skipped_reason: string | null
  latest_versions_only: boolean
  stale_versions_excluded: number
  allowed_document_ids: number | null
  stages: RetrievalStageOut[]
}

export type ChatResponse = {
  request_id: string
  system: 'traditional' | 'hybrid' | 'agentic'
  answer: string
  citations: Citation[]
  retrieved_chunks: RetrievedChunk[]
  insufficient_evidence: boolean
  timings: TimingBreakdown
  usage: TokenUsage
  retrieval_attempts: number
  model: string | null
  warnings: string[]
  diagnostics: RetrievalDiagnostics | null
}

export type DocumentOut = {
  id: string
  title: string
  source: string
  publication_date?: string | null
  version?: string | null
  category?: string | null
  document_key?: string | null
  filename: string
  content_type: string
  status: string
  error_message?: string | null
  chunk_count: number
  created_at: string
  updated_at: string
  indexed_at?: string | null
}

export type DocumentListOut = {
  items: DocumentOut[]
  total: number
  limit: number
  offset: number
}

export type UploadResponse = {
  document: DocumentOut
  duplicate: boolean
  message: string
}

export type ReindexItem = {
  document_id: string
  status: string
  chunk_count: number
  error?: string | null
}

export type ReindexResponse = {
  items: ReindexItem[]
}

export type RetrievalSearchHit = {
  index: number
  chunk_id: string
  document_id: string
  title: string
  source: string
  publication_date?: string | null
  version?: string | null
  category?: string | null
  section?: string | null
  page_number?: number | null
  text: string
  score: number
  score_kind: string
  dense_score?: number | null
  keyword_score?: number | null
  rrf_score?: number | null
  rerank_score?: number | null
  dense_rank?: number | null
  keyword_rank?: number | null
  matched: string[]
}

export type RetrievalSearchRequest = {
  query: string
  top_k?: number | null
  filters?: ChatFilters | null
}

export type RetrievalSearchResponse = {
  query: string
  hits: RetrievalSearchHit[]
  total_ms: number
  diagnostics: RetrievalDiagnostics
}

export type TraceOut = {
  id: string
  request_id?: string | null
  mode: string
  query_sha256: string
  query_preview: string
  steps: Record<string, unknown>[]
  status: string
  retrieval_attempts: number
  latency_ms: number
  prompt_tokens?: number | null
  completion_tokens?: number | null
  error?: string | null
  created_at: string
}

export type TraceListOut = {
  items: TraceOut[]
  total: number
  limit: number
  offset: number
}

export type BenchmarkResultsOut = {
  run: {
    id: string
    status: string
    created_at: string
    updated_at: string
    total_questions: number
    completed_questions: number
    error?: string | null
  }
  summary?: Record<string, unknown> | null
  results?: Record<string, unknown>[] | null
}

export type BenchmarkRunResponse = {
  run_id: string
  status: string
}

export type ApiError = {
  error: { code: string; message: string; details: Record<string, unknown> }
}
