const BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  if (!res.ok) {
    let body: unknown
    try {
      body = await res.json()
    } catch {
      body = { error: { code: 'http_error', message: res.statusText, details: {} } }
    }
    const err = new Error(
      (body as { error?: { message?: string } })?.error?.message ?? res.statusText,
    )
    ;(err as Error & { status?: number; body?: unknown }).status = res.status
    ;(err as Error & { status?: number; body?: unknown }).body = body
    throw err
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

export const api = {
  health: () => request<import('../types').HealthResponse>('/api/health'),

  listDocuments: (params?: { status?: string; search?: string; limit?: number; offset?: number }) => {
    const q = new URLSearchParams()
    if (params?.status) q.set('status', params.status)
    if (params?.search) q.set('search', params.search)
    if (params?.limit != null) q.set('limit', String(params.limit))
    if (params?.offset != null) q.set('offset', String(params.offset))
    const qs = q.toString()
    return request<import('../types').DocumentListOut>(`/api/documents${qs ? `?${qs}` : ''}`)
  },

  uploadDocument: async (file: File, meta: Record<string, string | undefined>) => {
    const form = new FormData()
    form.append('file', file)
    Object.entries(meta).forEach(([k, v]) => {
      if (v != null && v !== '') form.append(k, v)
    })
    const res = await fetch(`${BASE}/api/documents/upload`, { method: 'POST', body: form })
    if (!res.ok) {
      let body: unknown
      try { body = await res.json() } catch { body = null }
      throw new Error((body as { error?: { message?: string } })?.error?.message ?? res.statusText)
    }
    return (await res.json()) as import('../types').UploadResponse
  },

  deleteDocument: (id: string) =>
    request<void>(`/api/documents/${id}`, { method: 'DELETE' }),

  reindexDocuments: (document_ids?: string[]) =>
    request<import('../types').ReindexResponse>('/api/documents/reindex', {
      method: 'POST',
      body: JSON.stringify({ document_ids: document_ids ?? null }),
    }),

  chatTraditional: (payload: import('../types').ChatRequest, signal?: AbortSignal) =>
    request<import('../types').ChatResponse>('/api/chat/traditional', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    }),

  chatHybrid: (payload: import('../types').ChatRequest, signal?: AbortSignal) =>
    request<import('../types').ChatResponse>('/api/chat/hybrid', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    }),

  chatAgentic: (payload: import('../types').ChatRequest, signal?: AbortSignal) =>
    request<import('../types').ChatResponse>('/api/chat/agentic', {
      method: 'POST',
      body: JSON.stringify(payload),
      signal,
    }),

  retrievalSearch: (payload: import('../types').RetrievalSearchRequest) =>
    request<import('../types').RetrievalSearchResponse>('/api/retrieval/search', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),

  listTraces: (params?: { mode?: string; limit?: number; offset?: number }) => {
    const q = new URLSearchParams()
    if (params?.mode) q.set('mode', params.mode)
    if (params?.limit != null) q.set('limit', String(params.limit))
    if (params?.offset != null) q.set('offset', String(params.offset))
    const qs = q.toString()
    return request<import('../types').TraceListOut>(`/api/agents/traces${qs ? `?${qs}` : ''}`)
  },

  benchmarkRun: () =>
    request<import('../types').BenchmarkRunResponse>('/api/benchmark/run', { method: 'POST', body: JSON.stringify({}) }),

  benchmarkResults: () => request<import('../types').BenchmarkResultsOut>('/api/benchmark/results'),
}
