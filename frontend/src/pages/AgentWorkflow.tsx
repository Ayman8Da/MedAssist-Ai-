import * as React from 'react'
import { api } from '@/lib/api'
import type { TraceOut, TraceListOut } from '@/types'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'

const NODES = [
  'User Question',
  'Safety Layer',
  'Planner Agent',
  'Query Decomposition',
  'Hybrid Retrieval (BM25 + Vector)',
  'Reciprocal Rank Fusion',
  'Reranker',
  'Clinical Evidence Agent',
  'Verifier Agent',
  'Safety Agent',
  'Response Generator',
  'Final Answer',
]

export function AgentWorkflow() {
  const [traces, setTraces] = React.useState<TraceListOut | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [selected, setSelected] = React.useState<TraceOut | null>(null)

  React.useEffect(() => {
    api.listTraces({ limit: 50 }).then(setTraces).catch(() => setTraces(null)).finally(() => setLoading(false))
  }, [])

  const agenticTraces = traces?.items.filter((t) => t.mode === 'agentic') ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-text">Agent Workflow</h1>
        <p className="mt-1 text-sm text-text-muted">
          Visual representation of the Agentic RAG pipeline. Live execution traces appear when the backend
          provides them.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Pipeline</CardTitle>
          <CardDescription>
            Nodes are connected in execution order. Status reflects a selected trace when available.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-2">
            {NODES.map((n, i) => (
              <React.Fragment key={n}>
                <div className="rounded-xl border border-border bg-white px-3 py-2 text-xs font-medium text-text shadow-sm dark:bg-surface">
                  {n}
                </div>
                {i < NODES.length - 1 && <span className="text-text-muted">→</span>}
              </React.Fragment>
            ))}
          </div>
          <p className="mt-4 text-xs text-text-muted">
            The agentic orchestration (LangGraph) is a Phase 3 deliverable. Until it ships, this page
            shows the planned pipeline and any traditional/hybrid traces below. No fake execution is
            simulated.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Execution traces</CardTitle>
          <CardDescription>Recent backend traces (traditional / hybrid). Agentic traces will appear in Phase 3.</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-12" />)}</div>
          ) : !traces?.items.length ? (
            <div className="rounded-xl border border-dashed border-border bg-surface-muted/40 p-8 text-center text-sm text-text-muted">
              No traces yet. Run a chat query to generate one.
            </div>
          ) : (
            <div className="space-y-2">
              {traces.items.map((t) => (
                <button
                  key={t.id}
                  onClick={() => setSelected(t)}
                  className="flex w-full items-center justify-between rounded-xl border border-border bg-white px-4 py-3 text-left text-sm hover:bg-surface-muted dark:bg-surface"
                >
                  <div className="min-w-0">
                    <div className="truncate font-medium text-text">{t.query_preview}</div>
                    <div className="text-xs text-text-muted">{t.mode} · {t.status} · {new Date(t.created_at).toLocaleString()}</div>
                  </div>
                  <Badge variant={t.status === 'completed' ? 'success' : 'danger'}>{Math.round(t.latency_ms)} ms</Badge>
                </button>
              ))}
            </div>
          )}

          {selected && (
            <div className="mt-4 rounded-xl border border-border bg-surface-muted/40 p-4">
              <div className="mb-2 flex items-center justify-between">
                <div className="text-sm font-semibold text-text">Trace {selected.id.slice(0, 8)}…</div>
                <button onClick={() => setSelected(null)} className="text-xs text-medical-600 hover:underline">Close</button>
              </div>
              <div className="space-y-1">
                {selected.steps.map((s, i) => (
                  <div key={i} className="flex items-center justify-between rounded-lg bg-white px-3 py-2 text-xs dark:bg-surface">
                    <span className="font-medium text-text">{String(s.step ?? s.name ?? 'step')}</span>
                    <span className="text-text-muted">
                      {String(s.status ?? '')} · {s.latency_ms != null ? `${Number(s.latency_ms).toFixed(1)} ms` : ''}
                      {s.retrieved != null ? ` · retrieved ${s.retrieved}` : ''}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {agenticTraces.length === 0 && (
        <Card>
          <CardContent className="p-4 text-sm text-text-muted">
            No agentic traces are available yet because the agentic pipeline (Phase 3) is not implemented.
            Traditional and hybrid traces above are real.
          </CardContent>
        </Card>
      )}
    </div>
  )
}
