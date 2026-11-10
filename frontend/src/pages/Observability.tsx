import * as React from 'react'
import { api } from '@/lib/api'
import type { TraceOut, TraceListOut } from '@/types'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

export function Observability() {
  const [traces, setTraces] = React.useState<TraceListOut | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [mode, setMode] = React.useState('')

  const load = React.useCallback(async () => {
    setLoading(true)
    try {
      setTraces(await api.listTraces({ mode: mode || undefined, limit: 100 }))
    } finally {
      setLoading(false)
    }
  }, [mode])

  React.useEffect(() => { load() }, [load])

  const chartData = (traces?.items ?? []).map((t) => ({
    name: t.id.slice(0, 6),
    latency: Math.round(t.latency_ms),
    mode: t.mode,
  }))

  const totalTokens = (traces?.items ?? []).reduce((s, t) => s + (t.prompt_tokens ?? 0) + (t.completion_tokens ?? 0), 0)
  const failed = (traces?.items ?? []).filter((t) => t.status !== 'completed').length

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-text">Observability</h1>
          <p className="mt-1 text-sm text-text-muted">Execution traces, latency, and token consumption.</p>
        </div>
        <select
          value={mode}
          onChange={(e) => setMode(e.target.value)}
          className="h-10 rounded-[var(--radius-control)] border border-border bg-white px-3 text-sm dark:bg-surface"
        >
          <option value="">All modes</option>
          <option value="traditional">Traditional</option>
          <option value="hybrid">Hybrid</option>
          <option value="agentic">Agentic</option>
        </select>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Kpi label="Traces" value={String(traces?.total ?? 0)} />
        <Kpi label="Total tokens" value={String(totalTokens)} />
        <Kpi label="Failed executions" value={String(failed)} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Latency by trace</CardTitle>
          <CardDescription>Milliseconds per request</CardDescription>
        </CardHeader>
        <CardContent className="h-72">
          {loading ? <Skeleton className="h-full" /> : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#dde3ec" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="latency" fill="#2f81f7" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Recent traces</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-12" />)}</div>
          ) : (
            <div className="space-y-2">
              {(traces?.items ?? []).map((t: TraceOut) => (
                <div key={t.id} className="flex items-center justify-between rounded-xl border border-border bg-white px-4 py-3 text-sm dark:bg-surface">
                  <div className="min-w-0">
                    <div className="truncate font-medium text-text">{t.query_preview}</div>
                    <div className="text-xs text-text-muted">
                      {t.mode} · {new Date(t.created_at).toLocaleString()} · {t.prompt_tokens ?? 0}+{t.completion_tokens ?? 0} tokens
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant={t.status === 'completed' ? 'success' : 'danger'}>{t.status}</Badge>
                    <span className="text-xs text-text-muted">{Math.round(t.latency_ms)} ms</span>
                  </div>
                </div>
              ))}
              {!traces?.items.length && <div className="py-6 text-center text-sm text-text-muted">No traces.</div>}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="p-5">
        <div className="text-xs font-medium uppercase tracking-wide text-text-muted">{label}</div>
        <div className="mt-2 text-3xl font-semibold text-text">{value}</div>
      </CardContent>
    </Card>
  )
}
