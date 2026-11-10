import * as React from 'react'
import { Activity, Database, FileText, Gauge, Timer, TrendingUp, RefreshCw } from 'lucide-react'
import { api } from '@/lib/api'
import type { HealthResponse, DocumentListOut, TraceListOut, BenchmarkResultsOut } from '@/types'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, PieChart, Pie, Cell, Legend,
} from 'recharts'

const COLORS = ['#2f81f7', '#14b8a6', '#f59e0b', '#ef4444']

export function Dashboard() {
  const [health, setHealth] = React.useState<HealthResponse | null>(null)
  const [docs, setDocs] = React.useState<DocumentListOut | null>(null)
  const [traces, setTraces] = React.useState<TraceListOut | null>(null)
  const [bench, setBench] = React.useState<BenchmarkResultsOut | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)

  const load = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [h, d, t, b] = await Promise.allSettled([
        api.health(),
        api.listDocuments({ limit: 200 }),
        api.listTraces({ limit: 200 }),
        api.benchmarkResults(),
      ])
      if (h.status === 'fulfilled') setHealth(h.value)
      if (d.status === 'fulfilled') setDocs(d.value)
      if (t.status === 'fulfilled') setTraces(t.value)
      if (b.status === 'fulfilled') setBench(b.value)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  React.useEffect(() => { load() }, [load])

  const indexedChunks = docs?.items.reduce((s, d) => s + d.chunk_count, 0) ?? 0
  const indexedDocs = docs?.items.filter((d) => d.status === 'indexed').length ?? 0
  const totalQuestions = traces?.total ?? 0
  const completedTraces = traces?.items.filter((t) => t.status === 'completed').length ?? 0
  const failedTraces = traces?.items.filter((t) => t.status !== 'completed').length ?? 0
  const agenticTraces = traces?.items.filter((t) => t.mode === 'agentic') ?? []
  const agenticSuccess = agenticTraces.filter((t) => t.status === 'completed').length
  const agenticRate = agenticTraces.length ? Math.round((agenticSuccess / agenticTraces.length) * 100) : null
  const avgLatency = traces?.items.length
    ? Math.round(traces.items.reduce((s, t) => s + t.latency_ms, 0) / traces.items.length)
    : null

  // Questions over time
  const byDate: Record<string, number> = {}
  traces?.items.forEach((t) => {
    const day = new Date(t.created_at).toISOString().slice(0, 10)
    byDate[day] = (byDate[day] ?? 0) + 1
  })
  const questionsOverTime = Object.entries(byDate)
    .sort(([a], [b]) => (a < b ? -1 : 1))
    .map(([date, count]) => ({ date: date.slice(5), count }))

  // Mode usage
  const modeCounts: Record<string, number> = {}
  traces?.items.forEach((t) => { modeCounts[t.mode] = (modeCounts[t.mode] ?? 0) + 1 })
  const modeUsage = Object.entries(modeCounts).map(([mode, count]) => ({ mode, count }))

  // Latency by mode
  const latencyByMode: Record<string, { sum: number; n: number }> = {}
  traces?.items.forEach((t) => {
    if (!latencyByMode[t.mode]) latencyByMode[t.mode] = { sum: 0, n: 0 }
    latencyByMode[t.mode].sum += t.latency_ms
    latencyByMode[t.mode].n += 1
  })
  const latencyChart = Object.entries(latencyByMode).map(([mode, v]) => ({
    mode, avg: Math.round(v.sum / v.n),
  }))

  // Verification outcomes (trace status)
  const statusCounts: Record<string, number> = {}
  traces?.items.forEach((t) => { statusCounts[t.status] = (statusCounts[t.status] ?? 0) + 1 })
  const statusChart = Object.entries(statusCounts).map(([status, count]) => ({ status, count }))

  const kpis = [
    { label: 'Total medical documents', value: String(docs?.total ?? 0), hint: `${indexedDocs} indexed`, icon: FileText, tone: 'text-medical-600' },
    { label: 'Indexed knowledge chunks', value: String(indexedChunks), hint: 'Across all documents', icon: Database, tone: 'text-teal-500' },
    { label: 'Total questions', value: String(totalQuestions), hint: `${completedTraces} completed`, icon: TrendingUp, tone: 'text-indigo-500' },
    { label: 'Agentic success rate', value: agenticRate === null ? '—' : `${agenticRate}%`, hint: `${agenticSuccess}/${agenticTraces.length} agentic`, icon: Gauge, tone: 'text-emerald-500' },
    { label: 'Avg. response time', value: avgLatency === null ? '—' : `${avgLatency} ms`, hint: 'From traces', icon: Timer, tone: 'text-amber-500' },
    { label: 'Failed executions', value: String(failedTraces), hint: 'Traces not completed', icon: Activity, tone: 'text-rose-500' },
  ]

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-text">Dashboard</h1>
          <p className="mt-1 text-sm text-text-muted">
            Live overview of the MedAssist AI knowledge base, retrieval, and agent activity.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-4 w-4" /> Refresh
        </Button>
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4 text-sm text-red-700">Failed to load: {error}</CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {loading
          ? Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-28" />)
          : kpis.map((k) => (
              <Card key={k.label}>
                <CardContent className="p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="text-xs font-medium uppercase tracking-wide text-text-muted">{k.label}</div>
                      <div className="mt-2 text-3xl font-semibold text-text">{k.value}</div>
                      <div className="mt-1 text-xs text-text-muted">{k.hint}</div>
                    </div>
                    <k.icon className={`h-6 w-6 ${k.tone}`} />
                  </div>
                </CardContent>
              </Card>
            ))}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Backend health</CardTitle>
            <CardDescription>Component status from /api/health</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-40" /> : health ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Badge variant={health.status === 'ok' ? 'success' : 'warning'}>{health.status}</Badge>
                  <span className="text-sm text-text-muted">Phase {health.phase} · {health.app} {health.version}</span>
                </div>
                <div className="divide-y divide-border">
                  {health.checks.map((c) => (
                    <div key={c.name} className="flex items-center justify-between py-2 text-sm">
                      <span className="font-medium text-text">{c.name}</span>
                      <div className="flex items-center gap-3 text-text-muted">
                        <span className="truncate max-w-[260px]">{c.detail ?? '—'}</span>
                        <Badge variant={c.status === 'ok' ? 'success' : c.status === 'degraded' ? 'warning' : 'danger'}>
                          {c.status}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Indexed documents</CardTitle>
            <CardDescription>{docs?.total ?? 0} total · {indexedChunks} chunks</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? <Skeleton className="h-40" /> : (
              <div className="space-y-2">
                {(docs?.items ?? []).slice(0, 8).map((d) => (
                  <div key={d.id} className="flex items-center justify-between text-sm">
                    <div className="min-w-0 pr-4">
                      <div className="truncate font-medium text-text">{d.title}</div>
                      <div className="text-xs text-text-muted">{d.source} · {d.chunk_count} chunks</div>
                    </div>
                    <Badge variant={d.status === 'indexed' ? 'success' : 'warning'}>{d.status}</Badge>
                  </div>
                ))}
                {!docs?.items.length && <div className="text-sm text-text-muted">No documents yet.</div>}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Questions over time</CardTitle>
            <CardDescription>From execution traces</CardDescription>
          </CardHeader>
          <CardContent className="h-64">
            {loading ? <Skeleton className="h-full" /> : questionsOverTime.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={questionsOverTime}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#dde3ec" />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" stroke="#2f81f7" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : <EmptyChart text="No traces yet." />}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>RAG mode usage</CardTitle>
            <CardDescription>Traditional vs Hybrid vs Agentic</CardDescription>
          </CardHeader>
          <CardContent className="h-64">
            {loading ? <Skeleton className="h-full" /> : modeUsage.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={modeUsage}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#dde3ec" />
                  <XAxis dataKey="mode" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#14b8a6" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyChart text="No traces yet." />}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Avg. latency by mode</CardTitle>
            <CardDescription>Milliseconds</CardDescription>
          </CardHeader>
          <CardContent className="h-64">
            {loading ? <Skeleton className="h-full" /> : latencyChart.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={latencyChart}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#dde3ec" />
                  <XAxis dataKey="mode" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Bar dataKey="avg" fill="#f59e0b" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyChart text="No traces yet." />}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Execution outcomes</CardTitle>
            <CardDescription>Trace status distribution</CardDescription>
          </CardHeader>
          <CardContent className="h-64">
            {loading ? <Skeleton className="h-full" /> : statusChart.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={statusChart} dataKey="count" nameKey="status" outerRadius={80} label>
                    {statusChart.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : <EmptyChart text="No traces yet." />}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent activity</CardTitle>
          <CardDescription>Latest questions, uploads, executions, failures</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? <Skeleton className="h-40" /> : (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">Latest questions</div>
                <div className="space-y-1">
                  {(traces?.items ?? []).slice(0, 5).map((t) => (
                    <div key={t.id} className="truncate text-sm text-text">
                      {t.query_preview} <span className="text-xs text-text-muted">({t.mode})</span>
                    </div>
                  ))}
                  {!traces?.items.length && <div className="text-sm text-text-muted">None yet.</div>}
                </div>
              </div>
              <div>
                <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-muted">Failed executions</div>
                <div className="space-y-1">
                  {(traces?.items ?? []).filter((t) => t.status !== 'completed').slice(0, 5).map((t) => (
                    <div key={t.id} className="truncate text-sm text-red-600">
                      {t.query_preview} <span className="text-xs text-text-muted">({t.mode})</span>
                    </div>
                  ))}
                  {!failedTraces && <div className="text-sm text-text-muted">None. 🎉</div>}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {bench?.summary && (
        <Card>
          <CardHeader>
            <CardTitle>Benchmark summary</CardTitle>
            <CardDescription>Latest run: {bench.run.status}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-text-muted">
              Traditional vs Agentic averages are available on the Benchmark Center page.
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function EmptyChart({ text }: { text: string }) {
  return <div className="flex h-full items-center justify-center text-sm text-text-muted">{text}</div>
}
