import * as React from 'react'
import { Play, AlertCircle, Loader2 } from 'lucide-react'
import { api } from '@/lib/api'
import type { BenchmarkResultsOut } from '@/types'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

export function BenchmarkCenter() {
  const [running, setRunning] = React.useState(false)
  const [runId, setRunId] = React.useState<string | null>(null)
  const [results, setResults] = React.useState<BenchmarkResultsOut | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  const poll = React.useCallback(async () => {
    try {
      const r = (await api.benchmarkResults()) as BenchmarkResultsOut | null
      setResults(r)
      if (r?.run.status === 'completed' || r?.run.status === 'failed') setRunning(false)
    } catch (e) {
      // 204-ish / no content
    }
  }, [])

  React.useEffect(() => { poll() }, [poll])

  React.useEffect(() => {
    if (!running) return
    const t = setInterval(poll, 5000)
    return () => clearInterval(t)
  }, [running, poll])

  const run = async () => {
    setRunning(true)
    setError(null)
    try {
      const r = await api.benchmarkRun()
      setRunId(r.run_id)
    } catch (e) {
      setError((e as Error).message)
      setRunning(false)
    }
  }

  const runStatus = results?.run.status
  const summary = results?.summary as { modes?: Record<string, Record<string, unknown>> } | null

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-text">Benchmark Center</h1>
          <p className="mt-1 text-sm text-text-muted">
            Compare Traditional RAG vs Agentic RAG on the five sample questions.
          </p>
        </div>
        <Button onClick={run} disabled={running}>
          {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {running ? 'Running…' : 'Run Benchmark'}
        </Button>
      </div>

      {error && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4 text-sm text-red-700">{error}</CardContent>
        </Card>
      )}

      {running && (
        <Card>
          <CardContent className="flex items-center gap-3 p-4 text-sm text-text-muted">
            <Loader2 className="h-4 w-4 animate-spin" />
            Benchmark running (run {runId})… {results?.run.completed_questions ?? 0}/
            {results?.run.total_questions ?? '?'} questions. This can take a while on CPU.
          </CardContent>
        </Card>
      )}

      {results?.run.status === 'failed' && (
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-4 text-sm text-red-700">Run failed: {results.run.error}</CardContent>
        </Card>
      )}

      {summary?.modes && (
        <Card>
          <CardHeader>
            <CardTitle>Summary</CardTitle>
            <CardDescription>Averages across all questions (LLM-as-judge metrics, 0–1).</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs text-text-muted">
                    <th className="py-2">Metric</th>
                    <th className="py-2">Traditional RAG</th>
                    <th className="py-2">Agentic RAG</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    ['Questions', 'questions'],
                    ['Avg. faithfulness', 'avg_faithfulness'],
                    ['Avg. answer relevance', 'avg_answer_relevance'],
                    ['Avg. context precision', 'avg_context_precision'],
                    ['Avg. coherence', 'avg_coherence'],
                    ['Avg. latency (ms)', 'avg_latency_ms'],
                  ].map(([label, key]) => (
                    <tr key={key} className="border-b border-border/60">
                      <td className="py-2 font-medium text-text">{label}</td>
                      <td className="py-2 text-text-muted">{String(summary.modes!.traditional?.[key] ?? '—')}</td>
                      <td className="py-2 text-text-muted">{String(summary.modes!.agentic?.[key] ?? '—')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-xs text-text-muted">
              Context recall is null because no gold reference answers are provided (would be invented
              otherwise). Citation accuracy is reported per question below.
            </p>
          </CardContent>
        </Card>
      )}

      {results?.results && (
        <Card>
          <CardHeader>
            <CardTitle>Per-question results</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {results.results.map((r: any, i: number) => (
              <div key={i} className="rounded-xl border border-border p-4">
                <div className="mb-2 text-sm font-semibold text-text">
                  Q{i + 1}. {r.question}
                </div>
                <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                  {(['traditional', 'agentic'] as const).map((mode) => {
                    const m = r.modes?.[mode]
                    if (!m) return null
                    return (
                      <div key={mode} className="rounded-lg bg-surface-muted/40 p-3 text-xs">
                        <div className="mb-1 flex items-center gap-2">
                          <Badge variant={mode === 'agentic' ? 'default' : 'secondary'}>{mode}</Badge>
                          <span className="text-text-muted">{m.latency_ms} ms</span>
                        </div>
                        <div className="mb-2 line-clamp-4 text-text-muted">{String(m.answer ?? '')}</div>
                        <div className="flex flex-wrap gap-2">
                          {m.metrics &&
                            Object.entries(m.metrics).map(([k, v]) => (
                              <Badge key={k} variant="outline">
                                {k}: {String(v ?? 'n/a')}
                              </Badge>
                            ))}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {!results && !running && (
        <Card>
          <CardContent className="p-4 text-sm text-text-muted">
            No benchmark runs yet. Click "Run Benchmark" to execute the 5 sample questions through both
            Traditional and Agentic RAG. Requires the backend and a working LLM.
          </CardContent>
        </Card>
      )}
    </div>
  )
}
