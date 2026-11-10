import * as React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Copy, Trash2, Sparkles, BookOpen, ShieldAlert } from 'lucide-react'
import { api } from '@/lib/api'
import type { ChatResponse, Citation, RetrievedChunk } from '@/types'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Textarea } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'

type Mode = 'traditional' | 'hybrid' | 'agentic'

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  response?: ChatResponse
  error?: string
  pending?: boolean
}

const SUGGESTIONS = [
  'What are the contraindications of metformin in renal impairment?',
  'Compare blood pressure targets for adults with type 2 diabetes (2019 vs 2024).',
  'What information is needed to assess fever, cough, and shortness of breath?',
  'Antimicrobial stewardship principles for respiratory infections',
]

export function MedicalAssistant() {
  const [mode, setMode] = React.useState<Mode>('hybrid')
  const [messages, setMessages] = React.useState<Message[]>([])
  const [input, setInput] = React.useState('')
  const [pending, setPending] = React.useState(false)
  const [fastMode, setFastMode] = React.useState(false)
  const abortRef = React.useRef<AbortController | null>(null)
  const scrollRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const send = async (question: string) => {
    const q = question.trim()
    if (!q || pending) return
    const controller = new AbortController()
    abortRef.current = controller
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: q }
    const assistantMsg: Message = { id: crypto.randomUUID(), role: 'assistant', content: '', pending: true }
    setMessages((m) => [...m, userMsg, assistantMsg])
    setInput('')
    setPending(true)
    try {
      const payload = { question: q, top_k: fastMode ? 2 : undefined }
      const res =
        mode === 'traditional'
          ? await api.chatTraditional(payload, controller.signal)
          : mode === 'hybrid'
            ? await api.chatHybrid(payload, controller.signal)
            : await api.chatAgentic(payload, controller.signal)
      setMessages((m) =>
        m.map((msg) =>
          msg.id === assistantMsg.id ? { ...msg, content: res.answer, response: res, pending: false } : msg,
        ),
      )
    } catch (e) {
      if ((e as Error).name === 'AbortError') {
        setMessages((m) => m.map((msg) => (msg.id === assistantMsg.id ? { ...msg, content: 'Cancelled.', pending: false } : msg)))
      } else {
        setMessages((m) =>
          m.map((msg) =>
            msg.id === assistantMsg.id ? { ...msg, content: '', error: (e as Error).message, pending: false } : msg,
          ),
        )
      }
    } finally {
      setPending(false)
      abortRef.current = null
    }
  }

  const cancel = () => abortRef.current?.abort()

  const clear = () => setMessages([])

  const copy = (text: string) => navigator.clipboard.writeText(text)

  return (
    <div className="flex h-[calc(100vh-3rem)] flex-col gap-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-text">Medical Assistant</h1>
          <p className="mt-1 text-sm text-text-muted">
            Evidence-based answers from your indexed medical knowledge base.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 text-xs text-text-muted">
            <input type="checkbox" checked={fastMode} onChange={(e) => setFastMode(e.target.checked)} />
            Fast (top-2)
          </label>
          <div className="flex rounded-xl border border-border bg-white p-1 dark:bg-surface">
            {(['traditional', 'hybrid', 'agentic'] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium capitalize transition ${
                  mode === m ? 'bg-medical-600 text-white' : 'text-text-muted hover:text-text'
                }`}
              >
                {m} RAG
              </button>
            ))}
          </div>
          <Button variant="ghost" size="icon" onClick={clear} title="Clear chat">
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Card className="flex-1 overflow-hidden">
        <CardContent className="flex h-full flex-col p-0">
          <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-6">
            {messages.length === 0 && (
              <div className="mx-auto max-w-2xl space-y-4 py-10 text-center">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-medical-600/10 text-medical-600">
                  <Sparkles className="h-6 w-6" />
                </div>
                <h2 className="text-xl font-semibold text-text">Ask a clinical question</h2>
                <p className="text-sm text-text-muted">
                  Responses are grounded in your indexed documents and include citations. MedAssist AI is
                  clinical decision support — not a replacement for professional medical judgment.
                </p>
                <div className="flex flex-wrap justify-center gap-2 pt-2">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => send(s)}
                      className="rounded-full border border-border bg-white px-3 py-1.5 text-xs text-text-muted hover:border-medical-500 hover:text-medical-600 dark:bg-surface"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <AnimatePresence initial={false}>
            {messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.18 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-medical-600 text-white'
                      : 'bg-surface-muted text-text'
                  }`}
                >
                  {msg.pending && <span className="inline-block animate-pulse">Thinking…</span>}
                  {msg.error && <span className="text-red-600">Error: {msg.error}</span>}
                  {msg.content && <div className="whitespace-pre-wrap">{msg.content}</div>}

                  {msg.response && (
                    <div className="mt-3 space-y-3 border-t border-border/60 pt-3">
                      {msg.response.insufficient_evidence && (
                        <div className="flex items-center gap-2 text-amber-700">
                          <ShieldAlert className="h-4 w-4" />
                          <span className="text-xs font-medium">Insufficient evidence in knowledge base.</span>
                        </div>
                      )}

                      {msg.response.diagnostics && (
                        <div className="flex flex-wrap gap-2 text-[11px] text-text-muted">
                          <Badge variant="secondary">dense {msg.response.diagnostics.dense_candidates}</Badge>
                          <Badge variant="secondary">keyword {msg.response.diagnostics.keyword_candidates}</Badge>
                          <Badge variant="secondary">fused {msg.response.diagnostics.fused_candidates}</Badge>
                          <Badge variant={msg.response.diagnostics.reranker_used ? 'success' : 'warning'}>
                            reranker {msg.response.diagnostics.reranker_used ? 'used' : 'skipped'}
                          </Badge>
                        </div>
                      )}

                      {msg.response.citations.length > 0 && (
                        <div>
                          <div className="mb-2 flex items-center gap-1.5 text-xs font-semibold text-text-muted">
                            <BookOpen className="h-3.5 w-3.5" /> Sources & citations
                          </div>
                          <div className="space-y-2">
                            {msg.response.citations.map((c) => (
                              <CitationCard key={c.chunk_id} citation={c} chunk={msg.response!.retrieved_chunks.find((r) => r.chunk_id === c.chunk_id)} />
                            ))}
                          </div>
                        </div>
                      )}

                      {msg.response.warnings.length > 0 && (
                        <div className="text-[11px] text-text-muted">warnings: {msg.response.warnings.join(', ')}</div>
                      )}

                      <div className="flex gap-2">
                        <Button variant="ghost" size="sm" onClick={() => copy(msg.response!.answer)}>
                          <Copy className="h-3.5 w-3.5" /> Copy
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          </div>

          <div className="border-t border-border p-4">
            <form
              onSubmit={(e) => {
                e.preventDefault()
                send(input)
              }}
              className="flex items-end gap-2"
            >
              <Textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    send(input)
                  }
                }}
                placeholder="Ask a clinical question… (Enter to send, Shift+Enter for newline)"
                className="min-h-[44px] max-h-40"
              />
              <Button type="submit" disabled={pending || !input.trim()} size="icon" className="h-11 w-11 shrink-0">
                <Send className="h-4 w-4" />
              </Button>
              {pending && (
                <Button type="button" variant="outline" size="icon" className="h-11 w-11 shrink-0" onClick={cancel} title="Cancel">
                  <Trash2 className="h-4 w-4" />
                </Button>
              )}
            </form>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function CitationCard({ citation, chunk }: { citation: Citation; chunk?: RetrievedChunk }) {
  const [open, setOpen] = React.useState(false)
  return (
    <div className="rounded-xl border border-border bg-white/60 p-3 text-xs dark:bg-surface">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-medical-600">[{citation.index}]</span>
            <span className="truncate font-medium text-text">{citation.title}</span>
            {citation.used && <Badge variant="success">used</Badge>}
            {citation.score_kind && <Badge variant="outline">{citation.score_kind}</Badge>}
          </div>
          <div className="mt-1 text-text-muted">
            {citation.source} · {citation.publication_date ?? 'n/a'} · v{citation.version ?? 'n/a'}
            {citation.section ? ` · ${citation.section}` : ''}
            {citation.page_number != null ? ` · p.${citation.page_number}` : ''}
            {' · '}score {citation.score.toFixed(4)}
          </div>
        </div>
        <button onClick={() => setOpen((o) => !o)} className="text-medical-600 hover:underline">
          {open ? 'Hide' : 'Evidence'}
        </button>
      </div>
      {open && chunk && (
        <div className="mt-2 rounded-lg bg-surface-muted p-2 text-text">{chunk.text}</div>
      )}
    </div>
  )
}
