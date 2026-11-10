import * as React from 'react'
import { Upload, Trash2, RefreshCw, Search, FileText } from 'lucide-react'
import { api } from '@/lib/api'
import type { DocumentOut, DocumentListOut } from '@/types'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Skeleton } from '@/components/ui/skeleton'

export function KnowledgeBase() {
  const [docs, setDocs] = React.useState<DocumentListOut | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [search, setSearch] = React.useState('')
  const [status, setStatus] = React.useState('')
  const [uploading, setUploading] = React.useState(false)
  const [drag, setDrag] = React.useState(false)
  const [meta, setMeta] = React.useState({ source: 'MedAssist Synthetic Library', title: '', publication_date: '', version: '', category: '', document_key: '' })
  const fileRef = React.useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = React.useState<File | null>(null)

  const load = React.useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setDocs(await api.listDocuments({ search: search || undefined, status: status || undefined, limit: 200 }))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [search, status])

  React.useEffect(() => { load() }, [load])

  const onFiles = (files: FileList | null) => {
    if (!files || !files.length) return
    setSelectedFile(files[0])
    if (!meta.title) setMeta((m) => ({ ...m, title: files[0].name }))
  }

  const upload = async () => {
    if (!selectedFile) return
    setUploading(true)
    setError(null)
    try {
      await api.uploadDocument(selectedFile, {
        source: meta.source,
        title: meta.title || selectedFile.name,
        publication_date: meta.publication_date,
        version: meta.version,
        category: meta.category,
        document_key: meta.document_key,
      })
      setSelectedFile(null)
      setMeta({ source: meta.source, title: '', publication_date: '', version: '', category: '', document_key: '' })
      await load()
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setUploading(false)
    }
  }

  const remove = async (id: string) => {
    if (!confirm('Delete this document and its chunks?')) return
    await api.deleteDocument(id)
    await load()
  }

  const reindex = async (id: string) => {
    await api.reindexDocuments([id])
    await load()
  }

  const reindexAll = async () => {
    await api.reindexDocuments()
    await load()
  }

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-text">Knowledge Base</h1>
          <p className="mt-1 text-sm text-text-muted">Upload, index, and manage medical documents.</p>
        </div>
        <Button variant="outline" onClick={reindexAll}>
          <RefreshCw className="h-4 w-4" /> Reindex all
        </Button>
      </div>

      {error && <Card className="border-red-200 bg-red-50"><CardContent className="p-4 text-sm text-red-700">{error}</CardContent></Card>}

      <Card>
        <CardHeader>
          <CardTitle>Upload document</CardTitle>
          <CardDescription>PDF, TXT, or Markdown. Drag and drop or browse.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); onFiles(e.dataTransfer.files) }}
            className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 transition ${
              drag ? 'border-medical-500 bg-medical-600/5' : 'border-border bg-surface-muted/40'
            }`}
          >
            <Upload className="h-8 w-8 text-text-muted" />
            <p className="mt-2 text-sm text-text-muted">
              {selectedFile ? selectedFile.name : 'Drop a file here or click to browse'}
            </p>
            <input ref={fileRef} type="file" accept=".pdf,.txt,.md,.markdown" className="hidden" onChange={(e) => onFiles(e.target.files)} />
            <Button variant="outline" size="sm" className="mt-3" onClick={() => fileRef.current?.click()}>
              Browse
            </Button>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <Field label="Source (required)" value={meta.source} onChange={(v) => setMeta({ ...meta, source: v })} />
            <Field label="Title" value={meta.title} onChange={(v) => setMeta({ ...meta, title: v })} />
            <Field label="Publication date (YYYY-MM-DD)" value={meta.publication_date} onChange={(v) => setMeta({ ...meta, publication_date: v })} />
            <Field label="Version" value={meta.version} onChange={(v) => setMeta({ ...meta, version: v })} />
            <Field label="Category" value={meta.category} onChange={(v) => setMeta({ ...meta, category: v })} />
            <Field label="Document key (for version grouping)" value={meta.document_key} onChange={(v) => setMeta({ ...meta, document_key: v })} />
          </div>

          <div className="flex justify-end">
            <Button onClick={upload} disabled={!selectedFile || uploading || !meta.source}>
              {uploading ? 'Uploading…' : 'Upload & index'}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <CardTitle>Documents</CardTitle>
              <CardDescription>{docs?.total ?? 0} total</CardDescription>
            </div>
            <div className="flex gap-2">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
                <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search…" className="pl-9" />
              </div>
              <select
                value={status}
                onChange={(e) => setStatus(e.target.value)}
                className="h-10 rounded-[var(--radius-control)] border border-border bg-white px-3 text-sm dark:bg-surface"
              >
                <option value="">All statuses</option>
                <option value="indexed">Indexed</option>
                <option value="failed">Failed</option>
                <option value="pending">Pending</option>
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-14" />)}</div>
          ) : (
            <div className="divide-y divide-border">
              {(docs?.items ?? []).map((d: DocumentOut) => (
                <div key={d.id} className="flex items-center justify-between gap-4 py-3">
                  <div className="min-w-0 flex items-start gap-3">
                    <FileText className="mt-0.5 h-5 w-5 shrink-0 text-text-muted" />
                    <div className="min-w-0">
                      <div className="truncate font-medium text-text">{d.title}</div>
                      <div className="text-xs text-text-muted">
                        {d.source} · {d.category ?? '—'} · v{d.version ?? '—'} · {d.chunk_count} chunks · {d.status}
                        {d.document_key ? ` · key: ${d.document_key}` : ''}
                      </div>
                    </div>
                  </div>
                  <div className="flex shrink-0 gap-2">
                    <Button variant="outline" size="sm" onClick={() => reindex(d.id)}>
                      <RefreshCw className="h-3.5 w-3.5" /> Reindex
                    </Button>
                    <Button variant="ghost" size="icon" onClick={() => remove(d.id)}>
                      <Trash2 className="h-4 w-4 text-red-600" />
                    </Button>
                  </div>
                </div>
              ))}
              {!docs?.items.length && <div className="py-6 text-center text-sm text-text-muted">No documents.</div>}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function Field({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-text-muted">{label}</span>
      <Input value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  )
}
