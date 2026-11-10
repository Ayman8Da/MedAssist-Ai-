import * as React from 'react'
import { Settings as SettingsIcon, Globe, Sliders, Shield } from 'lucide-react'
import { useTheme } from '@/hooks/useTheme'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export function SettingsPage() {
  const { theme, toggle } = useTheme()
  const [apiBase, setApiBase] = React.useState(import.meta.env.VITE_API_BASE_URL || '(same origin / Vite proxy)')
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-text">Settings</h1>
        <p className="mt-1 text-sm text-text-muted">Application and backend configuration.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Globe className="h-5 w-5" /> API connection</CardTitle>
          <CardDescription>Configured via VITE_API_BASE_URL at build time.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="block">
            <span className="mb-1 block text-xs font-medium text-text-muted">API base URL</span>
            <Input value={apiBase} onChange={(e) => setApiBase(e.target.value)} disabled />
          </label>
          <p className="text-xs text-text-muted">
            In development, requests to <code>/api</code> are proxied to <code>http://localhost:8000</code>.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Sliders className="h-5 w-5" /> Appearance</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-text">Theme</div>
            <div className="text-xs text-text-muted">Current: {theme}</div>
          </div>
          <Button variant="outline" onClick={toggle}>{theme === 'dark' ? 'Switch to light' : 'Switch to dark'}</Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Shield className="h-5 w-5" /> Backend-managed settings</CardTitle>
          <CardDescription>LLM provider, embedding model, retrieval parameters, reranker, and verifier limits are configured on the backend via environment variables. They are not exposed here to avoid leaking secrets.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {['LLM provider / model', 'Embedding model', 'Top-K', 'Reranker', 'Verifier retries'].map((s) => (
              <Badge key={s} variant="secondary">{s}</Badge>
            ))}
          </div>
          <p className="mt-3 text-xs text-text-muted">
            See <code>.env.example</code> in the repository for the full list of backend settings.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><SettingsIcon className="h-5 w-5" /> Medical safety</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-text-muted">
          MedAssist AI provides evidence-based medical information for clinical decision support. It does not
          replace professional medical judgment, diagnosis, or treatment.
        </CardContent>
      </Card>
    </div>
  )
}
