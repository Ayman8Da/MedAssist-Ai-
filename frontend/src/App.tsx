import { Routes, Route } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { Dashboard } from '@/pages/Dashboard'
import { MedicalAssistant } from '@/pages/MedicalAssistant'
import { AgentWorkflow } from '@/pages/AgentWorkflow'
import { KnowledgeBase } from '@/pages/KnowledgeBase'
import { BenchmarkCenter } from '@/pages/BenchmarkCenter'
import { Observability } from '@/pages/Observability'
import { SettingsPage } from '@/pages/Settings'

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Dashboard />} />
        <Route path="/assistant" element={<MedicalAssistant />} />
        <Route path="/workflow" element={<AgentWorkflow />} />
        <Route path="/knowledge" element={<KnowledgeBase />} />
        <Route path="/benchmark" element={<BenchmarkCenter />} />
        <Route path="/observability" element={<Observability />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  )
}
