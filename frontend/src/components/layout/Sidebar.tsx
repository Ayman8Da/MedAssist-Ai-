import * as React from 'react'
import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  MessageSquareHeart,
  GitBranch,
  Database,
  BarChart3,
  Activity,
  Settings as SettingsIcon,
  ChevronLeft,
  ChevronRight,
  Sun,
  Moon,
  ShieldCheck,
} from 'lucide-react'
import { useTheme } from '@/hooks/useTheme'
import { cn } from '@/lib/utils'

const NAV = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/assistant', label: 'Medical Assistant', icon: MessageSquareHeart },
  { to: '/workflow', label: 'Agent Workflow', icon: GitBranch },
  { to: '/knowledge', label: 'Knowledge Base', icon: Database },
  { to: '/benchmark', label: 'Benchmark Center', icon: BarChart3 },
  { to: '/observability', label: 'Observability', icon: Activity },
  { to: '/settings', label: 'Settings', icon: SettingsIcon },
]

export function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const { theme, toggle } = useTheme()
  return (
    <aside
      className={cn(
        'flex h-full flex-col border-r border-border bg-white/80 backdrop-blur transition-all duration-200 dark:bg-surface',
        collapsed ? 'w-[72px]' : 'w-[248px]',
      )}
    >
      <div className="flex items-center gap-3 px-4 py-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-medical-600 to-teal-500 text-white shadow-sm">
          <ShieldCheck className="h-5 w-5" />
        </div>
        {!collapsed && (
          <div className="leading-tight">
            <div className="text-sm font-semibold text-text">MedAssist AI</div>
            <div className="text-[11px] text-text-muted">Clinical decision support</div>
          </div>
        )}
      </div>

      <nav className="flex-1 space-y-1 px-3">
        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-medical-600/10 text-medical-600'
                  : 'text-text-muted hover:bg-surface-muted hover:text-text',
              )
            }
          >
            <item.icon className="h-5 w-5 shrink-0" />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      <div className="space-y-2 border-t border-border px-3 py-4">
        <button
          onClick={toggle}
          className="flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm text-text-muted hover:bg-surface-muted"
        >
          {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          {!collapsed && <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>}
        </button>
        <button
          onClick={onToggle}
          className="flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm text-text-muted hover:bg-surface-muted"
        >
          {collapsed ? <ChevronRight className="h-5 w-5" /> : <ChevronLeft className="h-5 w-5" />}
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  )
}
