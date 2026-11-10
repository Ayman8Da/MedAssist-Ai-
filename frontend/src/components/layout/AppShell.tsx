import * as React from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { Sidebar } from './Sidebar'
import { cn } from '@/lib/utils'

export function AppShell() {
  const [collapsed, setCollapsed] = React.useState(false)
  const location = useLocation()
  return (
    <div className="flex h-full w-full bg-surface">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
      <main className="flex-1 overflow-auto">
        <div className="mx-auto max-w-[1400px] px-6 py-6 lg:px-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.16 }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </div>
        <footer className="border-t border-border bg-white/60 px-6 py-3 text-center text-[11px] text-text-muted dark:bg-surface">
          MedAssist AI provides evidence-based medical information for clinical decision support. It does not
          replace professional medical judgment, diagnosis, or treatment.
        </footer>
      </main>
    </div>
  )
}
