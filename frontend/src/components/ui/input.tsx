import * as React from 'react'
import { cn } from '@/lib/utils'

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type, ...props }, ref) => (
    <input
      type={type}
      ref={ref}
      className={cn(
        'flex h-10 w-full rounded-[var(--radius-control)] border border-border bg-white px-3 py-2 text-sm text-text placeholder:text-text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-medical-500/50 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-surface',
        className,
      )}
      {...props}
    />
  ),
)
Input.displayName = 'Input'

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        'flex min-h-[96px] w-full rounded-[var(--radius-control)] border border-border bg-white px-3 py-2 text-sm text-text placeholder:text-text-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-medical-500/50 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-surface',
        className,
      )}
      {...props}
    />
  ),
)
Textarea.displayName = 'Textarea'
