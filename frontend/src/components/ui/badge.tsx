import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors',
  {
    variants: {
      variant: {
        default: 'bg-medical-600/10 text-medical-600',
        secondary: 'bg-surface-muted text-text-muted',
        success: 'bg-emerald-100 text-emerald-700',
        warning: 'bg-amber-100 text-amber-700',
        danger: 'bg-red-100 text-red-700',
        outline: 'border border-border text-text-muted',
      },
    },
    defaultVariants: { variant: 'default' },
  },
)

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />
}
