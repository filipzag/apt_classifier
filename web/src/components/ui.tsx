import * as DialogPrimitive from '@radix-ui/react-dialog'
import * as TabsPrimitive from '@radix-ui/react-tabs'
import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import { motion } from 'framer-motion'
import { Info, Loader2, X } from 'lucide-react'
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, TextareaHTMLAttributes } from 'react'

import { cn } from '../lib/utils'

/* ---------- buttons ---------- */

type Variant = 'primary' | 'secondary' | 'ghost' | 'danger' | 'outline'
type Size = 'sm' | 'md' | 'lg'

const variantClass: Record<Variant, string> = {
  primary:
    'bg-accent text-bg font-semibold hover:bg-cyan-300 shadow-[0_0_0_1px_rgba(34,211,238,.4),0_8px_30px_-10px_rgba(34,211,238,.6)]',
  secondary: 'bg-surface-3 text-text hover:bg-border-strong border border-border-strong',
  outline: 'bg-transparent text-text border border-border-strong hover:bg-surface-2',
  ghost: 'bg-transparent text-muted hover:text-text hover:bg-surface-2',
  danger: 'bg-rose-soft text-rose border border-rose/30 hover:bg-rose/20',
}
const sizeClass: Record<Size, string> = {
  sm: 'h-8 px-3 text-xs gap-1.5 rounded-lg',
  md: 'h-9 px-3.5 text-sm gap-2 rounded-lg',
  lg: 'h-11 px-5 text-sm gap-2 rounded-xl',
}

export function Button({
  variant = 'secondary',
  size = 'md',
  loading = false,
  className,
  children,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant
  size?: Size
  loading?: boolean
}) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center whitespace-nowrap transition-colors focus-ring disabled:opacity-50 disabled:pointer-events-none',
        variantClass[variant],
        sizeClass[size],
        className,
      )}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : null}
      {children}
    </button>
  )
}

/* ---------- surfaces ---------- */

export function Card({
  className,
  children,
  glow = false,
}: {
  className?: string
  children: ReactNode
  glow?: boolean
}) {
  return (
    <section className={cn('card p-5', glow && 'shadow-(--shadow-glow) border-accent/30', className)}>
      {children}
    </section>
  )
}

export function SectionTitle({
  children,
  hint,
  action,
  className,
}: {
  children: ReactNode
  hint?: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('mb-4 flex items-start justify-between gap-4', className)}>
      <div>
        <h2 className="text-[15px] font-semibold tracking-tight">{children}</h2>
        {hint ? <p className="mt-0.5 text-xs leading-relaxed text-muted">{hint}</p> : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  )
}

export function Label({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('text-[11px] font-semibold uppercase tracking-[0.08em] text-muted', className)}>
      {children}
    </div>
  )
}

export function Callout({
  tone = 'info',
  title,
  children,
  icon,
  className,
}: {
  tone?: 'info' | 'warning' | 'success' | 'danger' | 'violet'
  title?: ReactNode
  children?: ReactNode
  icon?: ReactNode
  className?: string
}) {
  const tones = {
    info: 'border-sky/30 bg-sky/10 text-text',
    warning: 'border-amber/30 bg-amber-soft text-text',
    success: 'border-emerald/30 bg-emerald-soft text-text',
    danger: 'border-rose/30 bg-rose-soft text-text',
    violet: 'border-violet/30 bg-violet-soft text-text',
  }
  return (
    <div className={cn('flex gap-3 rounded-xl border px-4 py-3 text-sm', tones[tone], className)}>
      <div className="mt-0.5 shrink-0 text-muted">{icon ?? <Info className="size-4" />}</div>
      <div className="min-w-0 leading-relaxed">
        {title ? <div className="font-semibold">{title}</div> : null}
        {children ? <div className={cn(title && 'mt-0.5 text-muted')}>{children}</div> : null}
      </div>
    </div>
  )
}

/* ---------- badges ---------- */

export type BadgeTone = 'neutral' | 'accent' | 'violet' | 'emerald' | 'amber' | 'rose' | 'sky'

const badgeTone: Record<BadgeTone, string> = {
  neutral: 'bg-surface-3 text-muted border-border-strong',
  accent: 'bg-accent-soft text-accent border-accent/30',
  violet: 'bg-violet-soft text-violet border-violet/30',
  emerald: 'bg-emerald-soft text-emerald border-emerald/30',
  amber: 'bg-amber-soft text-amber border-amber/30',
  rose: 'bg-rose-soft text-rose border-rose/30',
  sky: 'bg-sky/10 text-sky border-sky/30',
}

export function Badge({
  tone = 'neutral',
  children,
  className,
  dot = false,
}: {
  tone?: BadgeTone
  children: ReactNode
  className?: string
  dot?: boolean
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium leading-4 whitespace-nowrap',
        badgeTone[tone],
        className,
      )}
    >
      {dot ? <span className="size-1.5 rounded-full bg-current" aria-hidden /> : null}
      {children}
    </span>
  )
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn('mono text-[12px] text-muted', className)}>{children}</span>
}

/* ---------- metrics ---------- */

export function Metric({
  label,
  value,
  hint,
  tone,
  className,
}: {
  label: ReactNode
  value: ReactNode
  hint?: ReactNode
  tone?: 'accent' | 'violet' | 'emerald' | 'amber' | 'rose'
  className?: string
}) {
  const tones = {
    accent: 'text-accent',
    violet: 'text-violet',
    emerald: 'text-emerald',
    amber: 'text-amber',
    rose: 'text-rose',
  }
  return (
    <div className={cn('rounded-xl border border-border bg-surface-2/60 px-4 py-3', className)}>
      <Label>{label}</Label>
      <div className={cn('mt-1 text-2xl font-semibold tracking-tight tabular', tone && tones[tone])}>
        {value}
      </div>
      {hint ? <div className="mt-0.5 text-xs text-muted">{hint}</div> : null}
    </div>
  )
}

/** Horizontal bar with an optional threshold marker; value and max in the same units. */
export function Bar({
  value,
  max = 1,
  threshold,
  tone = 'accent',
  height = 6,
  className,
  animate = true,
}: {
  value: number
  max?: number
  threshold?: number | null
  tone?: 'accent' | 'violet' | 'emerald' | 'amber' | 'rose' | 'muted'
  height?: number
  className?: string
  animate?: boolean
}) {
  const tones = {
    accent: 'bg-accent',
    violet: 'bg-violet',
    emerald: 'bg-emerald',
    amber: 'bg-amber',
    rose: 'bg-rose',
    muted: 'bg-faint',
  }
  const share = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0
  const mark = threshold !== null && threshold !== undefined && max > 0 ? Math.max(0, Math.min(1, threshold / max)) : null
  return (
    <div
      className={cn('relative w-full overflow-hidden rounded-full bg-surface-3', className)}
      style={{ height }}
      role="presentation"
    >
      <motion.div
        className={cn('h-full rounded-full', tones[tone])}
        initial={animate ? { width: 0 } : false}
        animate={{ width: `${share * 100}%` }}
        transition={{ type: 'spring', stiffness: 120, damping: 20 }}
      />
      {mark !== null ? (
        <div
          className="absolute top-0 h-full w-0.5 bg-text/70"
          style={{ left: `calc(${mark * 100}% - 1px)` }}
          aria-hidden
        />
      ) : null}
    </div>
  )
}

/* ---------- inputs ---------- */

export function Input({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn(
        'h-9 w-full rounded-lg border border-border-strong bg-surface-2 px-3 text-sm text-text placeholder:text-faint focus-ring',
        className,
      )}
      {...rest}
    />
  )
}

export function Textarea({ className, ...rest }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return (
    <textarea
      className={cn(
        'w-full resize-y rounded-lg border border-border-strong bg-surface-2 px-3 py-2 text-sm leading-relaxed text-text placeholder:text-faint focus-ring',
        className,
      )}
      {...rest}
    />
  )
}

export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: ReactNode
  hint?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <label className={cn('block', className)}>
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <span className="text-xs font-medium text-text">{label}</span>
        {hint ? <span className="text-[11px] text-faint">{hint}</span> : null}
      </div>
      {children}
    </label>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  className,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  label: ReactNode
  className?: string
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={cn('inline-flex items-center gap-2 text-xs text-muted hover:text-text focus-ring rounded-md', className)}
    >
      <span
        className={cn(
          'relative inline-flex h-5 w-9 items-center rounded-full border transition-colors',
          checked ? 'bg-accent/80 border-accent' : 'bg-surface-3 border-border-strong',
        )}
      >
        <span
          className={cn(
            'absolute size-3.5 rounded-full bg-bg transition-transform',
            checked ? 'translate-x-[18px]' : 'translate-x-[3px]',
          )}
        />
      </span>
      {label}
    </button>
  )
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  size = 'sm',
  className,
}: {
  value: T
  onChange: (value: T) => void
  options: { value: T; label: ReactNode; tone?: BadgeTone }[]
  size?: 'sm' | 'md'
  className?: string
}) {
  return (
    <div
      className={cn(
        'inline-flex items-center rounded-lg border border-border-strong bg-surface-2 p-0.5',
        className,
      )}
      role="radiogroup"
    >
      {options.map((option) => {
        const active = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(option.value)}
            className={cn(
              'rounded-md font-medium transition-colors focus-ring',
              size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3 py-1.5 text-sm',
              active ? 'bg-surface-3 text-text shadow-sm' : 'text-muted hover:text-text',
            )}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}

/* ---------- feedback ---------- */

export function Spinner({ label, className }: { label?: ReactNode; className?: string }) {
  return (
    <div className={cn('flex items-center gap-2 text-sm text-muted', className)}>
      <Loader2 className="size-4 animate-spin text-accent" aria-hidden />
      {label}
    </div>
  )
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('skeleton h-4 w-full', className)} aria-hidden />
}

export function EmptyState({
  icon,
  title,
  children,
  action,
  className,
}: {
  icon?: ReactNode
  title: ReactNode
  children?: ReactNode
  action?: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center rounded-xl border border-dashed border-border-strong px-6 py-12 text-center',
        className,
      )}
    >
      {icon ? <div className="mb-3 text-faint">{icon}</div> : null}
      <div className="text-sm font-semibold">{title}</div>
      {children ? <div className="mt-1 max-w-md text-xs leading-relaxed text-muted">{children}</div> : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  )
}

/* ---------- radix wrappers ---------- */

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <TooltipPrimitive.Root delayDuration={200}>
      <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          sideOffset={6}
          className="z-50 max-w-xs rounded-lg border border-border-strong bg-surface-3 px-3 py-2 text-xs leading-relaxed text-text shadow-xl"
        >
          {content}
          <TooltipPrimitive.Arrow className="fill-surface-3" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  )
}

export function SlideOver({
  open,
  onOpenChange,
  title,
  description,
  children,
  width = 'max-w-2xl',
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: ReactNode
  description?: ReactNode
  children: ReactNode
  width?: string
}) {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/60 backdrop-blur-[2px] data-[state=open]:animate-in data-[state=open]:fade-in" />
        <DialogPrimitive.Content
          className={cn(
            'fixed inset-y-0 right-0 z-50 flex w-full flex-col border-l border-border bg-surface shadow-2xl outline-none',
            width,
          )}
        >
          <div className="flex items-start justify-between gap-4 border-b border-border px-6 py-4">
            <div className="min-w-0">
              <DialogPrimitive.Title className="text-base font-semibold tracking-tight">
                {title}
              </DialogPrimitive.Title>
              {description ? (
                <DialogPrimitive.Description className="mt-0.5 text-xs text-muted">
                  {description}
                </DialogPrimitive.Description>
              ) : (
                <DialogPrimitive.Description className="sr-only">Details</DialogPrimitive.Description>
              )}
            </div>
            <DialogPrimitive.Close asChild>
              <Button variant="ghost" size="sm" aria-label="Close">
                <X className="size-4" />
              </Button>
            </DialogPrimitive.Close>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto px-6 py-5">{children}</div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

export function Tabs({
  tabs,
  defaultValue,
  className,
}: {
  tabs: { value: string; label: ReactNode; content: ReactNode; count?: number }[]
  defaultValue?: string
  className?: string
}) {
  return (
    <TabsPrimitive.Root defaultValue={defaultValue ?? tabs[0]?.value} className={className}>
      <TabsPrimitive.List className="mb-4 flex flex-wrap gap-1 border-b border-border">
        {tabs.map((tab) => (
          <TabsPrimitive.Trigger
            key={tab.value}
            value={tab.value}
            className="-mb-px inline-flex items-center gap-2 border-b-2 border-transparent px-3 py-2 text-sm text-muted transition-colors hover:text-text data-[state=active]:border-accent data-[state=active]:text-text focus-ring"
          >
            {tab.label}
            {tab.count !== undefined ? (
              <span className="rounded-full bg-surface-3 px-1.5 text-[10px] tabular text-muted">
                {tab.count}
              </span>
            ) : null}
          </TabsPrimitive.Trigger>
        ))}
      </TabsPrimitive.List>
      {tabs.map((tab) => (
        <TabsPrimitive.Content key={tab.value} value={tab.value} className="outline-none">
          {tab.content}
        </TabsPrimitive.Content>
      ))}
    </TabsPrimitive.Root>
  )
}

/* ---------- tables ---------- */

export function DataTable({
  columns,
  rows,
  empty = 'Nothing to show.',
  maxHeight,
  className,
  rowKey,
  onRowClick,
  selectedKey,
}: {
  columns: { key: string; label: ReactNode; className?: string; render?: (row: Record<string, unknown>) => ReactNode }[]
  rows: Record<string, unknown>[]
  empty?: ReactNode
  maxHeight?: number | string
  className?: string
  rowKey?: (row: Record<string, unknown>, index: number) => string
  onRowClick?: (row: Record<string, unknown>) => void
  selectedKey?: string | null
}) {
  if (!rows.length) return <div className="px-1 py-3 text-xs text-muted">{empty}</div>
  return (
    <div
      className={cn('overflow-auto rounded-xl border border-border', className)}
      style={maxHeight ? { maxHeight } : undefined}
    >
      <table className="data">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key} className={column.className}>
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const key = rowKey ? rowKey(row, index) : String(index)
            return (
              <tr
                key={key}
                className={cn(onRowClick && 'cursor-pointer', selectedKey === key && 'selected')}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
              >
                {columns.map((column) => (
                  <td key={column.key} className={column.className}>
                    {column.render ? column.render(row) : formatCell(row[column.key])}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function formatCell(value: unknown): ReactNode {
  if (value === null || value === undefined) return <span className="text-faint">–</span>
  if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(3)
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (Array.isArray(value)) return value.map(String).join(', ')
  return String(value)
}
