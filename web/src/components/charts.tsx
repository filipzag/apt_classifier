import { motion } from 'framer-motion'
import type { ReactNode } from 'react'

import { cn, fmt } from '../lib/utils'

/** Ranked horizontal bars; the first bar is the reference (100). */
export function RankedBars({
  items,
  max,
  onSelect,
  selected,
  valueFormatter = (v) => fmt(v, 1),
  className,
  tone = 'accent',
}: {
  items: { key: string; label: ReactNode; value: number; sub?: ReactNode; tone?: 'accent' | 'violet' | 'emerald' | 'amber' | 'rose' }[]
  max?: number
  onSelect?: (key: string) => void
  selected?: string | null
  valueFormatter?: (value: number) => string
  className?: string
  tone?: 'accent' | 'violet' | 'emerald' | 'amber' | 'rose'
}) {
  const top = max ?? Math.max(...items.map((i) => i.value), 0.0001)
  const tones = {
    accent: 'from-accent/90 to-accent/50',
    violet: 'from-violet/90 to-violet/50',
    emerald: 'from-emerald/90 to-emerald/50',
    amber: 'from-amber/90 to-amber/50',
    rose: 'from-rose/90 to-rose/50',
  }
  return (
    <ul className={cn('space-y-1.5', className)}>
      {items.map((item, index) => {
        const active = selected === item.key
        const Row = onSelect ? 'button' : 'div'
        return (
          <li key={item.key}>
            <Row
              type={onSelect ? 'button' : undefined}
              onClick={onSelect ? () => onSelect(item.key) : undefined}
              className={cn(
                'group grid w-full grid-cols-[minmax(0,1fr)_64px] items-center gap-3 rounded-lg px-2 py-1 text-left transition-colors',
                onSelect && 'hover:bg-surface-2 focus-ring',
                active && 'bg-accent-soft',
              )}
            >
              <div className="min-w-0">
                <div className="flex items-baseline gap-2">
                  <span className="text-[13px] font-medium truncate">{item.label}</span>
                  {item.sub ? <span className="text-[11px] text-faint truncate">{item.sub}</span> : null}
                </div>
                <div className="mt-1 h-2 w-full overflow-hidden rounded-full bg-surface-3">
                  <motion.div
                    className={cn('h-full rounded-full bg-gradient-to-r', tones[item.tone ?? tone])}
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(1.5, (item.value / top) * 100)}%` }}
                    transition={{ type: 'spring', stiffness: 100, damping: 22, delay: index * 0.03 }}
                  />
                </div>
              </div>
              <div className="text-right text-xs tabular text-muted">{valueFormatter(item.value)}</div>
            </Row>
          </li>
        )
      })}
    </ul>
  )
}

/** Sparse rank matrix (groups × classifiers) as a heat grid. */
export function RankHeatmap({
  classifiers,
  rows,
  limit = 15,
}: {
  classifiers: string[]
  rows: { group_id: string; name: string; ranks: Record<string, number | null> }[]
  limit?: number
}) {
  const shown = rows.slice(0, limit)
  return (
    <div className="overflow-auto rounded-xl border border-border">
      <table className="data">
        <thead>
          <tr>
            <th>Candidate</th>
            {classifiers.map((c) => (
              <th key={c} className="text-center">
                {c.replace(/_/g, ' ')}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((row) => (
            <tr key={row.group_id}>
              <td className="whitespace-nowrap">
                {row.name} <span className="mono text-faint">{row.group_id}</span>
              </td>
              {classifiers.map((c) => {
                const rank = row.ranks[c]
                return (
                  <td key={c} className="text-center">
                    <RankCell rank={rank} />
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RankCell({ rank }: { rank: number | null | undefined }) {
  if (rank === null || rank === undefined) return <span className="text-faint">–</span>
  const heat = rank <= 1 ? 1 : rank <= 3 ? 0.7 : rank <= 10 ? 0.4 : 0.15
  return (
    <span
      className="inline-grid size-7 place-items-center rounded-md text-[11px] font-medium tabular"
      style={{
        background: `rgba(34, 211, 238, ${heat * 0.35})`,
        color: heat > 0.5 ? 'var(--color-text)' : 'var(--color-muted)',
      }}
    >
      {rank}
    </span>
  )
}

/** A metric with a threshold: value bar, bar marker, and pass/fail text. */
export function GateMetric({
  label,
  value,
  threshold,
  help,
  informational = false,
  max = 1,
  format = (v: number) => fmt(v, 2),
}: {
  label: string
  value: number
  threshold: number | null
  help: string
  informational?: boolean
  max?: number
  format?: (v: number) => string
}) {
  const passes = threshold !== null && value >= threshold
  const negative = value < 0
  const share = Math.max(0, Math.min(1, Math.abs(value) / max))
  return (
    <div className="rounded-xl border border-border bg-surface-2/60 px-4 py-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted">{label}</span>
        <span
          className={cn(
            'text-[11px]',
            informational ? 'text-faint' : passes ? 'text-emerald' : 'text-amber',
          )}
        >
          {informational
            ? 'informational'
            : threshold === null
              ? ''
              : passes
                ? `≥ ${format(threshold)} bar`
                : `bar ${format(threshold)}`}
        </span>
      </div>
      <div className={cn('mt-1 text-2xl font-semibold tracking-tight tabular', !informational && (passes ? 'text-emerald' : 'text-text'))}>
        {format(value)}
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-surface-3 relative">
        <motion.div
          className={cn(
            'h-full rounded-full',
            negative ? 'bg-rose' : informational ? 'bg-faint' : passes ? 'bg-emerald' : 'bg-accent',
          )}
          initial={{ width: 0 }}
          animate={{ width: `${share * 100}%` }}
          transition={{ type: 'spring', stiffness: 100, damping: 22 }}
        />
        {threshold !== null && !informational ? (
          <div
            className="absolute top-0 h-full w-0.5 bg-text/70"
            style={{ left: `calc(${Math.max(0, Math.min(1, threshold / max)) * 100}% - 1px)` }}
            aria-hidden
          />
        ) : null}
      </div>
      <p className="mt-2 text-[11px] leading-relaxed text-faint">
        {negative ? 'Negative: the alternative is better supported than the leader. ' : ''}
        {help}
      </p>
    </div>
  )
}
