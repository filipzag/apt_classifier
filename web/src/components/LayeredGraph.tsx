import { useMemo, useState } from 'react'

import type { GraphEdge, GraphNode } from '../api/types'
import { cn, fmt, truncate } from '../lib/utils'

export interface Column {
  key: string
  title: string
  x: number
}

/**
 * Deterministic layered layout: each node is assigned to a column by `columnOf`, ordered by
 * score within the column, and edges are drawn as cubic curves. Pure SVG, no physics, so the
 * picture is identical on every render and screenshots are reproducible.
 */
export function LayeredGraph({
  nodes,
  edges,
  columns,
  columnOf,
  colorOf,
  onSelect,
  selected,
  height = 640,
  labelLimit = 34,
}: {
  nodes: GraphNode[]
  edges: GraphEdge[]
  columns: Column[]
  columnOf: (node: GraphNode) => string
  colorOf: (node: GraphNode) => string
  onSelect?: (node: GraphNode) => void
  selected?: string | null
  height?: number
  labelLimit?: number
}) {
  const [hover, setHover] = useState<{ x: number; y: number; title: string; body: string } | null>(null)
  const width = 1200
  const padX = 120
  const layout = useMemo(() => {
    const byColumn = new Map<string, GraphNode[]>()
    for (const node of nodes) {
      const key = columnOf(node)
      byColumn.set(key, [...(byColumn.get(key) ?? []), node])
    }
    const positions = new Map<string, { x: number; y: number; r: number; node: GraphNode }>()
    const span = width - padX * 2
    for (const column of columns) {
      const list = (byColumn.get(column.key) ?? []).sort(
        (a, b) => (a.kind ?? a.vertex ?? '').localeCompare(b.kind ?? b.vertex ?? '') || b.score - a.score || a.label.localeCompare(b.label),
      )
      list.forEach((node, index) => {
        const y = 48 + ((index + 1) / (list.length + 1)) * (height - 96)
        positions.set(node.id, {
          x: padX + column.x * span,
          y,
          r: 5 + 9 * Math.max(0, Math.min(1, node.score)),
          node,
        })
      })
    }
    return positions
  }, [nodes, columns, columnOf, height])

  const neighbours = useMemo(() => {
    const map = new Map<string, Set<string>>()
    for (const edge of edges) {
      map.set(edge.source, (map.get(edge.source) ?? new Set()).add(edge.target))
      map.set(edge.target, (map.get(edge.target) ?? new Set()).add(edge.source))
    }
    return map
  }, [edges])

  const focus = selected ?? null
  const related = focus ? neighbours.get(focus) ?? new Set<string>() : null

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img" aria-label="Layered evidence graph">
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {columns.map((column) => (
          <g key={column.key}>
            <text
              x={padX + column.x * (width - padX * 2)}
              y={22}
              textAnchor="middle"
              className="fill-[var(--color-faint)] text-[11px] font-semibold uppercase tracking-[0.12em]"
            >
              {column.title}
            </text>
            <line
              x1={padX + column.x * (width - padX * 2)}
              x2={padX + column.x * (width - padX * 2)}
              y1={34}
              y2={height - 20}
              stroke="var(--color-border)"
              strokeDasharray="2 6"
            />
          </g>
        ))}
        {edges.map((edge, index) => {
          const a = layout.get(edge.source)
          const b = layout.get(edge.target)
          if (!a || !b) return null
          const dim = focus ? !(edge.source === focus || edge.target === focus) : false
          const midX = (a.x + b.x) / 2
          const path = `M ${a.x} ${a.y} C ${midX} ${a.y}, ${midX} ${b.y}, ${b.x} ${b.y}`
          return (
            <path
              key={index}
              d={path}
              fill="none"
              stroke={dim ? 'var(--color-border)' : 'var(--color-muted)'}
              strokeOpacity={dim ? 0.25 : 0.25 + 0.6 * Math.max(0, Math.min(1, edge.weight))}
              strokeWidth={dim ? 1 : 1 + 1.5 * Math.max(0, Math.min(1, edge.weight))}
              className="transition-opacity"
              onMouseEnter={(e) =>
                setHover({
                  x: e.nativeEvent.offsetX,
                  y: e.nativeEvent.offsetY,
                  title: edge.relationship.replace(/_/g, ' '),
                  body: `${edge.details || ''} · weight ${fmt(edge.weight, 3)}`,
                })
              }
              onMouseLeave={() => setHover(null)}
            />
          )
        })}
        {Array.from(layout.values()).map(({ x, y, r, node }) => {
          const dim = focus ? node.id !== focus && !related?.has(node.id) : false
          const isSelected = node.id === focus
          const anchorRight = columns.find((c) => c.key === columnOf(node))?.x === 1
          return (
            <g
              key={node.id}
              transform={`translate(${x} ${y})`}
              className={cn(onSelect && 'cursor-pointer')}
              opacity={dim ? 0.3 : 1}
              onClick={onSelect ? () => onSelect(node) : undefined}
              onMouseEnter={(e) =>
                setHover({
                  x: e.nativeEvent.offsetX,
                  y: e.nativeEvent.offsetY,
                  title: node.label,
                  body: `${node.kind ?? node.vertex ?? ''}${node.status ? ` · ${node.status}` : ''} · score ${fmt(node.score, 3)}${node.details ? ` · ${node.details}` : ''}`,
                })
              }
              onMouseLeave={() => setHover(null)}
            >
              <circle r={r + (isSelected ? 4 : 0)} fill={colorOf(node)} fillOpacity={0.18} stroke={colorOf(node)} strokeWidth={isSelected ? 2 : 1} filter={isSelected ? 'url(#glow)' : undefined} />
              <circle r={Math.max(2.5, r * 0.45)} fill={colorOf(node)} />
              <text
                x={anchorRight ? -(r + 8) : r + 8}
                y={4}
                textAnchor={anchorRight ? 'end' : 'start'}
                className="fill-[var(--color-text)] text-[11.5px]"
                style={{ paintOrder: 'stroke', stroke: 'var(--color-bg)', strokeWidth: 3, strokeLinejoin: 'round' }}
              >
                {truncate(node.label, labelLimit)}
              </text>
            </g>
          )
        })}
      </svg>
      {hover ? (
        <div
          className="pointer-events-none absolute z-20 max-w-sm rounded-lg border border-border-strong bg-surface-3 px-3 py-2 text-xs shadow-2xl"
          style={{ left: Math.min(hover.x + 12, 900), top: hover.y + 12 }}
        >
          <div className="font-semibold">{hover.title}</div>
          <div className="mt-0.5 text-muted">{hover.body}</div>
        </div>
      ) : null}
    </div>
  )
}

export function Legend({ items }: { items: { key: string; label: string; color: string; active: boolean; onToggle: () => void }[] }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          onClick={item.onToggle}
          aria-pressed={item.active}
          className={cn(
            'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors focus-ring',
            item.active ? 'border-border-strong bg-surface-3 text-text' : 'border-border text-faint',
          )}
        >
          <span className="size-2 rounded-full" style={{ background: item.color, opacity: item.active ? 1 : 0.4 }} />
          {item.label}
        </button>
      ))}
    </div>
  )
}
