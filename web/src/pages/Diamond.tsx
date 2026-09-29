import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { api } from '../api/client'
import { describe, keys } from '../api/hooks'
import type { CasePayload, GraphNode } from '../api/types'
import { CaseFrame, PageHeader, useCaseId } from '../components/CaseFrame'
import { DossierPanel } from '../components/Dossier'
import { LayeredGraph, Legend } from '../components/LayeredGraph'
import { Badge, Callout, Card, Field, Spinner } from '../components/ui'
import { cn, fmt } from '../lib/utils'

const VERTICES = [
  { key: 'adversary', title: 'Adversary', x: 0, color: '#fb7185' },
  { key: 'capability', title: 'Capability', x: 0.34, color: '#fbbf24' },
  { key: 'infrastructure', title: 'Infrastructure', x: 0.68, color: '#60a5fa' },
  { key: 'victim', title: 'Victim', x: 1, color: '#4ade80' },
]
const STATUS_TONE: Record<string, 'emerald' | 'sky' | 'neutral' | 'rose'> = {
  observed: 'emerald',
  hypothesis: 'sky',
  unknown: 'neutral',
  contradicted: 'rose',
}

export function DiamondPage() {
  const id = useCaseId()
  return <CaseFrame>{(data) => <DiamondView caseId={id} data={data} />}</CaseFrame>
}

function DiamondView({ caseId, data }: { caseId: string; data: CasePayload }) {
  const [topGroups, setTopGroups] = useState(Math.min(8, Math.max(data.ranking.length, 1)))
  const [caps, setCaps] = useState(15)
  const [infra, setInfra] = useState(15)
  const [hidden, setHidden] = useState<Set<string>>(new Set())
  const [selected, setSelected] = useState<string | null>(null)
  const [dossier, setDossier] = useState<string | null>(null)
  const query = useQuery({
    queryKey: keys.diamond(caseId, topGroups, caps, infra),
    queryFn: () => api.diamond(caseId, topGroups, caps, infra),
    staleTime: 60_000,
  })
  const graph = query.data
  const nodes = useMemo(() => graph?.nodes.filter((n) => !hidden.has(n.vertex ?? '')) ?? [], [graph, hidden])
  const ids = useMemo(() => new Set(nodes.map((n) => n.id)), [nodes])
  const edges = useMemo(() => graph?.edges.filter((e) => ids.has(e.source) && ids.has(e.target)) ?? [], [graph, ids])
  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = {}
    for (const n of graph?.nodes ?? []) counts[n.status ?? 'unknown'] = (counts[n.status ?? 'unknown'] ?? 0) + 1
    return counts
  }, [graph])

  const onSelect = (node: GraphNode) => {
    setSelected((c) => (c === node.id ? null : node.id))
    if (node.vertex === 'adversary') {
      const match = node.id.match(/G\d{4}/)
      if (match) setDossier(match[0])
    }
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Diamond model"
        lead="The case organised into adversary, capability, infrastructure and victim vertices. An evidence map, not an attribution verdict: adversary nodes are the ranked hypotheses."
      />
      <Card className="flex flex-wrap items-end gap-5 py-4">
        <Field label={`Adversary hypotheses: ${topGroups}`} className="w-48">
          <input type="range" min={1} max={Math.max(2, Math.min(15, data.ranking.length))} value={topGroups} onChange={(e) => setTopGroups(Number(e.target.value))} className="w-full accent-cyan-400" />
        </Field>
        <Field label={`Capability nodes: ${caps}`} className="w-44">
          <input type="range" min={4} max={30} value={caps} onChange={(e) => setCaps(Number(e.target.value))} className="w-full accent-cyan-400" />
        </Field>
        <Field label={`Infrastructure nodes: ${infra}`} className="w-44">
          <input type="range" min={4} max={30} value={infra} onChange={(e) => setInfra(Number(e.target.value))} className="w-full accent-cyan-400" />
        </Field>
        <div className="flex-1">
          <div className="mb-1.5 text-xs font-medium">Vertices</div>
          <Legend
            items={VERTICES.map((v) => ({
              key: v.key,
              label: v.title,
              color: v.color,
              active: !hidden.has(v.key),
              onToggle: () =>
                setHidden((s) => {
                  const next = new Set(s)
                  if (next.has(v.key)) next.delete(v.key)
                  else next.add(v.key)
                  return next
                }),
            }))}
          />
        </div>
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(statusCounts).map(([status, count]) => (
            <Badge key={status} tone={STATUS_TONE[status] ?? 'neutral'}>
              {status} <span className="tabular opacity-70">{count}</span>
            </Badge>
          ))}
        </div>
      </Card>
      <Card className="p-3">
        {query.isLoading ? (
          <Spinner label="Building the diamond…" className="p-6" />
        ) : query.isError ? (
          <Callout tone="danger">{describe(query.error)}</Callout>
        ) : nodes.length ? (
          <LayeredGraph
            nodes={nodes}
            edges={edges}
            columns={VERTICES}
            columnOf={(n) => n.vertex ?? 'capability'}
            colorOf={(n) => VERTICES.find((v) => v.key === n.vertex)?.color ?? '#94a3b8'}
            onSelect={onSelect}
            selected={selected}
            height={Math.max(480, 34 * Math.max(nodes.length / 2, 10))}
            labelLimit={30}
          />
        ) : (
          <Callout tone="info">Select at least one vertex.</Callout>
        )}
      </Card>
      {graph?.caveats.length ? (
        <ul className="grid gap-2 md:grid-cols-2">
          {graph.caveats.map((caveat, i) => (
            <li key={i} className={cn('rounded-xl border border-border bg-surface-2/60 px-4 py-3 text-xs leading-relaxed text-muted')}>
              {caveat}
            </li>
          ))}
        </ul>
      ) : null}
      {selected && graph ? (
        <Card>
          {(() => {
            const node = graph.nodes.find((n) => n.id === selected)
            if (!node) return null
            return (
              <div className="flex flex-wrap items-center gap-3 text-sm">
                <Badge tone={STATUS_TONE[node.status ?? 'unknown'] ?? 'neutral'}>{node.status}</Badge>
                <span className="font-semibold">{node.label}</span>
                <span className="text-muted">{node.vertex} · score {fmt(node.score, 3)}</span>
                <span className="text-muted">{node.details}</span>
              </div>
            )
          })()}
        </Card>
      ) : null}
      <DossierPanel caseId={caseId} groupId={dossier} onClose={() => setDossier(null)} />
    </div>
  )
}
