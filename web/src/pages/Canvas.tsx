import { useQuery } from '@tanstack/react-query'
import { useMemo, useState } from 'react'

import { api } from '../api/client'
import { describe, keys } from '../api/hooks'
import type { CasePayload, GraphNode } from '../api/types'
import { CaseFrame, PageHeader, useCaseId } from '../components/CaseFrame'
import { DossierPanel } from '../components/Dossier'
import { LayeredGraph, Legend } from '../components/LayeredGraph'
import { Callout, Card, DataTable, Field, SectionTitle, Spinner } from '../components/ui'
import { titleCase } from '../lib/utils'

const KIND_LABELS: Record<string, string> = {
  evidence: 'Evidence',
  context: 'Strategic context',
  technique: 'ATT&CK techniques',
  tool: 'Tools / software',
  campaign: 'Campaigns',
  live_intel: 'Live threat intelligence',
  group: 'Candidate groups',
  country: 'Country hypotheses',
  assessment: 'Threat type',
}
const COLUMN_BY_KIND: Record<string, string> = {
  evidence: 'evidence',
  context: 'evidence',
  technique: 'attack',
  tool: 'attack',
  campaign: 'attack',
  live_intel: 'attack',
  group: 'groups',
  country: 'origin',
  assessment: 'origin',
}
const COLOR_BY_KIND: Record<string, string> = {
  evidence: '#60a5fa',
  context: '#5eead4',
  technique: '#fbbf24',
  tool: '#fb923c',
  campaign: '#c084fc',
  live_intel: '#f9a8d4',
  group: '#fb7185',
  country: '#4ade80',
  assessment: '#d6b49c',
}
const COLUMNS = [
  { key: 'evidence', title: 'Evidence & context', x: 0 },
  { key: 'attack', title: 'Techniques · tools · campaigns', x: 0.34 },
  { key: 'groups', title: 'Candidate groups', x: 0.68 },
  { key: 'origin', title: 'Origin · intrusion type', x: 1 },
]

export function CanvasPage() {
  const id = useCaseId()
  return <CaseFrame>{(data) => <CanvasView caseId={id} data={data} />}</CaseFrame>
}

function CanvasView({ caseId, data }: { caseId: string; data: CasePayload }) {
  const [topGroups, setTopGroups] = useState(Math.min(8, Math.max(data.ranking.length, 1)))
  const [maxEvidence, setMaxEvidence] = useState(Math.min(12, Math.max(data.observations.length, 4)))
  const [hiddenKinds, setHiddenKinds] = useState<Set<string>>(new Set())
  const [hiddenRelations, setHiddenRelations] = useState<Set<string>>(new Set())
  const [selected, setSelected] = useState<string | null>(null)
  const query = useQuery({
    queryKey: keys.canvas(caseId, topGroups, maxEvidence),
    queryFn: () => api.canvas(caseId, topGroups, maxEvidence),
    staleTime: 60_000,
  })
  const graph = query.data
  const kinds = useMemo(() => Object.keys(KIND_LABELS).filter((k) => graph?.nodes.some((n) => n.kind === k)), [graph])
  const relations = useMemo(() => Array.from(new Set(graph?.edges.map((e) => e.relationship) ?? [])).sort(), [graph])
  const visibleNodes = useMemo(() => graph?.nodes.filter((n) => !hiddenKinds.has(n.kind ?? '')) ?? [], [graph, hiddenKinds])
  const visibleIds = useMemo(() => new Set(visibleNodes.map((n) => n.id)), [visibleNodes])
  const visibleEdges = useMemo(
    () => graph?.edges.filter((e) => visibleIds.has(e.source) && visibleIds.has(e.target) && !hiddenRelations.has(e.relationship)) ?? [],
    [graph, visibleIds, hiddenRelations],
  )
  const linked = useMemo(
    () => new Set(graph?.edges.filter((e) => e.source.startsWith('evidence:')).map((e) => e.source.replace('evidence:', '')) ?? []),
    [graph],
  )
  const unresolved = data.observations.filter((o) => !linked.has(o.id))
  const [dossier, setDossier] = useState<string | null>(null)

  const onSelect = (node: GraphNode) => {
    setSelected((current) => (current === node.id ? null : node.id))
    if (node.kind === 'group') setDossier(node.id.replace(/^group:/, ''))
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Analyst canvas"
        lead="Evidence and context → ATT&CK techniques, tools and campaigns → candidate groups → documented origin and intrusion type. Hover a node or link for its provenance; click a group to open its dossier."
      />
      <Card className="flex flex-wrap items-end gap-5 py-4">
        <Field label={`Candidate groups: ${topGroups}`} className="w-48">
          <input type="range" min={1} max={Math.max(2, Math.min(15, data.ranking.length))} value={topGroups} onChange={(e) => setTopGroups(Number(e.target.value))} className="w-full accent-cyan-400" />
        </Field>
        <Field label={`Evidence items: ${maxEvidence}`} className="w-48">
          <input type="range" min={4} max={Math.max(20, Math.min(data.observations.length, 40))} value={maxEvidence} onChange={(e) => setMaxEvidence(Number(e.target.value))} className="w-full accent-cyan-400" />
        </Field>
        <div className="min-w-[280px] flex-1">
          <div className="mb-1.5 text-xs font-medium">Layers</div>
          <Legend
            items={kinds.map((kind) => ({
              key: kind,
              label: KIND_LABELS[kind],
              color: COLOR_BY_KIND[kind],
              active: !hiddenKinds.has(kind),
              onToggle: () =>
                setHiddenKinds((s) => {
                  const next = new Set(s)
                  if (next.has(kind)) next.delete(kind)
                  else next.add(kind)
                  return next
                }),
            }))}
          />
        </div>
        <div className="min-w-[280px] flex-1">
          <div className="mb-1.5 text-xs font-medium">Connections</div>
          <Legend
            items={relations.map((rel) => ({
              key: rel,
              label: titleCase(rel),
              color: 'var(--color-muted)',
              active: !hiddenRelations.has(rel),
              onToggle: () =>
                setHiddenRelations((s) => {
                  const next = new Set(s)
                  if (next.has(rel)) next.delete(rel)
                  else next.add(rel)
                  return next
                }),
            }))}
          />
        </div>
      </Card>
      <Card className="p-3">
        {query.isLoading ? (
          <Spinner label="Building the canvas…" className="p-6" />
        ) : query.isError ? (
          <Callout tone="danger">{describe(query.error)}</Callout>
        ) : visibleNodes.length ? (
          <LayeredGraph
            nodes={visibleNodes}
            edges={visibleEdges}
            columns={COLUMNS}
            columnOf={(n) => COLUMN_BY_KIND[n.kind ?? 'evidence'] ?? 'evidence'}
            colorOf={(n) => COLOR_BY_KIND[n.kind ?? 'evidence'] ?? '#94a3b8'}
            onSelect={onSelect}
            selected={selected}
            height={Math.max(520, 40 * Math.max(visibleNodes.length / 3, 8))}
          />
        ) : (
          <Callout tone="info">Select at least one layer.</Callout>
        )}
      </Card>
      {unresolved.length ? (
        <Card>
          <SectionTitle hint="These observations have no technique, tool or actor-behaviour connection and do not influence the ranking.">
            {unresolved.length} unresolved observation{unresolved.length === 1 ? '' : 's'}
          </SectionTitle>
          <DataTable
            columns={[
              { key: 'source', label: 'Source' },
              { key: 'kind', label: 'Kind' },
              { key: 'text', label: 'Observation', className: 'min-w-[400px]' },
            ]}
            rows={unresolved as unknown as Record<string, unknown>[]}
            maxHeight={260}
          />
        </Card>
      ) : null}
      <DossierPanel caseId={caseId} groupId={dossier} onClose={() => setDossier(null)} />
    </div>
  )
}
