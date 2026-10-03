import { ArrowRight, Check, Crosshair, MessageSquare, X } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { describe, useApplyDecisions, useDistinguish, useTop3 } from '../api/hooks'
import type { CasePayload, DistinguishResponse } from '../api/types'
import { CaseFrame, PageHeader, useCaseId } from '../components/CaseFrame'
import { GateMetric } from '../components/charts'
import { TopThreePanel } from '../components/TopThree'
import { Badge, Button, Callout, Card, Field, Label, SectionTitle, Spinner, Tabs, Tip } from '../components/ui'
import { cn, fmt, pct } from '../lib/utils'

export function DistinguishPage() {
  const id = useCaseId()
  return <CaseFrame>{(data) => <DistinguishView caseId={id} data={data} />}</CaseFrame>
}

function DeltaChip({ value, label }: { value: number; label?: string }) {
  const tone = value > 0.005 ? 'emerald' : value < -0.005 ? 'rose' : 'neutral'
  return (
    <Badge tone={tone} className="tabular">
      {label ? `${label} ` : ''}
      {value > 0 ? '+' : ''}
      {fmt(value, 2)}
    </Badge>
  )
}

function DistinguishView({ caseId, data }: { caseId: string; data: CasePayload }) {
  const navigate = useNavigate()
  // null = the naming gate's own window, so the separation here equals the decision's
  const [topK, setTopK] = useState<number | null>(null)
  const query = useDistinguish(caseId, topK)
  const apply = useApplyDecisions(caseId)
  const result = query.data

  const shortlist =
    query.isLoading ? (
        <Card>
          <Spinner label="Replaying the separation for every pending candidate and lead…" />
        </Card>
      ) : query.isError || !result ? (
        <Callout tone="danger" title="Could not compute the separation">
          {describe(query.error)}
        </Callout>
      ) : (
        <>
          <Card glow={result.separation >= result.thresholds.separation}>
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
              <div>
                <div className="flex flex-wrap items-center gap-3">
                  <span className="grid size-10 place-items-center rounded-xl bg-accent-soft text-accent">
                    <Crosshair className="size-5" />
                  </span>
                  <div>
                    <Label>Leader vs best-supported alternative</Label>
                    <div className="text-lg font-semibold tracking-tight">
                      {result.leader.name ?? '–'}
                      <span className="mx-2 text-faint">vs</span>
                      {result.runner_up.name ?? '–'}
                      {result.runner_up_related ? (
                        <Badge tone="amber" className="ml-2">documented-related</Badge>
                      ) : null}
                    </div>
                  </div>
                </div>
                <p className="mt-3 text-sm leading-relaxed text-muted">
                  Separation is the leader's discriminative evidence minus the alternative's, as a share of the larger one.
                  The naming gate ({result.thresholds.gate_mode}) requires at least {fmt(result.thresholds.separation)} with a
                  discriminative mass of {fmt(result.thresholds.min_discriminative, 1)}. Shared evidence counts for neither side,
                  so confirming a generic technique changes nothing; confirming an exclusive one does.
                </p>
                <div className="mt-3 flex flex-wrap items-end gap-4">
                  <Field
                    label={`Shortlist size: top ${result.top_k}${result.top_k === result.gate_top_k ? ' (naming gate window)' : ''}`}
                    className="w-64"
                  >
                    <input
                      type="range"
                      min={2}
                      max={Math.max(2, Math.min(15, data.ranking.length))}
                      value={result.top_k}
                      onChange={(e) => {
                        const value = Number(e.target.value)
                        setTopK(value === result.gate_top_k ? null : value)
                      }}
                      className="w-full accent-cyan-400"
                    />
                  </Field>
                  {result.top_k !== result.gate_top_k ? (
                    <Button size="sm" variant="ghost" onClick={() => setTopK(null)}>
                      Back to the gate window ({result.gate_top_k})
                    </Button>
                  ) : null}
                  {result.contrast_enabled ? (
                    <Badge tone="violet">contrastive near misses on</Badge>
                  ) : (
                    <Tip content="Only accepted techniques count. ATTRIBUTION_CONTRAST_MIN_PROBABILITY lets pending candidates with a calibrated probability above the floor enter as exclusivity-weighted near misses.">
                      <Badge>accepted techniques only</Badge>
                    </Tip>
                  )}
                </div>
              </div>
              <div className="grid gap-3 sm:grid-cols-2">
                <GateMetric
                  label="Separation"
                  value={result.separation}
                  threshold={result.thresholds.separation}
                  help="Leader minus alternative over the larger of the two (−1 to 1)."
                />
                <GateMetric
                  label="Discriminative mass"
                  value={result.leader_total}
                  threshold={result.thresholds.min_discriminative}
                  max={Math.max(result.thresholds.min_discriminative * 2, result.leader_total, 1)}
                  help="Weighted evidence the leader has that the shortlist cannot explain."
                  format={(v) => fmt(v, 2)}
                />
              </div>
            </div>
          </Card>

          <Card>
            <SectionTitle hint="Rows are techniques in this case, ordered by how much they separate the shortlist. Weight = confidence × exclusivity × documented specificity. A tick marks a group that documents the technique.">
              Distinguishing techniques
            </SectionTitle>
            <TechniqueMatrix result={result} />
          </Card>

          <div className="grid gap-5 xl:grid-cols-2">
            <Card>
              <SectionTitle hint="Pending candidates whose confirmation would move the separation. Green sharpens the current leader; red would challenge it. Confirm or reject straight from here.">
                Confirm to sharpen
              </SectionTitle>
              {result.pending.length ? (
                <ul className="space-y-2">
                  {result.pending.map((item) => (
                    <li key={item.technique_id} className="rounded-xl border border-border bg-surface-2/60 px-4 py-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <DeltaChip value={item.delta_separation} />
                        <span className="font-medium">{item.technique_name}</span>
                        <span className="mono text-[11px] text-faint">{item.technique_id}</span>
                        <span className="text-[11px] text-muted">p {fmt(item.probability)} · exclusivity {pct(item.exclusivity, 0)} · specificity {pct(item.specificity, 0)}</span>
                        {!item.keeps_leader ? (
                          <Tip content="The group with the most evidence the rest of the shortlist cannot explain would change: the naming gate would then be scored against it.">
                            <Badge tone="rose">evidence favourite → {item.new_leader_name}</Badge>
                          </Tip>
                        ) : null}
                      </div>
                      <p className="mt-1.5 text-xs leading-relaxed text-muted">{item.observation}</p>
                      <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[11px] text-faint">
                        documented by
                        {item.documented_by.map((g) => (
                          <Badge key={g.group_id}>{g.name}</Badge>
                        ))}
                      </div>
                      <div className="mt-2 flex gap-2">
                        <Button
                          size="sm"
                          variant="secondary"
                          loading={apply.isPending}
                          onClick={() =>
                            apply.mutate([
                              { observation_id: item.observation_id, technique_id: item.technique_id, action: 'confirm', rationale: 'confirmed from the Distinguish step' },
                            ])
                          }
                        >
                          <Check className="size-3.5" /> Confirm
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          loading={apply.isPending}
                          onClick={() =>
                            apply.mutate([
                              { observation_id: item.observation_id, technique_id: item.technique_id, action: 'reject', rationale: 'rejected from the Distinguish step' },
                            ])
                          }
                        >
                          <X className="size-3.5" /> Reject
                        </Button>
                      </div>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted">No pending candidate would change the separation of this shortlist.</p>
              )}
            </Card>
            <Card>
              <SectionTitle hint="Interview leads replayed as a yes and as a no. Answering in the interview records the evidence and recomputes the ranking.">
                Ask to sharpen
              </SectionTitle>
              {result.software?.length ? (
                <ul className="mb-3 space-y-2">
                  {result.software.map((item) => (
                    <li key={item.software_id} className="rounded-xl border border-border bg-surface-2/60 px-4 py-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <DeltaChip value={item.delta_found} label="found" />
                        <DeltaChip value={item.delta_not_found} label="not found" />
                        <span className="font-medium">{item.name}</span>
                        <span className="mono text-[11px] text-faint">{item.software_id}</span>
                      </div>
                      <div className="mt-1 text-[11px] text-muted">
                        malware ATT&CK documents only for {item.group_name} (top-three place {item.target_rank}) ·{' '}
                        {item.technique_count} documented techniques
                      </div>
                    </li>
                  ))}
                </ul>
              ) : null}
              {result.leads.length ? (
                <ul className="space-y-2">
                  {result.leads.map((lead) => (
                    <li key={lead.technique_id} className="rounded-xl border border-border bg-surface-2/60 px-4 py-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <DeltaChip value={lead.delta_yes} label="yes" />
                        <DeltaChip value={lead.delta_no} label="no" />
                        <span className="font-medium">{lead.technique_name}</span>
                        <span className="mono text-[11px] text-faint">{lead.technique_id}</span>
                      </div>
                      <div className="mt-1 text-[11px] text-muted">
                        priority {fmt(lead.priority, 3)} · documented in {lead.supporting_group_count}/{lead.candidate_group_count} candidate profiles
                        {lead.leader_after_yes && lead.leader_after_yes !== result.leader.name ? (
                          <span className="ml-2 text-rose">a yes would make {lead.leader_after_yes} lead</span>
                        ) : null}
                      </div>
                      {lead.first_step ? <p className="mt-1 text-xs text-muted">{lead.first_step}</p> : null}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted">No open lead would change the separation.</p>
              )}
              <Button variant="outline" className="mt-4" onClick={() => navigate(`/cases/${caseId}/interview`)}>
                <MessageSquare className="size-4" /> Go to the interview
              </Button>
            </Card>
          </div>
        </>
      )

  return (
    <div className="space-y-5">
      <PageHeader
        step="Step 3 · Distinguish"
        title="What separates the shortlist"
        lead="After the ensemble picks the top candidates, only evidence the other shortlisted groups cannot explain counts towards naming one of them. This view shows the techniques doing that work and which pending mappings or interview leads would change the separation the most."
        action={
          <Button variant="outline" onClick={() => navigate(`/cases/${caseId}/interview`)}>
            Continue to interview <ArrowRight className="size-4" />
          </Button>
        }
      />
      {!data.ranking.length ? (
        <Callout tone="warning" title="No shortlist yet">
          Add more concrete evidence first; nothing can be distinguished without candidate groups.
        </Callout>
      ) : (
        <Tabs
          defaultValue="top3"
          tabs={[
            { value: 'top3', label: 'Top three', content: <TopThreeTab caseId={caseId} /> },
            {
              value: 'shortlist',
              label: 'Whole shortlist',
              content: (
                <div className="space-y-5">
                  {shortlist}
                </div>
              ),
            },
          ]}
        />
      )}
    </div>
  )
}

function TopThreeTab({ caseId }: { caseId: string }) {
  const query = useTop3(caseId)
  if (query.isLoading) {
    return (
      <Card>
        <Spinner label="Comparing the first three candidates…" />
      </Card>
    )
  }
  if (query.isError || !query.data) {
    return (
      <Callout tone="danger" title="Could not compare the top three">
        {describe(query.error)}
      </Callout>
    )
  }
  return <TopThreePanel view={query.data} />
}

function TechniqueMatrix({ result }: { result: DistinguishResponse }) {
  const groups = result.shortlist
  const rows = result.techniques
  if (!rows.length) return <p className="text-sm text-muted">No accepted technique overlaps the shortlist.</p>
  const maxWeight = Math.max(...rows.map((r) => r.weight), 0.0001)
  return (
    <div className="overflow-auto rounded-xl border border-border">
      <table className="data">
        <thead>
          <tr>
            <th className="min-w-[220px]">Technique</th>
            <th className="min-w-[140px]">Weight</th>
            {groups.map((g) => (
              <th key={g.group_id} className="text-center">
                <Tip content={`${g.name} · relative index ${g.relative_score} · discriminative mass ${fmt(g.total ?? 0)}`}>
                  <span className="inline-block max-w-[84px] truncate align-bottom">
                    {g.rank}. {g.name}
                  </span>
                </Tip>
              </th>
            ))}
            <th className="text-right">Excl.</th>
            <th className="text-right">Spec.</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.technique_id} className={cn(row.pending && 'opacity-80')}>
              <td>
                <div className="font-medium">
                  {row.technique_name || row.technique_id}
                  {row.pending ? <Badge tone="violet" className="ml-2">near miss</Badge> : null}
                </div>
                <div className="mono text-[11px] text-faint">
                  {row.technique_id} · confidence {fmt(row.confidence)}
                </div>
              </td>
              <td>
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-20 overflow-hidden rounded-full bg-surface-3">
                    <div
                      className={cn('h-full rounded-full', row.weight > 0 ? 'bg-accent' : 'bg-faint')}
                      style={{ width: `${(row.weight / maxWeight) * 100}%` }}
                    />
                  </div>
                  <span className="tabular text-xs text-muted">{fmt(row.weight, 3)}</span>
                </div>
              </td>
              {groups.map((g) => {
                const documented = row.documented_by.includes(g.group_id)
                const contribution = row.contributions[g.group_id] ?? 0
                return (
                  <td key={g.group_id} className="text-center">
                    {documented ? (
                      <span
                        className="inline-grid size-6 place-items-center rounded-md text-accent"
                        style={{ background: `rgba(34, 211, 238, ${0.1 + 0.4 * Math.min(1, contribution / maxWeight)})` }}
                        title={`contribution ${fmt(contribution, 3)}`}
                      >
                        <Check className="size-3.5" />
                      </span>
                    ) : (
                      <span className="text-faint">·</span>
                    )}
                  </td>
                )
              })}
              <td className="text-right tabular text-muted">{pct(row.exclusivity, 0)}</td>
              <td className="text-right tabular text-muted">{pct(row.specificity, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
