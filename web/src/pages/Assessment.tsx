import { ArrowRight, Dna, Info, Link2, ListOrdered, Sliders } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { useRescore } from '../api/hooks'
import type { CasePayload } from '../api/types'
import { CaseFrame, PageHeader, useCaseId } from '../components/CaseFrame'
import { GateMetric, RankedBars, RankHeatmap } from '../components/charts'
import { DossierPanel } from '../components/Dossier'
import { STAGE_META, StageBadge, stageIcon } from '../components/StageBadge'
import { Top3StateBadge } from '../components/TopThree'
import { Badge, Button, Callout, Card, DataTable, Label, Metric, SectionTitle, Segmented, Tabs } from '../components/ui'
import { cn, fmt, pct, titleCase } from '../lib/utils'

export function AssessmentPage() {
  const id = useCaseId()
  return <CaseFrame>{(data) => <AssessmentView caseId={id} data={data} />}</CaseFrame>
}

function AssessmentView({ caseId, data }: { caseId: string; data: CasePayload }) {
  const navigate = useNavigate()
  const [selected, setSelected] = useState<string | null>(null)
  const [rankingView, setRankingView] = useState<'bars' | 'table'>('bars')
  const decision = data.decision
  const meta = decision ? STAGE_META[decision.stage] : null
  const gateMode = decision?.thresholds.gate_mode ?? 'maturity'
  const usesMaturity = gateMode === 'maturity' || gateMode === 'both'
  const usesSeparation = gateMode === 'separation' || gateMode === 'both'
  // only a link that may be read as one cluster under the gate's scope (docs 10.11)
  const sameActivity = decision?.related_groups.filter((g) => g.link === 'same_activity') ?? []
  const otherLinks = decision?.related_groups.filter((g) => g.link !== 'same_activity') ?? []

  return (
    <div className="space-y-6">
      <PageHeader
        step="Step 5 · Assessment"
        title="Attribution assessment"
        lead="The stage decision, the deterministic ranking and the evidence behind it. Nothing here is a probability; the relative index compares candidates within this case only."
        action={
          !data.interview.answered && !data.interview.complete ? (
            <Button variant="outline" onClick={() => navigate(`/cases/${caseId}/interview`)}>
              Sharpen with the interview <ArrowRight className="size-4" />
            </Button>
          ) : null
        }
      />

      {decision && meta ? (
        <Card glow={decision.named} className="relative overflow-hidden">
          <div
            className={cn(
              'pointer-events-none absolute -right-20 -top-20 size-64 rounded-full blur-3xl',
              decision.named || decision.named_cluster ? 'bg-emerald/15' : 'bg-accent/10',
            )}
          />
          <div className="relative grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <StageBadge stage={decision.stage} />
                <span className="text-xs text-muted">{meta.summary}</span>
                {data.top3 && data.top3.state !== 'empty' ? (
                  <button
                    type="button"
                    className="focus-ring rounded-full"
                    onClick={() => navigate(`/cases/${caseId}/distinguish`)}
                    aria-label="Open the top-three comparison"
                  >
                    <Top3StateBadge summary={data.top3} />
                  </button>
                ) : null}
              </div>
              <div className="mt-4 flex items-start gap-3">
                <span
                  className={cn(
                    'grid size-11 shrink-0 place-items-center rounded-xl',
                    decision.named || decision.named_cluster ? 'bg-emerald-soft text-emerald' : 'bg-surface-3 text-muted',
                  )}
                >
                  {stageIcon(decision.stage, 'size-5')}
                </span>
                <div className="min-w-0">
                  <Label>
                    {decision.named_cluster
                      ? 'Named cluster hypothesis (documented overlap)'
                      : decision.named
                        ? 'Named hypothesis'
                        : 'Closest documented group (not an attribution)'}
                  </Label>
                  {decision.named_cluster ? (
                    <>
                      <div className="mt-0.5 text-2xl font-semibold tracking-tight">{decision.named_cluster.label}</div>
                      <div className="mt-1 text-xs text-muted">
                        ATT&CK link: {decision.named_cluster.reason}. The pair separates from{' '}
                        {decision.named_cluster.runner_up_name ?? 'the rest of the shortlist'} (cluster separation{' '}
                        {fmt(decision.named_cluster.separation, 2)}, mass {fmt(decision.named_cluster.discriminative, 2)}
                        {decision.named_cluster.probability !== null
                          ? `, combined probability ${fmt(decision.named_cluster.probability, 2)}`
                          : ''}
                        ), but the evidence does not separate the two. Report the cluster, not either member.
                      </div>
                    </>
                  ) : (
                    <div className="mt-0.5 text-2xl font-semibold tracking-tight">
                      {decision.leading_group_name || 'No candidate'}
                      {decision.leading_group_id ? (
                        <span className="mono ml-2 text-sm font-normal text-faint">{decision.leading_group_id}</span>
                      ) : null}
                    </div>
                  )}
                  {decision.runner_up_name ? (
                    <div className="mt-1 text-xs text-muted">
                      Best-supported alternative:{' '}
                      <button
                        type="button"
                        className="text-text underline-offset-2 hover:underline"
                        onClick={() => setSelected(decision.runner_up_group_id)}
                      >
                        {decision.runner_up_name}
                      </button>
                      {decision.runner_up_related ? ' (documented-related)' : ''}
                    </div>
                  ) : null}
                </div>
              </div>
              <p className="mt-4 text-sm leading-relaxed text-muted">{decision.rationale}</p>
              <div className="mt-4 space-y-2">
                {sameActivity.length ? (
                  <Callout tone="warning" icon={<Link2 className="size-4" />} title="Documented overlap">
                    ATT&CK documents the leader and{' '}
                    {sameActivity.map((g, i) => (
                      <span key={g.group_id}>
                        {i ? '; ' : ''}
                        <span className="text-text">{g.name}</span> ({g.reason})
                      </span>
                    ))}{' '}
                    as the same or overlapping activity.{' '}
                    {decision.stage === 'known_group_hypothesis'
                      ? 'Here the evidence separates the leader from them, so the leader is named alone.'
                      : decision.stage === 'known_cluster_hypothesis'
                        ? 'The named cluster above reports them together.'
                        : 'When the evidence cannot separate them, report the cluster rather than one name.'}
                  </Callout>
                ) : null}
                {otherLinks.length ? (
                  <Callout tone="info" icon={<Link2 className="size-4" />} title="ATT&CK-linked groups">
                    ATT&CK links the leader to{' '}
                    {otherLinks.map((g, i) => (
                      <span key={g.group_id}>
                        {i ? '; ' : ''}
                        <span className="text-text">{g.name}</span> ({g.reason})
                      </span>
                    ))}
                    , but does not document them as the same actor. Keep them apart: look for evidence only one of
                    them documents.
                  </Callout>
                ) : null}
                {decision.decisive_software ? (
                  <Callout tone="violet" icon={<Dna className="size-4" />} title="Decisive tooling">
                    The evidence names{' '}
                    {decision.decisive_software.software.map((s, i) => (
                      <span key={s.id}>
                        {i ? ', ' : ''}
                        <span className="text-text">{s.name}</span> <span className="mono">{s.id}</span>
                      </span>
                    ))}
                    , malware that ATT&CK documents for very few groups, so the software vote was weighted ×
                    {fmt(decision.decisive_software.factor, 1)}.
                  </Callout>
                ) : null}
                {!decision.named && !decision.named_cluster ? (
                  <Callout tone="info" title="No named group meets the evidence bar">
                    {meta.advice} The ranking below is a shortlist of the closest documented groups.
                  </Callout>
                ) : null}
              </div>
              {decision.cluster_id ? (
                <div className="mt-3 text-[11px] text-faint">
                  Activity cluster id (stable hash of the evidence): <span className="mono">{decision.cluster_id}</span>
                </div>
              ) : null}
            </div>
            <div className="space-y-3">
              <div className="grid gap-3 sm:grid-cols-3">
                <GateMetric
                  label="Maturity"
                  value={decision.maturity_score}
                  threshold={usesMaturity ? decision.thresholds.maturity : null}
                  informational={!usesMaturity}
                  help="Evidence volume, independent families, classifier agreement, margin and technique specificity (0–1)."
                />
                <GateMetric
                  label="Margin"
                  value={decision.margin}
                  threshold={usesMaturity ? decision.thresholds.margin : null}
                  informational={!usesMaturity}
                  help="Gap between the first and second candidate's relative index (0–1)."
                />
                <GateMetric
                  label="Separation"
                  value={decision.separation}
                  threshold={usesSeparation ? decision.thresholds.separation : null}
                  informational={!usesSeparation}
                  help="Evidence the leader has that its best alternative cannot explain, as a share of the leader's discriminative evidence (−1 to 1). Shared evidence counts for neither side."
                />
              </div>
              {decision.calibrated_probability !== null ? (
                <div className="rounded-xl border border-violet/30 bg-violet-soft px-4 py-3">
                  <div className="flex items-baseline justify-between gap-2">
                    <Label>Leader probability</Label>
                    <span className="text-[11px] text-faint">learned combiner</span>
                  </div>
                  <div className="mt-1 text-2xl font-semibold tracking-tight tabular text-violet">
                    {pct(decision.calibrated_probability, 0)}
                  </div>
                  <p className="mt-1 text-[11px] leading-relaxed text-faint">{decision.calibration_note}</p>
                </div>
              ) : null}
              <div className="rounded-xl border border-border bg-surface-2/60 px-4 py-3">
                <div className="flex items-center justify-between">
                  <Label>Maturity factors</Label>
                  <span className="text-[11px] text-faint">gate: {gateMode}</span>
                </div>
                <div className="mt-2 space-y-1.5">
                  {Object.entries(decision.maturity_factors)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([name, value]) => (
                      <div key={name} className="grid grid-cols-[130px_1fr_40px] items-center gap-2 text-[11px]">
                        <span className="truncate text-muted">{name.replace(/_/g, ' ')}</span>
                        <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
                          <div className="h-full rounded-full bg-violet/80" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
                        </div>
                        <span className="text-right tabular text-muted">{fmt(value)}</span>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        </Card>
      ) : (
        <Callout tone="info">No attribution decision is available for this case.</Callout>
      )}

      <Card>
        <SectionTitle
          hint="Family-aware reciprocal-rank fusion over the documented ATT&CK overlap, re-ordered by the candidate-relative discriminative mass. Select a candidate to open its dossier."
          action={
            <Segmented
              value={rankingView}
              onChange={setRankingView}
              options={[
                { value: 'bars', label: 'Bars' },
                { value: 'table', label: <span className="inline-flex items-center gap-1"><ListOrdered className="size-3" /> Table</span> },
              ]}
            />
          }
        >
          Candidate groups
        </SectionTitle>
        {!data.ranking.length ? (
          <Callout tone="danger">No candidate has documented technical support. Add evidence in Intake.</Callout>
        ) : rankingView === 'bars' ? (
          <RankedBars
            items={data.ranking.slice(0, 15).map((e) => ({
              key: e.group_id,
              label: (
                <span>
                  <span className="mr-1.5 text-faint tabular">{e.rank}.</span>
                  {e.name}
                  {e.contested ? <Badge tone="amber" className="ml-2">contested</Badge> : null}
                </span>
              ),
              sub: e.group_id,
              value: e.relative_score,
              tone: e.rank === 1 ? (decision?.named ? 'emerald' : 'accent') : 'accent',
            }))}
            max={100}
            onSelect={setSelected}
            selected={selected}
          />
        ) : (
          <DataTable
            columns={[
              { key: 'rank', label: '#', className: 'tabular w-10' },
              {
                key: 'name',
                label: 'Candidate',
                render: (r) => (
                  <span>
                    {String(r.name)} <span className="mono text-faint">{String(r.group_id)}</span>
                  </span>
                ),
              },
              { key: 'relative_score', label: 'Relative index', className: 'tabular', render: (r) => fmt(Number(r.relative_score), 1) },
              {
                key: 'discrimination',
                label: 'Discriminative mass',
                className: 'tabular',
                render: (r) => {
                  const d = r.discrimination as CasePayload['ranking'][number]['discrimination']
                  return d ? `${fmt(d.total)} (t ${fmt(d.technique, 1)} · b ${fmt(d.behavior, 1)} · s ${fmt(d.software, 1)})` : '–'
                },
              },
              { key: 'rank_min', label: 'Best rank', className: 'tabular' },
              { key: 'rank_max', label: 'Worst rank', className: 'tabular' },
              { key: 'rank_std', label: 'Rank σ', className: 'tabular', render: (r) => fmt(Number(r.rank_std)) },
              { key: 'voting', label: 'Voting', className: 'tabular', render: (r) => Object.keys(r.classifier_ranks as object).length },
              { key: 'contested', label: 'Contested' },
            ]}
            rows={data.ranking as unknown as Record<string, unknown>[]}
            rowKey={(r) => String(r.group_id)}
            onRowClick={(r) => setSelected(String(r.group_id))}
            selectedKey={selected}
            maxHeight={520}
          />
        )}
      </Card>

      <Card>
        <SectionTitle hint="Techniques, behaviours, context and origin: what the ranking is made of.">
          Evidence detail
        </SectionTitle>
        <Tabs
          tabs={[
            {
              value: 'techniques',
              label: 'Accepted techniques',
              count: data.techniques.length,
              content: (
                <DataTable
                  columns={[
                    {
                      key: 'technique_id',
                      label: 'Technique',
                      render: (r) => (
                        <span>
                          {String(r.technique_name)} <span className="mono text-faint">{String(r.technique_id)}</span>
                        </span>
                      ),
                    },
                    { key: 'confidence', label: 'Confidence', className: 'tabular', render: (r) => fmt(Number(r.confidence), 3) },
                    { key: 'observation_ids', label: 'Observations', className: 'tabular', render: (r) => (r.observation_ids as string[]).length },
                    { key: 'matched_text', label: 'Matched ATT&CK text', className: 'min-w-[320px] text-muted' },
                  ]}
                  rows={data.techniques as unknown as Record<string, unknown>[]}
                  empty="No technique was accepted."
                  maxHeight={420}
                />
              ),
            },
            {
              value: 'behaviour',
              label: 'Actor behaviour matches',
              count: data.behaviour_matches.length,
              content: (
                <DataTable
                  columns={[
                    { key: 'name', label: 'Candidate' },
                    { key: 'technique_id', label: 'Technique', className: 'mono' },
                    { key: 'similarity', label: 'Similarity', className: 'tabular' },
                    { key: 'observed', label: 'Observed', className: 'min-w-[260px] text-muted' },
                    { key: 'documented', label: 'Documented', className: 'min-w-[260px] text-muted' },
                  ]}
                  rows={data.behaviour_matches as unknown as Record<string, unknown>[]}
                  empty="No actor-specific behaviour matched above the threshold."
                  maxHeight={420}
                />
              ),
            },
            {
              value: 'strategic',
              label: 'Strategic context',
              count: data.strategic_matches.length,
              content: (
                <DataTable
                  columns={[
                    { key: 'rank', label: '#', className: 'tabular' },
                    { key: 'name', label: 'Candidate' },
                    { key: 'coverage', label: 'Coverage', className: 'tabular' },
                    { key: 'factors', label: 'Factors' },
                  ]}
                  rows={data.strategic_matches as unknown as Record<string, unknown>[]}
                  empty="No strategic context was provided."
                />
              ),
            },
            {
              value: 'origin',
              label: 'Origin and intrusion type',
              content: <OriginPanel data={data} />,
            },
          ]}
        />
      </Card>

      <Diagnostics caseId={caseId} data={data} />

      <DossierPanel caseId={caseId} groupId={selected} onClose={() => setSelected(null)} />
    </div>
  )
}

function OriginPanel({ data }: { data: CasePayload }) {
  const origin = data.origin
  return (
    <div className="space-y-4">
      <Callout tone="info" icon={<Info className="size-4" />}>
        Aggregated from ATT&CK reporting on the ranked groups. Never a geolocation or proof of sponsorship.
      </Callout>
      <div className="grid gap-3 sm:grid-cols-3">
        <Metric label="Most supported origin" value={origin.top_country ?? 'Unresolved'} hint={origin.confidence} />
        <Metric label="Intrusion type" value={origin.threat_type.label} hint={origin.threat_type.confidence} />
        <Metric label="Origin-bearing coverage" value={pct(origin.coverage / 100, 0)} hint="of ranked candidates" />
      </div>
      {origin.hypotheses.length ? (
        <DataTable
          columns={[
            { key: 'country', label: 'Country' },
            { key: 'relative_support', label: 'Relative support', className: 'tabular', render: (r) => fmt(Number(r.relative_support), 1) },
            {
              key: 'supporting',
              label: 'Supporting candidates',
              render: (r) => (r.supporting as { name: string }[]).map((s) => s.name).join(', '),
            },
          ]}
          rows={origin.hypotheses as unknown as Record<string, unknown>[]}
        />
      ) : null}
      {origin.threat_type.rationale.length ? (
        <ul className="list-disc space-y-1 pl-5 text-xs text-muted">
          {origin.threat_type.rationale.map((line, i) => (
            <li key={i}>{line}</li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

function Diagnostics({ caseId, data }: { caseId: string; data: CasePayload }) {
  const rescore = useRescore(caseId)
  const [weights, setWeights] = useState<Record<string, number>>(data.weights)
  const dirty = useMemo(
    () => Object.entries(weights).some(([k, v]) => Math.abs((data.weights[k] ?? 0) - v) > 1e-9),
    [weights, data.weights],
  )
  return (
    <Card>
      <details>
        <summary className="cursor-pointer text-[15px] font-semibold tracking-tight">Classifier diagnostics and weights</summary>
        <div className="mt-4 space-y-5">
          <div>
            <Label className="mb-2">Per-classifier ranking matrix</Label>
            <RankHeatmap classifiers={data.classifier_matrix.classifiers} rows={data.classifier_matrix.rows} />
          </div>
          <div className="grid gap-5 lg:grid-cols-2">
            <div>
              <Label className="mb-2">Effective weights after family normalisation</Label>
              <DataTable
                columns={[
                  { key: 'name', label: 'Classifier' },
                  { key: 'weight', label: 'Effective weight', className: 'tabular' },
                ]}
                rows={Object.entries(data.effective_weights).map(([name, weight]) => ({ name: name.replace(/_/g, ' '), weight }))}
              />
            </div>
            <div>
              <Label className="mb-2">Why each candidate is eligible</Label>
              <DataTable
                columns={[
                  { key: 'name', label: 'Group' },
                  { key: 'via', label: 'Eligible via' },
                ]}
                rows={data.eligibility as unknown as Record<string, unknown>[]}
                maxHeight={260}
              />
            </div>
          </div>
          <div>
            <div className="mb-2 flex items-center justify-between">
              <Label>
                <span className="inline-flex items-center gap-1.5"><Sliders className="size-3" /> Ensemble weights</span>
              </Label>
              <div className="flex gap-2">
                <Button size="sm" variant="ghost" disabled={!dirty} onClick={() => setWeights(data.weights)}>
                  Reset
                </Button>
                <Button size="sm" variant="primary" disabled={!dirty} loading={rescore.isPending} onClick={() => rescore.mutate(weights)}>
                  Re-score
                </Button>
              </div>
            </div>
            <div className="grid gap-x-6 gap-y-2 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(weights)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([name, value]) => (
                  <label key={name} className="grid grid-cols-[1fr_44px] items-center gap-2 text-xs">
                    <span className="flex items-center justify-between">
                      <span className="text-muted">{titleCase(name)}</span>
                    </span>
                    <span className="text-right tabular">{fmt(value, 2)}</span>
                    <input
                      type="range"
                      min={0}
                      max={3}
                      step={0.05}
                      value={value}
                      onChange={(e) => setWeights((w) => ({ ...w, [name]: Number(e.target.value) }))}
                      className="col-span-2 accent-cyan-400"
                    />
                  </label>
                ))}
            </div>
          </div>
        </div>
      </details>
    </Card>
  )
}
