import { BrainCircuit, ScanSearch, Scale } from 'lucide-react'
import { useMemo, useState } from 'react'

import { useCyberRE, useReview, useSystem } from '../api/hooks'
import type { CasePayload } from '../api/types'
import { CaseFrame, PageHeader, useCaseId } from '../components/CaseFrame'
import { Badge, Button, Callout, Card, DataTable, Label, Metric, SectionTitle, Toggle } from '../components/ui'
import { fmt } from '../lib/utils'

export function ReviewPage() {
  const id = useCaseId()
  return <CaseFrame>{(data) => <ReviewView caseId={id} data={data} />}</CaseFrame>
}

function ReviewView({ caseId, data }: { caseId: string; data: CasePayload }) {
  const system = useSystem()
  const local = system.data?.mode === 'local'
  const review = useReview(caseId)
  const cyberre = useCyberRE(caseId)
  const [usePrefilter, setUsePrefilter] = useState(true)
  const observations = useMemo(
    () => data.observations.filter((o) => o.status === 'positive' && o.kind !== 'ioc' && o.kind !== 'software'),
    [data.observations],
  )
  const [observationId, setObservationId] = useState<string>(observations[0]?.id ?? '')
  const results = data.cyberre.results
  const pending = observations.filter((o) => !results[o.id])
  const [inspect, setInspect] = useState<string>(Object.keys(results)[0] ?? '')
  const inspected = results[inspect] ?? Object.values(results)[0]

  return (
    <div className="space-y-6">
      <PageHeader
        title="AI review and CyberRE-LLM"
        lead="LLM passes you run on demand: the independent review is advisory and blinded from live intelligence; CyberRE-LLM adds a bounded vote to the ranking. Extraction and the technique verifier run with every analysis."
      />
      {local ? (
        <Callout tone="warning" title="Unavailable in local mode">
          Set OPENROUTER_API_KEY and restart the service to enable the independent review and CyberRE-LLM.
        </Callout>
      ) : null}

      <Card>
        <SectionTitle
          hint="Four blinded and unblinded reviews plus a devil's advocate, reconciled into an advisory opinion. Costs several model calls."
          action={
            <Button variant="primary" disabled={local} loading={review.isPending} onClick={() => review.mutate()}>
              <BrainCircuit className="size-4" /> {data.review ? 'Re-run review' : 'Run independent review'}
            </Button>
          }
        >
          Independent AI review
        </SectionTitle>
        {review.isPending ? (
          <Callout tone="violet">Running blinded, unblinded, devil's-advocate and reconciliation reviews… this takes a minute.</Callout>
        ) : null}
        {data.review ? (
          <div className="space-y-4">
            <div className="rounded-xl border border-violet/30 bg-violet-soft px-5 py-4">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone="violet">Review confidence: {data.review.confidence_label.toUpperCase()}</Badge>
                <span className="text-xs text-muted">advisory only</span>
              </div>
              <h3 className="mt-2 text-lg font-semibold tracking-tight">{data.review.headline}</h3>
              <p className="mt-2 whitespace-pre-line text-sm leading-relaxed text-text/90">{data.review.assessment}</p>
            </div>
            {data.review.advisory_ranking.length ? (
              <div>
                <Label className="mb-2">Advisory ordering (not applied to the deterministic ranking)</Label>
                <DataTable
                  columns={[
                    { key: 'rank', label: 'Advisory rank', className: 'tabular' },
                    { key: 'name', label: 'Candidate' },
                    { key: 'deterministic_rank', label: 'Deterministic rank', className: 'tabular' },
                    { key: 'justification', label: 'Justification', className: 'min-w-[360px] text-muted' },
                  ]}
                  rows={data.review.advisory_ranking as unknown as Record<string, unknown>[]}
                />
              </div>
            ) : null}
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-xl border border-border bg-surface-2/60 p-4">
                <Label className="mb-2">
                  <span className="inline-flex items-center gap-1.5"><Scale className="size-3" /> Fame-bias check</span>
                </Label>
                <div className="grid grid-cols-2 gap-3">
                  <Metric label="Blinded vs named τ" value={fmt(data.review.fame_bias.kendall_tau, 3)} />
                  <Metric label="Prominence correlation" value={fmt(data.review.fame_bias.prominence_correlation, 3)} />
                </div>
                {data.review.fame_bias.largest_promotions.length ? (
                  <p className="mt-2 text-xs text-muted">Largest promotions when named: {data.review.fame_bias.largest_promotions.join(', ')}</p>
                ) : null}
              </div>
              <div className="rounded-xl border border-border bg-surface-2/60 p-4">
                <Label className="mb-2">Devil's advocate</Label>
                <p className="text-sm leading-relaxed text-text/90">{data.review.advocate.counter_case}</p>
                {data.review.advocate.missing_evidence.length ? (
                  <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-muted">
                    {data.review.advocate.missing_evidence.map((item, i) => (
                      <li key={i}>Missing: {item}</li>
                    ))}
                  </ul>
                ) : null}
              </div>
            </div>
            {data.review.limitations.length || data.review.disconfirmation_tests.length ? (
              <div className="grid gap-4 lg:grid-cols-2">
                <div>
                  <Label className="mb-2">Limitations</Label>
                  <ul className="list-disc space-y-1 pl-4 text-sm text-muted">
                    {data.review.limitations.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <Label className="mb-2">Disconfirmation tests</Label>
                  <ul className="list-disc space-y-1 pl-4 text-sm text-muted">
                    {data.review.disconfirmation_tests.map((item, i) => (
                      <li key={i}>{item}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : null}
            {data.review.audit_path ? <p className="text-[11px] text-faint">Audit written to {data.review.audit_path}</p> : null}
          </div>
        ) : (
          <p className="text-sm text-muted">No review has been run for this case.</p>
        )}
      </Card>

      <Card>
        <SectionTitle hint="An independent LLM pass that filters ATT&CK techniques against one observation and ranks the relevant ones. Its result is a bounded, non-authoritative vote in the ranking.">
          CyberRE-LLM
        </SectionTitle>
        {!observations.length ? (
          <p className="text-sm text-muted">No positive action observations are available.</p>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap items-end gap-3">
              <label className="min-w-[320px] flex-1 text-xs">
                <span className="mb-1.5 block font-medium">Observation</span>
                <select
                  value={observationId}
                  onChange={(e) => setObservationId(e.target.value)}
                  className="h-9 w-full rounded-lg border border-border-strong bg-surface-2 px-2 text-sm"
                >
                  {observations.map((o) => (
                    <option key={o.id} value={o.id}>
                      {o.index}. {o.text.slice(0, 120)}
                      {results[o.id] ? ' (scanned)' : ''}
                    </option>
                  ))}
                </select>
              </label>
              <Toggle checked={usePrefilter} onChange={setUsePrefilter} label="ATTACK-BERT pre-filter (recommended)" />
              <Button variant="primary" disabled={local || !observationId} loading={cyberre.isPending} onClick={() => cyberre.mutate({ observationId, usePrefilter })}>
                <ScanSearch className="size-4" /> Scan this observation
              </Button>
              <Button disabled={local || !pending.length} loading={cyberre.isPending} onClick={() => cyberre.mutate({ observationId: null, usePrefilter })}>
                Scan all remaining ({pending.length})
              </Button>
            </div>
            <p className="text-[11px] text-faint">
              About {usePrefilter ? (system.data ? 50 : 50) : system.data?.techniques ?? 0} filter calls plus one ranking call per observation ·{' '}
              {observations.length - pending.length} of {observations.length} observations scanned.
            </p>
            {inspected ? (
              <div className="space-y-3 rounded-xl border border-border bg-surface-2/60 p-4">
                <div className="flex flex-wrap items-center gap-3">
                  <select value={inspect || inspected.observation_id} onChange={(e) => setInspect(e.target.value)} className="h-9 min-w-[320px] rounded-lg border border-border-strong bg-surface px-2 text-sm">
                    {Object.values(results).map((r) => (
                      <option key={r.observation_id} value={r.observation_id}>
                        {r.observation_text.slice(0, 120)}
                      </option>
                    ))}
                  </select>
                  <Badge tone={inspected.applicable ? 'emerald' : 'neutral'}>{inspected.applicable ? 'Vote active' : 'Abstained'}</Badge>
                  <span className="text-xs text-muted">
                    {inspected.model} · {inspected.filter_calls} filter calls · {inspected.kept} kept · {inspected.tokens.toLocaleString()} tokens · ${fmt(inspected.cost_usd, 4)}
                  </span>
                </div>
                <DataTable
                  columns={[
                    { key: 'rank', label: '#', className: 'tabular' },
                    { key: 'technique_id', label: 'Technique', render: (r) => <span>{String(r.technique_name)} <span className="mono text-faint">{String(r.technique_id)}</span></span> },
                    { key: 'reason', label: 'Reason', className: 'text-muted' },
                  ]}
                  rows={inspected.ranking as unknown as Record<string, unknown>[]}
                  maxHeight={320}
                />
              </div>
            ) : null}
          </div>
        )}
      </Card>
    </div>
  )
}
