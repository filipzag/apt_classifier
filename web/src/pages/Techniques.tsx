import { ArrowRight, Check, HelpCircle, Search, X } from 'lucide-react'
import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api } from '../api/client'
import { useApplyDecisions } from '../api/hooks'
import type { Candidate, CandidateState, CasePayload, DecisionInput } from '../api/types'
import { CaseFrame, PageHeader, useCaseId } from '../components/CaseFrame'
import { Badge, Button, Card, Input, Segmented, Tip, Toggle } from '../components/ui'
import { cn, fmt } from '../lib/utils'
import { useQuery } from '@tanstack/react-query'

type Action = 'keep' | 'confirm' | 'reject' | 'replace'
interface Draft {
  action: Action
  replacement: string
  rationale: string
}
const EMPTY_DRAFT: Draft = { action: 'keep', replacement: '', rationale: '' }

const STATE_META: Record<CandidateState, { label: string; tone: 'emerald' | 'amber' | 'rose' }> = {
  auto_accepted: { label: 'Accepted', tone: 'emerald' },
  needs_review: { label: 'Needs review', tone: 'amber' },
  auto_rejected: { label: 'Rejected', tone: 'rose' },
}

function evidenceSummary(candidate: Candidate): string[] {
  const parts: string[] = []
  for (const item of candidate.evidence) {
    if (item.source === 'acceptance_model' && item.score !== null) parts.push(`p=${fmt(item.score)}`)
    else if (item.source === 'dense' && item.score !== null) parts.push(`dense ${fmt(item.score)} (${item.field ?? '?'})`)
    else if (item.source === 'lexical' && item.rank !== null) parts.push(`lexical #${item.rank}`)
    else if (item.source === 'explicit_cue' || item.source === 'structured_signal') parts.push('cue')
    else if (item.source === 'corroboration') parts.push('corroborated')
    else if (item.source === 'analyst_review') parts.push('analyst')
  }
  return Array.from(new Set(parts))
}

export function TechniquesPage() {
  const id = useCaseId()
  return <CaseFrame>{(data) => <TechniquesView caseId={id} data={data} />}</CaseFrame>
}

function TechniquesView({ caseId, data }: { caseId: string; data: CasePayload }) {
  const navigate = useNavigate()
  const apply = useApplyDecisions(caseId)
  const [states, setStates] = useState<Record<CandidateState, boolean>>({
    auto_accepted: true,
    needs_review: true,
    auto_rejected: false,
  })
  const [topOnly, setTopOnly] = useState(data.candidate_counts.needs_review > 30)
  const [search, setSearch] = useState('')
  const [drafts, setDrafts] = useState<Record<string, Draft>>({})

  const observations = useMemo(
    () => new Map(data.observations.map((o) => [o.id, o])),
    [data.observations],
  )
  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return data.candidates.filter((c) => {
      if (!states[c.state]) return false
      if (topOnly && c.rank > 3) return false
      if (!needle) return true
      const text = observations.get(c.observation_id)?.text ?? ''
      return (
        text.toLowerCase().includes(needle) ||
        c.technique_id.toLowerCase().includes(needle) ||
        c.technique_name.toLowerCase().includes(needle)
      )
    })
  }, [data.candidates, states, topOnly, search, observations])

  const grouped = useMemo(() => {
    const map = new Map<string, Candidate[]>()
    for (const c of visible) {
      const list = map.get(c.observation_id) ?? []
      list.push(c)
      map.set(c.observation_id, list)
    }
    return Array.from(map.entries())
      .map(([oid, list]) => ({ oid, list: list.sort((a, b) => a.rank - b.rank), index: list[0].observation_index }))
      .sort((a, b) => a.index - b.index)
  }, [visible])

  const decisions: DecisionInput[] = useMemo(
    () =>
      Object.entries(drafts)
        .filter(([, d]) => d.action !== 'keep')
        .map(([key, d]) => {
          const [observation_id, technique_id] = key.split('|')
          return {
            observation_id,
            technique_id,
            action: d.action as 'confirm' | 'reject' | 'replace',
            replacement: d.action === 'replace' ? d.replacement.trim().toUpperCase() : null,
            rationale: d.rationale.trim() || null,
          }
        }),
    [drafts],
  )
  const problems = decisions.filter((d) => d.action === 'replace' && !/^T\d{4}(\.\d{3})?$/.test(d.replacement ?? ''))

  const setDraft = (key: string, patch: Partial<Draft>) =>
    setDrafts((all) => ({
      ...all,
      [key]: { ...EMPTY_DRAFT, ...(all[key] ?? {}), ...patch },
    }))

  const counts = data.candidate_counts
  return (
    <div className="space-y-5 pb-24">
      <PageHeader
        step="Step 2 · Technique review"
        title="Decide which mappings vote"
        lead="Every observation was matched against ATT&CK. Accepted mappings vote in the group ranking; candidates that need review are shown for you to confirm, reject or replace. Decisions are saved with the case and the ranking is recomputed."
        action={
          <Button variant="outline" onClick={() => navigate(`/cases/${caseId}/interview`)}>
            Continue to interview <ArrowRight className="size-4" />
          </Button>
        }
      />

      <Card className="flex flex-wrap items-center gap-3 py-3">
        <div className="flex flex-wrap items-center gap-1.5">
          {(Object.keys(STATE_META) as CandidateState[]).map((state) => (
            <button
              key={state}
              type="button"
              onClick={() => setStates((s) => ({ ...s, [state]: !s[state] }))}
              className={cn(
                'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors focus-ring',
                states[state]
                  ? 'border-border-strong bg-surface-3 text-text'
                  : 'border-border text-faint hover:text-muted',
              )}
              aria-pressed={states[state]}
            >
              <span
                className={cn(
                  'size-1.5 rounded-full',
                  STATE_META[state].tone === 'emerald' && 'bg-emerald',
                  STATE_META[state].tone === 'amber' && 'bg-amber',
                  STATE_META[state].tone === 'rose' && 'bg-rose',
                )}
              />
              {STATE_META[state].label}
              <span className="tabular text-faint">{counts[state]}</span>
            </button>
          ))}
        </div>
        <Toggle checked={topOnly} onChange={setTopOnly} label="Top 3 per observation" />
        <div className="relative ml-auto w-full max-w-xs">
          <Search className="pointer-events-none absolute left-2.5 top-2.5 size-4 text-faint" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search observation or technique…"
            className="pl-8"
          />
        </div>
        <span className="text-xs text-faint">
          {visible.length} of {data.candidates.length} candidates
        </span>
      </Card>

      {grouped.map(({ oid, list, index }) => {
        const observation = observations.get(oid)
        return (
          <Card key={oid} className="p-0">
            <div className="flex items-start gap-3 border-b border-border px-5 py-3">
              <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-md bg-surface-3 text-[11px] font-semibold tabular text-muted">
                {index}
              </span>
              <p className="text-sm leading-relaxed">{observation?.text ?? oid}</p>
              {observation ? (
                <Badge className="ml-auto shrink-0">{observation.source}</Badge>
              ) : null}
            </div>
            <div className="overflow-x-auto">
              <table className="data">
                <thead>
                  <tr>
                    <th className="w-10">#</th>
                    <th>Technique</th>
                    <th className="min-w-[150px]">Status</th>
                    <th>Evidence</th>
                    <th>Decision</th>
                    <th className="w-[280px]">Replacement / rationale</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((c) => {
                    const draft = drafts[c.key] ?? EMPTY_DRAFT
                    const meta = STATE_META[c.state]
                    return (
                      <tr key={c.key} className={cn(draft.action !== 'keep' && 'selected')}>
                        <td className="tabular text-muted">{c.rank}</td>
                        <td className="min-w-[220px]">
                          <div className="font-medium">{c.technique_name}</div>
                          <div className="mono text-[11px] text-faint">
                            {c.technique_id}
                            {c.is_subtechnique && c.parent_technique_id ? ` · sub of ${c.parent_technique_id}` : ''}
                          </div>
                        </td>
                        <td>
                          <div className="flex items-center gap-1.5">
                            <Badge tone={meta.tone}>{meta.label}</Badge>
                            {c.state_reason ? (
                              <Tip content={c.state_reason}>
                                <HelpCircle className="size-3.5 text-faint" />
                              </Tip>
                            ) : null}
                          </div>
                          {c.verification ? (
                            <div className="mt-1">
                              <Tip content={c.verification.reason || 'No reason given.'}>
                                <span
                                  className={cn(
                                    'inline-flex cursor-help items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px]',
                                    c.verification.verdict === 'supported' && 'border-emerald/30 text-emerald',
                                    c.verification.verdict === 'refuted' && 'border-rose/30 text-rose',
                                    !['supported', 'refuted'].includes(c.verification.verdict) && 'border-border-strong text-muted',
                                  )}
                                >
                                  verifier: {c.verification.verdict}
                                </span>
                              </Tip>
                            </div>
                          ) : null}
                        </td>
                        <td className="whitespace-nowrap text-xs text-muted">
                          {evidenceSummary(c).join(' · ') || '–'}
                        </td>
                        <td>
                          <Segmented<Action>
                            value={draft.action}
                            onChange={(action) => setDraft(c.key, { action })}
                            options={[
                              { value: 'keep', label: 'Keep' },
                              { value: 'confirm', label: <span className="inline-flex items-center gap-1"><Check className="size-3" /> Confirm</span> },
                              { value: 'reject', label: <span className="inline-flex items-center gap-1"><X className="size-3" /> Reject</span> },
                              { value: 'replace', label: 'Replace' },
                            ]}
                          />
                        </td>
                        <td>
                          {draft.action === 'replace' ? (
                            <TechniquePicker
                              value={draft.replacement}
                              onChange={(value) => setDraft(c.key, { replacement: value })}
                            />
                          ) : null}
                          {draft.action !== 'keep' ? (
                            <Input
                              value={draft.rationale}
                              onChange={(e) => setDraft(c.key, { rationale: e.target.value })}
                              placeholder="Rationale (optional)"
                              className="mt-1 h-8 text-xs"
                            />
                          ) : null}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        )
      })}
      {!grouped.length ? (
        <Card className="text-center text-sm text-muted">No candidate matches the current filters.</Card>
      ) : null}

      {data.decisions.length ? (
        <Card>
          <details>
            <summary className="cursor-pointer text-sm font-semibold">
              {data.decisions.length} decision{data.decisions.length === 1 ? '' : 's'} already applied to this case
            </summary>
            <ul className="mt-3 space-y-1.5 text-xs">
              {data.decisions.map((d, i) => (
                <li key={i} className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface-2/60 px-3 py-2">
                  <Badge tone={d.action === 'confirm' ? 'emerald' : d.action === 'reject' ? 'rose' : 'violet'}>
                    {d.action}
                  </Badge>
                  <span className="mono">{d.technique_id}</span>
                  {d.replacement ? <span className="mono text-violet">→ {d.replacement}</span> : null}
                  <span className="text-muted">{d.observation}</span>
                  {d.rationale ? <span className="text-faint">— {d.rationale}</span> : null}
                </li>
              ))}
            </ul>
          </details>
        </Card>
      ) : null}

      <div className="fixed inset-x-0 bottom-0 z-20 border-t border-border bg-bg/85 backdrop-blur md:left-[248px]">
        <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-3 px-6 py-3 lg:px-8">
          <span className="text-sm">
            <span className="font-semibold tabular">{decisions.length}</span>{' '}
            <span className="text-muted">pending decision{decisions.length === 1 ? '' : 's'}</span>
          </span>
          {problems.length ? (
            <span className="text-xs text-rose">
              {problems.length} replacement{problems.length === 1 ? '' : 's'} need a valid ATT&CK id (T#### or T####.###)
            </span>
          ) : null}
          <div className="ml-auto flex gap-2">
            <Button variant="ghost" onClick={() => setDrafts({})} disabled={!decisions.length}>
              Reset
            </Button>
            <Button
              variant="primary"
              disabled={!decisions.length || problems.length > 0}
              loading={apply.isPending}
              onClick={() => apply.mutate(decisions, { onSuccess: () => setDrafts({}) })}
            >
              Apply and re-rank
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function TechniquePicker({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  const [open, setOpen] = useState(false)
  const query = useQuery({
    queryKey: ['technique-search', value],
    queryFn: () => api.techniques(value),
    enabled: open && value.trim().length >= 2,
    staleTime: 300_000,
  })
  return (
    <div className="relative">
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder="T1059.001 or a name…"
        className="h-8 text-xs mono"
      />
      {open && query.data?.length ? (
        <ul className="absolute z-30 mt-1 max-h-56 w-72 overflow-auto rounded-lg border border-border-strong bg-surface-3 p-1 shadow-2xl">
          {query.data.map((t) => (
            <li key={t.id}>
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  onChange(t.id)
                  setOpen(false)
                }}
                className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs hover:bg-surface-2"
              >
                <span className="mono text-accent">{t.id}</span>
                <span className="truncate text-muted">{t.name}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}
