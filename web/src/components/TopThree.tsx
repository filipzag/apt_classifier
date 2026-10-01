import { BookOpen, GitCompareArrows, Link2, Scale, Sparkles } from 'lucide-react'
import type { ReactNode } from 'react'

import type { Top3Candidate, Top3OneSided, Top3Pair, Top3State, Top3Summary, Top3View } from '../api/types'
import { cn, fmt, pct } from '../lib/utils'
import { Badge, Callout, Card, Label, SectionTitle, Tip } from './ui'

/** Tone and icon for each Top-three state (shared with the Assessment badge). */
const TOP3_STATE_META: Record<
  Top3State,
  { callout: 'info' | 'warning' | 'success'; badge: 'emerald' | 'amber' | 'neutral'; icon: ReactNode }
> = {
  named: { callout: 'success', badge: 'emerald', icon: <Scale className="size-4" /> },
  overlap: { callout: 'warning', badge: 'amber', icon: <Link2 className="size-4" /> },
  split: { callout: 'warning', badge: 'amber', icon: <GitCompareArrows className="size-4" /> },
  thin: { callout: 'info', badge: 'neutral', icon: <Sparkles className="size-4" /> },
  not_separated: { callout: 'info', badge: 'neutral', icon: <Scale className="size-4" /> },
  empty: { callout: 'info', badge: 'neutral', icon: <Scale className="size-4" /> },
}

export function Top3StateBadge({ summary, className }: { summary: Top3Summary; className?: string }) {
  const meta = TOP3_STATE_META[summary.state] ?? TOP3_STATE_META.not_separated
  return (
    <Tip content={summary.summary}>
      <span className={className}>
        <Badge tone={meta.badge}>
          {meta.icon}
          Top three: {summary.label}
        </Badge>
      </span>
    </Tip>
  )
}

const CHANNELS: { key: 'technique' | 'behavior' | 'software' | 'soft'; label: string; tone: string }[] = [
  { key: 'technique', label: 'techniques', tone: 'bg-accent' },
  { key: 'behavior', label: 'actor behaviour', tone: 'bg-emerald' },
  { key: 'software', label: 'software', tone: 'bg-amber' },
  { key: 'soft', label: 'near misses', tone: 'bg-sky' },
]

function MassBar({
  candidate,
  max,
  weights,
}: {
  candidate: Top3Candidate
  max: number
  weights: Top3View['weights']
}) {
  const channels = candidate.channels
  if (!channels) return <p className="text-xs text-faint">Not scored by the naming gate.</p>
  return (
    <div>
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-surface-3">
        {CHANNELS.map((channel) => {
          const value = channels[channel.key] * (weights?.[channel.key] ?? 1)
          if (value <= 0 || max <= 0) return null
          return (
            <div
              key={channel.key}
              className={cn('h-full', channel.tone)}
              style={{ width: `${Math.min(100, (value / max) * 100)}%` }}
              title={`${channel.label} ${fmt(value, 2)}`}
            />
          )
        })}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-muted">
        {CHANNELS.filter((c) => channels[c.key] > 0).map((c) => (
          <span key={c.key} className="inline-flex items-center gap-1">
            <span className={cn('size-1.5 rounded-full', c.tone)} />
            {c.label} {fmt(channels[c.key], 2)}
          </span>
        ))}
        {channels.negative > 0 ? <span className="text-rose">negative answers −{fmt(channels.negative, 2)}</span> : null}
      </div>
    </div>
  )
}

function CandidateCard({ candidate, view, max }: { candidate: Top3Candidate; view: Top3View; max: number }) {
  const contender = view.contest?.promoted === candidate.group_id
  const documented = candidate.documented
  return (
    <div className={cn('rounded-xl border bg-surface-2/60 p-4', candidate.rank === 1 ? 'border-accent/40' : 'border-border')}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <Label>#{candidate.rank}</Label>
          <div className="mt-0.5 truncate text-base font-semibold tracking-tight">{candidate.name}</div>
          <div className="mono text-[11px] text-faint">{candidate.group_id}</div>
        </div>
        {contender ? (
          <Tip content="Holds clearly more evidence that only it explains than the leader. On held-out test folds that did not make it the right answer more often; it is shown, not promoted.">
            <span>
              <Badge tone="amber">
                <GitCompareArrows className="size-3" /> more exclusive evidence
              </Badge>
            </span>
          </Tip>
        ) : null}
      </div>

      <div className="mt-3">
        <div className="flex items-baseline justify-between text-xs">
          <Tip content="Weighted evidence this group explains that the rest of the naming gate's window does not (shared evidence counts for nobody).">
            <span className="text-muted underline decoration-dotted underline-offset-2">Exclusive evidence</span>
          </Tip>
          <span className="tabular font-semibold">{fmt(candidate.channels?.total ?? 0, 2)}</span>
        </div>
        <div className="mt-1.5">
          <MassBar candidate={candidate} max={max} weights={view.weights} />
        </div>
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5 text-[11px]">
        {candidate.combiner_probability !== null ? (
          <Tip content={view.combiner_note ?? 'Learned combiner probability (fitted on ATT&CK folds).'}>
            <span>
              <Badge>combiner p {fmt(candidate.combiner_probability, 2)}</Badge>
            </span>
          </Tip>
        ) : null}
        {candidate.rrf_rank !== null ? (
          <Tip content="Place in the plain rank-fusion order before the learned combiner re-ordered the shortlist.">
            <span>
              <Badge>rank fusion #{candidate.rrf_rank}</Badge>
            </span>
          </Tip>
        ) : null}
        <Tip
          content={
            candidate.broad_profile
              ? 'One of the largest ATT&CK profiles: a broad profile explains many techniques by default, so shared techniques say little about it.'
              : 'Techniques ATT&CK documents for this group.'
          }
        >
          <span>
            <Badge tone={candidate.broad_profile ? 'amber' : 'neutral'}>
              {candidate.profile_size} documented techniques{candidate.broad_profile ? ' · broad' : ''}
            </Badge>
          </span>
        </Tip>
      </div>

      {candidate.named_software.length ? (
        <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px] text-muted">
          named software
          {candidate.named_software.map((s) => (
            <Badge key={s.id} tone="amber">
              {s.name}
            </Badge>
          ))}
        </div>
      ) : null}

      <div className="mt-3 rounded-lg border border-border bg-surface-1/60 px-3 py-2">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] text-faint">
          <BookOpen className="size-3" /> ATT&CK documents (not inferred from this case)
        </div>
        <div className="mt-1.5 flex flex-wrap gap-1.5 text-[11px]">
          {documented.origins.length ? (
            documented.origins.map((origin) => (
              <Tip key={origin.country} content={origin.sentence}>
                <span>
                  <Badge>{origin.country}</Badge>
                </span>
              </Tip>
            ))
          ) : (
            <span className="text-faint">no documented origin</span>
          )}
          {documented.type ? (
            <Tip content={`ATT&CK wording: ${[...documented.state_markers, ...documented.criminal_markers].join(', ')}`}>
              <span>
                <Badge>{documented.type}</Badge>
              </span>
            </Tip>
          ) : null}
          {documented.motivations.map((motivation) => (
            <Badge key={motivation}>{motivation}</Badge>
          ))}
        </div>
      </div>
    </div>
  )
}

function OneSided({ title, side }: { title: string; side: Top3OneSided }) {
  const empty = !side.technique_count && !side.software.length && !side.behavior_win_count
  return (
    <div className="min-w-0">
      <div className="text-[11px] font-medium text-muted">{title}</div>
      {empty ? (
        <p className="mt-1 text-xs text-faint">Nothing in this case.</p>
      ) : (
        <div className="mt-1.5 space-y-1.5">
          {side.software.length ? (
            <div className="flex flex-wrap gap-1">
              {side.software.map((s) => (
                <Badge key={s.id} tone="amber">
                  {s.name}
                </Badge>
              ))}
            </div>
          ) : null}
          {side.techniques.length ? (
            <div className="flex flex-wrap gap-1">
              {side.techniques.map((t) => (
                <Tip key={t.technique_id} content={`${t.name || t.technique_id} · confidence ${fmt(t.confidence)} · documented specificity ${pct(t.specificity, 0)}`}>
                  <span>
                    <Badge tone="accent" className="mono">
                      {t.technique_id}
                    </Badge>
                  </span>
                </Tip>
              ))}
              {side.technique_count > side.techniques.length ? (
                <span className="text-[11px] text-faint">+{side.technique_count - side.techniques.length} more</span>
              ) : null}
            </div>
          ) : null}
          {side.behavior_win_count ? (
            <Tip
              content={
                <ul className="space-y-1">
                  {side.behavior_wins.map((w) => (
                    <li key={w.observation_id}>{w.text}</li>
                  ))}
                </ul>
              }
            >
              <span className="inline-block text-[11px] text-emerald underline decoration-dotted underline-offset-2">
                closer actor-behaviour match on {side.behavior_win_count} observation{side.behavior_win_count === 1 ? '' : 's'}
              </span>
            </Tip>
          ) : null}
        </div>
      )}
    </div>
  )
}

function SeparationScale({ pair }: { pair: Top3Pair }) {
  // −1 (fully favours b) … +1 (fully favours a); the naming bar sits at ±0.5
  const position = (Math.max(-1, Math.min(1, pair.separation)) + 1) / 2
  return (
    <div className="relative h-2 w-full rounded-full bg-surface-3" role="presentation">
      <div className="absolute inset-y-0 left-1/4 w-px bg-text/30" />
      <div className="absolute inset-y-0 left-1/2 w-px bg-text/50" />
      <div className="absolute inset-y-0 left-3/4 w-px bg-text/30" />
      <div
        className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-surface-1 bg-accent"
        style={{ left: `${position * 100}%` }}
      />
    </div>
  )
}

function PairRow({ pair }: { pair: Top3Pair }) {
  const favoured = pair.favours === pair.a ? pair.a_name : pair.favours === pair.b ? pair.b_name : null
  return (
    <li className="rounded-xl border border-border bg-surface-2/60 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm font-medium">
          {pair.a_name} <span className="text-faint">vs</span> {pair.b_name}
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {pair.related_reason ? (
            <Tip content={`ATT&CK links these groups: ${pair.related_reason}. Documented profile overlap ${pct(pair.containment, 0)}.`}>
              <span>
                <Badge tone="amber">
                  <Link2 className="size-3" /> {pair.overlap_cluster ? 'documented overlap · one cluster' : 'documented-related'}
                </Badge>
              </span>
            </Tip>
          ) : null}
          <Badge className="tabular">
            separation {pair.separation > 0 ? '+' : ''}
            {fmt(pair.separation, 2)}
            {favoured ? ` · favours ${favoured}` : ''}
          </Badge>
        </div>
      </div>
      <div className="mt-2 grid grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 text-[11px] text-faint">
        <span className="max-w-[120px] truncate">{pair.b_name}</span>
        <SeparationScale pair={pair} />
        <span className="max-w-[120px] truncate text-right">{pair.a_name}</span>
      </div>
      <div className="mt-3 grid gap-4 sm:grid-cols-2">
        <OneSided title={`Only ${pair.a_name} documents`} side={pair.a_only} />
        <OneSided title={`Only ${pair.b_name} documents`} side={pair.b_only} />
      </div>
    </li>
  )
}

export function TopThreePanel({ view }: { view: Top3View }) {
  const meta = TOP3_STATE_META[view.state] ?? TOP3_STATE_META.not_separated
  if (view.state === 'empty' || !view.candidates.length) {
    return <p className="text-sm text-muted">No candidate groups yet.</p>
  }
  const weightedMax = Math.max(
    view.thresholds.min_discriminative,
    ...view.candidates.map((c) => c.channels?.total ?? 0),
    1,
  )
  return (
    <div className="space-y-5">
      <Callout tone={meta.callout} icon={meta.icon} title={view.label}>
        {view.summary}
        {view.display_order_differs ? (
          <span className="mt-1 block text-xs text-faint">
            The ranking shown elsewhere includes the AI vote; this comparison uses the deterministic order the naming decision used.
          </span>
        ) : null}
      </Callout>

      <div className={cn('grid gap-3', view.candidates.length >= 3 ? 'lg:grid-cols-3' : 'lg:grid-cols-2')}>
        {view.candidates.map((candidate) => (
          <CandidateCard key={candidate.group_id} candidate={candidate} view={view} max={weightedMax} />
        ))}
      </div>

      <Card>
        <SectionTitle
          hint={`Separation of each pair on its exclusive evidence, as the naming gate computes it (−1 to 1). The ticks mark ±${fmt(view.thresholds.separation, 1)}, the naming bar. Below: the evidence in this case that only one side of the pair documents.`}
        >
          Head to head
        </SectionTitle>
        <ul className="space-y-2">
          {view.pairs.map((pair) => (
            <PairRow key={`${pair.a}-${pair.b}`} pair={pair} />
          ))}
        </ul>
      </Card>

    </div>
  )
}
