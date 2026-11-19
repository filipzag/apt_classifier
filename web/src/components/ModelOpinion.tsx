import { AlertTriangle, BadgeCheck, MessageCircleQuestion } from 'lucide-react'
import { useState } from 'react'

import { useModelOpinion, useModelOpinionPending, useSystem } from '../api/hooks'
import type { ModelOpinion, ModelOpinionCandidate, ModelOpinionDecision, ModelOpinionStatus } from '../api/types'
import { cn, fmt } from '../lib/utils'
import { Badge, Button, Callout, Card, Label, SectionTitle, Spinner, Tip } from './ui'

/** The ATT&CK name when matched, else the model's own text; the model can leave the name empty. */
function shownName(candidate: ModelOpinionCandidate): string {
  return candidate.attack_name ?? (candidate.name.trim() || 'no name given')
}

/** The decision in words; each completes "The model …". */
const DECISION_TEXT: Record<ModelOpinionDecision, string> = {
  name: 'would name this group',
  open: 'would not single out one group',
  not_in_attack: 'thinks the actor is not in ATT&CK',
}

/** What to do about each reason the model cannot be asked (the status text itself is shown too). */
const UNAVAILABLE_HELP: Partial<Record<ModelOpinionStatus['status'], string>> = {
  'needs an OpenRouter key':
    'Set OPENROUTER_API_KEY and restart the service. A local model never answers this question.',
  'needs a cloud model':
    'MODEL_OPINION_MODEL names a local model. Set it to an OpenRouter model and restart the service.',
}

/** "99%" or "98.5%": cut, never rounded up, so a stated 0.985 does not read as the 99% of the mark. */
function statedPercent(probability: number): string {
  const tenths = Math.floor(probability * 1000 + 1e-6)
  return `${tenths % 10 === 0 ? tenths / 10 : (tenths / 10).toFixed(1)}%`
}

function plural(count: number, word: string): string {
  return `${count.toLocaleString()} ${word}${count === 1 ? '' : 's'}`
}

function askedAt(iso: string): string {
  const when = new Date(iso)
  return Number.isNaN(when.getTime()) ? iso : when.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })
}

/** True when the model used another name (an alias) than the ATT&CK name shown for the group. */
function wroteAnotherName(candidate: ModelOpinionCandidate): boolean {
  return (
    candidate.attack_name !== null && candidate.name.trim().toLowerCase() !== candidate.attack_name.trim().toLowerCase()
  )
}

function decisionSentence(opinion: ModelOpinion): string {
  if (!opinion.first && opinion.decision === 'name') return 'The model would name a group but gave none.'
  return `The model ${DECISION_TEXT[opinion.decision] ?? `answered "${opinion.decision}"`}.`
}

function Rationale({ text }: { text: string }) {
  const [open, setOpen] = useState(false)
  const long = text.length > 420
  return (
    <div>
      <Label>The model's reasoning</Label>
      <p className={cn('mt-1 whitespace-pre-line text-sm leading-relaxed text-muted', long && !open && 'line-clamp-4')}>
        {text}
      </p>
      {long ? (
        <button
          type="button"
          className="focus-ring mt-1 rounded text-xs text-text underline-offset-2 hover:underline"
          aria-expanded={open}
          onClick={() => setOpen(!open)}
        >
          {open ? 'Show less' : 'Show all'}
        </button>
      ) : null}
    </div>
  )
}

function Answer({ opinion, status }: { opinion: ModelOpinion; status: ModelOpinionStatus }) {
  const first = opinion.first
  // the first choice is the first candidate; the rest are the model's alternatives
  const others = opinion.candidates.slice(first ? 1 : 0)
  const bar = statedPercent(opinion.confident_rule.probability)
  const cut = opinion.statements_total > opinion.statements_sent
  const footer = [
    { text: opinion.model },
    { text: `effort ${opinion.effort}` },
    {
      text: cut
        ? `the first ${opinion.statements_sent.toLocaleString()} of ${plural(opinion.statements_total, 'statement')} sent`
        : `${plural(opinion.statements_sent, 'statement')} sent`,
    },
    {
      text: opinion.names_masked
        ? `${plural(opinion.names_masked, 'group-name mention')} hidden`
        : 'no group name found to hide',
    },
    {
      text: `$${fmt(opinion.cost_usd, 4)}`,
      title: `${opinion.prompt_tokens.toLocaleString()} prompt and ${opinion.completion_tokens.toLocaleString()} completion tokens for the answer shown. If a first answer could not be read, the request was sent once more and that attempt was charged as well; it is not in this figure.`,
    },
    { text: `asked ${askedAt(opinion.asked_at)}` },
  ]

  return (
    <div className="space-y-4">
      {opinion.stale ? (
        <Callout
          tone="warning"
          icon={<AlertTriangle className="size-4" />}
          title="The case text changed after this answer"
        >
          The answer below was given for the earlier statements. Ask again for the current ones (another paid call).
        </Callout>
      ) : null}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
        <div className="self-start rounded-xl border border-border bg-surface-2/60 px-4 py-3">
          <Label>The model's first choice</Label>
          {first ? (
            <>
              <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1">
                <span className="text-xl font-semibold tracking-tight">{shownName(first)}</span>
                {first.group_id ? (
                  <span className="mono text-sm text-faint">{first.group_id}</span>
                ) : (
                  <Badge tone="neutral">no ATT&CK match</Badge>
                )}
              </div>
              {wroteAnotherName(first) ? (
                <div className="mt-0.5 text-xs text-muted">The model wrote “{first.name}”.</div>
              ) : null}
              <div className="mt-1 text-xs text-muted">
                {first.probability !== null ? (
                  <>
                    Stated probability <span className="tabular text-text">{statedPercent(first.probability)}</span>: the
                    model's own number, not calibrated.
                  </>
                ) : (
                  'The model stated no probability.'
                )}
              </div>
            </>
          ) : (
            <div className="mt-1 text-xl font-semibold tracking-tight">No candidate</div>
          )}
          <p className="mt-3 text-sm leading-relaxed">{decisionSentence(opinion)}</p>
          {opinion.confident_name ? (
            <div className="mt-2 flex">
              <Tip
                content={`The model would name an ATT&CK group and states ${bar} or more. This cut-off was tested for ${status.registered_model} only. It is the model's own number, not a calibrated probability, and the product's decision never uses it.`}
              >
                <span className="inline-flex max-w-full">
                  <Badge tone="neutral" className="whitespace-normal text-left">
                    <BadgeCheck className="size-3 shrink-0" />
                    confident name (stated {bar} or more)
                  </Badge>
                </span>
              </Tip>
            </div>
          ) : null}
          {!opinion.confident_rule.applies ? (
            <p className="mt-2 text-[11px] leading-relaxed text-faint">
              The confident-name mark (stated {bar} or more) was measured for {status.registered_model} only, at one
              reasoning effort. This answer came from {opinion.model} at effort {opinion.effort}, so the mark is not
              used here.
            </p>
          ) : null}
        </div>
        <div className="space-y-3">
          <div>
            <Label>Compared with the product's answer</Label>
            <p className="mt-1 text-sm leading-relaxed">{opinion.agreement.text}</p>
          </div>
          {others.length ? (
            <div>
              <Label>The model's other candidates</Label>
              <ul className="mt-1 space-y-1 text-sm">
                {others.map((candidate, index) => (
                  <li key={`${index}-${candidate.name}`} className="flex flex-wrap items-baseline gap-x-2">
                    <span className="tabular text-faint">{index + (first ? 2 : 1)}.</span>
                    <span title={wroteAnotherName(candidate) ? `The model wrote “${candidate.name}”.` : undefined}>
                      {shownName(candidate)}
                    </span>
                    {candidate.group_id ? (
                      <span className="mono text-xs text-faint">{candidate.group_id}</span>
                    ) : (
                      <span className="text-xs text-faint">no ATT&CK match</span>
                    )}
                    <span className="tabular text-xs text-muted">
                      {candidate.probability !== null
                        ? `stated ${statedPercent(candidate.probability)}`
                        : 'no stated probability'}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {opinion.rationale ? <Rationale text={opinion.rationale} /> : null}
        </div>
      </div>
      <div className="border-t border-border pt-3 text-[11px] leading-relaxed text-faint">
        <div>
          {footer.map((part, index) => (
            <span key={index}>
              {index ? ' · ' : ''}
              <span title={part.title}>{part.text}</span>
            </span>
          ))}
        </div>
        <div className="mt-1">{opinion.note}</div>
      </div>
    </div>
  )
}

/**
 * The analyst-requested second opinion of a language model, beside the product's own answer.
 * Display only and neutral in tone: nothing here is the product's decision. Shown only when
 * the service has the feature switched on (`system.model_opinion.enabled`); one paid call
 * per click, never on load.
 */
export function ModelOpinionCard({ caseId, opinion }: { caseId: string; opinion: ModelOpinion | null | undefined }) {
  const system = useSystem()
  const ask = useModelOpinion(caseId)
  // not ask.isPending: that forgets a call still under way when the page is left and reopened
  const asking = useModelOpinionPending(caseId)
  const status = system.data?.model_opinion
  // off by default: then the page shows nothing of it, not even a disabled button
  if (!status?.enabled) return null

  const help = UNAVAILABLE_HELP[status.status]
  // the cost was measured for the tested model; another model's price is not known here
  const asMeasured = status.model === status.registered_model && status.effort === status.registered_effort
  const cost = !status.cost_note
    ? 'A paid call each time you ask.'
    : asMeasured
      ? `A paid call each time you ask: ${status.cost_note}.`
      : `A paid call each time you ask. For ${status.registered_model} at effort ${status.registered_effort} the cost was ${status.cost_note}; the cost of this model and effort may differ.`
  return (
    <Card>
      <SectionTitle
        hint={`A second opinion from a language model (${status.model}). ${cost} The case statements are sent to OpenRouter with ATT&CK group names hidden. It never changes the product's decision.`}
        action={
          <Button
            variant={!opinion || opinion.stale ? 'secondary' : 'outline'}
            disabled={!status.available}
            loading={asking}
            onClick={() => ask.mutate()}
          >
            {asking ? null : <MessageCircleQuestion className="size-4" />}
            {opinion ? 'Ask again' : 'Ask the model'}
          </Button>
        }
      >
        Model second opinion
      </SectionTitle>
      {!status.available ? (
        <p className="mb-3 text-xs leading-relaxed text-muted">
          Not available: {status.status}.{help ? ` ${help}` : ''}
        </p>
      ) : null}
      {asking ? <Spinner className="mb-3" label="Asking the model. This can take up to a minute." /> : null}
      {opinion ? (
        <div className={cn(asking && 'opacity-60')}>
          {/* keyed by the answer so a new one starts with its reasoning collapsed */}
          <Answer key={opinion.asked_at} opinion={opinion} status={status} />
        </div>
      ) : asking ? null : (
        <p className="text-sm text-muted">The model has not been asked about this case.</p>
      )}
    </Card>
  )
}
