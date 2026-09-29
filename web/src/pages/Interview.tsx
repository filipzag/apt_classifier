import { AnimatePresence, motion } from 'framer-motion'
import { ArrowRight, Bot, CheckCircle2, Eye, EyeOff, HelpCircle, RotateCcw, Send, User, XCircle } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { describe, useInterview, useInterviewMutations, useSystem } from '../api/hooks'
import type { CasePayload, InterviewState } from '../api/types'
import { CaseFrame, PageHeader, useCaseId } from '../components/CaseFrame'
import { RankedBars } from '../components/charts'
import { Badge, Button, Callout, Card, DataTable, Input, Label, SectionTitle, Spinner } from '../components/ui'
import { cn, fmt, pct } from '../lib/utils'

const STOP_MESSAGES: Record<string, string> = {
  question_limit: 'The configured question limit was reached. Restart the interview or continue to the assessment.',
  no_candidates: 'No unanswered ATT&CK behaviour is supported by the current candidates. Add more concrete evidence.',
  low_priority: 'The remaining predicted behaviours are below the minimum investigation priority. Continue to the assessment.',
  concentrated_low_value: 'The candidate weights are concentrated and the remaining behaviours would not discriminate. Continue to the assessment.',
  selection_failed: 'The question selector did not return a usable lead. Restart the interview to retry.',
}

const ANSWERS = [
  { status: 'positive', label: 'Yes, observed', icon: CheckCircle2, tone: 'emerald' },
  { status: 'negative', label: 'No, not observed', icon: XCircle, tone: 'rose' },
  { status: 'unknown', label: 'Unknown', icon: HelpCircle, tone: 'neutral' },
  { status: 'not_visible', label: 'Not visible in telemetry', icon: EyeOff, tone: 'amber' },
] as const

const ANSWER_LABEL: Record<string, string> = {
  positive: 'Yes, observed',
  negative: 'No, not observed',
  unknown: 'Unknown',
  not_visible: 'Not visible in available telemetry',
}

export function InterviewPage() {
  const id = useCaseId()
  return <CaseFrame>{(data) => <InterviewView caseId={id} data={data} />}</CaseFrame>
}

function InterviewView({ caseId, data }: { caseId: string; data: CasePayload }) {
  const navigate = useNavigate()
  const interview = useInterview(caseId)
  const { answer, restart } = useInterviewMutations(caseId)
  const system = useSystem()
  const [note, setNote] = useState('')
  const names = new Map(data.ranking.map((e) => [e.group_id, e.name]))
  const state = interview.data

  return (
    <div className="space-y-5">
      <PageHeader
        step="Step 4 · Interview"
        title="Separate the shortlist"
        lead="Each question is the unconfirmed ATT&CK behaviour with the highest expected information gain across the current candidates, phrased as a neutral collection lead. Answers are recorded as evidence and the ranking is recomputed."
        action={
          <div className="flex gap-2">
            <Button
              variant="ghost"
              onClick={() => restart.mutate()}
              loading={restart.isPending}
              disabled={!state || (!state.answered.length && !state.complete)}
            >
              <RotateCcw className="size-4" /> Restart
            </Button>
            <Button variant="outline" onClick={() => navigate(`/cases/${caseId}`)}>
              Assessment <ArrowRight className="size-4" />
            </Button>
          </div>
        }
      />
      {!data.ranking.length ? (
        <Callout tone="warning" title="No candidate group to interview about">
          Add more concrete evidence first; the interview needs at least one documented group with overlap.
        </Callout>
      ) : null}
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-4">
          {interview.isLoading ? (
            <Card>
              <Spinner label={system.data?.mode === 'local' ? 'Selecting the next discriminating question…' : 'Asking the model to phrase the next lead…'} />
            </Card>
          ) : interview.isError || !state ? (
            <Callout tone="danger" title="The interview could not be loaded">
              {describe(interview.error)}
            </Callout>
          ) : (
            <>
              <Transcript state={state} />
              <AnimatePresence mode="wait">
                {state.complete ? (
                  <motion.div key="done" initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
                    <Card>
                      <div className="flex items-start gap-3">
                        <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-emerald-soft text-emerald">
                          <CheckCircle2 className="size-4" />
                        </span>
                        <div>
                          <div className="text-sm font-semibold">Interview complete</div>
                          <p className="mt-1 text-sm text-muted">
                            {STOP_MESSAGES[state.stop_reason ?? ''] ?? 'The interview stopped.'}
                          </p>
                          <Button variant="primary" className="mt-3" onClick={() => navigate(`/cases/${caseId}`)}>
                            Continue to assessment <ArrowRight className="size-4" />
                          </Button>
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                ) : state.question ? (
                  <motion.div
                    key={state.question.technique_id + state.answered.length}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -6 }}
                  >
                    <Card glow>
                      <div className="flex items-start gap-3">
                        <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent">
                          <Bot className="size-4" />
                        </span>
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge tone="accent">Question {state.answered.length + 1} of {state.max_questions}</Badge>
                            <span className="mono text-[11px] text-faint">{state.question.technique_id}</span>
                          </div>
                          <p className="mt-2 text-base font-semibold leading-snug">{state.question.question}</p>
                          <p className="mt-2 text-xs leading-relaxed text-muted">{state.question.hypothesis}</p>
                          <details className="mt-3 rounded-lg border border-border bg-surface-2/60 px-3 py-2">
                            <summary className="cursor-pointer text-xs font-medium text-text">
                              What to inspect before answering
                            </summary>
                            <p className="mt-2 text-[11px] text-faint">
                              Event IDs are collection pivots, not proof: correlate host, user, time, process tree and command line.
                            </p>
                            <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-muted">
                              {state.question.investigation_steps.map((step, i) => (
                                <li key={i}>{step}</li>
                              ))}
                            </ul>
                          </details>
                          <div className="mt-3 grid grid-cols-3 gap-2 text-[11px] text-muted">
                            <Stat label="Profile support" value={`${pct(state.question.profile_support)} (${state.question.supporting_group_count}/${state.question.candidate_group_count})`} />
                            <Stat label="Information gain" value={`${fmt(state.question.information_gain, 3)} bits`} />
                            <Stat label="P(yes)" value={pct(state.question.probability_yes)} />
                          </div>
                          <div className="mt-4 grid gap-2 sm:grid-cols-2">
                            {ANSWERS.map((option) => (
                              <Button
                                key={option.status}
                                variant="secondary"
                                size="lg"
                                disabled={answer.isPending}
                                onClick={() => answer.mutate({ status: option.status, text: note }, { onSuccess: () => setNote('') })}
                                className={cn(
                                  'justify-start',
                                  option.tone === 'emerald' && 'hover:border-emerald/50 hover:text-emerald',
                                  option.tone === 'rose' && 'hover:border-rose/50 hover:text-rose',
                                  option.tone === 'amber' && 'hover:border-amber/50 hover:text-amber',
                                )}
                              >
                                <option.icon className="size-4" /> {option.label}
                              </Button>
                            ))}
                          </div>
                          <div className="mt-2 flex gap-2">
                            <Input
                              value={note}
                              onChange={(e) => setNote(e.target.value)}
                              placeholder="Optional note: where it was seen, event ids, caveats…"
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' && note.trim()) {
                                  const parsed = note.trim().toLowerCase()
                                  const status = parsed.startsWith('y') ? 'positive' : parsed.startsWith('no') ? 'negative' : parsed.startsWith('not') ? 'not_visible' : parsed.startsWith('un') ? 'unknown' : null
                                  if (status) answer.mutate({ status, text: note }, { onSuccess: () => setNote('') })
                                }
                              }}
                            />
                            <Button variant="ghost" disabled title="A note starting with yes / no / unknown / not visible is submitted as the answer">
                              <Send className="size-4" />
                            </Button>
                          </div>
                          {answer.isPending ? <Spinner className="mt-3" label="Recording the answer and re-ranking…" /> : null}
                        </div>
                      </div>
                    </Card>
                  </motion.div>
                ) : (
                  <Callout tone="danger" title="The interview could not produce a question">
                    {state.selection_error ?? 'Restart it to retry.'}
                  </Callout>
                )}
              </AnimatePresence>
            </>
          )}
        </div>
        <aside className="space-y-4">
          <Card>
            <SectionTitle hint="Normalised squared relative scores of the top candidates. Not probabilities.">
              Which candidates drive the questions
            </SectionTitle>
            {state?.weights.length ? (
              <RankedBars
                items={state.weights.slice(0, 8).map((w) => ({
                  key: w.group_id,
                  label: names.get(w.group_id) ?? w.group_id,
                  value: w.weight,
                }))}
                max={Math.max(...state.weights.map((w) => w.weight), 0.0001)}
                valueFormatter={(v) => pct(v, 0)}
              />
            ) : (
              <p className="text-xs text-muted">No weights yet.</p>
            )}
          </Card>
          <Card>
            <SectionTitle hint="The shortlist the selector chooses from, ordered by investigation priority.">
              Predicted leads
            </SectionTitle>
            <DataTable
              columns={[
                { key: 'priority', label: 'Priority', className: 'tabular', render: (r) => fmt(Number(r.priority), 3) },
                {
                  key: 'technique_id',
                  label: 'Technique',
                  render: (r) => (
                    <span>
                      {String(r.technique_name)} <span className="mono text-faint">{String(r.technique_id)}</span>
                    </span>
                  ),
                },
                {
                  key: 'support',
                  label: 'Profiles',
                  className: 'whitespace-nowrap tabular',
                  render: (r) => `${String(r.supporting_group_count)}/${String(r.candidate_group_count)}`,
                },
              ]}
              rows={(state?.leads ?? []) as unknown as Record<string, unknown>[]}
              empty="No leads."
              maxHeight={320}
            />
          </Card>
          {system.data?.elastic_mcp ? (
            <Callout tone="info" icon={<Eye className="size-4" />} title="Elasticsearch gathering">
              Read-only search through Elastic Agent Builder is configured. Use the API endpoint
              /api/cases/{'{id}'}/elastic/search to stage events for the open lead.
            </Callout>
          ) : null}
        </aside>
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface-2/60 px-2.5 py-1.5">
      <Label className="text-[9px]">{label}</Label>
      <div className="mt-0.5 text-xs tabular text-text">{value}</div>
    </div>
  )
}

function Transcript({ state }: { state: InterviewState }) {
  if (!state.answered.length) return null
  return (
    <div className="space-y-3">
      {state.answered.map((item, index) => (
        <div key={index} className="space-y-2">
          <div className="flex items-start gap-3">
            <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-surface-3 text-muted">
              <Bot className="size-3.5" />
            </span>
            <div className="max-w-3xl rounded-xl rounded-tl-sm border border-border bg-surface px-4 py-2.5 text-sm">
              {item.question}
              {item.technique_id ? <div className="mono mt-1 text-[11px] text-faint">{item.technique_id}</div> : null}
            </div>
          </div>
          <div className="flex items-start justify-end gap-3">
            <div
              className={cn(
                'max-w-xl rounded-xl rounded-tr-sm border px-4 py-2.5 text-sm',
                item.status === 'positive' && 'border-emerald/30 bg-emerald-soft',
                item.status === 'negative' && 'border-rose/30 bg-rose-soft',
                item.status === 'not_visible' && 'border-amber/30 bg-amber-soft',
                item.status === 'unknown' && 'border-border bg-surface-2',
              )}
            >
              <span className="font-semibold">{ANSWER_LABEL[item.status] ?? item.status}</span>
              {item.text ? <div className="mt-0.5 text-xs text-muted">{item.text}</div> : null}
            </div>
            <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-surface-3 text-muted">
              <User className="size-3.5" />
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}
