import { useMutation } from '@tanstack/react-query'
import { ChevronDown, FileUp, Loader2, Sparkles, Trash2, Wand2, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { toast } from 'sonner'

import { api } from '../api/client'
import { describe, keys, useSystem } from '../api/hooks'
import { useQueryClient } from '@tanstack/react-query'
import { PageHeader } from '../components/CaseFrame'
import { PlatformPicker } from '../components/PlatformPicker'
import { Button, Callout, Card, Field, Input, Label, SectionTitle, Textarea } from '../components/ui'
import { PLATFORM_HELP } from '../lib/platforms'
import { cn } from '../lib/utils'

const PHASES = [
  'Initial access',
  'Execution',
  'Persistence',
  'Privilege escalation',
  'Credential access',
  'Lateral movement',
  'Command and control',
  'Exfiltration / impact',
]

export function IntakePage() {
  const navigate = useNavigate()
  const [params] = useSearchParams()
  const system = useSystem()
  const client = useQueryClient()
  const [name, setName] = useState('')
  const [narrative, setNarrative] = useState('')
  const [phases, setPhases] = useState<Record<string, string>>({})
  const [showPhases, setShowPhases] = useState(false)
  const [alerts, setAlerts] = useState('')
  const [iocs, setIocs] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [platforms, setPlatforms] = useState<string[]>([])
  const [context, setContext] = useState({
    target_geographies: '',
    target_industries: '',
    motivations: '',
    objectives: '',
    suspected_origins: '',
  })
  const [showContext, setShowContext] = useState(false)
  const [dragging, setDragging] = useState(false)

  const loadSample = useMutation({
    mutationFn: api.sample,
    onSuccess: (result) => {
      setNarrative(result.text)
      if (!name) setName('Sample incident')
      toast.success('Sample incident loaded')
    },
    onError: (error) => toast.error('Could not load the sample', { description: describe(error) }),
  })

  useEffect(() => {
    if (params.get('sample') === '1' && !narrative) loadSample.mutate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const hasInput = useMemo(
    () =>
      Boolean(
        narrative.trim() ||
          Object.values(phases).some((v) => v.trim()) ||
          alerts.trim() ||
          iocs.trim() ||
          files.length,
      ),
    [narrative, phases, alerts, iocs, files],
  )

  const analyse = useMutation({
    mutationFn: () => {
      const form = new FormData()
      form.set('name', name)
      form.set('narrative', narrative)
      form.set('phases', JSON.stringify(phases))
      form.set('alerts', alerts)
      form.set('iocs', iocs)
      form.set('target_platforms', platforms.join(','))
      Object.entries(context).forEach(([key, value]) => form.set(key, value))
      files.forEach((file) => form.append('files', file, file.name))
      return api.createCase(form)
    },
    onSuccess: (payload) => {
      client.setQueryData(keys.case(payload.case_id), payload)
      void client.invalidateQueries({ queryKey: keys.cases })
      void client.invalidateQueries({ queryKey: keys.system })
      toast.success(`${payload.observations.length} observations analysed`, {
        description: `${payload.candidate_counts.auto_accepted} techniques accepted, ${payload.candidate_counts.needs_review} need review.`,
      })
      navigate(`/cases/${payload.case_id}/techniques`)
    },
    onError: (error) => toast.error('Analysis stopped', { description: describe(error) }),
  })

  const local = system.data?.mode === 'local'

  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader
        step="Step 1 · Evidence intake"
        title="Describe what was observed"
        lead="Paste the incident narrative, alerts and indicators, or upload logs. The evidence is normalised into atomic observations, mapped to ATT&CK techniques and ranked against documented groups. Nothing leaves this machine except the configured model provider."
        action={
          <div className="flex gap-2">
            <Button onClick={() => loadSample.mutate()} loading={loadSample.isPending}>
              <Wand2 className="size-4" /> Load sample
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setName('')
                setNarrative('')
                setPhases({})
                setAlerts('')
                setIocs('')
                setFiles([])
                setPlatforms([])
              }}
            >
              <Trash2 className="size-4" /> Clear
            </Button>
          </div>
        }
      />

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="space-y-5">
          <Card>
            <div className="grid gap-4">
              <Field label="Case name" hint="optional">
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Finance workstation intrusion, 2026-09"
                />
              </Field>
              <Field
                label="What happened?"
                hint={local ? 'split into sentences locally' : 'normalised by the extraction model'}
              >
                <Textarea
                  rows={10}
                  value={narrative}
                  onChange={(e) => setNarrative(e.target.value)}
                  placeholder="The actor gained access through a phishing email carrying a macro-enabled document, executed a PowerShell downloader, created a scheduled task for persistence, dumped LSASS with a renamed procdump, and moved laterally over SMB using stolen credentials…"
                />
              </Field>
              <div>
                <div className="mb-1.5 flex items-baseline justify-between gap-3">
                  <span className="text-xs font-medium text-text">Target OS</span>
                  <span className="text-[11px] text-faint">where the activity ran · optional</span>
                </div>
                <PlatformPicker value={platforms} onChange={setPlatforms} />
                <p className="mt-1.5 text-[11px] leading-relaxed text-faint">{PLATFORM_HELP}</p>
              </div>
            </div>
          </Card>

          <Card>
            <button
              type="button"
              onClick={() => setShowPhases((v) => !v)}
              className="flex w-full items-center justify-between text-left focus-ring rounded-lg"
            >
              <SectionTitle className="mb-0" hint="Optional. One box per kill-chain phase keeps observations atomic.">
                Observations by phase
              </SectionTitle>
              <ChevronDown className={cn('size-4 text-muted transition-transform', showPhases && 'rotate-180')} />
            </button>
            {showPhases ? (
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {PHASES.map((phase) => (
                  <Field key={phase} label={phase}>
                    <Textarea
                      rows={3}
                      value={phases[phase] ?? ''}
                      onChange={(e) => setPhases((p) => ({ ...p, [phase]: e.target.value }))}
                      placeholder={`Observed ${phase.toLowerCase()}…`}
                    />
                  </Field>
                ))}
              </div>
            ) : null}
          </Card>

          <Card>
            <SectionTitle hint="Alert titles or exported JSON; domains, hashes and named tooling; Windows/Sysmon JSON or JSONL, Elastic exports, Windows Event XML or eCAR. The format is detected from the content.">
              Alerts, indicators and log files
            </SectionTitle>
            <div className="grid gap-4">
              <Field label="EDR / SIEM alerts">
                <Textarea
                  rows={4}
                  value={alerts}
                  onChange={(e) => setAlerts(e.target.value)}
                  placeholder="Alert titles, descriptions or exported alert JSON…"
                />
              </Field>
              <Field label="Indicators and named tooling">
                <Textarea
                  rows={3}
                  value={iocs}
                  onChange={(e) => setIocs(e.target.value)}
                  placeholder="Domains, IPs, hashes, Mimikatz, Cobalt Strike…"
                />
              </Field>
              <div
                onDragOver={(e) => {
                  e.preventDefault()
                  setDragging(true)
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault()
                  setDragging(false)
                  setFiles((f) => [...f, ...Array.from(e.dataTransfer.files)])
                }}
                className={cn(
                  'rounded-xl border border-dashed px-4 py-5 text-center transition-colors',
                  dragging ? 'border-accent bg-accent-soft' : 'border-border-strong bg-surface-2/40',
                )}
              >
                <FileUp className="mx-auto size-5 text-muted" />
                <p className="mt-2 text-xs text-muted">
                  Drop structured evidence files here, or{' '}
                  <label className="cursor-pointer text-accent underline-offset-2 hover:underline">
                    browse
                    <input
                      type="file"
                      multiple
                      accept=".json,.jsonl,.ndjson,.xml,.txt"
                      className="sr-only"
                      onChange={(e) => setFiles((f) => [...f, ...Array.from(e.target.files ?? [])])}
                    />
                  </label>
                  . Files above 16 MB are truncated.
                </p>
                {files.length ? (
                  <ul className="mt-3 flex flex-wrap justify-center gap-1.5">
                    {files.map((file, index) => (
                      <li
                        key={`${file.name}-${index}`}
                        className="inline-flex items-center gap-1 rounded-full border border-border-strong bg-surface-3 px-2.5 py-0.5 text-[11px]"
                      >
                        {file.name} <span className="text-faint">{(file.size / 1024).toFixed(0)} KB</span>
                        <button
                          type="button"
                          aria-label={`Remove ${file.name}`}
                          onClick={() => setFiles((f) => f.filter((_, i) => i !== index))}
                          className="ml-1 text-faint hover:text-rose"
                        >
                          <X className="size-3" />
                        </button>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            </div>
          </Card>

          <Card>
            <button
              type="button"
              onClick={() => setShowContext((v) => !v)}
              className="flex w-full items-center justify-between text-left focus-ring rounded-lg"
            >
              <SectionTitle className="mb-0" hint="Only facts supported by incident or intelligence reporting. Context re-ranks technically supported candidates; it cannot introduce a group with no TTP or tool link.">
                Strategic targeting context
              </SectionTitle>
              <ChevronDown className={cn('size-4 text-muted transition-transform', showContext && 'rotate-180')} />
            </button>
            {showContext ? (
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {(
                  [
                    ['target_geographies', 'Victim country / region'],
                    ['target_industries', 'Targeted industry / sector'],
                    ['motivations', 'Suspected motivation'],
                    ['objectives', 'Likely end objective'],
                    ['suspected_origins', 'Suspected sponsor / origin'],
                  ] as const
                ).map(([key, label]) => (
                  <Field
                    key={key}
                    label={label}
                    hint={key === 'suspected_origins' ? 'never from this tool’s output' : undefined}
                    className={key === 'suspected_origins' ? 'sm:col-span-2' : undefined}
                  >
                    <Input
                      value={context[key]}
                      onChange={(e) => setContext((c) => ({ ...c, [key]: e.target.value }))}
                    />
                  </Field>
                ))}
              </div>
            ) : null}
          </Card>
        </div>

        <aside className="space-y-4 lg:sticky lg:top-20 lg:self-start">
          <Card glow>
            <Label>Run the analysis</Label>
            <p className="mt-2 text-xs leading-relaxed text-muted">
              {local
                ? 'Local mode: prose is split into sentences, every sentence is retrieved against ATT&CK and the calibrated acceptance model decides which mappings vote.'
                : 'The extraction model normalises the evidence into atomic observations, the verifier checks ambiguous mappings, then the ensemble ranks documented groups.'}
            </p>
            <Button
              variant="primary"
              size="lg"
              className="mt-4 w-full"
              disabled={!hasInput}
              loading={analyse.isPending}
              onClick={() => analyse.mutate()}
            >
              {analyse.isPending ? 'Analysing…' : 'Analyse evidence'}
            </Button>
            {analyse.isPending ? (
              <ol className="mt-4 space-y-2 text-xs text-muted">
                <Step active>Normalising evidence into observations</Step>
                <Step>Retrieving and accepting ATT&CK techniques</Step>
                <Step>Ranking documented groups and gating the decision</Step>
              </ol>
            ) : (
              <p className="mt-3 text-[11px] text-faint">
                {local ? 'Usually a few seconds.' : 'Usually 15–60 seconds, depending on the amount of text.'}
              </p>
            )}
          </Card>
          <Callout tone="info" icon={<Sparkles className="size-4" />} title="What makes good evidence">
            Concrete behaviour: commands, process trees, persistence artefacts, lateral-movement
            protocols, named tooling. Vague summaries map to generic techniques that most groups share
            and cannot separate candidates.
          </Callout>
        </aside>
      </div>
    </div>
  )
}

function Step({ children, active = false }: { children: React.ReactNode; active?: boolean }) {
  return (
    <li className="flex items-center gap-2">
      {active ? (
        <Loader2 className="size-3.5 animate-spin text-accent" />
      ) : (
        <span className="size-3.5 rounded-full border border-border-strong" />
      )}
      <span className={cn(active && 'text-text')}>{children}</span>
    </li>
  )
}
