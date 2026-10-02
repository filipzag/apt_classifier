import { motion } from 'framer-motion'
import { ArrowRight, Crosshair, FolderOpen, Plus, Sparkles } from 'lucide-react'
import { Link, useNavigate } from 'react-router-dom'

import { describe, useCases, useSystem } from '../api/hooks'
import { TAGLINE } from '../components/Brand'
import { STAGE_META, StageBadge } from '../components/StageBadge'
import { Button, Callout, EmptyState, Skeleton } from '../components/ui'
import { relativeTime } from '../lib/utils'

export function DashboardPage() {
  const cases = useCases()
  const system = useSystem()
  const navigate = useNavigate()
  const s = system.data
  return (
    <div className="space-y-8">
      <section className="relative overflow-hidden rounded-2xl border border-border bg-surface p-8 grid-bg">
        <div className="pointer-events-none absolute -right-24 -top-24 size-72 rounded-full bg-accent/10 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-32 left-1/3 size-72 rounded-full bg-violet/10 blur-3xl" />
        <div className="relative flex flex-wrap items-end justify-between gap-6">
          <div className="max-w-2xl">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-accent/30 bg-accent-soft px-3 py-1 text-[11px] font-medium text-accent">
              <Sparkles className="size-3" /> {TAGLINE}
            </div>
            <h1 className="text-3xl font-semibold tracking-tight">
              Map observations to ATT&CK, rank documented groups, name only what the evidence supports.
            </h1>
            <p className="mt-3 text-sm leading-relaxed text-muted">
              Every ranking is deterministic and auditable: hybrid ATTACK-BERT retrieval with a
              calibrated acceptance model, a family-aware ensemble over documented tradecraft, and a
              separation gate that refuses to name a group its evidence cannot distinguish from the
              alternatives.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              <Button variant="primary" size="lg" onClick={() => navigate('/new')}>
                <Plus className="size-4" /> New case
              </Button>
              <Button variant="outline" size="lg" onClick={() => navigate('/new?sample=1')}>
                Try the sample incident <ArrowRight className="size-4" />
              </Button>
            </div>
          </div>
          <dl className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4 lg:grid-cols-2">
            <Stat label="Cases" value={s ? String(s.cases) : '…'} />
            <Stat label="ATT&CK" value={s ? `v${s.attack_version}` : '…'} />
            <Stat label="Groups documented" value={s ? String(s.groups) : '…'} />
            <Stat
              label="Indexed sentences"
              value={s ? (s.index.ok ? String(s.index.technique_sentences) : 'offline') : '…'}
            />
          </dl>
        </div>
      </section>

      {s?.mode === 'local' ? (
        <Callout tone="warning" title="Local mode">
          No OpenRouter key is configured. Prose is split into sentences instead of LLM-normalised,
          the technique verifier is skipped and the AI review and CyberRE-LLM are
          unavailable. Retrieval, acceptance, ranking, the interview and every export still work.
        </Callout>
      ) : null}
      {s && !s.index.ok ? (
        <Callout tone="danger" title="The vector index is unreachable">
          {s.index.error} — start ChromaDB (make up) and ingest ATT&CK (make ingest).
        </Callout>
      ) : null}

      <section>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-[0.08em] text-muted">Recent cases</h2>
          {cases.data?.length ? (
            <span className="text-xs text-faint">{cases.data.length} saved locally</span>
          ) : null}
        </div>
        {cases.isLoading ? (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
            <Skeleton className="h-32" />
          </div>
        ) : cases.isError ? (
          <Callout tone="danger" title="Could not list cases">
            {describe(cases.error)}
          </Callout>
        ) : !cases.data?.length ? (
          <EmptyState
            icon={<FolderOpen className="size-8" />}
            title="No cases yet"
            action={
              <Button variant="primary" onClick={() => navigate('/new?sample=1')}>
                Analyse the sample incident
              </Button>
            }
          >
            Cases are saved automatically after every analysis, decision and interview answer under
            the data directory.
          </EmptyState>
        ) : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {cases.data.map((item, index) => {
              const stage = item.summary.stage ?? null
              const meta = stage ? STAGE_META[stage] : null
              return (
                <motion.div
                  key={item.case_id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.04 }}
                >
                  <Link
                    to={`/cases/${item.case_id}`}
                    className="card group flex h-full flex-col p-4 transition-colors hover:border-accent/40 focus-ring"
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold tracking-tight group-hover:text-accent">
                          {item.name}
                        </div>
                        <div className="mono mt-0.5 text-[11px] text-faint">{item.case_id}</div>
                      </div>
                      <StageBadge stage={stage} short />
                    </div>
                    <div className="mt-3 flex items-center gap-2 text-xs text-muted">
                      <Crosshair className="size-3.5 text-faint" />
                      {item.summary.leading_group_name ? (
                        <span className="truncate">
                          {stage === 'known_group_hypothesis'
                            ? 'Named'
                            : stage === 'known_cluster_hypothesis'
                              ? 'Named cluster'
                              : 'Closest'}
                          :{' '}
                          <span className="text-text">{item.summary.leading_group_name}</span>
                        </span>
                      ) : (
                        <span>No candidate group yet</span>
                      )}
                    </div>
                    <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-faint">
                      {meta?.summary ?? 'Not analysed.'}
                    </p>
                    <div className="mt-auto flex items-center justify-between gap-2 pt-3 text-[11px] text-faint">
                      <span className="truncate">
                        {item.summary.observations ?? 0} observations
                        {item.owner ? ` · ${item.owner.name}` : ''}
                      </span>
                      <span className="shrink-0">updated {relativeTime(item.updated_at)}</span>
                    </div>
                  </Link>
                </motion.div>
              )
            })}
          </div>
        )}
      </section>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface-2/70 px-4 py-3 backdrop-blur">
      <dt className="text-[10px] font-semibold uppercase tracking-[0.1em] text-faint">{label}</dt>
      <dd className="mt-1 text-xl font-semibold tabular">{value}</dd>
    </div>
  )
}
