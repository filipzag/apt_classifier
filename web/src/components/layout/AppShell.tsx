import * as DropdownMenu from '@radix-ui/react-dropdown-menu'
import * as TooltipPrimitive from '@radix-ui/react-tooltip'
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  Check,
  ChevronDown,
  Crosshair,
  Diamond,
  Download,
  FileText,
  FolderOpen,
  ListChecks,
  MessageSquare,
  Network,
  Pencil,
  Plus,
  Radar,
  Radio,
  ShieldAlert,
  Trash2,
} from 'lucide-react'
import { useState } from 'react'
import { Link, NavLink, Outlet, useMatch, useNavigate } from 'react-router-dom'

import { api } from '../../api/client'
import { useCase, useDeleteCase, useRename, useSystem } from '../../api/hooks'
import type { CasePayload, SystemInfo } from '../../api/types'
import { cn, fmt } from '../../lib/utils'
import { StageBadge } from '../StageBadge'
import { Badge, Button, Tip } from '../ui'

export function AppShell() {
  const match = useMatch('/cases/:id/*')
  const caseId = match?.params.id
  const caseQuery = useCase(caseId)
  const system = useSystem()
  return (
    <TooltipPrimitive.Provider>
      <div className="flex h-full min-h-screen">
        <Sidebar caseId={caseId} data={caseQuery.data} system={system.data} />
        <div className="flex min-w-0 flex-1 flex-col">
          <TopBar caseId={caseId} data={caseQuery.data} system={system.data} />
          <main className="min-w-0 flex-1 px-6 py-6 lg:px-8">
            <div className="mx-auto w-full max-w-[1400px]">
              <Outlet />
            </div>
          </main>
        </div>
      </div>
    </TooltipPrimitive.Provider>
  )
}

function NavItem({
  to,
  icon,
  children,
  end = false,
  trailing,
  step,
}: {
  to: string
  icon: React.ReactNode
  children: React.ReactNode
  end?: boolean
  trailing?: React.ReactNode
  step?: number
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        cn(
          'group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] transition-colors focus-ring',
          isActive
            ? 'bg-accent-soft text-text shadow-[inset_2px_0_0_0_var(--color-accent)]'
            : 'text-muted hover:bg-surface-2 hover:text-text',
        )
      }
    >
      {step !== undefined ? (
        <span className="grid size-5 shrink-0 place-items-center rounded-md border border-border-strong bg-surface-2 text-[10px] font-semibold text-muted group-aria-[current=page]:border-accent group-aria-[current=page]:text-accent">
          {step}
        </span>
      ) : (
        <span className="shrink-0 text-faint group-hover:text-muted group-aria-[current=page]:text-accent">
          {icon}
        </span>
      )}
      <span className="min-w-0 flex-1 truncate">{children}</span>
      {trailing ? <span className="shrink-0 text-[10px] text-faint">{trailing}</span> : null}
    </NavLink>
  )
}

function Sidebar({
  caseId,
  data,
  system,
}: {
  caseId?: string
  data?: CasePayload
  system?: SystemInfo
}) {
  const counts = data?.candidate_counts
  return (
    <aside className="sticky top-0 hidden h-screen w-[248px] shrink-0 flex-col border-r border-border bg-surface/80 backdrop-blur md:flex">
      <Link to="/" className="flex items-center gap-2.5 px-4 pt-5 pb-4 focus-ring rounded-lg">
        <span className="grid size-8 place-items-center rounded-lg bg-accent-soft text-accent ring-1 ring-accent/40">
          <Crosshair className="size-4" />
        </span>
        <span className="leading-tight">
          <span className="block text-[13px] font-semibold tracking-tight">APT Attribution</span>
          <span className="block text-[11px] text-muted">Assistant</span>
        </span>
      </Link>
      <nav className="flex-1 space-y-5 overflow-y-auto px-2.5 pb-4">
        <div>
          <div className="px-2.5 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-faint">
            Workspace
          </div>
          <NavItem to="/" end icon={<FolderOpen className="size-4" />}>
            Cases
          </NavItem>
          <NavItem to="/new" icon={<Plus className="size-4" />}>
            New case
          </NavItem>
        </div>
        {caseId ? (
          <>
            <div>
              <div className="px-2.5 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-faint">
                Workflow
              </div>
              <NavItem
                to={`/cases/${caseId}/evidence`}
                step={1}
                icon={<FileText className="size-4" />}
                trailing={data ? `${data.observations.length} obs` : undefined}
              >
                Evidence
              </NavItem>
              <NavItem
                to={`/cases/${caseId}/techniques`}
                step={2}
                icon={<ListChecks className="size-4" />}
                trailing={counts ? `${counts.needs_review} to review` : undefined}
              >
                Technique review
              </NavItem>
              <NavItem
                to={`/cases/${caseId}/interview`}
                step={3}
                icon={<MessageSquare className="size-4" />}
                trailing={
                  data
                    ? data.interview.complete
                      ? 'done'
                      : `${data.interview.answered} answered`
                    : undefined
                }
              >
                Interview
              </NavItem>
              <NavItem to={`/cases/${caseId}`} end step={4} icon={<ShieldAlert className="size-4" />}>
                Assessment
              </NavItem>
            </div>
            <div>
              <div className="px-2.5 pb-1.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-faint">
                Views
              </div>
              <NavItem to={`/cases/${caseId}/canvas`} icon={<Network className="size-4" />}>
                Analyst canvas
              </NavItem>
              <NavItem to={`/cases/${caseId}/diamond`} icon={<Diamond className="size-4" />}>
                Diamond model
              </NavItem>
              <NavItem to={`/cases/${caseId}/intel`} icon={<Radio className="size-4" />}>
                Live intelligence
              </NavItem>
              <NavItem to={`/cases/${caseId}/review`} icon={<BrainCircuit className="size-4" />}>
                AI review
              </NavItem>
            </div>
          </>
        ) : null}
      </nav>
      <SystemFooter system={system} />
    </aside>
  )
}

function SystemFooter({ system }: { system?: SystemInfo }) {
  if (!system) return <div className="border-t border-border px-4 py-3 text-[11px] text-faint">Connecting…</div>
  const local = system.mode === 'local'
  return (
    <div className="border-t border-border px-4 py-3">
      <div className="flex items-center justify-between">
        <Badge tone={local ? 'amber' : 'emerald'} dot>
          {local ? 'Local mode' : 'Production'}
        </Badge>
        <Tip
          content={
            <div className="space-y-1">
              <div>ATT&CK {system.attack_version} · {system.groups} groups · {system.techniques} techniques</div>
              <div>
                Index: {system.index.ok ? `${system.index.technique_sentences} sentences, ${system.index.group_documents} group docs` : `unavailable (${system.index.error})`}
              </div>
              <div>Embedder {system.embedding_model}</div>
              {!local ? <div>Extraction {system.extract_model} · review {system.review_model}</div> : null}
              <div>Preset {system.preset} · gate {system.gate_mode}</div>
            </div>
          }
        >
          <span className="inline-flex items-center gap-1 text-[11px] text-faint hover:text-muted">
            <Activity className={cn('size-3', system.index.ok ? 'text-emerald' : 'text-rose')} />
            ATT&CK {system.attack_version}
          </span>
        </Tip>
      </div>
      <div className="mt-1.5 text-[11px] text-faint">
        {local
          ? 'No OpenRouter key: sentence split, no verifier, no AI review.'
          : `LLM spend this session $${fmt(system.llm_spend_usd, 2)}`}
      </div>
    </div>
  )
}

function TopBar({
  caseId,
  data,
  system,
}: {
  caseId?: string
  data?: CasePayload
  system?: SystemInfo
}) {
  const navigate = useNavigate()
  const rename = useRename(caseId ?? '')
  const remove = useDeleteCase()
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')

  const commit = () => {
    setEditing(false)
    if (caseId && draft.trim() && draft.trim() !== data?.name) rename.mutate(draft.trim())
  }
  const stage = data?.decision?.stage ?? null
  const next = data?.next_step
  const nextHref = next
    ? next.page === 'assessment'
      ? `/cases/${caseId}`
      : next.page === 'intake'
        ? `/cases/${caseId}/evidence`
        : `/cases/${caseId}/${next.page}`
    : null

  return (
    <header className="sticky top-0 z-30 flex h-14 items-center gap-3 border-b border-border bg-bg/75 px-6 backdrop-blur lg:px-8">
      <Link to="/" className="md:hidden">
        <Crosshair className="size-5 text-accent" />
      </Link>
      {caseId ? (
        <>
          <div className="flex min-w-0 items-center gap-2">
            {editing ? (
              <input
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onBlur={commit}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commit()
                  if (e.key === 'Escape') {
                    setDraft(data?.name ?? '')
                    setEditing(false)
                  }
                }}
                className="h-8 w-72 rounded-md border border-accent/50 bg-surface-2 px-2 text-sm font-semibold focus-ring"
                aria-label="Case name"
              />
            ) : (
              <button
                type="button"
                onClick={() => {
                  setDraft(data?.name ?? '')
                  setEditing(true)
                }}
                className="group flex min-w-0 items-center gap-2 rounded-md px-1 text-sm font-semibold tracking-tight hover:bg-surface-2 focus-ring"
                title="Rename case"
              >
                <span className="truncate">{data?.name ?? 'Loading…'}</span>
                <Pencil className="size-3 shrink-0 text-faint opacity-0 transition-opacity group-hover:opacity-100" />
              </button>
            )}
            <StageBadge stage={stage} short />
            {data?.decision?.leading_group_name ? (
              <span className="hidden truncate text-xs text-muted lg:inline">
                {data.decision.named ? 'Named:' : 'Closest:'}{' '}
                <span className="text-text">{data.decision.leading_group_name}</span>
              </span>
            ) : null}
          </div>
          <div className="ml-auto flex items-center gap-2">
            {next && nextHref ? (
              <Link
                to={nextHref}
                className="hidden items-center gap-1.5 rounded-full border border-border-strong bg-surface-2 px-3 py-1 text-[11px] text-muted hover:text-text xl:inline-flex"
              >
                <Radar className="size-3 text-accent" />
                <span className="max-w-[360px] truncate">{next.message}</span>
                <ArrowRight className="size-3" />
              </Link>
            ) : null}
            <ExportMenu caseId={caseId} />
            <Tip content="Delete this case permanently">
              <Button
                variant="ghost"
                size="sm"
                aria-label="Delete case"
                onClick={() => {
                  if (window.confirm(`Delete case "${data?.name ?? caseId}"? This cannot be undone.`)) {
                    remove.mutate(caseId, { onSuccess: () => navigate('/') })
                  }
                }}
              >
                <Trash2 className="size-4" />
              </Button>
            </Tip>
          </div>
        </>
      ) : (
        <div className="ml-auto flex items-center gap-2">
          {system ? (
            <span className="text-xs text-muted">
              {system.cases} case{system.cases === 1 ? '' : 's'} on disk
            </span>
          ) : null}
          <Button variant="primary" size="sm" onClick={() => navigate('/new')}>
            <Plus className="size-4" /> New case
          </Button>
        </div>
      )}
    </header>
  )
}

function ExportMenu({ caseId }: { caseId: string }) {
  const items: { kind: 'markdown' | 'json' | 'navigator' | 'attack-flow'; label: string; hint: string }[] = [
    { kind: 'markdown', label: 'Markdown report', hint: 'Analyst assessment' },
    { kind: 'json', label: 'JSON audit', hint: 'Ledger, ranking, decision, graphs' },
    { kind: 'navigator', label: 'ATT&CK Navigator layers', hint: 'Observed and candidate techniques' },
    { kind: 'attack-flow', label: 'Attack Flow (STIX 2.1)', hint: 'Sequenced technique bundle' },
  ]
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button variant="outline" size="sm">
          <Download className="size-4" /> Export <ChevronDown className="size-3 text-muted" />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={6}
          className="z-50 w-72 rounded-xl border border-border-strong bg-surface-2 p-1.5 shadow-2xl"
        >
          {items.map((item) => (
            <DropdownMenu.Item key={item.kind} asChild>
              <a
                href={api.exportUrl(caseId, item.kind)}
                download
                className="flex cursor-pointer flex-col rounded-lg px-3 py-2 outline-none hover:bg-surface-3 data-[highlighted]:bg-surface-3"
              >
                <span className="flex items-center gap-2 text-sm">
                  <Check className="size-3.5 text-accent" /> {item.label}
                </span>
                <span className="pl-5.5 text-[11px] text-muted">{item.hint}</span>
              </a>
            </DropdownMenu.Item>
          ))}
          <DropdownMenu.Separator className="my-1 h-px bg-border" />
          <div className="px-3 py-1.5 text-[11px] text-faint">
            Available at any point; the report states which steps were run.
          </div>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  )
}
