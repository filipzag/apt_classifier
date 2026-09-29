import { useQuery } from '@tanstack/react-query'
import { ExternalLink, RefreshCw } from 'lucide-react'
import { useMemo, useState } from 'react'

import { api } from '../api/client'
import { describe, keys, useSystem } from '../api/hooks'
import type { CasePayload } from '../api/types'
import { CaseFrame, PageHeader, useCaseId } from '../components/CaseFrame'
import { Badge, Button, Callout, Card, Field, Input, Metric, SectionTitle, Spinner, Toggle } from '../components/ui'
import { titleCase } from '../lib/utils'

export function IntelPage() {
  const id = useCaseId()
  return <CaseFrame>{(data) => <IntelView caseId={id} data={data} />}</CaseFrame>
}

function IntelView({ caseId, data }: { caseId: string; data: CasePayload }) {
  const system = useSystem()
  const [days, setDays] = useState(30)
  const [claims, setClaims] = useState(true)
  const [refreshToken, setRefreshToken] = useState(0)
  const [search, setSearch] = useState('')
  const [kind, setKind] = useState('all')
  const [source, setSource] = useState('all')
  const query = useQuery({
    queryKey: [...keys.intel(caseId, days, claims), refreshToken],
    queryFn: () => api.intel(caseId, days, claims, refreshToken > 0),
    staleTime: 30 * 60_000,
  })
  const result = query.data
  const kinds = useMemo(() => Array.from(new Set(result?.items.map((i) => i.kind) ?? [])).sort(), [result])
  const sources = useMemo(() => Array.from(new Set(result?.items.map((i) => i.source) ?? [])).sort(), [result])
  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return (result?.items ?? []).filter(
      (item) =>
        (kind === 'all' || item.kind === kind) &&
        (source === 'all' || item.source === source) &&
        (!needle ||
          [item.title, item.summary, ...item.actors, ...item.cves, ...item.products, ...item.industries]
            .join(' ')
            .toLowerCase()
            .includes(needle)),
    )
  }, [result, search, kind, source])
  const offline = result?.snapshot.sources.filter((s) => !s.ok) ?? []
  void system
  void data

  return (
    <div className="space-y-5">
      <PageHeader
        title="Live intelligence"
        lead="Public feeds (CISA KEV and advisories, ransomware claim sites). Matches enrich the case but never vote, and are withheld from the independent reviewer."
        action={
          <Button onClick={() => setRefreshToken((t) => t + 1)} loading={query.isFetching}>
            <RefreshCw className="size-4" /> Refresh feeds
          </Button>
        }
      />
      <Card className="flex flex-wrap items-end gap-5 py-4">
        <Field label={`Lookback: ${days} days`} className="w-56">
          <input type="range" min={1} max={90} value={days} onChange={(e) => setDays(Number(e.target.value))} className="w-full accent-cyan-400" />
        </Field>
        <Toggle checked={claims} onChange={setClaims} label="Include ransomware victim claims (copied from extortion sites; not confirmed breaches)" />
      </Card>
      {query.isLoading ? (
        <Card>
          <Spinner label="Fetching public feeds…" />
        </Card>
      ) : query.isError ? (
        <Callout tone="danger" title="Feeds could not be fetched">
          {describe(query.error)}
        </Callout>
      ) : result ? (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            <Metric label="Records" value={result.snapshot.records} />
            <Metric label="Authoritative" value={result.snapshot.authoritative} tone="emerald" />
            <Metric label="Unverified claims" value={result.snapshot.claims} tone="amber" />
            <Metric label="Matches to this case" value={result.matches.length} tone={result.matches.length ? 'accent' : undefined} />
            <Metric label="Fetched" value={result.snapshot.fetched_at.slice(11, 16)} hint={`${result.snapshot.fetched_at.slice(0, 10)} UTC`} />
          </div>
          {offline.length ? (
            <Callout tone="warning" title="Unavailable sources">
              {offline.map((s) => `${s.source} (${s.error ?? 'error'})`).join(', ')}
            </Callout>
          ) : null}
          <Card>
            <SectionTitle hint="Explicit overlap only: actor alias, CVE, named tool, product, industry or geography.">
              Matches to this case
            </SectionTitle>
            {result.matches.length ? (
              <ul className="space-y-2">
                {result.matches.map((m) => (
                  <li key={m.item_id} className="flex flex-wrap items-start gap-3 rounded-xl border border-border bg-surface-2/60 px-4 py-3">
                    <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-accent-soft text-sm font-semibold tabular text-accent">
                      {m.score.toFixed(1)}
                    </span>
                    <div className="min-w-0 flex-1">
                      <a href={m.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-sm font-medium hover:text-accent">
                        {m.title} <ExternalLink className="size-3 text-faint" />
                      </a>
                      <div className="mt-0.5 text-xs text-muted">
                        {m.published_at} · {m.source} · {m.reasons.join('; ')}
                      </div>
                    </div>
                    <Badge tone={m.verification === 'authoritative' ? 'emerald' : 'amber'}>{titleCase(m.verification)}</Badge>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-muted">No explicit overlap between this case and the fetched records.</p>
            )}
          </Card>
          <Card>
            <SectionTitle
              hint={`${filtered.length} of ${result.items.length} records`}
              action={
                <div className="flex flex-wrap gap-2">
                  <select value={kind} onChange={(e) => setKind(e.target.value)} className="h-9 rounded-lg border border-border-strong bg-surface-2 px-2 text-xs">
                    <option value="all">All types</option>
                    {kinds.map((k) => (
                      <option key={k} value={k}>
                        {titleCase(k)}
                      </option>
                    ))}
                  </select>
                  <select value={source} onChange={(e) => setSource(e.target.value)} className="h-9 rounded-lg border border-border-strong bg-surface-2 px-2 text-xs">
                    <option value="all">All sources</option>
                    {sources.map((s) => (
                      <option key={s} value={s}>
                        {s}
                      </option>
                    ))}
                  </select>
                  <Input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Actor, CVE, product, victim, sector…" className="w-64" />
                </div>
              }
            >
              Feed records
            </SectionTitle>
            <div className="max-h-[600px] overflow-auto rounded-xl border border-border">
              <table className="data">
                <thead>
                  <tr>
                    <th>Published</th>
                    <th>Source</th>
                    <th>Type</th>
                    <th>Title</th>
                    <th>Actors</th>
                    <th>CVEs</th>
                    <th>Verification</th>
                    <th>Match</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((item) => (
                    <tr key={item.id}>
                      <td className="whitespace-nowrap text-muted">{item.published_at}</td>
                      <td className="whitespace-nowrap">{item.source}</td>
                      <td className="whitespace-nowrap">{titleCase(item.kind)}</td>
                      <td className="min-w-[320px]">
                        <a href={item.url} target="_blank" rel="noreferrer" className="hover:text-accent">
                          {item.title}
                        </a>
                      </td>
                      <td className="text-muted">{item.actors.join(', ')}</td>
                      <td className="mono text-muted">{item.cves.join(', ')}</td>
                      <td>
                        <Badge tone={item.verification === 'authoritative' ? 'emerald' : 'amber'}>{titleCase(item.verification)}</Badge>
                      </td>
                      <td className="tabular">{item.match_score !== null ? item.match_score.toFixed(2) : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      ) : null}
    </div>
  )
}
