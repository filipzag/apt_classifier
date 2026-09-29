import { Plus } from 'lucide-react'
import { Link } from 'react-router-dom'

import { CaseFrame, PageHeader } from '../components/CaseFrame'
import { Badge, Button, Callout, Card, DataTable, Label, SectionTitle } from '../components/ui'
import { titleCase } from '../lib/utils'

export function EvidencePage() {
  return (
    <CaseFrame>
      {(data) => {
        const ingestion = data.ingestion as Record<string, unknown>
        const structured = 'parsed' in ingestion
        return (
          <div className="space-y-5">
            <PageHeader
              step="Step 1 · Evidence"
              title="Evidence ledger"
              lead="The atomic observations this case is built from. Every technique mapping, ranking entry and interview question traces back to one of these rows."
              action={
                <Button variant="outline" onClick={() => {}}>
                  <Link to="/new" className="inline-flex items-center gap-2">
                    <Plus className="size-4" /> New case from other evidence
                  </Link>
                </Button>
              }
            />
            {!data.ranking.length ? (
              <Callout tone="warning" title="No group can be ranked yet">
                {data.techniques.length === 0
                  ? 'No ATT&CK technique cleared the evidence bar. Add more specific behaviour or logs.'
                  : `${data.techniques.length} technique(s) were mapped but no documented group overlaps them. Add named tooling or more distinctive behaviour.`}
              </Callout>
            ) : null}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <Label>Observations</Label>
                <div className="mt-1 text-2xl font-semibold tabular">{data.observations.length}</div>
                <div className="mt-1 text-xs text-muted">
                  {ingestion.normalization ? 'local sentence split (no LLM)' : 'LLM-normalised'}
                </div>
              </Card>
              <Card>
                <Label>Structured ingestion</Label>
                <div className="mt-1 text-2xl font-semibold tabular">
                  {structured ? String(ingestion.parsed ?? 0) : '–'}
                </div>
                <div className="mt-1 text-xs text-muted">
                  {structured
                    ? `${String(ingestion.skipped ?? 0)} skipped · ${String(ingestion.source_count ?? 0)} sources`
                    : 'no log files in this case'}
                </div>
              </Card>
              <Card>
                <Label>Strategic context</Label>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {Object.keys(data.strategic_context).length ? (
                    Object.entries(data.strategic_context).map(([field, values]) => (
                      <Badge key={field} tone="sky">
                        {titleCase(field)}: {values.join(', ')}
                      </Badge>
                    ))
                  ) : (
                    <span className="text-xs text-muted">none provided</span>
                  )}
                </div>
              </Card>
            </div>
            <Card>
              <SectionTitle hint="Source, kind and occurrence count come from the extractor; line numbers point back into uploaded files.">
                Observations
              </SectionTitle>
              <DataTable
                columns={[
                  { key: 'index', label: '#', className: 'w-10 text-muted tabular' },
                  { key: 'source', label: 'Source', className: 'whitespace-nowrap' },
                  { key: 'kind', label: 'Kind', render: (r) => <Badge>{String(r.kind)}</Badge> },
                  {
                    key: 'status',
                    label: 'Status',
                    render: (r) => (
                      <Badge tone={r.status === 'positive' ? 'emerald' : r.status === 'negative' ? 'rose' : 'neutral'}>
                        {String(r.status)}
                      </Badge>
                    ),
                  },
                  { key: 'text', label: 'Observation', className: 'min-w-[360px] leading-relaxed' },
                  { key: 'occurrences', label: 'Occurrences', className: 'text-right tabular' },
                  { key: 'lines', label: 'Lines', className: 'text-faint' },
                ]}
                rows={data.observations as unknown as Record<string, unknown>[]}
                maxHeight={640}
              />
            </Card>
          </div>
        )
      }}
    </CaseFrame>
  )
}
