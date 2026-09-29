import { useDossier } from '../api/hooks'
import { titleCase } from '../lib/utils'
import { Badge, DataTable, Label, Metric, SlideOver, Spinner, Tabs } from './ui'

function genericColumns(rows: Record<string, unknown>[]) {
  const keys = rows.length ? Object.keys(rows[0]) : []
  return keys.map((key) => ({ key, label: titleCase(key) }))
}

export function DossierPanel({
  caseId,
  groupId,
  onClose,
}: {
  caseId: string
  groupId: string | null
  onClose: () => void
}) {
  const query = useDossier(caseId, groupId)
  const data = query.data
  return (
    <SlideOver
      open={Boolean(groupId)}
      onOpenChange={(open) => !open && onClose()}
      title={data ? `${data.name}` : 'Candidate dossier'}
      description={
        data ? (
          <span className="mono">{data.group_id}</span>
        ) : (
          'Why this group ranks where it does, its documented profile and the gaps.'
        )
      }
    >
      {query.isLoading || !data ? (
        <Spinner label="Building the dossier…" />
      ) : (
        <div className="space-y-5">
          <div className="grid grid-cols-3 gap-3">
            <Metric label="Rank" value={`#${data.rank}`} tone="accent" />
            <Metric label="Relative index" value={data.relative_score.toFixed(1)} hint="within this case" />
            <Metric
              label="Classifiers"
              value={data.contested ? 'disagree' : 'agree'}
              tone={data.contested ? 'amber' : 'emerald'}
            />
          </div>
          <Tabs
            tabs={[
              {
                value: 'why',
                label: 'Why it ranks here',
                content: (
                  <div className="space-y-4">
                    <div>
                      <Label className="mb-2">Observed techniques this group documents</Label>
                      <DataTable
                        columns={genericColumns(data.matched_techniques)}
                        rows={data.matched_techniques}
                        empty="No accepted technique overlaps this group."
                        maxHeight={260}
                      />
                    </div>
                    {data.specific_behaviors.length ? (
                      <div>
                        <Label className="mb-2">Actor-specific behaviour matches</Label>
                        <DataTable
                          columns={genericColumns(data.specific_behaviors)}
                          rows={data.specific_behaviors}
                          maxHeight={220}
                        />
                      </div>
                    ) : null}
                    {data.tools.length ? (
                      <div>
                        <Label className="mb-2">Named tools</Label>
                        <div className="flex flex-wrap gap-1.5">
                          {data.tools.map((tool) => (
                            <Badge key={tool.software_id} tone="amber">
                              {tool.name} <span className="mono opacity-70">{tool.software_id}</span>
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ) : null}
                    <div>
                      <Label className="mb-2">Classifier ranks</Label>
                      <div className="grid grid-cols-2 gap-1.5 sm:grid-cols-3">
                        {Object.entries(data.classifier_ranks)
                          .sort(([a], [b]) => a.localeCompare(b))
                          .map(([name, rank]) => (
                            <div
                              key={name}
                              className="flex items-center justify-between rounded-lg border border-border bg-surface-2/60 px-2.5 py-1.5 text-xs"
                            >
                              <span className="truncate text-muted">{name.replace(/_/g, ' ')}</span>
                              <span className="tabular font-semibold">#{rank}</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  </div>
                ),
              },
              {
                value: 'profile',
                label: 'Documented profile',
                content: (
                  <div className="space-y-4">
                    <p className="text-sm leading-relaxed text-text/90">
                      {data.description || 'No ATT&CK group description available.'}
                    </p>
                    {data.country_evidence.length ? (
                      <div>
                        <Label className="mb-2">Origin evidence in ATT&CK reporting</Label>
                        <ul className="space-y-1.5">
                          {data.country_evidence.map((item, index) => (
                            <li key={index} className="rounded-lg border border-border bg-surface-2/60 px-3 py-2 text-xs">
                              <span className="font-semibold">{item.country}:</span>{' '}
                              <span className="text-muted">{item.excerpt}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                ),
              },
              {
                value: 'gaps',
                label: 'Gaps',
                count: data.unmatched_techniques.length + data.contradicting_techniques.length,
                content: (
                  <div className="space-y-4">
                    <div>
                      <Label className="mb-2">Observed but not documented for this group</Label>
                      <DataTable
                        columns={genericColumns(data.unmatched_techniques)}
                        rows={data.unmatched_techniques}
                        empty="Every accepted technique is documented for this group."
                        maxHeight={260}
                      />
                    </div>
                    {data.contradicting_techniques.length ? (
                      <div>
                        <Label className="mb-2">Contradicted by interview answers</Label>
                        <div className="flex flex-wrap gap-1.5">
                          {data.contradicting_techniques.map((tid) => (
                            <Badge key={tid} tone="rose">
                              {tid}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                ),
              },
            ]}
          />
        </div>
      )}
    </SlideOver>
  )
}
