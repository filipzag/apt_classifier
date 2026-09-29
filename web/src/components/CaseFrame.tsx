import { AlertTriangle } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link, useParams } from 'react-router-dom'

import { describe, useCase } from '../api/hooks'
import type { CasePayload } from '../api/types'
import { Button, EmptyState, Skeleton } from './ui'

export function useCaseId(): string {
  const { id } = useParams<{ id: string }>()
  return id ?? ''
}

/** Loads the case for the current route and renders children with it, or a loading/error frame. */
export function CaseFrame({ children }: { children: (data: CasePayload) => ReactNode }) {
  const id = useCaseId()
  const query = useCase(id)
  if (query.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-64" />
        <div className="grid gap-4 md:grid-cols-3">
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
          <Skeleton className="h-28" />
        </div>
        <Skeleton className="h-64" />
        <p className="text-xs text-muted">
          Loading the case. A saved case is re-analysed on first open, which takes a few seconds.
        </p>
      </div>
    )
  }
  if (query.isError || !query.data) {
    return (
      <EmptyState
        icon={<AlertTriangle className="size-8 text-amber" />}
        title="This case could not be loaded"
        action={
          <div className="flex gap-2">
            <Button onClick={() => query.refetch()}>Retry</Button>
            <Button variant="ghost">
              <Link to="/">Back to cases</Link>
            </Button>
          </div>
        }
      >
        {describe(query.error)}
      </EmptyState>
    )
  }
  return <>{children(query.data)}</>
}

export function PageHeader({
  step,
  title,
  lead,
  action,
}: {
  step?: string
  title: ReactNode
  lead?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        {step ? (
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-accent">
            {step}
          </div>
        ) : null}
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {lead ? <p className="mt-1 max-w-3xl text-sm leading-relaxed text-muted">{lead}</p> : null}
      </div>
      {action ? <div className="shrink-0">{action}</div> : null}
    </div>
  )
}
