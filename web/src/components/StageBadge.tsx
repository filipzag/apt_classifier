import { CircleDashed, CircleDot, Layers, ShieldCheck } from 'lucide-react'

import type { Stage } from '../api/types'
import { cn } from '../lib/utils'
import { Badge, type BadgeTone } from './ui'

export const STAGE_META: Record<
  Stage,
  { label: string; short: string; tone: BadgeTone; summary: string; advice: string }
> = {
  insufficient_evidence: {
    label: 'Insufficient evidence',
    short: 'Insufficient',
    tone: 'neutral',
    summary: 'Too little technique evidence to rank groups meaningfully.',
    advice: 'Add concrete behaviour (commands, persistence, lateral movement) or logs.',
  },
  unresolved_activity_cluster: {
    label: 'Unresolved activity cluster',
    short: 'Cluster',
    tone: 'amber',
    summary: 'Techniques were mapped but no documented group stands out.',
    advice: 'Treat this as an unattributed cluster; the shortlist is a starting point only.',
  },
  known_group_shortlist: {
    label: 'Known-group shortlist',
    short: 'Shortlist',
    tone: 'sky',
    summary: 'Several documented groups overlap the evidence; none clears the naming bar.',
    advice: 'Answer interview questions or add discriminating evidence to separate them.',
  },
  known_group_hypothesis: {
    label: 'Named group hypothesis',
    short: 'Named',
    tone: 'emerald',
    summary: 'One documented group clears the evidence bar.',
    advice: 'Check the documented-overlap note and run the independent review before reporting.',
  },
}

export function stageIcon(stage: Stage | null | undefined, className = 'size-4') {
  switch (stage) {
    case 'known_group_hypothesis':
      return <ShieldCheck className={className} aria-hidden />
    case 'known_group_shortlist':
      return <Layers className={className} aria-hidden />
    case 'unresolved_activity_cluster':
      return <CircleDot className={className} aria-hidden />
    default:
      return <CircleDashed className={className} aria-hidden />
  }
}

export function StageBadge({
  stage,
  short = false,
  className,
}: {
  stage: Stage | null | undefined
  short?: boolean
  className?: string
}) {
  if (!stage) {
    return (
      <Badge tone="neutral" className={className}>
        {stageIcon(null, 'size-3')} No analysis yet
      </Badge>
    )
  }
  const meta = STAGE_META[stage]
  return (
    <Badge tone={meta.tone} className={cn('px-2.5 py-1 text-xs', className)}>
      {stageIcon(stage, 'size-3.5')}
      {short ? meta.short : meta.label}
    </Badge>
  )
}
