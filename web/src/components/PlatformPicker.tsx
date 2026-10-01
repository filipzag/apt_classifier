import { Check } from 'lucide-react'

import type { PlatformScope } from '../api/types'
import { PLATFORMS } from '../lib/platforms'
import { cn } from '../lib/utils'
import { Badge, Tip } from './ui'

/** Multi-select chips for the operating systems the affected hosts run. */
export function PlatformPicker({
  value,
  onChange,
  disabled = false,
  className,
}: {
  value: string[]
  onChange: (next: string[]) => void
  disabled?: boolean
  className?: string
}) {
  const toggle = (platform: string) => {
    const next = value.includes(platform) ? value.filter((p) => p !== platform) : [...value, platform]
    onChange(PLATFORMS.filter((p) => next.includes(p)))
  }
  return (
    <div role="group" aria-label="Target operating systems" className={cn('flex flex-wrap gap-1.5', className)}>
      {PLATFORMS.map((platform) => {
        const active = value.includes(platform)
        return (
          <button
            key={platform}
            type="button"
            aria-pressed={active}
            disabled={disabled}
            onClick={() => toggle(platform)}
            className={cn(
              'inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors focus-ring disabled:opacity-50',
              active
                ? 'border-accent/60 bg-accent-soft text-accent'
                : 'border-border-strong bg-surface-2 text-muted hover:text-text',
            )}
          >
            {active ? <Check className="size-3" aria-hidden /> : null}
            {platform}
          </button>
        )
      })}
    </div>
  )
}

const SCOPE_SOURCE: Record<PlatformScope['source'], string> = {
  log: 'recorded by the log source',
  text: 'named in this observation',
  case: 'the case Target OS',
}

/** Which operating systems an observation was mapped under; the case-wide scope is shown once
 * on the page, so only observation-specific scopes get a badge. */
export function PlatformScopeBadge({ scope }: { scope: PlatformScope | null | undefined }) {
  if (!scope || scope.source === 'case') return null
  return (
    <Tip content={`Mapped only to techniques documented for ${scope.platforms.join(' or ')} (${SCOPE_SOURCE[scope.source]}).`}>
      <Badge tone="sky" className="cursor-help">
        {scope.platforms.join(' · ')}
      </Badge>
    </Tip>
  )
}
