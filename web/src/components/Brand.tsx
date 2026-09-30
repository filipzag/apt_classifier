import { cn } from '../lib/utils'

/**
 * The Vestige mark: three traces fading in from the edge and converging on a solid origin,
 * the evidence trail leading back to the actor. Drawn as vectors so it scales from the
 * favicon to the sign-in page without assets.
 */
export function VestigeMark({ size = 32, className }: { size?: number; className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      width={size}
      height={size}
      className={cn('shrink-0', className)}
      role="img"
      aria-label="Vestige"
    >
      <defs>
        <linearGradient id="vestige-bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#0f1a2e" />
          <stop offset="1" stopColor="#0a1120" />
        </linearGradient>
        <linearGradient id="vestige-trail" x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#22d3ee" stopOpacity="0.15" />
          <stop offset="1" stopColor="#22d3ee" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#vestige-bg)" stroke="#1f2b45" strokeWidth="2" />
      {/* trail: three arcs, growing and brightening toward the origin */}
      <path d="M14 50 C 20 38, 28 34, 40 30" fill="none" stroke="url(#vestige-trail)" strokeWidth="3" strokeLinecap="round" />
      <path d="M19 55 C 27 44, 34 38, 43 35" fill="none" stroke="#22d3ee" strokeOpacity="0.45" strokeWidth="3" strokeLinecap="round" />
      <path d="M9 44 C 16 34, 26 28, 38 25" fill="none" stroke="#22d3ee" strokeOpacity="0.25" strokeWidth="3" strokeLinecap="round" />
      {/* origin */}
      <circle cx="46" cy="22" r="9" fill="#22d3ee" fillOpacity="0.18" stroke="#22d3ee" strokeWidth="2" />
      <circle cx="46" cy="22" r="3.5" fill="#22d3ee" />
    </svg>
  )
}

export function Wordmark({ className, subtitle = true }: { className?: string; subtitle?: boolean }) {
  return (
    <span className={cn('leading-tight', className)}>
      <span className="block text-[15px] font-semibold tracking-[0.18em] uppercase">Vestige</span>
      {subtitle ? <span className="block text-[11px] text-muted">Evidence-first attribution</span> : null}
    </span>
  )
}

export const TAGLINE = 'Attribution from the traces left behind.'
