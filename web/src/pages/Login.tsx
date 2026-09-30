import { ShieldCheck } from 'lucide-react'

import type { AuthInfo } from '../api/types'
import { TAGLINE, VestigeMark } from '../components/Brand'
import { Callout } from '../components/ui'

const ERRORS: Record<string, string> = {
  not_allowed: 'This Google account is not on the allowlist for this workspace. Ask an administrator to add your address or domain.',
  unverified: 'Google reported the address as unverified. Verify it with Google and try again.',
  inactive: 'This account has been deactivated by an administrator.',
  provider: 'Google did not complete the sign-in. Try again.',
}

export function LoginPage({ auth, error }: { auth: AuthInfo | null; error: string | null }) {
  return (
    <div className="grid min-h-screen place-items-center px-4 grid-bg">
      <div className="w-full max-w-md">
        <div className="card relative overflow-hidden p-8">
          <div className="pointer-events-none absolute -right-16 -top-16 size-48 rounded-full bg-accent/10 blur-3xl" />
          <div className="flex items-center gap-3">
            <VestigeMark size={44} />
            <div>
              <div className="text-lg font-semibold uppercase tracking-[0.2em]">Vestige</div>
              <div className="text-xs text-muted">{TAGLINE}</div>
            </div>
          </div>
          <h1 className="mt-6 text-xl font-semibold tracking-tight">Sign in to your workspace</h1>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            Cases, decisions and interview answers are shared with your team and attributed to you.
            Sign-in uses your organisation's Google account; this application never sees a password.
          </p>
          {error ? (
            <Callout tone="danger" className="mt-4">
              {ERRORS[error] ?? 'Sign-in failed.'}
            </Callout>
          ) : null}
          {auth && !auth.allowlist_configured ? (
            <Callout tone="warning" className="mt-4" title="No allowlist configured">
              The server refuses every sign-in until AUTH_ALLOWED_DOMAINS or AUTH_ALLOWED_EMAILS is set.
            </Callout>
          ) : null}
          <a
            href="/auth/login"
            className="mt-6 flex h-11 w-full items-center justify-center gap-3 rounded-xl bg-white text-sm font-semibold text-[#1f1f1f] shadow-[0_1px_0_0_rgba(0,0,0,.08),0_8px_30px_-12px_rgba(255,255,255,.5)] transition hover:bg-[#f2f2f2] focus-ring"
          >
            <GoogleMark />
            Sign in with Google
          </a>
          <p className="mt-4 flex items-start gap-2 text-[11px] leading-relaxed text-faint">
            <ShieldCheck className="mt-0.5 size-3.5 shrink-0" />
            Only addresses on the workspace allowlist can sign in. Sessions expire automatically.
          </p>
        </div>
      </div>
    </div>
  )
}

function GoogleMark() {
  return (
    <svg viewBox="0 0 48 48" className="size-4" aria-hidden>
      <path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9 3.5l6.7-6.7C35.6 2.6 30.2 0 24 0 14.6 0 6.5 5.4 2.6 13.3l7.8 6C12.3 13.5 17.7 9.5 24 9.5z" />
      <path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.7c-.6 3-2.3 5.5-4.8 7.2l7.5 5.8c4.4-4.1 7.1-10.1 7.1-17.5z" />
      <path fill="#FBBC05" d="M10.4 28.7c-.5-1.5-.8-3-.8-4.7s.3-3.2.8-4.7l-7.8-6C.9 16.5 0 20.1 0 24s.9 7.5 2.6 10.7l7.8-6z" />
      <path fill="#34A853" d="M24 48c6.2 0 11.4-2 15.2-5.6l-7.5-5.8c-2 1.4-4.6 2.2-7.7 2.2-6.3 0-11.7-4-13.6-9.8l-7.8 6C6.5 42.6 14.6 48 24 48z" />
    </svg>
  )
}
