import { useQuery } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { createContext, useContext } from 'react'
import { useLocation } from 'react-router-dom'

import { api, ApiError } from '../api/client'
import type { AuthInfo, Me, User } from '../api/types'
import { LoginPage } from '../pages/Login'
import { Callout, Spinner } from './ui'

const MeContext = createContext<Me | null>(null)

export function useMe(): Me {
  const me = useContext(MeContext)
  if (!me) throw new Error('useMe outside AuthGate')
  return me
}

export function useCurrentUser(): User {
  return useMe().user
}

/** Renders the app once /api/me answers; a 401 shows the sign-in page instead. */
export function AuthGate({ children }: { children: ReactNode }) {
  const location = useLocation()
  const me = useQuery({
    queryKey: ['me'],
    queryFn: api.me,
    retry: false,
    staleTime: 5 * 60_000,
  })
  const authInfo = useQuery({
    queryKey: ['auth-info'],
    queryFn: api.authInfo,
    enabled: me.isError,
    retry: false,
  })
  if (me.isLoading) {
    return (
      <div className="grid min-h-screen place-items-center">
        <Spinner label="Connecting…" />
      </div>
    )
  }
  if (me.isError) {
    const status = me.error instanceof ApiError ? me.error.status : 0
    if (status === 401) {
      const params = new URLSearchParams(location.search)
      return <LoginPage auth={(authInfo.data as AuthInfo | undefined) ?? null} error={params.get('error')} />
    }
    return (
      <div className="grid min-h-screen place-items-center px-4">
        <Callout tone="danger" title="The service is not reachable" className="max-w-md">
          {me.error instanceof Error ? me.error.message : 'Unknown error'}. Start the API (make api) and reload.
        </Callout>
      </div>
    )
  }
  return <MeContext.Provider value={me.data!}>{children}</MeContext.Provider>
}
