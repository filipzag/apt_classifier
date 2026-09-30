import { ShieldCheck, UserX } from 'lucide-react'
import { Navigate } from 'react-router-dom'

import { describe, useUpdateUser, useUsers } from '../api/hooks'
import { useCurrentUser, useMe } from '../components/AuthGate'
import { PageHeader } from '../components/CaseFrame'
import { Badge, Callout, Card, Segmented, Spinner, Toggle } from '../components/ui'
import { relativeTime } from '../lib/utils'

export function UsersPage() {
  const me = useMe()
  const current = useCurrentUser()
  const users = useUsers(current.role === 'admin')
  const update = useUpdateUser()
  if (current.role !== 'admin') return <Navigate to="/" replace />
  return (
    <div className="space-y-5">
      <PageHeader
        title="Users"
        lead="Everyone who has signed in to this workspace. Analysts work on cases; administrators also manage users and can delete any case. A user is created on first sign-in when the address is on the allowlist."
      />
      {me.auth.mode !== 'google' ? (
        <Callout tone="info" title="Sign-in is off">
          The app runs as one local analyst with administrator rights. Set GOOGLE_CLIENT_ID,
          GOOGLE_CLIENT_SECRET and an allowlist to enable Google sign-in and per-user cases.
        </Callout>
      ) : null}
      <Card className="p-0">
        {users.isLoading ? (
          <Spinner label="Loading users…" className="p-5" />
        ) : users.isError ? (
          <Callout tone="danger" className="m-4">
            {describe(users.error)}
          </Callout>
        ) : (
          <div className="overflow-x-auto">
            <table className="data">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Role</th>
                  <th>Status</th>
                  <th>Last sign-in</th>
                  <th>Since</th>
                </tr>
              </thead>
              <tbody>
                {(users.data ?? []).map((user) => {
                  const self = user.id === current.id
                  const busy = update.isPending && update.variables?.id === user.id
                  return (
                    <tr key={user.id}>
                      <td className="min-w-[260px]">
                        <div className="flex items-center gap-3">
                          <Avatar name={user.name} picture={user.picture} />
                          <div>
                            <div className="font-medium">
                              {user.name}
                              {self ? <Badge className="ml-2">you</Badge> : null}
                            </div>
                            <div className="text-[11px] text-muted">{user.email}</div>
                          </div>
                        </div>
                      </td>
                      <td>
                        <Segmented
                          value={user.role}
                          onChange={(role) => update.mutate({ id: user.id, role })}
                          options={[
                            { value: 'analyst', label: 'Analyst' },
                            { value: 'admin', label: <span className="inline-flex items-center gap-1"><ShieldCheck className="size-3" /> Admin</span> },
                          ]}
                        />
                      </td>
                      <td>
                        <Toggle
                          checked={user.active}
                          onChange={(active) => update.mutate({ id: user.id, active })}
                          label={user.active ? 'active' : <span className="inline-flex items-center gap-1 text-rose"><UserX className="size-3" /> deactivated</span>}
                          className={busy ? 'opacity-50' : undefined}
                        />
                      </td>
                      <td className="whitespace-nowrap text-muted">{relativeTime(user.last_login_at)}</td>
                      <td className="whitespace-nowrap text-muted">{user.created_at.slice(0, 10)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}

export function Avatar({ name, picture, size = 'size-8' }: { name: string; picture?: string; size?: string }) {
  const initials = name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? '')
    .join('')
  if (picture) {
    return <img src={picture} alt="" referrerPolicy="no-referrer" className={`${size} rounded-full object-cover ring-1 ring-border-strong`} />
  }
  return (
    <span className={`grid ${size} place-items-center rounded-full bg-surface-3 text-[11px] font-semibold text-muted ring-1 ring-border-strong`}>
      {initials || '?'}
    </span>
  )
}
