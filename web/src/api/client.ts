import type {
  CaseListItem,
  CasePayload,
  DecisionInput,
  DistinguishResponse,
  Dossier,
  ElasticSearchResult,
  Graph,
  IntelResponse,
  InterviewState,
  Me,
  SystemInfo,
  Top3View,
  User,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      Accept: 'application/json',
      // marks requests from our own script; the API refuses state changes without it
      'X-Requested-With': 'Vestige',
      ...(init.body && !(init.body instanceof FormData)
        ? { 'Content-Type': 'application/json' }
        : {}),
      ...(init.headers ?? {}),
    },
  })
  if (response.status === 204) return undefined as T
  const text = await response.text()
  let data: unknown = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail =
      data && typeof data === 'object' && 'detail' in data
        ? String((data as { detail: unknown }).detail)
        : response.statusText || `HTTP ${response.status}`
    throw new ApiError(response.status, detail)
  }
  return data as T
}

const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) })

export const api = {
  me: () => request<Me>('/api/me'),
  authInfo: () => request<{ mode: 'off' | 'google'; provider: string | null; allowlist_configured: boolean }>('/api/auth'),
  logout: () => request<void>('/auth/logout', { method: 'POST' }),
  users: () => request<User[]>('/api/users'),
  updateUser: (id: string, patch: { role?: string; active?: boolean }) =>
    request<User>(`/api/users/${id}`, { method: 'PATCH', body: JSON.stringify(patch) }),
  system: () => request<SystemInfo>('/api/system'),
  sample: () => request<{ text: string }>('/api/sample'),
  techniques: (q: string) =>
    request<{ id: string; name: string }[]>(`/api/techniques?q=${encodeURIComponent(q)}&limit=12`),
  cases: () => request<CaseListItem[]>('/api/cases'),
  case: (id: string) => request<CasePayload>(`/api/cases/${id}`),
  createCase: (form: FormData) => request<CasePayload>('/api/cases', { method: 'POST', body: form }),
  rename: (id: string, name: string) =>
    request<{ case_id: string; name: string }>(`/api/cases/${id}`, {
      method: 'PATCH',
      body: JSON.stringify({ name }),
    }),
  remove: (id: string) => request<void>(`/api/cases/${id}`, { method: 'DELETE' }),
  decisions: (id: string, decisions: DecisionInput[]) =>
    request<CasePayload>(`/api/cases/${id}/decisions`, json({ decisions })),
  weights: (id: string, weights: Record<string, number>) =>
    request<CasePayload>(`/api/cases/${id}/weights`, json({ weights })),
  platforms: (id: string, targetPlatforms: string[]) =>
    request<CasePayload>(`/api/cases/${id}/platforms`, json({ target_platforms: targetPlatforms })),
  interview: (id: string) => request<InterviewState>(`/api/cases/${id}/interview`),
  answer: (id: string, status: string, text?: string) =>
    request<InterviewState>(`/api/cases/${id}/interview/answer`, json({ status, text })),
  restart: (id: string) =>
    request<InterviewState>(`/api/cases/${id}/interview/restart`, { method: 'POST' }),
  review: (id: string) => request<CasePayload>(`/api/cases/${id}/review`, { method: 'POST' }),
  cyberre: (id: string, observationId: string | null, usePrefilter: boolean) =>
    request<CasePayload>(
      `/api/cases/${id}/cyberre`,
      json({ observation_id: observationId, use_prefilter: usePrefilter }),
    ),
  /** topK null = the naming gate's own window (the default view) */
  distinguish: (id: string, topK: number | null) =>
    request<DistinguishResponse>(
      `/api/cases/${id}/distinguish${topK === null ? '' : `?top_k=${topK}`}`,
    ),
  top3: (id: string) => request<Top3View>(`/api/cases/${id}/top3`),
  dossier: (id: string, groupId: string) => request<Dossier>(`/api/cases/${id}/dossier/${groupId}`),
  canvas: (id: string, topGroups: number, maxEvidence: number) =>
    request<Graph>(`/api/cases/${id}/canvas?top_groups=${topGroups}&max_evidence=${maxEvidence}`),
  diamond: (id: string, topGroups: number, caps: number, infra: number) =>
    request<Graph>(
      `/api/cases/${id}/diamond?top_groups=${topGroups}&max_capabilities=${caps}&max_infrastructure=${infra}`,
    ),
  intel: (id: string, days: number, claims: boolean, refresh: boolean) =>
    request<IntelResponse>(
      `/api/cases/${id}/intel?days=${days}&claims=${claims}&refresh=${refresh}`,
    ),
  elasticSearch: (id: string, lookbackHours: number | null) =>
    request<ElasticSearchResult>(
      `/api/cases/${id}/elastic/search`,
      json({ lookback_hours: lookbackHours }),
    ),
  elasticImport: (id: string, candidateIds: string[]) =>
    request<CasePayload>(`/api/cases/${id}/elastic/import`, json({ candidate_ids: candidateIds })),
  exportUrl: (id: string, kind: 'markdown' | 'json' | 'navigator' | 'attack-flow') =>
    `/api/cases/${id}/export/${kind}`,
}
