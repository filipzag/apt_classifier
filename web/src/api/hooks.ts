import { useIsMutating, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'

import { api, ApiError } from './client'
import type { CasePayload, DecisionInput, InterviewState } from './types'

export const keys = {
  system: ['system'] as const,
  cases: ['cases'] as const,
  case: (id: string) => ['case', id] as const,
  interview: (id: string) => ['interview', id] as const,
  dossier: (id: string, group: string) => ['dossier', id, group] as const,
  distinguish: (id: string, k: number | null) => ['distinguish', id, k ?? 'gate'] as const,
  top3: (id: string) => ['top3', id] as const,
  canvas: (id: string, g: number, e: number) => ['canvas', id, g, e] as const,
  diamond: (id: string, g: number, c: number, i: number) => ['diamond', id, g, c, i] as const,
  intel: (id: string, d: number, c: boolean) => ['intel', id, d, c] as const,
  /** mutation key: lets the page see a call that is still under way */
  modelOpinion: (id: string) => ['model-opinion', id] as const,
}

export function describe(error: unknown): string {
  if (error instanceof ApiError) return error.message
  if (error instanceof Error) return error.message
  return String(error)
}

export function useSystem() {
  return useQuery({ queryKey: keys.system, queryFn: api.system, staleTime: 60_000 })
}

export function useCases() {
  return useQuery({ queryKey: keys.cases, queryFn: api.cases })
}

export function useCase(id: string | undefined) {
  return useQuery({
    queryKey: keys.case(id ?? ''),
    queryFn: () => api.case(id!),
    enabled: Boolean(id),
    staleTime: 30_000,
  })
}

export function useInterview(id: string | undefined) {
  return useQuery({
    queryKey: keys.interview(id ?? ''),
    queryFn: () => api.interview(id!),
    enabled: Boolean(id),
    staleTime: 15_000,
  })
}

export function useDistinguish(id: string | undefined, topK: number | null) {
  return useQuery({
    queryKey: keys.distinguish(id ?? '', topK),
    queryFn: () => api.distinguish(id!, topK),
    enabled: Boolean(id),
    staleTime: 30_000,
  })
}

export function useTop3(id: string | undefined) {
  return useQuery({
    queryKey: keys.top3(id ?? ''),
    queryFn: () => api.top3(id!),
    enabled: Boolean(id),
    staleTime: 30_000,
  })
}

export function useDossier(id: string | undefined, group: string | null) {
  return useQuery({
    queryKey: keys.dossier(id ?? '', group ?? ''),
    queryFn: () => api.dossier(id!, group!),
    enabled: Boolean(id && group),
    staleTime: 60_000,
  })
}

/** Writes the case payload returned by a mutation into the cache and refreshes dependants. */
export function useCaseWriter(id: string) {
  const client = useQueryClient()
  return (payload: CasePayload) => {
    client.setQueryData(keys.case(id), payload)
    void client.invalidateQueries({ queryKey: ['interview', id] })
    void client.invalidateQueries({ queryKey: ['dossier', id] })
    void client.invalidateQueries({ queryKey: ['distinguish', id] })
    void client.invalidateQueries({ queryKey: ['top3', id] })
    void client.invalidateQueries({ queryKey: ['canvas', id] })
    void client.invalidateQueries({ queryKey: ['diamond', id] })
    void client.invalidateQueries({ queryKey: keys.cases })
  }
}

export function useApplyDecisions(id: string) {
  const write = useCaseWriter(id)
  return useMutation({
    mutationFn: (decisions: DecisionInput[]) => api.decisions(id, decisions),
    onSuccess: (payload, decisions) => {
      write(payload)
      toast.success(`Applied ${decisions.length} decision${decisions.length === 1 ? '' : 's'}`, {
        description: 'Technique mappings and the group ranking were recomputed.',
      })
    },
    onError: (error) => toast.error('Could not apply decisions', { description: describe(error) }),
  })
}

export function useSetPlatforms(id: string) {
  const write = useCaseWriter(id)
  return useMutation({
    mutationFn: (platforms: string[]) => api.platforms(id, platforms),
    onSuccess: (payload) => {
      write(payload)
      const chosen = payload.target_platforms
      toast.success(chosen.length ? `Target OS: ${chosen.join(', ')}` : 'Target OS cleared', {
        description: 'Observations were mapped again and the group ranking was recomputed.',
      })
    },
    onError: (error) => toast.error('Could not change the target OS', { description: describe(error) }),
  })
}

export function useRescore(id: string) {
  const write = useCaseWriter(id)
  return useMutation({
    mutationFn: (weights: Record<string, number>) => api.weights(id, weights),
    onSuccess: (payload) => {
      write(payload)
      toast.success('Ranking recomputed with the new weights')
    },
    onError: (error) => toast.error('Re-score failed', { description: describe(error) }),
  })
}

export function useInterviewMutations(id: string) {
  const client = useQueryClient()
  const settle = (state: InterviewState) => {
    client.setQueryData(keys.interview(id), state)
    void client.invalidateQueries({ queryKey: keys.case(id) })
    void client.invalidateQueries({ queryKey: ['distinguish', id] })
    void client.invalidateQueries({ queryKey: ['top3', id] })
    void client.invalidateQueries({ queryKey: keys.cases })
  }
  const answer = useMutation({
    mutationFn: ({ status, text }: { status: string; text?: string }) => api.answer(id, status, text),
    onSuccess: settle,
    onError: (error) => toast.error('Answer not recorded', { description: describe(error) }),
  })
  const restart = useMutation({
    mutationFn: () => api.restart(id),
    onSuccess: (state) => {
      settle(state)
      toast.success('Interview restarted')
    },
    onError: (error) => toast.error('Restart failed', { description: describe(error) }),
  })
  return { answer, restart }
}

export function useReview(id: string) {
  const write = useCaseWriter(id)
  return useMutation({
    mutationFn: () => api.review(id),
    onSuccess: (payload) => {
      write(payload)
      toast.success('Independent review complete')
    },
    onError: (error) => toast.error('Review failed', { description: describe(error) }),
  })
}

/** Asks the language model for its second opinion (one paid call, only on the analyst's click). */
export function useModelOpinion(id: string) {
  const write = useCaseWriter(id)
  return useMutation({
    mutationKey: keys.modelOpinion(id),
    mutationFn: () => api.modelOpinion(id),
    onSuccess: (payload) => {
      write(payload)
      toast.success('The model answered', {
        description: "Its second opinion is shown below the decision. The product's answer is unchanged.",
      })
    },
    onError: (error) => {
      // 400 and 409 are refusals before any call; anything else can follow a paid one
      const refused = error instanceof ApiError && (error.status === 400 || error.status === 409)
      toast.error(refused ? 'The model could not be asked' : 'No answer from the model', {
        description: refused ? describe(error) : `${describe(error)} The call may still have been charged.`,
      })
    },
  })
}

/** True while a second-opinion call for this case is under way, also after leaving the page
 * and coming back: the button stays busy, because a second click is a second paid call. */
export function useModelOpinionPending(id: string): boolean {
  return useIsMutating({ mutationKey: keys.modelOpinion(id) }) > 0
}

export function useCyberRE(id: string) {
  const write = useCaseWriter(id)
  return useMutation({
    mutationFn: ({ observationId, usePrefilter }: { observationId: string | null; usePrefilter: boolean }) =>
      api.cyberre(id, observationId, usePrefilter),
    onSuccess: (payload) => {
      write(payload)
      toast.success('CyberRE scan complete')
    },
    onError: (error) => toast.error('CyberRE scan failed', { description: describe(error) }),
  })
}

export function useRename(id: string) {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.rename(id, name),
    onSuccess: (result) => {
      client.setQueryData<CasePayload | undefined>(keys.case(id), (old) =>
        old ? { ...old, name: result.name } : old,
      )
      void client.invalidateQueries({ queryKey: keys.cases })
    },
    onError: (error) => toast.error('Rename failed', { description: describe(error) }),
  })
}

export function useDeleteCase() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.remove(id),
    onSuccess: (_, id) => {
      client.removeQueries({ queryKey: keys.case(id) })
      void client.invalidateQueries({ queryKey: keys.cases })
      toast.success('Case deleted')
    },
    onError: (error) => toast.error('Delete failed', { description: describe(error) }),
  })
}

export function useUsers(enabled = true) {
  return useQuery({ queryKey: ['users'], queryFn: api.users, enabled, staleTime: 30_000 })
}

export function useUpdateUser() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: ({ id, ...patch }: { id: string; role?: string; active?: boolean }) =>
      api.updateUser(id, patch),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ['users'] })
      toast.success('User updated')
    },
    onError: (error) => toast.error('User not updated', { description: describe(error) }),
  })
}

export function useLogout() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: api.logout,
    onSuccess: () => {
      client.clear()
      window.location.assign('/login')
    },
    onError: (error) => toast.error('Sign-out failed', { description: describe(error) }),
  })
}
