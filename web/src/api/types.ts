// Mirrors api/serializers.py. Keep the two in sync; the API is the source of truth.

export type Stage =
  | 'insufficient_evidence'
  | 'unresolved_activity_cluster'
  | 'known_group_shortlist'
  | 'known_group_hypothesis'

export type CandidateState = 'auto_accepted' | 'needs_review' | 'auto_rejected'

export interface SystemInfo {
  mode: 'local' | 'production'
  attack_version: string
  groups: number
  techniques: number
  embedding_model: string
  extract_model: string
  review_model: string
  preset: string
  gate_mode: string
  acceptance_model: unknown
  combiner: string
  /** labelled-sentence exemplar index; when not active the acceptance model is off too */
  exemplars?: string
  index: {
    ok: boolean
    technique_sentences?: number
    group_documents?: number
    actor_behavior_documents?: number
    error?: string
  }
  llm_spend_usd: number
  cases: number
  elastic_mcp: boolean
  classifier_weights: Record<string, number>
}

export interface CaseSummary {
  observations?: number
  stage?: Stage | null
  leading_group?: string | null
  leading_group_name?: string | null
  accepted_techniques?: number
  answers?: number
}

export interface CaseListItem {
  case_id: string
  name: string
  created_at: string | null
  updated_at: string | null
  owner: CaseOwner | null
  summary: CaseSummary
}

export interface Observation {
  id: string
  index: number
  text: string
  kind: string
  status: string
  source: string
  source_name: string
  occurrences: number
  lines: number[]
}

export interface EvidenceItem {
  source: string
  field: string | null
  score: number | null
  rank: number | null
  text: string | null
  /** labelled corpus of an exemplar sentence (tram, expert, derived_procedures) */
  dataset?: string | null
}

export interface Candidate {
  key: string
  observation_id: string
  observation_index: number
  technique_id: string
  technique_name: string
  parent_technique_id: string | null
  is_subtechnique: boolean
  state: CandidateState
  state_reason: string
  rank: number
  fused_score: number
  probability: number | null
  evidence: EvidenceItem[]
  verification: { verdict: string; reason: string } | null
}

export interface AppliedDecision {
  observation_id: string
  observation: string
  technique_id: string
  action: 'confirm' | 'reject' | 'replace'
  replacement: string | null
  rationale: string
}

export interface AcceptedTechnique {
  technique_id: string
  technique_name: string
  confidence: number
  observation_ids: string[]
  matched_text: string
}

export interface Discrimination {
  group_id: string
  rrf: number
  technique: number
  behavior: number
  software: number
  negative: number
  total: number
}

export interface RankingEntry {
  group_id: string
  name: string
  rank: number
  score: number
  relative_score: number
  classifier_ranks: Record<string, number>
  rank_min: number
  rank_max: number
  rank_std: number
  contested: boolean
  discrimination: Discrimination | null
}

export interface DecisionInfo {
  stage: Stage
  label: string
  summary: string
  advice: string
  named: boolean
  leading_group_id: string | null
  leading_group_name: string
  leading_score: number
  margin: number
  maturity_score: number
  maturity_factors: Record<string, number>
  separation: number
  separation_factors: Record<string, unknown>
  runner_up_group_id: string | null
  runner_up_name: string | null
  runner_up_related: boolean
  related_groups: { group_id: string; name: string; reason: string }[]
  rationale: string
  cluster_id: string | null
  gate_mode: string
  calibrated_probability: number | null
  calibration_note: string
  thresholds: {
    maturity: number
    margin: number
    separation: number
    min_discriminative: number
    gate_mode: string
  }
  decisive_software: { factor: number; software: { id: string; name: string }[] } | null
}

export interface OriginInfo {
  top_country: string | null
  confidence: string
  coverage: number
  caveats: string[]
  hypotheses: {
    country: string
    relative_support: number
    supporting: { group_id: string; name: string }[]
  }[]
  threat_type: {
    label: string
    confidence: string
    likely_apt: boolean | null
    apt_support: number
    criminal_support: number
    unresolved_support: number
    rationale: string[]
  }
}

export interface BehaviourMatch {
  group_id: string
  name: string
  technique_id: string
  similarity: number
  observed: string
  documented: string
}

export interface StrategicMatch {
  rank: number
  group_id: string
  name: string
  coverage: number
  factors: string[]
}

export interface ClassifierMatrix {
  classifiers: string[]
  rows: { group_id: string; name: string; ranks: Record<string, number | null> }[]
}

export interface ReviewInfo {
  headline: string
  confidence_label: string
  assessment: string
  advisory_ranking: {
    rank: number | null
    group_id: string
    name: string
    deterministic_rank: number | null
    justification: string
  }[]
  limitations: string[]
  disconfirmation_tests: string[]
  fame_bias: { kendall_tau: number; prominence_correlation: number; largest_promotions: string[] }
  advocate: { counter_case: string; missing_evidence: string[] }
  audit_path: string | null
}

export interface ZeroShotInfo {
  model: string
  ranking: Record<string, unknown>[]
  cost_usd: number
}

export interface CyberREResult {
  observation_id: string
  observation_text: string
  model: string
  filter_calls: number
  kept: number
  applicable: boolean
  tokens: number
  cost_usd: number
  ranking: { rank: number; technique_id: string; technique_name: string; reason: string }[]
  filtered: { technique_id: string; technique_name: string; reason: string }[]
}

export interface IntelMatch {
  item_id: string
  score: number
  reasons: string[]
  verification: string
  group_ids: string[]
  published_at: string
  source: string
  title: string
  url: string
}

export interface IntelSnapshotSummary {
  fetched_at: string
  records: number
  authoritative: number
  claims: number
  sources: { source: string; ok: boolean; error: string | null }[]
}

export interface IntelItem {
  id: string
  published_at: string
  source: string
  kind: string
  title: string
  summary: string
  actors: string[]
  cves: string[]
  products: string[]
  industries: string[]
  verification: string
  match_score: number | null
  url: string
}

export interface IntelResponse {
  snapshot: IntelSnapshotSummary
  matches: IntelMatch[]
  items: IntelItem[]
}

export interface NextStep {
  message: string
  page: 'intake' | 'techniques' | 'interview' | 'assessment'
}

export interface CasePayload {
  case_id: string
  name: string
  created_at: string | null
  updated_at: string | null
  owner: CaseOwner | null
  ingestion: Record<string, unknown>
  summary: CaseSummary
  next_step: NextStep
  observations: Observation[]
  strategic_context: Record<string, string[]>
  candidate_counts: Record<CandidateState, number>
  candidates: Candidate[]
  decisions: AppliedDecision[]
  techniques: AcceptedTechnique[]
  ranking: RankingEntry[]
  ranking_includes_llm_vote: boolean
  deterministic_leader: string | null
  decision: DecisionInfo | null
  origin: OriginInfo
  behaviour_matches: BehaviourMatch[]
  strategic_matches: StrategicMatch[]
  classifier_matrix: ClassifierMatrix
  effective_weights: Record<string, number>
  weights: Record<string, number>
  eligibility: { group_id: string; name: string; via: string[] }[]
  retrieval_diagnostics: Record<string, unknown>
  interview: { complete: boolean; stop_reason: string | null; answered: number; has_question: boolean }
  review: ReviewInfo | null
  zero_shot: ZeroShotInfo | null
  cyberre: { results: Record<string, CyberREResult> }
  intel: { snapshot: IntelSnapshotSummary | null; matches: IntelMatch[] }
  group_label: string | null
}

export interface InterviewQuestion {
  technique_id: string
  question: string
  hypothesis: string
  investigation_steps: string[]
  profile_support: number
  supporting_group_count: number
  candidate_group_count: number
  probability_yes: number
  information_gain: number
  investigation_priority: number
}

export interface InterviewLead {
  technique_id: string
  technique_name: string
  priority: number
  profile_support: number
  supporting_group_count: number
  candidate_group_count: number
  first_step: string
}

export interface InterviewState {
  complete: boolean
  stop_reason: string | null
  selection_error: string | null
  answered: { question: string; technique_id: string | null; status: string; text: string | null }[]
  max_questions: number
  question: InterviewQuestion | null
  leads: InterviewLead[]
  weights: { group_id: string; weight: number }[]
}

export interface Dossier {
  group_id: string
  name: string
  rank: number
  relative_score: number
  contested: boolean
  description: string
  matched_techniques: Record<string, unknown>[]
  unmatched_techniques: Record<string, unknown>[]
  specific_behaviors: Record<string, unknown>[]
  tools: { software_id: string; name: string }[]
  classifier_ranks: Record<string, number>
  classifier_signals?: Record<string, unknown>
  country_evidence: { country: string; excerpt: string }[]
  contradicting_techniques: string[]
}

export interface GraphNode {
  id: string
  label: string
  score: number
  details: string
  kind?: string
  vertex?: string
  status?: string
}

export interface GraphEdge {
  source: string
  target: string
  relationship: string
  weight: number
  details: string
}

export interface Graph {
  nodes: GraphNode[]
  edges: GraphEdge[]
  caveats: string[]
}

export interface DecisionInput {
  observation_id: string
  technique_id: string
  action: 'confirm' | 'reject' | 'replace'
  replacement?: string | null
  rationale?: string | null
}

export interface ElasticSearchResult {
  ok: boolean
  error: string | null
  esql_query: string | null
  candidates: {
    id: string
    timestamp: string | null
    index: string | null
    event_id: string | null
    text: string
  }[]
}

export interface DistinguishTechnique {
  technique_id: string
  technique_name: string
  confidence: number
  pending: boolean
  documented_by: string[]
  exclusivity: number
  specificity: number
  weight: number
  contributions: Record<string, number>
}

export interface DistinguishPending {
  technique_id: string
  technique_name: string
  probability: number
  observation_id: string
  observation: string
  documented_by: { group_id: string; name: string }[]
  exclusivity: number
  specificity: number
  delta_separation: number
  new_leader_id: string | null
  new_leader_name: string | null
  keeps_leader: boolean
}

export interface DistinguishLead {
  technique_id: string
  technique_name: string
  priority: number
  profile_support: number
  supporting_group_count: number
  candidate_group_count: number
  delta_yes: number
  delta_no: number
  leader_after_yes: string | null
  first_step: string
}

export interface DistinguishResponse {
  top_k: number
  separation: number
  leader_total: number
  leader: { group_id: string | null; name: string | null }
  runner_up: { group_id: string | null; name: string | null }
  runner_up_related: boolean
  thresholds: { separation: number; min_discriminative: number; gate_mode: string }
  contrast_enabled: boolean
  shortlist: {
    group_id: string
    name: string
    rank: number
    relative_score: number
    technique?: number
    behavior?: number
    software?: number
    soft?: number
    negative?: number
    total?: number
  }[]
  techniques: DistinguishTechnique[]
  pending: DistinguishPending[]
  leads: DistinguishLead[]
}

export interface User {
  id: string
  email: string
  name: string
  picture: string
  role: 'admin' | 'analyst'
  active: boolean
  created_at: string
  last_login_at: string | null
}

export interface AuthInfo {
  mode: 'off' | 'google'
  provider: string | null
  allowlist_configured: boolean
}

export interface Me {
  user: User
  auth: AuthInfo
}

export interface CaseOwner {
  id: string
  name: string
  email: string
}
