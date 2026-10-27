// Mirrors api/serializers.py. Keep the two in sync; the API is the source of truth.

export type Stage =
  | 'insufficient_evidence'
  | 'unresolved_activity_cluster'
  | 'known_group_shortlist'
  | 'known_group_hypothesis'
  | 'known_cluster_hypothesis'

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
  /** where the exemplar library is served from: "npz", "chroma (exact)", "chroma (hnsw)" or "off" */
  exemplar_backend?: string
  /** the fine-tuned technique head's local service: "active", "active (unpinned)", "disabled" or "unavailable: ..." */
  technique_head?: string
  /** random-forest group and source-country models, e.g. "active (cuda)" */
  profile_forest?: string
  group_mlp?: string
  /** free-text TF-IDF group model; "disabled (weight 0)" unless TEXT_GROUP_WEIGHT > 0 */
  text_group?: string
  index: {
    ok: boolean
    technique_sentences?: number
    group_documents?: number
    actor_behavior_documents?: number
    error?: string
  }
  llm_spend_usd: number
  /** who answers the LLM tasks: OpenRouter, the local Ollama model, both, or nothing */
  llm_provider?: 'openrouter' | 'local' | 'hybrid' | 'none'
  /** model that answers each task (`local/…` runs on Ollama); null when unavailable */
  llm_models?: Record<'extraction' | 'verifier' | 'cyberre' | 'review', string | null>
  /** how report prose becomes observations: OpenRouter whole-report extraction, the local
   * model rewriting each split sentence, or the plain sentence split */
  prose_normalisation?: 'extract' | 'sentences' | 'split'
  cases: number
  elastic_mcp: boolean
  classifier_weights: Record<string, number>
}

export interface CaseSummary {
  observations?: number
  stage?: Stage | null
  leading_group?: string | null
  leading_group_name?: string | null
  /** members of a named documented-overlap cluster, leader first */
  named_cluster?: string[]
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

/** Operating systems an observation was mapped under: recorded by the log source, named in
 * the observation's text, or the case's Target OS. */
export interface PlatformScope {
  platforms: string[]
  source: 'log' | 'text' | 'case'
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
  platform: PlatformScope | null
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

export interface NamedCluster {
  members: { group_id: string; name: string }[]
  label: string
  reason: string
  separation: number
  discriminative: number
  probability: number | null
  runner_up_group_id: string | null
  runner_up_name: string | null
}

export interface DecisionInfo {
  stage: Stage
  label: string
  summary: string
  advice: string
  /** a single group is named (a named cluster is reported through named_kind / named_cluster) */
  named: boolean
  named_kind?: 'group' | 'cluster' | null
  named_cluster?: NamedCluster | null
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
  // ATT&CK-related groups; only "same_activity" documents the same actor (a cluster), the rest are
  // different actors ATT&CK merely links (shared rare malware, or a description calling them distinct)
  related_groups: { group_id: string; name: string; reason: string; link: string }[]
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
  /** Case Target OS: every OS the activity touched; empty = unknown (no constraint). */
  target_platforms: string[]
  platform_options: string[]
  strategic_context: Record<string, string[]>
  candidate_counts: Record<CandidateState, number>
  candidates: Candidate[]
  decisions: AppliedDecision[]
  techniques: AcceptedTechnique[]
  ranking: RankingEntry[]
  decision: DecisionInfo | null
  /** state of the first three candidates (null when it could not be computed) */
  top3: Top3Summary | null
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
  /** "software": a fixed-wording question about one malware entry (found / not found / not checked) */
  kind?: 'technique' | 'software'
  software_id?: string | null
  /** top-three place of the one candidate ATT&CK documents the malware for */
  target_rank?: number | null
  /** predicted change of the naming gate's separation if the malware is found */
  delta_found?: number | null
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
  answered: {
    question: string
    technique_id: string | null
    status: string
    text: string | null
    kind?: 'technique' | 'software'
    software_id?: string | null
  }[]
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

export interface DistinguishSoftware {
  software_id: string
  name: string
  group_id: string
  group_name: string
  target_rank: number
  users: string[]
  delta_found: number
  delta_not_found: number
  technique_count: number
}

export interface DistinguishResponse {
  top_k: number
  /** size of the naming gate's window (combiner top 5 plus RRF top 5), the default top_k */
  gate_top_k: number
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
  /** malware only one top-three candidate documents: what the software questions ask about */
  software?: DistinguishSoftware[]
}

export type Top3State =
  | 'named'
  | 'named_cluster'
  | 'overlap'
  | 'split'
  | 'thin'
  | 'not_separated'
  | 'empty'

export interface Top3Summary {
  state: Top3State
  label: string
  summary: string
  leader?: { group_id: string; name: string }
  other?: { group_id: string; name: string } | null
  /** calibrated confidence label (display only, never a decision); null when off */
  confidence?: Top3Confidence | null
}

/** per-label counts of one evaluation set, from the calibration artifact */
export interface Top3ConfidenceMeasured {
  source: string
  rows: number
  strong: { n: number; right: number }
  set: { n: number; right: number }
  open: { n: number; truth_in_top3: number }
}

/** How often cases like this one were right on ATT&CK test folds (confidence_labels.py) */
export interface Top3Confidence {
  label: 'strong' | 'set' | 'open'
  /** wording: named_lower = named by the gate, calibrated label lower; named_cluster = the gate's pair */
  text_key: 'strong' | 'one_of_these' | 'open' | 'named_lower' | 'named_cluster'
  /** calibrated probability that the shown answer holds (null when open) */
  probability: number | null
  p_leader: number | null
  p_top2: number | null
  p_top3: number | null
  set_size: number
  group_ids: string[]
  members: { group_id: string; name: string }[]
  /** set is null in a strong-only artifact: "one of these" is never shown */
  thresholds: { strong: number; set: number | null }
  measured: { folds: Top3ConfidenceMeasured | null; real_reports: Top3ConfidenceMeasured | null }
  note: string
}

/** another top-three member holds clearly more exclusive evidence than the leader */
export interface Top3Contest {
  promoted: string
  previous_leader: string
  from_rank: number
  gap: number
  leader_advantage: number
  advantage_channel: string | null
  blocked: boolean
  margin: number
  tolerance: number
}

export interface Top3Channels {
  technique: number
  behavior: number
  software: number
  soft: number
  negative: number
  total: number
}

export interface Top3Candidate {
  group_id: string
  name: string
  rank: number
  relative_score: number
  channels: Top3Channels | null
  combiner_probability: number | null
  rrf_rank: number | null
  profile_size: number
  broad_profile: boolean
  named_software: { id: string; name: string }[]
  /** groups ATT&CK documents as overlapping with this one, with their rank in this case */
  related: {
    group_id: string
    name: string
    reason: string
    link: 'same_activity' | 'nesting' | 'description' | 'shared_software'
    rank: number | null
  }[]
  /** ATT&CK's own description and campaigns, not an inference from this case */
  documented: {
    origins: { country: string; sentence: string }[]
    type: 'state-sponsored' | 'criminal' | 'mixed' | null
    state_markers: string[]
    criminal_markers: string[]
    motivations: string[]
  }
}

export interface Top3OneSided {
  techniques: { technique_id: string; name: string; confidence: number; specificity: number }[]
  technique_count: number
  software: { id: string; name: string }[]
  behavior_wins: { observation_id: string; text: string }[]
  behavior_win_count: number
}

export interface Top3Pair {
  a: string
  b: string
  a_name: string
  b_name: string
  separation: number
  favours: string | null
  related_reason: string | null
  // how ATT&CK links the pair: the same activity (reviewed table), nesting profiles, a description mention
  // that is not the same activity, or only rare shared malware (null: not linked)
  link: 'same_activity' | 'nesting' | 'description' | 'shared_software' | null
  containment: number
  // one cluster under the naming gate's scope (not for a shared-malware-only link by default)
  overlap_cluster: boolean
  a_only: Top3OneSided
  b_only: Top3OneSided
}

export interface Top3View extends Top3Summary {
  contest: Top3Contest | null
  candidates: Top3Candidate[]
  pairs: Top3Pair[]
  /** channel weights the naming gate scored the exclusive evidence with */
  weights: Record<'technique' | 'behavior' | 'software' | 'soft' | 'negative', number> | null
  display_order_differs?: boolean
  /** "not found" software interview answers and their effect on the leader's separation */
  negative_software?: {
    software_ids: string[]
    applied: string[]
    penalty: number
    total_before: number
    total: number
    separation_before: number
    separation: number
    members: string[]
  } | null
  combiner_note?: string
  thresholds: {
    separation: number
    min_discriminative: number
    clear_lead: number
    tolerance: number
    thin_evidence: number
  }
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
