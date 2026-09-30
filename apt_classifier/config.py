from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: SecretStr | None = None
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_ssl: bool = False
    embedding_model: str = "basel/ATTACK-BERT"
    embedding_model_revision: str = "81e30f983a822c606825507b63ae318a6830a8a2"
    extract_model: str = "openai/gpt-5.6-sol"
    review_model: str = "anthropic/claude-sonnet-5"
    extract_effort: str = "low"
    question_effort: str = "medium"
    review_effort: str = "max"
    attack_version: str = "latest"
    similarity_threshold: float = 0.45
    retrieval_top_k: int = 30
    technique_mapping_preset: str = "balanced"
    technique_retrieval_mode: str = "hybrid"
    technique_candidate_pool: int = 12
    technique_include_detection: bool = False
    technique_parent_policy: str = "ancestor_compatible"
    technique_verifier_mode: str = "ambiguous"
    technique_verifier_model: str = "openai/gpt-5.6-sol"
    technique_verifier_effort: str = "low"
    reranker_enabled: bool = False
    reranker_model: str = ""
    reranker_model_revision: str | None = None
    reranker_expected_base_model: str | None = None
    reranker_expected_base_model_revision: str | None = None
    reranker_expected_artifact_fingerprint: str | None = None
    reranker_max_candidates: int = 20
    behavior_similarity_threshold: float = 0.55
    behavior_retrieval_top_k: int = 20
    technique_acceptance_threshold: float = 0.75
    technique_relative_margin: float = 0.08
    # Calibrated per-candidate acceptance (fitted on public dev splits by
    # scripts/fit_acceptance_model.py). Falls back to the rule-based gate when the
    # artifact is missing or incompatible with the current ATT&CK version.
    acceptance_model_enabled: bool = True
    # Bonus added to a technique's observed confidence for each additional independent
    # observation that accepted it (capped at three), 0 disables. Measured with the
    # attribution benchmark before adoption.
    technique_corroboration_bonus: float = 0.0
    # LLM-generated observation sentences per technique (scripts/generate_synthetic_sentences.py),
    # indexed as the 'synthetic' field; the flag lets benchmarks ablate them.
    synthetic_sentences_path: Path = PROJECT_ROOT / "data" / "kb" / "synthetic_sentences.json"
    # Off by default: with the partial set (50 techniques) held-out F1 moved 0.320 -> 0.315 and
    # CTI-HAL was unchanged; re-measure after the full generation before enabling.
    technique_include_synthetic: bool = False
    # Long observations are retrieved and accepted per sentence (own pool, ranks, margins).
    # Held-out result: F1-neutral with the merged-pool acceptance model (expert +0.07, pooled
    # precision 0.50 -> 0.38); worse with a per-sentence refit. Off by default; opt in for
    # long-report workloads where paragraph recall matters more than precision.
    technique_segment_acceptance: bool = False
    # Hierarchy consistency: 'swap_only' replaces an accepted sub-technique that is not
    # separable from a plausible sibling with the parent; 'swap_and_fill' also accepts the
    # parent when plausible siblings exist and none was accepted; 'off' disables the rule.
    # Held-out result: 'swap_only' neutral, 'swap_and_fill' collapses precision. Off by default.
    technique_hierarchy_parent_mode: str = "off"
    # Order each observation's candidates by the calibrated acceptance probability instead of
    # the raw RRF score (the model is fitted on "is this mapping correct"). Adopted after the
    # held-out gate: hit@1 0.35 -> 0.40, MRR 0.47 -> 0.52, accepted set unchanged.
    technique_rank_by_acceptance: bool = True
    # Accept a technique still in review when at least this many different observations of
    # the case retrieved it with a calibrated probability >= the floor (0 = off). Adopted
    # 2026-09-29: real-report accepted recall 0.26 -> 0.31 at flat precision, naming gate clean.
    technique_corroboration_accept_observations: int = 2
    technique_corroboration_accept_min_probability: float = 0.1
    acceptance_model_path: Path = (
        PROJECT_ROOT / "data" / "models" / "acceptance" / "acceptance_model.json"
    )
    max_questions: int = 10
    threat_intel_days: int = 30
    threat_intel_timeout: float = 20.0
    threat_intel_max_items_per_source: int = 100
    include_ransomware_claims: bool = True
    elastic_mcp_url: str | None = None
    elastic_mcp_api_key: SecretStr | None = None
    elastic_mcp_timeout: float = 60.0
    elastic_mcp_lookback_hours: int = 72
    elastic_mcp_max_results: int = 20
    elastic_mcp_auto_gather: bool = False
    # Group-ensemble RRF constant. Lower values spread relative scores and margins; the
    # attribution benchmark (section 21.3.5) is used to choose it.
    rrf_k: int = 20
    # Technique-level dense/lexical RRF constant; kept separate because the acceptance
    # model's features are fitted against it.
    technique_rrf_k: int = 60
    software_idf_enabled: bool = True
    # Cap on the rarity weight of ATT&CK *tool* entries (legitimate utilities); malware keeps
    # its full rarity weight. See classifiers/software.py.
    software_tool_weight_cap: float = 1.5
    # Extra multiplier on the software vote (classifier weight and family budget) when the
    # case names a malware family documented for at most two groups (half of it for up to
    # four). 0 disables. See classifiers/software.py::decisive_software_factor.
    software_decisive_boost: float = 2.0
    technique_family_weight: float = 2.00
    context_family_weight: float = 1.00
    motivation_family_weight: float = 1.50
    actor_behavior_family_weight: float = 1.50
    software_family_weight: float = 1.20
    campaign_family_weight: float = 1.00
    llm_zero_shot_weight: float = 0.50
    llm_zero_shot_family_weight: float = 0.50
    cyberre_model: str = "openai/gpt-5.6-sol"
    cyberre_effort: str = "low"
    cyberre_weight: float = 0.35
    cyberre_family_weight: float = 0.35
    cyberre_max_ranked_techniques: int = 50
    cyberre_prefilter_top_k: int = 50
    cyberre_prefilter_enabled: bool = True
    specific_behavior_weight: float = 1.50
    # Per-classifier weights inside the technique-overlap family (the family budget still caps
    # their sum). Naive Bayes is the most reliable of the three on real reports; the two
    # size-normalised overlaps penalise groups with long ATT&CK technique lists.
    naive_bayes_weight: float = 1.00
    idf_overlap_weight: float = 1.00
    jaccard_weight: float = 1.00
    # Group-size normalisation exponent of the IDF cosine classifier (1.0 = full cosine, 0.5 =
    # square-root damping, 0 = ignore group size). See classifiers/idf_overlap.py.
    idf_overlap_length_norm: float = 1.0
    campaign_weight: float = 0.00
    group_description_weight: float = 0.40
    semantic_centroid_weight: float = 0.70
    strategic_context_weight: float = 0.70
    motivation_weight: float = 1.50
    software_weight: float = 1.20
    nb_alpha: float = 0.60
    nb_beta: float = 0.05
    negative_evidence_weight: float = 0.30
    posterior_temperature: float = 2.0
    # Telemetry-aware negative evidence
    telemetry_validation_enabled: bool = True
    # Attribution stages
    attribution_min_techniques: int = 2
    attribution_min_independent_families: int = 2
    attribution_min_maturity: float = 0.5
    # 0.78 came from the ATT&CK self-consistency folds; 0.82 is the lowest value that is also
    # clean on the 55 CTI-HAL real reports (3 of 3 named correct, was 4 of 7 at 0.78).
    attribution_hypothesis_maturity: float = 0.82
    attribution_hypothesis_margin: float = 0.10
    # Soft technique evidence for the group ranking: NEEDS_REVIEW candidates with a calibrated
    # acceptance probability >= this value count with probability * weight in the technique
    # classifiers (never in the accepted set, the interview or the naming gate). 0 = off.
    attribution_soft_evidence_min_probability: float = 0.0
    attribution_soft_evidence_weight: float = 1.0
    # Only techniques documented for at most this many groups may count as soft evidence
    # (0 = no restriction).
    attribution_soft_evidence_max_groups: int = 0
    # Candidate-relative separation (separation.py). The gate mode decides whether naming uses
    # the maturity bar ("maturity", current), separation and discriminative mass ("separation")
    # or both. Bars are chosen with scripts/attribution_gate_sweep.py; off until measured.
    # Adopted 2026-09-29 (docs 10.7): separation gate with bars from the offline sweep and a
    # 0.5 re-ranking blend. Folds: 35 named all correct (was 14), 0/25 removed actors named
    # (was 1), top-1 0.52 -> 0.60; real reports: 12 named all correct (was 3), top-1 0.44 -> 0.45.
    attribution_gate_mode: str = "separation"
    attribution_separation_top_k: int = 5
    attribution_separation_technique_weight: float = 1.0
    attribution_separation_behavior_weight: float = 2.0
    attribution_separation_software_weight: float = 1.0
    attribution_separation_negative_weight: float = 0.3
    attribution_separation_floor: float = 1.0
    attribution_hypothesis_separation: float = 0.5
    attribution_hypothesis_min_discriminative: float = 3.0
    attribution_hypothesis_specificity: float = 0.0
    # Blend the discriminative mass into the RRF score of the top-K (0 = pure RRF order).
    attribution_rerank_blend: float = 0.5
    # Shortlist size for the discriminative re-ranking; 0 = the gate's shortlist
    # (ATTRIBUTION_SEPARATION_TOP_K). Adopted at 10 on 2026-09-29: folds top-5 0.804 -> 0.915,
    # CTI-HAL top-1 0.455 -> 0.509, every named group still correct on both sets (docs 21.3.8).
    attribution_rerank_top_k: int = 10
    # Contrastive near misses: NEEDS_REVIEW candidates with acceptance probability >= this enter
    # the separation's ``soft`` channel (exclusivity- and specificity-weighted, never a
    # classifier vote). 0 disables. Off until measured on both attribution benchmarks.
    attribution_contrast_min_probability: float = 0.0
    attribution_contrast_weight: float = 1.0
    attribution_separation_soft_weight: float = 1.0
    # Activity clustering (opt-in)
    activity_clusters_enabled: bool = False
    activity_clusters_db_path: Path = PROJECT_ROOT / "data" / "activity_clusters.db"
    activity_clusters_min_similarity: float = 0.3
    data_dir: Path = PROJECT_ROOT / "data"
    model_cache_dir: Path = PROJECT_ROOT / ".huggingface"

    # ----- users and sign-in (web app) -----
    # "auto" = Google sign-in when GOOGLE_CLIENT_ID is set, otherwise a single local analyst.
    auth_mode: str = "auto"  # auto | off | google
    google_client_id: str | None = None
    google_client_secret: SecretStr | None = None
    # Signs the session cookie; generate one (python -c "import secrets;print(secrets.token_urlsafe(48))").
    auth_secret_key: SecretStr | None = None
    # Who may sign in: comma-separated e-mail domains and/or addresses (at least one required).
    auth_allowed_domains: str = ""
    auth_allowed_emails: str = ""
    # Administrators (comma-separated addresses); the first user to sign in is an admin too.
    auth_admin_emails: str = ""
    auth_session_hours: int = 8
    auth_cookie_secure: bool = False  # true behind HTTPS
    # External origin the OAuth redirect must use when behind a proxy (e.g. https://apt.example.com).
    auth_public_url: str | None = None

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def kb_dir(self) -> Path:
        return self.data_dir / "kb"

    @property
    def audit_dir(self) -> Path:
        return self.data_dir / "audits"

    @property
    def kb_path(self) -> Path:
        return self.kb_dir / "attack_kb.json"

    @property
    def attack_bundle_path(self) -> Path:
        return self.raw_dir / "enterprise-attack.json"

    def require_api_key(self) -> str:
        if self.openrouter_api_key is None or not self.openrouter_api_key.get_secret_value():
            raise RuntimeError(
                "OPENROUTER_API_KEY is required. Add it to .env before starting the app."
            )
        return self.openrouter_api_key.get_secret_value()

    @property
    def users_db_path(self) -> Path:
        return self.data_dir / "users.db"

    def resolved_auth_mode(self) -> str:
        mode = str(self.auth_mode or "auto").lower()
        if mode == "auto":
            return "google" if self.google_client_id else "off"
        return mode if mode in {"off", "google"} else "off"

    def ensure_directories(self) -> None:
        for path in (self.raw_dir, self.kb_dir, self.audit_dir, self.model_cache_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
