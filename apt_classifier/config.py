from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    # Local LLM (Ollama, docs 16.4a). Model ids prefixed ``local/`` in the per-task settings
    # below run on it; without an OpenRouter key every other model id falls back to
    # local_llm_fallback_model. On by default (product owner, 2026-10-02): it is used only
    # when the server answers with the fallback model pulled; with an OpenRouter key nothing
    # changes unless a model id starts with ``local/``.
    local_llm_enabled: bool = True
    local_llm_url: str = "http://localhost:11434"
    # The 4B fits the 8 GB laptop GPU next to the app and a benchmark; ``-text`` is the same
    # model without its vision tower (``make llm-pull`` builds it; 740 MiB less GPU memory).
    local_llm_fallback_model: str = "local/qwen3.5:4b-text"
    local_llm_num_ctx: int = 32768
    # GPU only: every layer must sit on the GPU, or the model refuses to load (num_gpu=999)
    # instead of spilling layers onto the CPU, which slows the whole machine down.
    local_llm_gpu_only: bool = True
    # CPU threads the local model may use; with every layer on the GPU they only sample and
    # wait, and a power-capped laptop GPU otherwise leaves them spinning on several cores.
    local_llm_num_thread: int = 2
    # A local verifier's 'supported' with P(supported) below this becomes 'uncertain' (left for
    # the analyst); 0 keeps every verdict. OpenRouter verdicts carry no probability (docs 16.4a).
    local_llm_verifier_min_confidence: float = 0.98
    # Idle time before Ollama frees the GPU memory again (the app's own GPU models need it).
    local_llm_keep_alive: str = "5m"
    local_llm_timeout: float = 300.0
    # Efforts that switch Qwen's thinking on, and the extra token budget it gets for it.
    local_llm_think_efforts: str = "high,xhigh,max"
    local_llm_think_tokens: int = 8192
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
    # An observation that names an operating system ("on the Linux server") is mapped only to
    # techniques documented for it, overriding the case's Target OS for that observation
    # (apt_classifier/platforms.py). The case Target OS itself is set per case by the analyst.
    # Default since 2026-10-01: neutral on every benchmark (public held-out F1 0.416 -> 0.417,
    # attribution unchanged, docs 21.3.12), which rarely describe non-Windows hosts.
    technique_platform_from_text: bool = True
    # Single-answer cap (2026-10-19, prereg cribl_single_answer_cap_2026-10-19): "model" follows
    # the acceptance artifact's own gate record (at most one probability-only acceptance per
    # observation, a second one above the artifact's bar; artifacts without a record run
    # uncapped), "off" ignores it, a number in (0, 1] forces that bar for an ablation (1.0:
    # strictly one acceptance on probability per observation).
    technique_single_answer_cap: str = "model"
    # Agreement filter (2026-10-19, prereg cribl_head_agreement): "technique" / "family" /
    # "top3" keep an accepted technique only when the technique head picks it too (its top
    # technique, its top family, one of its top-3 families); "off" (default) no filter.
    technique_head_agreement: str = "off"
    # Head-first acceptance (2026-10-20, prereg cribl_head_first): "model" follows the acceptance
    # artifact's gate record (a candidate the head gives at least that joint probability is
    # accepted outright; artifacts without the record change nothing), "off" ignores it, a
    # number forces a bar for an ablation.
    technique_head_first: str = "model"
    # Sibling replacement (2026-10-20, prereg cribl_sibling_replacement): "model" follows the
    # artifact's gate record (the head's preferred member of a family replaces an accepted
    # sibling when it prefers it by the recorded ratio), "off" ignores it, a number forces it.
    technique_head_sibling: str = "model"
    acceptance_model_path: Path = (
        PROJECT_ROOT / "data" / "models" / "acceptance" / "acceptance_model.json"
    )
    # Same fit without exemplar features; used only when the primary model cannot run (exemplars
    # disabled or the index missing), so the calibrated gate survives. `make fit-acceptance`.
    acceptance_model_fallback_path: Path | None = (
        PROJECT_ROOT / "data" / "models" / "acceptance" / "acceptance_model_no_exemplars.json"
    )
    # ----- the fine-tuned technique head (apt_classifier/technique_head.py, docs 21.3.18) -----
    # A local service (`make technique-head-serve`, scripts/cribl_finetune/serve.py, its own
    # .venv-cribl and about 3.5 GB of GPU memory) scoring every observation with the 4B technique
    # model: evidence on every candidate (acceptance feature schema 4) and up to
    # TECHNIQUE_HEAD_NEW_CANDIDATES proposals retrieval missed. On by default since 2026-10-20
    # (product owner; the mapping gate of cribl_product_integration_2026-10-19 passed and the
    # cluster volume guard settled the open-set naming): it is used only while the service
    # answers with the pinned checkpoint, otherwise the pipeline runs exactly as without it.
    # Prose only, for now: log-derived observations (the telemetry sources of cyberre.
    # is_telemetry_source) are not scored. Fail closed: unavailable, wrong ATT&CK version or a
    # fingerprint other than the pinned one leaves the pipeline exactly as without it; an
    # acceptance model fitted with the head runs only on that head.
    technique_head_enabled: bool = True
    technique_head_url: str = "http://localhost:8102"
    technique_head_timeout: float = 120.0
    # SHA-256 of the served checkpoint (its /health "fingerprint"); pinned to E2 step_52706
    # (the registered head); empty accepts any checkpoint
    technique_head_expected_fingerprint: str = (
        "5b5d015d8766dd27cdeceaf0ea1aaa0ea7c613f397d2bf0fecb07eea41fcea01"
    )
    technique_head_top_families: int = 3
    technique_head_new_candidates: int = 3
    technique_head_batch: int = 16
    technique_head_cache_size: int = 2048
    # The acceptance model fitted with the head's features (schema 4; `fit_acceptance_model.py
    # --require-technique-head`). Read only while the head is active: it then judges the
    # observations the head scored and ACCEPTANCE_MODEL_PATH the rest (acceptance.
    # HeadAwareAcceptance); with the service stopped the shipped gate runs unchanged.
    acceptance_model_head_path: Path | None = (
        PROJECT_ROOT / "data" / "models" / "acceptance" / "acceptance_model_technique_head.json"
    )
    max_questions: int = 10
    # Interview questions about malware only one top-three group uses (software_questions.py,
    # docs 21.3.15 C): asked first while the case is not named, with fixed wording and the answers
    # found / not found / not checked. A "found" counts like the software named in the text; a
    # "not found" can only lower the leader's (or a named cluster's) separation, by this weight
    # times the malware's rarity and exclusivity. Off until the pre-registered simulation passes.
    interview_software_questions: bool = False
    interview_software_per_group: int = 1
    interview_software_max_questions: int = 3
    interview_software_max_users: int = 1
    interview_software_when_named: bool = False
    interview_software_negative_weight: float = 0.3
    # a name must not rest on a "found" alone: without the interview software the leader's (or
    # the cluster's) separation in the gate window must be >= 0
    interview_software_found_requires_lead: bool = True
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
    # Dense technique hits within this similarity of an observation's best hit become candidates
    # (_best_technique_hits), and the number of sentence hits fetched per query (0 keeps
    # max(3 x retrieval_top_k, 4 x technique_candidate_pool), 90 at the defaults). Both feed the
    # acceptance features; its retrieval contract refuses values the model was not fitted with.
    technique_dense_relative_margin: float = 0.12
    technique_dense_overfetch: int = 0
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
    # Random-forest group and source-country models (apt_classifier/profile_forest.py, fitted
    # from the ATT&CK knowledge base on the GPU when available: make fit-forest). They vote with
    # the weights below; 0 keeps them visible in the classifier matrix without moving the ranking.
    profile_forest_enabled: bool = True
    profile_forest_path: Path = PROJECT_ROOT / "data" / "models" / "forest" / "profile_forest.npz"
    profile_forest_device: str = "auto"
    profile_forest_weight: float = 0.0
    profile_forest_family_weight: float = 1.0
    # Default since 2026-10-01 by the product owner's decision (docs 21.3.11): real reports improve
    # (anonymized top-1 0.436 -> 0.455, 8 -> 10 names) at the cost of one ATT&CK fold (0.837 -> 0.830).
    # It needs the combiner fitted with it (data/models/combiner/attribution_combiner_0930_fsum.json);
    # to turn it off use 0.0 with attribution_combiner_no_country_fsum.json. The group forest
    # stays at 0: it hurts
    # real reports at every weight tried.
    source_country_weight: float = 1.0
    source_country_family_weight: float = 1.0
    # PyTorch group network (apt_classifier/group_mlp.py, make fit-mlp): accepted techniques +
    # named software -> group. It joins the ensemble, in the learned_profile family, only with a
    # positive weight; at 0 it is not loaded and the fusion and combiner fingerprint are unchanged.
    # Measured 2026-10-01 at 1.0 with its own combiner (data/models/combiner/
    # attribution_combiner_group_mlp.json, on earlier classifier code; refit 2026-10-03 as
    # attribution_combiner_group_mlp_fsum.json, docs 21.3.9): CTI-HAL anonymized top-1 0.455 -> 0.473, as reported
    # 0.564 -> 0.564, no wrong names; but on the ATT&CK folds top-1 0.830 -> 0.817 and one wrong
    # name (Silence named as Kimsuky), so it fails the naming gate and stays at 0.
    group_mlp_enabled: bool = True
    group_mlp_path: Path = PROJECT_ROOT / "data" / "models" / "mlp" / "group_mlp.npz"
    group_mlp_device: str = "auto"
    group_mlp_weight: float = 0.0
    # Free-text group model (apt_classifier/text_group.py, make fit-text-group): a TF-IDF word
    # document per ATT&CK group scored against the case text. Like the group network it votes, in
    # the learned_profile family, only with a positive weight; at 0 it is not loaded and the fusion
    # and combiner fingerprint are unchanged. It never counts as an independent evidence family.
    # Gate trial 2026-10-03 at 1.0 with its own combiner (data/models/combiner/
    # attribution_combiner_text_group.json, docs 21.3.16): CTI-HAL anonymized top-1 0.455 -> 0.491,
    # MRR 0.512 -> 0.572, as reported MRR 0.636 -> 0.666, Elastic strict 4 better / 0 worse, no
    # wrong names anywhere; but ATT&CK folds top-1 0.830 -> 0.824 (5 gained / 6 lost), so the
    # pre-registered no-regression rule fails by one case and it stays at 0 (the product owner decides).
    text_group_enabled: bool = True
    text_group_path: Path = PROJECT_ROOT / "data" / "models" / "text_group" / "text_group.json.gz"
    text_group_weight: float = 0.0
    # Behaviour-text member (apt_classifier/behavior_text.py, prereg
    # attribution_behavior_text_2026-10-20): the text member's corpus indexed sentence by
    # sentence, every case statement voting for the groups of its nearest procedure sentences.
    # Reads TEXT_GROUP_PATH; joins the ensemble only with a positive weight (0 until the gate
    # passes); a learned profile that never counts as an independent evidence family. The
    # parameters are the dev choice of scripts/fit_behavior_text.py (vendor set).
    behavior_text_enabled: bool = True
    behavior_text_weight: float = 0.0
    # dev choice 2026-10-21 (vendor set, member alone): top_k 10, the five best statements, no
    # prior: MRR 0.809, top-1 0.676, top-3 0.912 (mean over all statements: top-1 0.706, MRR 0.784)
    behavior_text_top_k: int = 10
    behavior_text_min_similarity: float = 0.0
    behavior_text_aggregate: str = "top5"
    behavior_text_prior: float = 0.0
    # Technique-conditioned procedure member (apt_classifier/procedure_match.py, prereg
    # attribution_procedure_match_2026-10-21): each accepted technique's statements against the
    # procedure sentences documented under that technique (the group's own, its software's, its
    # campaigns'), with ATT&CK's software-to-group links and a bonus for software the report
    # names. Reads TEXT_GROUP_PATH; joins the ensemble only with a positive weight (0 until the
    # gate passes); never an independent evidence family. The parameters are the dev choice of
    # scripts/fit_procedure_match.py (vendor set + ATT&CK campaigns).
    procedure_match_enabled: bool = True
    procedure_match_weight: float = 0.0
    # dev choice 2026-10-21 (vendor set + campaigns, member alone, each report's own statements
    # hidden from the corpus, 64 variants): floor 0.30, exact technique, no software-sentence
    # route, the named-software link on, mean over the accepted techniques: MRR 0.820, top-1 0.782
    # (vendor 0.794 / 0.845, campaigns 0.762 / 0.779); the text routes alone reach MRR 0.64, the
    # software sentences add nothing on top of the link
    procedure_match_min_similarity: float = 0.30
    procedure_match_family: str = "exact"
    procedure_match_software_route: bool = False
    procedure_match_seen_bonus: float = 0.0
    procedure_match_link_weight: float = 1.0
    procedure_match_aggregate: str = "mean"
    procedure_match_prior: float = 0.0
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
    # Learned combiner (apt_classifier/combiner.py): orders the shortlist by a conditional-logit
    # model over per-classifier ranks, discrimination channels and evidence-regime features,
    # blended half-and-half with the RRF order; the naming gate then scores the leader against
    # the union of both shortlists. Adopted 2026-09-30 (docs 21.3.9): folds top-1 0.601 -> 0.830,
    # CTI-HAL 0.509 -> 0.527, every named group correct on both sets. Since 2026-10-04 the same
    # weights, stamped for the hardened classifiers (identical rankings); the 2026-10-03 refit
    # (attribution_combiner_fsum.json) lost 5 of 153 leak-free folds and is not used. Fail-closed.
    attribution_combiner_enabled: bool = True
    # the 2026-09-30 weights stamped for the exactly rounded classifier sums (docs 21.3.9); the
    # earlier artifacts stay next to it for checkouts with the previous classifier code
    attribution_combiner_path: Path = (
        PROJECT_ROOT / "data" / "models" / "combiner" / "attribution_combiner_0930_fsum.json"
    )
    attribution_combiner_top_n: int = 20
    attribution_combiner_blend: float = 0.5
    # Documented-overlap clusters (overlap_cluster.py, docs 10.11 / 21.3.15): name the gate
    # leader together with its ATT&CK-documented twin when the merged pair clears the separation
    # bar, this mass (3.0 for a nesting pair) and a combined combiner probability. The full-scope
    # rule passed on fresh seeds 43/47; the product owner narrowed it to the "documented" scope
    # below, which passed its own pre-registered confirmation on the unused seeds 59/61 on
    # 2026-10-02 (docs 21.3.15 A′: no wrong cluster, single names unchanged, replay exact).
    attribution_cluster_naming: bool = True
    attribution_cluster_min_probability: float = 0.8
    attribution_cluster_min_discriminative: float = 3.5
    # Which ATT&CK links make a twin: "documented" (ATT&CK documents the pair as the same activity,
    # reviewed in attack/same_activity.py, or the profiles nest; product-owner decision 2026-10-02:
    # groups that only share rare malware or that ATT&CK calls distinct are different actors),
    # "full" (the pre-registered rule: any relation, shared rare malware too), "nested".
    attribution_cluster_scope: Literal["documented", "full", "nested"] = "documented"
    # Evidence-volume guard for the cluster step (2026-10-19, prereg cribl_cluster_volume_guard,
    # docs 10.11 / 21.3.18): a cluster is held at shortlist on more accepted techniques than this
    # unless decisive software supports the leader. On by default since the registered checks
    # passed (inert on the folds, CTI-HAL and the vendor set; removes the two wrongly named
    # commodity posts of the Elastic open set); 0 switches it off.
    attribution_cluster_max_techniques: int = 30
    # Calibrated confidence labels on the Top-three view (apt_classifier/confidence_labels.py,
    # scripts/fit_confidence_labels.py): display only, never a gate, pinned to the combiner
    # artifact above. Adopted 2026-10-02 after the pre-registered check on fresh seeds 43/47
    # (docs 21.3.15: "strong" wrong 4.2% pooled, calibration error 0.041). Fails closed (no
    # labels) when the artifact is missing or was fitted against a different combiner.
    confidence_labels_enabled: bool = True
    # the 2026-10-02 labels re-pinned to attribution_combiner_0930_fsum.json (same weights, same
    # rankings); confidence_labels.json stays for checkouts with the earlier combiner
    confidence_labels_path: Path = (
        PROJECT_ROOT / "data" / "models" / "calibration" / "confidence_labels_fsum.json"
    )
    # Activity clustering (opt-in)
    activity_clusters_enabled: bool = False
    activity_clusters_db_path: Path = PROJECT_ROOT / "data" / "activity_clusters.db"
    activity_clusters_min_similarity: float = 0.3
    data_dir: Path = PROJECT_ROOT / "data"
    model_cache_dir: Path = PROJECT_ROOT / ".huggingface"

    # ----- exemplars: labelled real-report sentences as mapping evidence -----
    # Built by `make exemplars` from the public *train* splits (tram, expert,
    # derived_procedures; CC-BY-4.0); the acceptance model is fitted with them on dev.
    technique_exemplars_enabled: bool = True
    technique_exemplar_index_path: Path = (
        PROJECT_ROOT / "data" / "models" / "exemplars" / "exemplar_index.npz"
    )
    technique_exemplar_k: int = 64
    technique_exemplar_tau: float = 0.05
    # an exemplar at or above this cosine to the query is treated as a copy and never votes
    technique_exemplar_duplicate_similarity: float = 0.95
    technique_exemplar_new_candidates: int = 3
    # How a neighbour's weight is shared among its labels: "none" (each label gets the full
    # weight, the shipped behaviour) or "label_split" (divided by the neighbour's label count).
    technique_exemplar_vote_norm: str = "none"
    # "off" queries the exemplars once per observation (the shipped behaviour). "max" / "mean"
    # also query each sentence of a long (segmented) observation and pool the votes per
    # technique over the whole text and its sentences. Tuning knobs (docs 21.3.15): the
    # acceptance model refuses to run on values it was not fitted with.
    technique_exemplar_segment_votes: str = "off"
    # Where the library is read from: "npz" (the artifact above), "chroma" (the collection
    # below, written once by scripts/sync_exemplars_to_chroma.py; unusable -> no exemplars) or
    # "auto" (the collection, else the npz with a visible reason). Chroma re-normalises vectors
    # on insert, so the Chroma library is its own library with its own fingerprint (recomputed
    # from the vectors it returns): an acceptance model fitted on the npz does not run on it
    # (apt_classifier/exemplar_store.py).
    technique_exemplar_backend: str = "npz"
    # The Chroma library is accepted only when every vector norm is within this of 1 and, when
    # the npz it was copied from is present, every component within this of the npz's.
    technique_exemplar_tolerance: float = 1e-6
    # The fingerprint the served library must have (empty: not pinned). For a Chroma library it
    # is the one recomputed from the vectors Chroma returns (the sync report's
    # library_fingerprint; T1.env pins it): any changed vector is refused, with or without the
    # npz. Without the npz and without a pin the Chroma library is only norm-checked and its
    # status reads "active (unverified: ...)". Checked on both backends.
    technique_exemplar_library_fingerprint: str = ""
    # empty: technique_exemplars-<npz fingerprint>-s1-m<M>-efc<ef_construction>
    technique_exemplar_collection: str = ""
    # "exact" searches the verified local copy; "hnsw" also asks Chroma for the candidates and
    # counts a recall miss whenever they lack part of the exact window (the votes stay exact)
    technique_exemplar_search: str = "exact"
    technique_exemplar_hnsw_m: int = 32
    technique_exemplar_hnsw_ef_construction: int = 400
    # the collection must be searched with at least this ef (checked at load)
    technique_exemplar_hnsw_ef_search: int = 8192
    # Chroma results per query as a multiple of the exact window (2k+16)
    technique_exemplar_overfetch: float = 2.0
    # ATT&CK collections. A name other than the default is a read-only copy made by
    # scripts/sync_attack_collection_copy.py; it is opened with get_collection and refused unless
    # its recorded source (metadata, id and count) still matches the default collection.
    technique_collection_name: str = "attack_techniques"
    group_collection_name: str = "attack_groups"

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
