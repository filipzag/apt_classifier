# Vestige (package `apt_classifier`): project commands

- Create/update environment: `make setup`
- Start local ChromaDB: `make up`
- Download ATT&CK and build vector index: `make ingest`
- Run the web app: `make api` (FastAPI on 8100; ChromaDB owns 8000) plus `make web-dev` (Vite on 5173) for development, or `make run-next` (build + serve from one process); `make api-local` for no-LLM mode; `make web-check` (tsc + oxlint) before committing web changes
- Run Streamlit (legacy UI, preserved on branch `streamlit-stable`): `make run`
- Tests: `make test`
- Lint: `make lint`
- Validate Compose without the daemon: `docker compose config --quiet`
- Held-out public technique-mapping gate (primary): `make benchmark-public` (writes `data/audits/public_heldout_current.json`; per-case rows, bootstrap CIs, with/without cues)
- Regression check against the committed baseline: `make check-regression`
- Curated retrieval benchmark (development diagnostic): `make benchmark-curated` or `PYTHONPATH=. .venv/bin/python scripts/curated_retrieval_benchmark.py --modes legacy_dense hybrid lexical --ablate-cues`
- OTRF benchmark: `PYTHONPATH=. .venv/bin/python scripts/otrf_benchmark.py`
- Ensemble benchmark: `PYTHONPATH=. .venv/bin/python scripts/ensemble_benchmark.py --coverage 0.4 --seed 7`
- Exemplar index and acceptance model: `make exemplars` (embeds the public train sentences, a few minutes) then `make fit-acceptance` (public dev only, about 25 minutes). The acceptance model stores the exemplar index fingerprint and is disabled (fail closed) when the index is missing or different, so rebuild both together.
- Synthetic technique sentences: `make generate-synthetic` (OpenRouter, ~$2 for ~575 techniques; resumable), then `make ingest`; ablate with `TECHNIQUE_INCLUDE_SYNTHETIC=false`.
- CTI-HAL benchmarks (local evaluation only, no licence published): `scripts/prepare_cti_hal.py --source <clone>` then `make benchmark-cti-hal` (statement-level mapping + real-report attribution over 55 reports / 6 groups; the attribution headline `attribution_cti_hal_current.json` replaces every ATT&CK group name and alias in the statements with "the adversary" via `--anonymize-actors`, and `attribution_cti_hal_as_reported.json` keeps the text as reported, docs 21.3.4b); `make benchmark-cti-hal-production` runs the same reports through the technique verifier (spends OpenRouter credit, never the adoption gate). Only the LLM zero-shot vote was removed (2026-09-30, at the product owner's request); the LLM extraction, the technique verifier, the independent AI review and CyberRE-LLM all remain.
- Elastic Security Labs real-report benchmark (local evaluation only, post text (c) Elastic): `make prepare-elastic` (downloads the RSS feed once into `data/benchmarks/elastic/`; `--download` refreshes) then `make benchmark-elastic` (8 Elastic-attributed reports / 6 groups plus a 22-post open set; headline `attribution_elastic_current.json` at the strict anonymization level of `scripts/report_anonymizer.py`; `_names`, `_tradecraft`, `_as_reported` beside it, docs 21.3.4c). Labels and rules are pre-registered in `data/elastic_security_labs/labels.json`: never tune on this set. `make benchmark-elastic-production` spends OpenRouter credit (~$15-20).
- Random-forest group and source-country models: `make fit-forest` (ATT&CK knowledge base only; trains on the GPU when available, about 40 s, CPU fallback via `PROFILE_FOREST_DEVICE=cpu`; writes `data/models/forest/profile_forest.npz`). Changing the forest or its weights changes the combiner's inputs, so refit the combiner afterwards.
- PyTorch group network: `make fit-mlp` (ATT&CK knowledge base only, GPU when available, one network by default (`members` in `DEFAULT_PARAMS`), about 35 s to fit on the GPU; input = accepted techniques + named software; writes `data/models/mlp/group_mlp.npz` and `data/audits/group_mlp_eval.json` with top-1/top-5/MRR/macro-F1 on ATT&CK campaigns (development, used for tuning) and the CTI-HAL reports (test), next to the group forest). Ensemble member `group_mlp` (learned_profile family) behind `GROUP_MLP_WEIGHT`, default 0 (not loaded, fusion and combiner fingerprint unchanged); at 1.0 it needs a combiner fitted with it (docs 21.3.9) and failed the naming gate on 2026-10-01 (one wrong name on the ATT&CK folds).
- CTI-HAL train/validation/test for the group network: `PYTHONPATH=. .venv/bin/python scripts/cti_hal_split_eval.py` (5 rotating group-stratified parts, ~20 min GPU; writes `data/benchmarks/cti_hal/splits.json` and `data/audits/group_mlp_cti_hal_splits.json`, compares against the closed-set shortcut; a model trained on CTI-HAL is local evaluation only and is never saved).
- Learned attribution combiner: `make fit-combiner` (ATT&CK folds only, ~1 h local, no credit; writes `data/models/combiner/attribution_combiner_fsum.json` + `data/audits/attribution_combiner_cv.json`), then rerun both attribution benchmarks (`make benchmark-attribution`, `make benchmark-cti-hal`). The combiner is on by default; the benchmark refuses to run or write when it is enabled but did not load (a fingerprinted source or weight changed) unless `--allow-inactive-combiner`
- Top-three reorder replay: `PYTHONPATH=. .venv/bin/python scripts/top3_replay.py <attribution artifacts> --margin 1.25` (offline, seconds; the pre-registered rule failed on fresh seeds 41/53 and is not wired, docs 10.10/21.3.13). Design new top-three rules on fresh fold seeds (`scripts/attribution_benchmark.py --seed <n> --output <file>`; without `--output` it overwrites `attribution_heldout_current.json`), never on seed 7 or CTI-HAL alone.
- Verification artifact: `make verify-results` reruns `benchmark-public`, `benchmark-curated` and `benchmark-attribution`, then composes with the pytest/ruff/compose/index gates (`scripts/collect_verification.py --gates`); `scripts/collect_verification.py` alone composes the existing artifacts. Never hand-edit `verification_results_current.json`

The project must use its `.venv`; system transformers v5 is incompatible with ATTACK-BERT's `.bin` checkpoint. Tests mock OpenRouter and do not require an API key. The Streamlit application runs in local mode without `OPENROUTER_API_KEY` (no LLM steps). Benchmarks and `tests/test_app_flow.py` require a running ChromaDB with ingested ATT&CK data (the flow test skips otherwise).

## Key modules

- `apt_classifier/service.py` — `CaseService`: headless engine + case store + every analyst operation (incl. `distinguish`, the counterfactual separation replay); both front ends must go through it.
- `apt_classifier/combiner.py` — learned listwise combiner over the shortlist (features only, fail-closed artifact); `scripts/fit_attribution_combiner.py` fits it.
- `apt_classifier/top3_view.py` — the analyst's Top-three view (display only; state banner, pair contrasts, ATT&CK-documented origin/motivation); `apt_classifier/top3.py` — the measured, not adopted, exclusive-evidence promotion rule it uses for the "evidence pulls both ways" state.
- `apt_classifier/separation.py` — candidate-relative discrimination (`discriminate`, contrastive `soft` channel, `distinguishing_techniques`) and the re-ranking blend; the naming gate lives in `attribution.py`.
- `apt_classifier/report.py` — shared Markdown/JSON/Navigator/Attack Flow export builders.
- `api/` — FastAPI app (`app.py` routes and error mapping, `serializers.py` state → JSON, `auth.py` Google sign-in / sessions / user admin, `main.py` ASGI entry); `apt_classifier/users.py` — SQLite user store and allowlist policy. Tests run with sign-in off; `tests/test_api_auth.py` fakes the Google exchange. `web/` — React 19 + Vite + Tailwind v4 SPA (`src/api/types.ts` mirrors the serializers).
- `app/ui/` — Streamlit UI package: `runtime` (per-session engine, analysis state, autosave), `cases` (on-disk case store), `guard` (`run_step` error boundary), `components` (status strip, stage glossary, frames), `sidebar`, `local_extract` (no-LLM sentence split) and `pages/` (intake, technique_review, interview, assessment, canvas, diamond, cyberre, intel). Pages expose `render(engine, settings, weights)`.
- `apt_classifier/attack/parse.py` — ATT&CK STIX parser (techniques, groups, software, campaigns, detection strategies, analytics, data components).
- `apt_classifier/attack/kb.py` — Knowledge base data model with defensive objects.
- `apt_classifier/telemetry.py` — Telemetry-aware collection guidance and negative-evidence validation.
- `apt_classifier/platforms.py` — Target OS: each observation's platform scope (log source, OS named in its text, the case Target OS in `ledger.metadata["target_platforms"]`) and the rule that drops techniques documented only for other operating systems (docs 5.7).
- `apt_classifier/evidence/quality.py` — Admiralty-style evidence quality, corroboration, and documented specificity.
- `apt_classifier/attribution.py` — Open-set attribution stages and calibrated rejection.
- `apt_classifier/attack_flow.py` — Attack Flow construction and STIX 2.1 export.
- `apt_classifier/activity_clusters.py` — Opt-in local longitudinal activity clustering (SQLite).
- `apt_classifier/navigator.py` — ATT&CK Navigator layer export.
- `apt_classifier/reranker.py` — Optional local cross-encoder technique reranker.
- `apt_classifier/profile_forest.py` — Random-forest group and source-country models (profiles, synthetic training cases, fail-closed `.npz` artifact); `apt_classifier/torch_forest.py` — the PyTorch forest trainer/predictor they run on (GPU when available); ensemble members in `apt_classifier/classifiers/profile_forest.py`.
- `apt_classifier/group_mlp.py` — PyTorch MLP (one network by default; `members` averages several) over pre-extracted techniques + named software → group (synthetic report-like cases incl. the group's tools, prominence prior, fail-closed `.npz`).
- `apt_classifier/exemplars.py` — Exemplar index: labelled public report sentences (CC-BY-4.0, train splits only) whose nearest-neighbour votes are acceptance-model features.

Always brief me on the changes you make.

## Evaluation notes

- Leak-resistant EvidenceForge evaluator: `make evidenceforge-eval` (or `scripts/evidenceforge_eval.py --pipeline local|production|both --mode narrative|windows|ecar|all`). Never add `GROUND_TRUTH.activity` or scenario prose to model input. The branch-office scenario is development data; use a separately generated scenario for generalization claims.
- Technique vector index carries a schema version (`TECHNIQUE_SCHEMA_VERSION` in `apt_classifier/vectorstore.py`); bumping it or changing fields forces a rebuild on `make ingest`, and retrieval fails closed on an incompatible index.
- Curated benchmark reports `accepted_set` metrics (authoritative AUTO_ACCEPTED mappings) separately from visible top-k ranking.
- Current verification artifact: `data/audits/verification_results_current.json`, composed by `scripts/collect_verification.py` from `public_heldout_current.json`, `curated_hybrid_current.json`, the three `evidenceforge_*_local.json` files, `evidenceforge_clinic_production.json`, `attribution_heldout_current.json`, `cti_hal_mapping_current.json`, `attribution_cti_hal_current.json` (headline), `attribution_cti_hal_as_reported.json` and `attribution_cti_hal_production.json`. Superseded artifacts live in `data/audits/superseded/`.
- Held-out policy: the public gate uses the `tram`, `expert` and `derived_procedures` test splits. The `procedures` split is leaky (about 22% of its texts are ATT&CK procedure sentences that are embedded in the index) and is refused unless `--allow-leaky`. Dev splits are for calibration only.
- EvidenceForge bundles: `branch-office-example` and `heldout-manufacturing` are development data (both were used to tune signals/gating); `heldout-clinic` is the only untouched bundle and the only one that supports generalization claims.
- Regex cue tables (`_EXPLICIT_TECHNIQUE_CUES`, `_BENIGN_INDICATORS`, `technique_signals.SIGNALS`) were written against the curated and branch-office cases. Report benchmark numbers with and without cues (`--ablate-cues`); never add a cue to fix a single benchmark case.
