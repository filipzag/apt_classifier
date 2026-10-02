PYTHON := .venv/bin/python
PIP := .venv/bin/pip
EVIDENCEFORGE_DIR ?= ../evidenceforge_output
EVIDENCEFORGE_SCENARIO ?= ../EvidenceForge/scenarios/branch-office-example/scenario.yaml
EVIDENCEFORGE_MFG_DIR ?= ../evidenceforge_heldout
EVIDENCEFORGE_CLINIC_DIR ?= ../evidenceforge_clinic
PUBLIC_BASELINE ?= data/audits/public_heldout_baseline.json
PUBLIC_CURRENT ?= data/audits/public_heldout_current.json

.PHONY: exemplars exemplars-chroma exemplars-chroma-verify exemplar-equivalence fit-forest fit-mlp fit-combiner api api-dev api-local api-auth-demo web-install web-dev web-build web-check run-next setup up down ingest run run-local test lint evidenceforge-eval evidenceforge-eval-all benchmark-public benchmark-curated verify-results check-regression fit-acceptance benchmark-attribution train-reranker generate-synthetic benchmark-cti-hal benchmark-cti-hal-production benchmark-attribution-production prepare-elastic benchmark-elastic benchmark-elastic-production

setup:
	python3 -m venv --system-site-packages .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

up:
	docker compose up -d chroma

down:
	docker compose down

ingest:
	PYTHONPATH=. $(PYTHON) -m apt_classifier.ingest

run:
	PYTHONPATH=. $(PYTHON) -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.headless=true --browser.gatherUsageStats=false

# Same app without any OpenRouter call (blank key overrides .env): local retrieval, gating,
# ranking and the deterministic interview only.
run-local:
	OPENROUTER_API_KEY= PYTHONPATH=. $(PYTHON) -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.headless=true --browser.gatherUsageStats=false

test:
	PYTHONPATH=. $(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check apt_classifier app api tests scripts
	$(PYTHON) -m ruff format --check apt_classifier app api tests scripts

evidenceforge-eval:
	PYTHONPATH=. .venv/bin/python scripts/evidenceforge_eval.py \
		--evidenceforge-dir $(EVIDENCEFORGE_DIR) \
		--scenario $(EVIDENCEFORGE_SCENARIO) \
		--pipeline local --mode all

# Held-out public technique-mapping gate (tram/expert/derived_procedures test splits).
benchmark-public:
	PYTHONPATH=. $(PYTHON) scripts/benchmark_mapping.py --ablate-cues --output $(PUBLIC_CURRENT)

benchmark-curated:
	PYTHONPATH=. $(PYTHON) scripts/curated_retrieval_benchmark.py --modes legacy_dense hybrid lexical --ablate-cues --output data/audits/curated_hybrid_current.json

# Regenerate all three EvidenceForge local artifacts (development, development, held-out).
evidenceforge-eval-all:
	PYTHONPATH=. $(PYTHON) scripts/evidenceforge_eval.py --evidenceforge-dir $(EVIDENCEFORGE_DIR) --scenario $(EVIDENCEFORGE_SCENARIO) --pipeline local --mode all --output-local data/audits/evidenceforge_local_leakfree.json
	PYTHONPATH=. $(PYTHON) scripts/evidenceforge_eval.py --evidenceforge-dir $(EVIDENCEFORGE_MFG_DIR) --scenario data/evidenceforge/heldout-manufacturing/scenario.yaml --pipeline local --mode all --output-local data/audits/evidenceforge_heldout_local.json
	PYTHONPATH=. $(PYTHON) scripts/evidenceforge_eval.py --evidenceforge-dir $(EVIDENCEFORGE_CLINIC_DIR) --scenario data/evidenceforge/heldout-clinic/scenario.yaml --pipeline local --mode all --output-local data/audits/evidenceforge_clinic_local.json

# Regenerate the public + curated artifacts and compose verification_results_current.json.
benchmark-attribution:
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py

verify-results: benchmark-public benchmark-curated benchmark-attribution
	PYTHONPATH=. $(PYTHON) scripts/collect_verification.py --gates

check-regression:
	PYTHONPATH=. $(PYTHON) scripts/check_regression.py --baseline $(PUBLIC_BASELINE) --current $(PUBLIC_CURRENT)

# Fit the calibrated technique-acceptance model on the public dev splits (never test).
# Learned attribution combiner: ATT&CK folds only (training seeds), randomised evidence regimes,
# unknown-actor lists, group-level CV; writes data/models/combiner/attribution_combiner.json.
fit-combiner:
	PYTHONPATH=. $(PYTHON) scripts/fit_attribution_combiner.py --seeds 11 23 37 --variants 5 --unknown-all --overwrite

# Labelled real-report sentences (public train splits) used as technique-mapping evidence;
# needs scripts/prepare_public_benchmarks.py output. Run before fit-acceptance. Re-embedding gives
# a new fingerprint: the acceptance model must be refit (make fit-acceptance) and the Chroma copy
# resynced (make exemplars-chroma writes a new collection named after the new fingerprint).
exemplars:
	PYTHONPATH=. $(PYTHON) scripts/build_exemplar_index.py --overwrite

# Copy the npz library once into a new Chroma collection (technique_exemplars-<fingerprint>-...),
# the only Chroma write of the exemplar migration (ask before running it on the shared server);
# an existing collection is verified, never changed. Serve it with TECHNIQUE_EXEMPLAR_BACKEND.
exemplars-chroma:
	PYTHONPATH=. $(PYTHON) scripts/sync_exemplars_to_chroma.py --overwrite

exemplars-chroma-verify:
	PYTHONPATH=. $(PYTHON) scripts/sync_exemplars_to_chroma.py --verify-only --report data/audits/exemplar_chroma_verify.json --overwrite

# npz against the Chroma copy, levels L1-L4 (read-only; one GPU job to embed the query texts).
exemplar-equivalence:
	PYTHONPATH=. $(PYTHON) scripts/exemplar_backend_equivalence.py \
		--sets self public:dev public:test cti_hal evidenceforge:branch-office evidenceforge:manufacturing \
		--evidenceforge-bundle branch-office $(EVIDENCEFORGE_DIR) $(EVIDENCEFORGE_SCENARIO) \
		--evidenceforge-bundle manufacturing $(EVIDENCEFORGE_MFG_DIR) data/evidenceforge/heldout-manufacturing/scenario.yaml \
		--vector-cache data/audits/exemplar_equivalence_vectors.npz \
		--output data/audits/exemplar_backend_equivalence.json --overwrite

# Shipped configuration: public dev split, no per-technique priors (priors were measured and
# rejected, docs 21.3), exemplar features from `make exemplars`.
# Random-forest group and source-country models from the ATT&CK knowledge base (GPU when available).
fit-forest:
	PYTHONPATH=. $(PYTHON) scripts/fit_profile_forest.py

# PyTorch group network from the ATT&CK knowledge base (GPU when available), with its evaluation
# on ATT&CK campaigns (development) and the CTI-HAL reports (test): data/audits/group_mlp_eval.json.
fit-mlp:
	PYTHONPATH=. $(PYTHON) scripts/fit_group_mlp.py --save --quiet

fit-acceptance:
	PYTHONPATH=. $(PYTHON) scripts/fit_acceptance_model.py --priors-split none --overwrite
	PYTHONPATH=. $(PYTHON) scripts/fit_acceptance_model.py --priors-split none --no-exemplars --output data/models/acceptance/acceptance_model_no_exemplars.json --overwrite

# Train the local cross-encoder reranker on the public train splits (dev splits calibrate the
# promote/demote thresholds). Mined negatives are cached so a retry skips the mining pass.
PUBLIC_NORMALIZED := data/benchmarks/public/normalized
train-reranker:
	PYTHONPATH=. $(PYTHON) scripts/train_technique_reranker.py \
		--train $(PUBLIC_NORMALIZED)/tram_train.jsonl $(PUBLIC_NORMALIZED)/expert_train.jsonl $(PUBLIC_NORMALIZED)/derived_procedures_train.jsonl \
		--dev $(PUBLIC_NORMALIZED)/tram_dev.jsonl $(PUBLIC_NORMALIZED)/expert_dev.jsonl $(PUBLIC_NORMALIZED)/derived_procedures_dev.jsonl \
		--use-retrieval-negatives --epochs 1 --batch-size 8 \
		--pairs-cache data/models/reranker_pairs_cache.json --output data/models/technique-reranker

# LLM-generated observation sentences for sparsely documented techniques (indexed as the
# 'synthetic' field; re-run `make ingest` afterwards). Resumable; about $2 for ~575 techniques.
generate-synthetic:
	PYTHONPATH=. $(PYTHON) scripts/generate_synthetic_sentences.py --workers 4

# CTI-HAL (dessertlab/CTI-HAL, no licence published: local evaluation only). Convert with
# scripts/prepare_cti_hal.py --source <clone>, then run statement-level mapping and real-report
# attribution. The attribution headline replaces every ATT&CK group name and alias in the
# statements with "the adversary", as the ATT&CK folds do (docs 21.3.4b); the run on the
# statements as reported is kept beside it.
CTI_HAL_ROOT ?= data/benchmarks/cti_hal
benchmark-cti-hal:
	PYTHONPATH=. $(PYTHON) scripts/benchmark_mapping.py --root $(CTI_HAL_ROOT) --datasets cti_hal --output data/audits/cti_hal_mapping_current.json
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --cti-hal $(CTI_HAL_ROOT)/reports.jsonl --anonymize-actors --output data/audits/attribution_cti_hal_current.json
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --cti-hal $(CTI_HAL_ROOT)/reports.jsonl --output data/audits/attribution_cti_hal_as_reported.json

# Production pipeline (OpenRouter verifier on ambiguous candidates), anonymized like the headline;
# spends credit and records calls/errors/cost per report. Never used as the adoption gate.
benchmark-cti-hal-production:
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --pipeline production --cti-hal $(CTI_HAL_ROOT)/reports.jsonl --anonymize-actors --output data/audits/attribution_cti_hal_production.json

benchmark-attribution-production:
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --pipeline production --output data/audits/attribution_heldout_production.json

# Elastic Security Labs posts attributed by Elastic to one ATT&CK group, plus an open set of
# posts on actors outside ATT&CK (labels pre-registered in data/elastic_security_labs/labels.json;
# post text (c) Elastic, local evaluation only). prepare-elastic downloads the RSS feed once
# (--download refreshes it). The headline reads every prose sentence at the strict level (the
# report's attribution prose, cluster names, places and sponsors removed; docs 21.3.4c); names,
# tradecraft (ATT&CK malware masked too) and as-reported runs sit beside it. Local, no credit.
ELASTIC_ROOT ?= data/benchmarks/elastic
prepare-elastic:
	PYTHONPATH=. $(PYTHON) scripts/prepare_elastic_reports.py --output $(ELASTIC_ROOT)

benchmark-elastic:
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --report-set elastic --cti-hal $(ELASTIC_ROOT)/reports.jsonl --open-set $(ELASTIC_ROOT)/reports_open_set.jsonl --anonymize strict --output data/audits/attribution_elastic_current.json
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --report-set elastic --cti-hal $(ELASTIC_ROOT)/reports.jsonl --anonymize names --output data/audits/attribution_elastic_names.json
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --report-set elastic --cti-hal $(ELASTIC_ROOT)/reports.jsonl --anonymize tradecraft --output data/audits/attribution_elastic_tradecraft.json
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --report-set elastic --cti-hal $(ELASTIC_ROOT)/reports.jsonl --output data/audits/attribution_elastic_as_reported.json

# Production pipeline (technique verifier) on the strict headline reports. Spends OpenRouter
# credit: about $0.02 and 11 s per sentence, so roughly $15-20 and 2.5 h for the 8 reports.
# Probe first: add --limit 2 --output data/audits/attribution_elastic_production_sample.json.
benchmark-elastic-production:
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --pipeline production --report-set elastic --cti-hal $(ELASTIC_ROOT)/reports.jsonl --anonymize strict --output data/audits/attribution_elastic_production.json

# ----- modern front end (FastAPI + React) -----
api:
	PYTHONPATH=. $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8100

api-dev:
	PYTHONPATH=. $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8100 --reload --reload-dir api --reload-dir apt_classifier

# Shows the sign-in page without real Google credentials (the Google redirect itself fails).
api-auth-demo:
	AUTH_MODE=google GOOGLE_CLIENT_ID=demo GOOGLE_CLIENT_SECRET=demo AUTH_SECRET_KEY=demo-only AUTH_ALLOWED_DOMAINS=example.com PYTHONPATH=. $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8100

api-local:
	OPENROUTER_API_KEY= PYTHONPATH=. $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8100

web-install:
	cd web && npm ci

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

web-check:
	cd web && npx tsc -b && npm run lint

run-next: web-build api
