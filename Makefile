PYTHON := .venv/bin/python
PIP := .venv/bin/pip
EVIDENCEFORGE_DIR ?= ../evidenceforge_output
EVIDENCEFORGE_SCENARIO ?= ../EvidenceForge/scenarios/branch-office-example/scenario.yaml
EVIDENCEFORGE_MFG_DIR ?= ../evidenceforge_heldout
EVIDENCEFORGE_CLINIC_DIR ?= ../evidenceforge_clinic
PUBLIC_BASELINE ?= data/audits/public_heldout_baseline.json
PUBLIC_CURRENT ?= data/audits/public_heldout_current.json

.PHONY: llm-pull llm-check exemplars fit-forest fit-mlp fit-text-group fit-combiner fit-confidence-labels api api-dev api-local api-auth-demo web-install web-dev web-build web-check run-next setup up down ingest run run-local test lint evidenceforge-eval evidenceforge-eval-all benchmark-public benchmark-curated verify-results check-regression fit-acceptance benchmark-attribution train-reranker generate-synthetic benchmark-cti-hal benchmark-cti-hal-production benchmark-attribution-production

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

# Same app without any LLM call (blank key and local LLM off override .env): local retrieval, gating,
# ranking and the deterministic interview only.
run-local:
	OPENROUTER_API_KEY= LOCAL_LLM_ENABLED=false PYTHONPATH=. $(PYTHON) -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.headless=true --browser.gatherUsageStats=false

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
	PYTHONPATH=. $(PYTHON) scripts/fit_attribution_combiner.py --seeds 11 23 37 --variants 5 --unknown-all --overwrite-production

# Calibrated confidence labels (display only; offline, seconds): fitted on the design folds
# (seeds 41/53), seed 7 and CTI-HAL scored as report-only numbers. Refit after fit-combiner.
fit-confidence-labels:
	PYTHONPATH=. $(PYTHON) scripts/fit_confidence_labels.py --evaluate data/audits/attribution_heldout_current.json data/audits/attribution_cti_hal_current.json data/audits/attribution_cti_hal_as_reported.json

# Labelled real-report sentences (public train splits) used as technique-mapping evidence;
# needs scripts/prepare_public_benchmarks.py output. Run before fit-acceptance.
exemplars:
	PYTHONPATH=. $(PYTHON) scripts/build_exemplar_index.py

# Shipped configuration: public dev split, no per-technique priors (priors were measured and
# rejected, docs 21.3), exemplar features from `make exemplars`.
# Random-forest group and source-country models from the ATT&CK knowledge base (GPU when available).
fit-forest:
	PYTHONPATH=. $(PYTHON) scripts/fit_profile_forest.py

# PyTorch group network from the ATT&CK knowledge base (GPU when available), with its evaluation
# on ATT&CK campaigns (development) and the CTI-HAL reports (test): data/audits/group_mlp_eval.json.
fit-mlp:
	PYTHONPATH=. $(PYTHON) scripts/fit_group_mlp.py --save --quiet

# Free-text group model (TF-IDF word document per ATT&CK group) from the KB + raw STIX bundle.
fit-text-group:
	PYTHONPATH=. $(PYTHON) scripts/fit_text_group.py

fit-acceptance:
	PYTHONPATH=. $(PYTHON) scripts/fit_acceptance_model.py --priors-split none
	PYTHONPATH=. $(PYTHON) scripts/fit_acceptance_model.py --priors-split none --no-exemplars --output data/models/acceptance/acceptance_model_no_exemplars.json

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

# ----- local LLM (Ollama; LOCAL_LLM_ENABLED=true + local/ model ids, see .env.example) -----
LOCAL_LLM_MODELS ?= qwen3.5:4b
OLLAMA ?= ollama
# pulls the model, then registers qwen3.5:4b-text: the same weights without the vision tower,
# which Ollama otherwise loads onto the GPU (about 740 MiB; docs 16.4a)
llm-pull:
	for model in $(LOCAL_LLM_MODELS); do $(OLLAMA) pull $$model || exit 1; done
	PYTHONPATH=. $(PYTHON) scripts/ollama_text_only.py --ollama $(OLLAMA) --source qwen3.5:4b --name qwen3.5:4b-text

llm-check:
	PYTHONPATH=. $(PYTHON) scripts/local_llm_check.py

# ----- modern front end (FastAPI + React) -----
api:
	PYTHONPATH=. $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8100

api-dev:
	PYTHONPATH=. $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8100 --reload --reload-dir api --reload-dir apt_classifier

# Shows the sign-in page without real Google credentials (the Google redirect itself fails).
api-auth-demo:
	AUTH_MODE=google GOOGLE_CLIENT_ID=demo GOOGLE_CLIENT_SECRET=demo AUTH_SECRET_KEY=demo-only AUTH_ALLOWED_DOMAINS=example.com PYTHONPATH=. $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8100

api-local:
	OPENROUTER_API_KEY= LOCAL_LLM_ENABLED=false PYTHONPATH=. $(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8100

web-install:
	cd web && npm ci

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

web-check:
	cd web && npx tsc -b && npm run lint

run-next: web-build api
