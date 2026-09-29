PYTHON := .venv/bin/python
PIP := .venv/bin/pip
EVIDENCEFORGE_DIR ?= ../evidenceforge_output
EVIDENCEFORGE_SCENARIO ?= ../EvidenceForge/scenarios/branch-office-example/scenario.yaml
EVIDENCEFORGE_MFG_DIR ?= ../evidenceforge_heldout
EVIDENCEFORGE_CLINIC_DIR ?= ../evidenceforge_clinic
PUBLIC_BASELINE ?= data/audits/public_heldout_baseline.json
PUBLIC_CURRENT ?= data/audits/public_heldout_current.json

.PHONY: setup up down ingest run run-local test lint evidenceforge-eval evidenceforge-eval-all benchmark-public benchmark-curated verify-results check-regression fit-acceptance benchmark-attribution train-reranker generate-synthetic benchmark-cti-hal benchmark-cti-hal-production benchmark-attribution-production

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
	$(PYTHON) -m ruff check apt_classifier app tests scripts
	$(PYTHON) -m ruff format --check apt_classifier app tests scripts

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
fit-acceptance:
	PYTHONPATH=. $(PYTHON) scripts/fit_acceptance_model.py

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
# attribution.
CTI_HAL_ROOT ?= data/benchmarks/cti_hal
benchmark-cti-hal:
	PYTHONPATH=. $(PYTHON) scripts/benchmark_mapping.py --root $(CTI_HAL_ROOT) --datasets cti_hal --output data/audits/cti_hal_mapping_current.json
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --cti-hal $(CTI_HAL_ROOT)/reports.jsonl --output data/audits/attribution_cti_hal_current.json

# Production pipeline (OpenRouter verifier on ambiguous candidates + bounded zero-shot vote);
# spends credit and records calls/errors/cost per report. Never used as the adoption gate.
benchmark-cti-hal-production:
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --pipeline production --cti-hal $(CTI_HAL_ROOT)/reports.jsonl --output data/audits/attribution_cti_hal_production.json

benchmark-attribution-production:
	PYTHONPATH=. $(PYTHON) scripts/attribution_benchmark.py --pipeline production --output data/audits/attribution_heldout_production.json
