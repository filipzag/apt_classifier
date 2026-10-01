# Vestige

**Attribution from the traces left behind.** Vestige maps incident observations to MITRE
ATT&CK, ranks documented threat groups, and names one only when the evidence separates it from
every alternative. (Python package: `apt_classifier`.)

A local, evidence-driven Streamlit application for ranking threat-group hypotheses from mixed incident evidence. It accepts analyst observations, EDR/SIEM alerts, Windows/Sysmon JSON, IOCs, and named tools.

The application does **not** prove attribution. It ranks groups whose documented MITRE ATT&CK tradecraft overlaps the supplied evidence. Shared tools, incomplete reporting, and false-flag activity can invalidate the result.

See [`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md) for the detailed architecture, data flow, algorithms, operations, security boundaries, and limitations.

## Pipeline

1. Normalize prose and alerts into atomic attack-action sentences with `openai/gpt-5.6-sol` through OpenRouter.
2. Parse supported Windows/Sysmon events deterministically without sending bulk logs to an LLM.
3. Embed observations with `basel/ATTACK-BERT` and retrieve sentence-level ATT&CK techniques plus actor-specific behavior sentences from local ChromaDB. With a Target OS set for the case (Windows, Linux, macOS, ESXi), techniques documented only for other operating systems are not mapped (docs 5.7).
4. Capture optional strategic context: target geography, industry, motivation, intended objective, and independently supported sponsor/origin.
5. Rank groups with deterministic classifiers plus the analyst-requested CyberRE-LLM vote (the LLM zero-shot vote was removed; the LLM extraction, technique verifier and independent review in steps 1 and 7 remain):
   - Bernoulli Naive Bayes
   - TF-IDF cosine
   - weighted Jaccard
   - actor-specific behavior similarity
   - semantic technique-profile centroid
   - group-description similarity
   - strategic-context profile matching
   - motivation alignment
   - campaign-profile overlap
   - software/tool links
   - source-country random forest: predicts the source-country class (China, Russia, Iran, North Korea,
     another state, no documented state origin) from the mapped techniques, motivation and target, and votes
     for each group through the group's documented country (default since 2026-10-01: real reports improve, the
     ATT&CK folds lose one case; docs 9.13, 21.3.11). Trained on the GPU from ATT&CK profiles.
   - group random forest over the same features (wired at weight 0: it helped the ATT&CK folds but hurt real reports)
   - opt-in CyberRE-LLM technique filtering and ranking
6. Fuse applicable classifiers with family-aware Reciprocal Rank Fusion. Naive Bayes, TF-IDF, Jaccard, and the technique centroid share one technique-overlap budget instead of casting four full correlated votes. Actor-specific behavior, software, and campaign classifiers have their own family budgets. Context, motivation, and description signals may rerank only candidates with documented technical support; the analyst-requested CyberRE vote has its own low family budget and cannot introduce a candidate. Abstaining classifiers cast no vote. Groups with zero technical overlap are excluded from authoritative RRF.
7. Aggregate documented sponsor/origin evidence across the ranked candidates and classify the profile as APT-like state/espionage activity, financially motivated criminal intrusion, mixed, or indeterminate.
8. Fetch a cached, non-voting live-intelligence layer from CISA KEV, CISA Alerts & Advisories, and explicitly labeled ransomware.live extortion claims; match only explicit case overlaps and add them to the Canvas.
9. Run a hypothesis-driven chat interview that predicts unobserved behaviors from the current ensemble candidates, recommends ATT&CK-informed telemetry, Windows channels and Event IDs to inspect, and automatically asks the next prioritized question after each analyst response.
10. Optionally search API-key-scoped Elasticsearch security data through Elastic Agent Builder MCP for the current lead; normalize results locally, stage them for analyst approval, and reanalyze only accepted events.
11. Run isolated `anthropic/claude-sonnet-5` reviews: blinded ACH, named ACH, fame-bias measurement, devil's advocate, then advisory reconciliation. The deterministic RRF ranking remains authoritative; review-proposed rank changes are recorded but not applied.

### Technique mapping pipeline

Technique extraction uses hybrid dense (ATTACK-BERT) plus lexical (BM25-style) retrieval with behavior-focused candidate generation:

- Dense and lexical results are fused at the technique level using RRF, preserving the best supporting sentence and field for each candidate.
- Generic or non-mappable observations can abstain rather than forcing up to six labels.
- Parent/sub-technique hierarchy is handled explicitly and non-duplicatively: each evidence claim contributes at most once per group. Observed sub-techniques can support groups that document only the parent via ancestor-compatible matching, but this is discounted and never double-counted.
- Bounded OpenRouter verification checks ambiguous candidates against the normalized observation and source excerpt. The verifier receives only technique definitions and retrieval evidence—no group names, attribution ranking, or strategic context. Supported candidates are auto-accepted; unsupported are auto-rejected; uncertain cases go to analyst review.
- Analyst Technique Review allows confirming, rejecting, or replacing mappings before rescoring. Rejection removes only the mapping; it does not create negative incident evidence.

### Mapping presets

Five presets bundle retrieval, verification, hierarchy, and gating options:

- **Balanced (default):** hybrid retrieval, OpenRouter verification on ambiguous candidates, ancestor-compatible hierarchy matching, strict positive group eligibility.
- **Conservative:** stronger acceptance requirements, exact-only hierarchy matching, verification on all candidates.
- **Exploratory:** wider candidate pool; unsupported candidates remain non-voting.
- **Local only:** hybrid retrieval without OpenRouter; uncertain candidates go to review.
- **Legacy:** original dense-only ATTACK-BERT retrieval for comparison and rollback.

Actor-specific behavior matching requires distinctive shared operational details and a clear semantic margin over competing group descriptions. Generic observations such as scheduled-task creation or LSASS access make this classifier abstain rather than cast an actor vote.

Country and APT-like assessments are derived from public ATT&CK sponsor/origin descriptions for technically supported candidates. They never interpret victim geography or source IP addresses as attacker origin and return unresolved/low confidence when attribution coverage or candidate separation is weak.

After analysis, the application opens an analyst Canvas linking normalized evidence to ATT&CK techniques, actor-specific behavior, tools, campaigns, strategic context, candidate groups, country hypotheses, intrusion type, and matching live-intelligence records. The CyberRE-LLM tab provides an analyst-triggered full ATT&CK technique filter/ranker for a selected observation; generated CyberRE output is a bounded, non-authoritative RRF vote and does not replace deterministic retrieval. The Diamond Model tab separately organizes the intrusion around adversary, capability, infrastructure, and victim vertices, with explicit status and provenance for each node. Canvas layers and connection types are filterable, and the hypothesis inspector exposes support, contradictions, unmatched observations, and classifier ranks.

CISA KEV and CISA Alerts & Advisories are labeled authoritative for what CISA reports. ransomware.live records are public extortion-site claims, not confirmed breaches; they require independent victim, regulator, law-enforcement, or press corroboration. Live intelligence is excluded from RRF and the isolated reviewer by default to prevent recency and reporting bias.

When no named group meets the evidence bar (stages below *known group hypothesis*), the Assessment
view says so explicitly and labels the leader as the closest documented group, not an attribution;
on the attribution benchmarks every named known group was correct (33 of 33 on the ATT&CK folds; 8 of 8
on the CTI-HAL real reports with the group names removed from the text, 13 of 13 with them left in) and
all 25 actors removed from ATT&CK were correctly left unnamed. When the leader has documented overlap with another
group (ATT&CK cross-references or rarely shared tooling, for example Carbanak and FIN7), the view says
so, because the evidence usually cannot separate them.

The LLM zero-shot opinion was removed from the ensemble on 2026-09-30 to keep ranking deterministic and cheaper; it had changed the analyst-facing rank in one of ten measured reports. CyberRE-LLM output has its own bounded family and re-orders only the ranking the analyst sees: the stage decision (and so the naming gate) and the independent reviewer read a fusion computed without it (docs 9.11, 9.12, 16.2).

### Optional local reranker

A local reranker adapter (`apt_classifier/reranker.py`) can re-score the hybrid retrieval candidate pool with a sentence-transformers cross-encoder trained by `make train-reranker` (ATTACK-BERT base, public train splits, retrieval-mined negatives, dev-calibrated promote/demote band). It only moves candidates it is confident about to the front, can promote or demote acceptance within the calibrated band, never bypasses the benign/sparse guards or cue-corroborated mappings, and never introduces techniques. It is **disabled by default**: the first trained model lowered held-out F1 (0.32 -> 0.26) and clinic hit@1 (0.22 -> 0.06); see `docs/TECHNICAL_DOCUMENTATION.md` 21.3.6 for the evaluation and the fixes it triggered. Enable a model only after it passes the held-out gate, with `RERANKER_ENABLED=true`, `RERANKER_MODEL=<dir>` and the manifest pins (`RERANKER_EXPECTED_*`).

### Benchmarks

Benchmark scripts under `scripts/` measure technique mapping; there is no labelled
attribution-accuracy benchmark yet. Results are composed into
`data/audits/verification_results_current.json` by `make verify-results` and are never
typed by hand.

- `scripts/benchmark_mapping.py` (`make benchmark-public`) — **primary gate.** About 1,400
  held-out public cases (`tram`, `expert`, `derived_procedures` test splits of
  `tumeteor/mitre-ttp-mapping`), per-case rows, bootstrap confidence intervals, and a
  with/without-cues ablation. The leaky `procedures` split is refused by default.
  `make check-regression` compares a fresh run against `data/audits/public_heldout_baseline.json`.
- `scripts/curated_retrieval_benchmark.py` — 36 curated observations (30 positive incl.
  multi-label and hierarchy, 6 hard negatives). Development diagnostic: the regex cue tables
  were written against it, so numbers are reported with and without cues.
- `scripts/otrf_benchmark.py` — 9 OTRF-style Sysmon log narratives (every expected technique
  has a dedicated cue).
- `scripts/evidenceforge_eval.py` (`make evidenceforge-eval-all`) — three synthetic
  EvidenceForge bundles; only `heldout-clinic` is untouched by tuning. See
  `docs/EVIDENCEFORGE_EVALUATION.md`.
- `scripts/attribution_benchmark.py` (`make benchmark-attribution`; `--pipeline production` for the technique verifier, spends credit) — attribution gate: held-out,
  anonymized procedure sentences per group run through the real pipeline (text in, group out),
  with a per-fold leak filter and an unknown-actor arm for open-set metrics. Self-consistency over
  the ATT&CK answer key, NOT real-world attribution accuracy.
- `scripts/ensemble_benchmark.py` — legacy technique-ID self-consistency diagnostic.
- `scripts/prepare_cti_hal.py` + `make benchmark-cti-hal` — CTI-HAL (dessertlab): 968 human-annotated
  statements from 55 real reports on six named groups, used for statement-level mapping and the only
  real-report attribution measurement (local evaluation only; the corpus publishes no licence).
  The attribution headline first replaces every ATT&CK group name and alias in the statements with
  "the adversary", as the ATT&CK folds do (`--anonymize-actors`); the run on the statements as reported
  is kept beside it (docs 21.3.4b).

#### Current results (artifacts regenerated 2026-09-22, local pipeline, no LLM verifier)

**Public held-out (hybrid):**

| Split | n | Micro F1 [95% CI] | P | R | hit@1 | hit@3 | hit@10 |
|---|---|---|---|---|---|---|---|
| tram (baseline) | 713 | 0.30 [0.26, 0.33] | 0.36 | 0.25 | 0.33 | 0.50 | 0.67 |
| tram (current) | 713 | 0.43 [0.39, 0.46] | 0.54 | 0.35 | 0.43 | 0.58 | 0.75 |
| expert (baseline) | 157 | 0.13 [0.10, 0.16] | 0.42 | 0.08 | 0.30 | 0.54 | 0.74 |
| expert (current) | 157 | 0.18 [0.13, 0.22] | 0.70 | 0.10 | 0.39 | 0.59 | 0.83 |
| derived_procedures (baseline) | 521 | 0.18 [0.15, 0.22] | 0.26 | 0.14 | 0.21 | 0.33 | 0.55 |
| derived_procedures (current) | 521 | 0.26 [0.22, 0.29] | 0.39 | 0.19 | 0.24 | 0.43 | 0.67 |
| **pooled (baseline, 2026-09-22 morning)** | 1391 | 0.22 [0.21, 0.25] | 0.33 | 0.17 | 0.28 | 0.44 | 0.63 |
| **pooled (current, rule gate only)** | 1391 | 0.30 [0.27, 0.32] | 0.41 | 0.23 | 0.35 | 0.53 | 0.73 |
| **pooled (current, acceptance model)** | 1391 | 0.32 [0.30, 0.34] | 0.50 | 0.24 | 0.35 | 0.53 | 0.73 |
| **pooled (current, + ranking by acceptance probability)** | 1391 | 0.32 [0.30, 0.34] | 0.50 | 0.24 | **0.40** | **0.58** | **0.75** |
| opt-in priors model (not shipped, see docs 21.3.6) | 1391 | 0.36 [0.34, 0.39] | 0.51 | 0.28 | 0.44 | 0.64 | 0.76 |
| **pooled (current, + labelled-sentence exemplars, 2026-09-30)** | 1391 | **0.42 [0.39, 0.44]** | 0.52 | 0.35 | **0.50** | 0.67 | **0.80** |

Baseline = before the acceptance-gate work; current = after (see docs section 21.3.6). The last row
probability-ranking row was the shipped default from 2026-09-28; since 2026-09-30 the acceptance model also
reads what the nearest labelled public report sentences say about each candidate (exemplars, docs 8.2 and
21.3.10). Part of that public gain is within-collection, because the exemplars are the same datasets' train
splits; the CTI-HAL gain below is the cleaner estimate. The per-technique priors model named a wrong group on
a real report, so it is kept as an opt-in artifact only (docs 8.2, 21.3.4b).

**Curated retrieval (30 positive + 6 hard-negative cases):**

| Mode | Cues | Top-1 | Top-3 | Top-10 | Accepted-set P / R / F1 | Hard-negative accepted-FP cases |
|---|---|---|---|---|---|---|
| **hybrid (default)** | on | 25/30 | 29/30 | 30/30 | 0.79 / 0.77 / 0.78 | 0/6 |
| hybrid | off | 15/30 | 24/30 | 29/30 | 0.52 / 0.43 / 0.47 | 0/6 |
| legacy_dense | on | 26/30 | 29/30 | 30/30 | 0.85 / 0.83 / 0.84 | 0/6 |
| lexical | on | 26/30 | 26/30 | 26/30 | 1.00 / 0.63 / 0.77 | 0/6 |

**OTRF Sysmon (9 cases):** 9/9 top-1 with cues.

**EvidenceForge held-out clinic bundle (narrative, 21 cases):** micro-F1 0.36 (was 0.23 before the
exemplars), hit@1 0.33 (was 0.22); all three benign red herrings are now accepted (was two), and on the
development bundles narrated telemetry loses some ranking, so structured-telemetry deployments can set
`TECHNIQUE_EXEMPLARS_ENABLED=false` (docs 21.3.10). With the OpenRouter verifier enabled (production pipeline,
measured before the exemplars) narrative micro-F1 was 0.53, recall 0.56.

**Attribution (text in, group out; 153 held-out groups):** top-1 0.83 (was 0.38 before the fusion work and
0.60 before the learned combiner; 0.84 before the source-country vote, which costs one fold here and gains on real
reports), top-5 0.96, MRR 0.88. With the separation gate 32 known groups are named, all correct, and 0/25 removed
actors are named (was 4): named risk 0.00 (was 0.56) at 18% coverage.

**CTI-HAL real reports (55 reports, 6 groups; local evaluation only):** statement mapping F1 0.38 (was 0.32 before the
exemplars), recall 0.34 (was 0.25), hit@1 0.38 (was 0.33); median accepted-technique recall per report 0.42 (was 0.31).
Attribution, with every ATT&CK group name and alias in the statements replaced by "the adversary" as in the ATT&CK
folds: **top-1 0.46** (0.44 before the source-country vote), cluster-aware top-1 0.58, top-5 0.64, MRR 0.53; 10 reports
reached a named hypothesis, all correct (8 before). With the statements as reported (27 of the 55 reports name their own
group in them) the same pipeline gives top-1 0.56 (was 0.36 before the tool cap, decisive-malware boost, discriminative
re-ranking, combiner, exemplars and country vote), cluster-aware top-1 0.71, top-5 0.73, MRR 0.64, and 15 named, all
correct (was 3 under the maturity gate; at the
previous bar of 0.78 the gate named 7 and 3 were wrong, which is why the bar was raised); every earlier before/after
comparison in this README and the docs was measured that way. Four of the six top-1 answers lost to the replacement
are malware evidence removed with the name (Carbanak is both a group and its malware; a diagnostic run keeping such
aliases gives top-1 0.51), so attribution from tradecraft alone lies between 0.44 and 0.51 (docs 21.3.4b).
Ordinary-word software aliases (ATT&CK lists "Page" for Elise) now match only in their exact spelling;
before that fix the exemplar change named Lotus Blossom on a FIN6 report (docs 5.5). With the production pipeline as measured on 2026-09-28 (verifier plus the since-removed LLM vote; 10-report sample, $3.37) technique
recall on the statements doubles (0.23 -> 0.48, precision 0.49 -> 0.40) while ranking and naming stay the
same. This is the only attribution number measured on real report text. Refitting the
acceptance model with the CTI-HAL statements did not help (public F1 0.320 -> 0.313) and was not adopted.

**Ensemble self-consistency (153 groups, 40% coverage):** 147/153 (96%) top-1 without
campaign voting, MRR 0.977.

These are engineering diagnostics, not production accuracy claims. See
`docs/TECHNICAL_DOCUMENTATION.md` section 21.3 for definitions and caveats.

## Requirements

- Python 3.11–3.13
- Docker with permission to access the Docker daemon
- NVIDIA GPU optional; CPU works but ingestion is slower
- OpenRouter API key with access to the configured extraction and review models

## Setup

```bash
make setup
cp .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env` (optional: without it the app runs in **local mode**, with
sentence-split extraction and no verifier, CyberRE or AI review), then start Chroma and
build the index:

```bash
make up
make ingest
make run
```

Open the URL printed by Streamlit. Initial ingestion downloads approximately 54 MB of ATT&CK data and the ATTACK-BERT model.

To run the UI in a container next to Chroma instead of from the `.venv`:

```bash
docker compose up --build
```

The `app` service reads `.env`, keeps its data in the `app_data` volume and listens on
`127.0.0.1:8501`. Ingest once from the host (`make ingest` with `CHROMA_HOST=localhost`) or inside
the container. There is no authentication; put it behind a reverse proxy with SSO before exposing it.

### Using the app

1. **Intake**: paste the incident narrative (or load the sample), optionally pick the Target OS
   and add kill-chain notes, alerts, indicators, log files and strategic context, then
   **Analyse evidence**. The case is saved automatically under `data/cases/` and can be reopened
   from the sidebar.
2. **Technique review**: one table of every candidate mapping with its gate reason and evidence.
   Confirm, reject or replace, then apply; the ranking is recomputed. Changing the Target OS here
   maps the observations again.
3. **Interview**: answer predicted-behaviour questions (yes / no / unknown / not visible) to
   separate the shortlisted groups; optional Elasticsearch gathering per lead.
4. **Assessment**: the stage decision with its bars and glossary, the ranking chart, a dossier
   per candidate, the optional independent AI review, and exports (Markdown, JSON audit,
   Navigator layers, Attack Flow) that are always available.

Every page shows the case status strip and a **Next** hint. Errors from OpenRouter (credit, key,
rate limit) or Chroma are reported in plain language and never lose the case state.

`transformers==4.57.5` is intentionally pinned below v5 because ATTACK-BERT currently publishes a `.bin` checkpoint that transformers v5 cannot load reliably.

### Learned attribution combiner

The shortlist is ordered by a learned combiner (`apt_classifier/combiner.py`): a conditional-logit
model over per-classifier ranks, the discrimination channels and the case's evidence regime,
blended half-and-half with reciprocal-rank fusion. It is fitted only on ATT&CK-derived folds
(`make fit-combiner`, no credit, about an hour) and loads fail-closed on an ATT&CK-version or
pipeline mismatch. On the held-out ATT&CK folds it lifts top-1 from 0.60 to 0.83 and on the 55
CTI-HAL real reports (statements as reported) from 0.51 to 0.53 with every named group still correct; the naming gate scores
the leader against the union of both shortlists so a re-ordering cannot hide a competitor. The
assessment page shows its calibrated leader probability. Disable with
`ATTRIBUTION_COMBINER_ENABLED=false`.

### Web front end (FastAPI + React)

The primary interface is a React single-page app served by a FastAPI backend; the Streamlit
app remains available (`make run`, and unchanged on the `streamlit-stable` branch). The workflow
is Evidence → Technique review → Distinguish (which techniques separate the shortlist, and which
pending mappings or interview leads would sharpen or challenge the leader) → Interview →
Assessment, plus the canvas, diamond, live-intelligence and AI-review views.

```bash
make web-install   # once: npm ci inside web/
make api           # API on http://127.0.0.1:8100 (loads the embedder and Chroma at startup)
make web-dev       # Vite dev server on http://127.0.0.1:5173, /api proxied to 8100
make run-next      # production: build web/dist and serve API + app from port 8100
```

`make api-local` runs the same API without any OpenRouter call (local mode: sentence split, no
verifier, no AI review). The API surface is documented at `/docs` (OpenAPI) when the server
runs; every payload is produced by `api/serializers.py` from an `AnalysisState`, and every
analyst operation (intake, technique decisions, interview, LLM steps, live intel, exports)
goes through `apt_classifier/service.py`, the same headless service any other front end would
use. Cases are saved automatically under `data/cases/` after every change, including the verifier's verdicts, so reopening a case replays them instead of calling the model again. Docker builds the
web bundle in a Node stage and serves it from the Python image (`docker compose up`, port 8100).

### Users and Google sign-in

The web app can run as a shared team workspace with Google sign-in (OpenID Connect):

1. In the Google Cloud console create an OAuth client ID of type *Web application* and add the
   authorised redirect URI(s): `http://127.0.0.1:5173/auth/callback` for the Vite dev server,
   `http://127.0.0.1:8100/auth/callback` when the API serves the built app, and your production
   URL (`https://<host>/auth/callback`, with `AUTH_PUBLIC_URL=https://<host>`).
2. Set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `AUTH_SECRET_KEY` and an allowlist
   (`AUTH_ALLOWED_DOMAINS` and/or `AUTH_ALLOWED_EMAILS`) in `.env`; without an allowlist the
   server refuses to start in google mode. `AUTH_ADMIN_EMAILS` names administrators; the first
   user to sign in is an administrator as well.
3. Restart the API. Every `/api` route now needs a session; the app shows a sign-in page.

Users live in `data/users.db` (SQLite): address, name, picture, role (`admin` or `analyst`),
active flag, sign-in times. Administrators manage users on the *Users* page and can delete any
case; analysts see every case in the workspace, create cases attributed to them, and can delete
only their own. Without Google credentials (`AUTH_MODE=auto` or `off`) the app runs as one
local analyst with administrator rights, which is also how the tests run. The legacy Streamlit
UI has no sign-in and should only be run locally.

### Optional Elastic Agent Builder MCP

The supported connector targets Elastic Agent Builder MCP on Elasticsearch 9.2+/9.3+ or Elastic Serverless, not the deprecated standalone Elasticsearch MCP server. Create an API key restricted to intended security indices with `read` and `view_index_metadata`, plus Kibana `feature_agentBuilder.read` access, then configure:

```text
ELASTIC_MCP_URL=https://your-kibana.example/api/agent_builder/mcp
ELASTIC_MCP_API_KEY=<scoped API key>
ELASTIC_MCP_AUTO_GATHER=false
```

For a custom Kibana space, use `https://your-kibana.example/s/<space>/api/agent_builder/mcp`. The connector only calls `platform.core.search`, sends no explicit index so the API key defines the scope, caps returned records, and requires analyst selection before imported events can affect attribution.

## Supported Windows events

- Sysmon: 1, 3, 7, 8, 10, 11, 12, 13, 14, 22, 23, 25
- Windows Security/System: 4624, 4625, 4688, 4698, 4720, 7045, 1102

Inputs may be a JSON array, JSON Lines, an Elastic `hits` export, or EVTX-exported nested JSON.

## Configuration

All settings are environment variables documented in `.env.example`. Important values:

- `EMBEDDING_MODEL=basel/ATTACK-BERT` (pinned by `EMBEDDING_MODEL_REVISION`)
- `EXTRACT_MODEL=openai/gpt-5.6-sol`
- `REVIEW_MODEL=anthropic/claude-sonnet-5`
- `SIMILARITY_THRESHOLD=0.45`
- `BEHAVIOR_SIMILARITY_THRESHOLD=0.55`
- `THREAT_INTEL_DAYS=30`
- `THREAT_INTEL_TIMEOUT=20.0`
- `THREAT_INTEL_MAX_ITEMS_PER_SOURCE=100`
- `INCLUDE_RANSOMWARE_CLAIMS=true`
- `ELASTIC_MCP_URL=` (optional Elastic Agent Builder MCP endpoint)
- `ELASTIC_MCP_API_KEY=` (optional, scoped Pydantic secret)
- `ELASTIC_MCP_LOOKBACK_HOURS=72`
- `ELASTIC_MCP_MAX_RESULTS=20`
- `ELASTIC_MCP_AUTO_GATHER=false`
- `RRF_K=20` (group ensemble; `TECHNIQUE_RRF_K=60` for technique-level fusion)
- `TECHNIQUE_FAMILY_WEIGHT=2.00`
- `CONTEXT_FAMILY_WEIGHT=1.00`
- `SPECIFIC_BEHAVIOR_WEIGHT=1.50`
- `CAMPAIGN_WEIGHT=0.00` (available for inspection, disabled from fusion by default)
- `GROUP_DESCRIPTION_WEIGHT=0.40`
- `SEMANTIC_CENTROID_WEIGHT=0.70`
- `STRATEGIC_CONTEXT_WEIGHT=0.70`
- `SOFTWARE_WEIGHT=1.20` (software weighted by rarity across groups; `SOFTWARE_IDF_ENABLED=true`; ATT&CK *tool*
  entries capped at `SOFTWARE_TOOL_WEIGHT_CAP=1.5`; a named malware family documented for at most two groups
  multiplies the software vote by `1 + SOFTWARE_DECISIVE_BOOST` (2.0), see docs 9.10)
- `ATTRIBUTION_GATE_MODE=separation` (named hypotheses require candidate-relative separation >= 0.5 and
  discriminative evidence >= 3.0 among the top-5, chosen by `scripts/attribution_gate_sweep.py` with zero wrong names
  on both benchmarks; `ATTRIBUTION_RERANK_BLEND=0.5` blends the same evidence into the ranking; docs 10.7)
- `ATTRIBUTION_HYPOTHESIS_MATURITY=0.82` (the previous maturity/margin gate, `ATTRIBUTION_GATE_MODE=maturity`; see docs 10.6)
- `TECHNIQUE_INCLUDE_SYNTHETIC=false` (LLM-generated observation sentences per technique in the index;
  `make generate-synthetic` then `make ingest`; off until the full set is generated and measured, see
  docs 7.3 and 21.3.6)
- `SOURCE_COUNTRY_WEIGHT=1.0`, `PROFILE_FOREST_WEIGHT=0.0` (source-country vote on, group forest silent; `make fit-forest`
  trains both on the GPU when available, `PROFILE_FOREST_DEVICE=cpu` otherwise). To turn the country vote off set
  `SOURCE_COUNTRY_WEIGHT=0.0` and `ATTRIBUTION_COMBINER_PATH=data/models/combiner/attribution_combiner_no_country.json`
  (docs 9.13 and 21.3.11)
- `TECHNIQUE_EXEMPLARS_ENABLED=true` (labelled public report sentences as acceptance evidence, `make exemplars`
  then `make fit-acceptance`; CTI-HAL median accepted-technique recall per report 0.31 -> 0.42; `false` restores
  the previous calibrated gate through `ACCEPTANCE_MODEL_FALLBACK_PATH`, see docs 8.2 and 21.3.10)
- `TECHNIQUE_PLATFORM_FROM_TEXT=true` (an observation that names an operating system, "on the Linux server",
  is mapped only to techniques documented for it, overriding the case's Target OS for that observation;
  neutral on every benchmark, see docs 5.7 and 21.3.12). The Target OS itself is set per case: pick every
  operating system the activity touched, leave it empty when unknown (a wrong one hides the right techniques)
- `TECHNIQUE_RANK_BY_ACCEPTANCE=true` (order each observation's candidates by the acceptance model's
  probability; held-out hit@1 0.35 -> 0.40 with the accepted set unchanged, see docs 8.2)
- `TECHNIQUE_CORROBORATION_ACCEPT_OBSERVATIONS=2`, `TECHNIQUE_CORROBORATION_ACCEPT_MIN_PROBABILITY=0.1`
  (accept a technique left in review when two different observations of the case retrieved it; real-report
  accepted recall 0.26 -> 0.31 at flat precision, see docs 8.2)
- `TECHNIQUE_SEGMENT_ACCEPTANCE=false`, `TECHNIQUE_HIERARCHY_PARENT_MODE=off`,
  `TECHNIQUE_CORROBORATION_BONUS=0.0` (optional per-sentence acceptance for long paragraphs,
  hierarchy-consistent acceptance, cross-observation corroboration; all measured on the held-out
  benchmarks and off by default, see docs 21.3.6)
- `NEGATIVE_EVIDENCE_WEIGHT=0.30`

The OpenRouter key is loaded as a Pydantic secret, is never written to an audit artifact, and is never sent to Chroma.

## Verification

```bash
make test
make lint
docker compose config --quiet
```

The LLM boundary is mocked in tests, so tests run without network access or API spend. Review prompts/responses from real cases are persisted under `data/audits/` for auditability and are ignored by Git.

## Data and privacy

- ATT&CK vectors and application state live locally in Dockerized ChromaDB.
- Raw logs are parsed locally.
- Elastic MCP searches and result normalization run locally between this application and Kibana; the Elastic API key is never sent to OpenRouter.
- Elastic MCP results remain staged and non-voting until an analyst selects them for import. Imported event text may later enter an OpenRouter review payload if the analyst explicitly runs AI review.
- Free-text observations, alert text, adaptive-question context, and review payloads are sent to OpenRouter when those features are used.
- Do not submit evidence to OpenRouter if organizational policy prohibits third-party processing.
