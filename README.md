# Vestige

**Attribution from the traces left behind.** Vestige maps incident evidence to MITRE ATT&CK, ranks the
documented threat groups that fit it, and names one only when the evidence separates it from every
alternative. It is a local web application (React on FastAPI; Python package `apt_classifier`) that accepts
report prose, analyst notes, EDR/SIEM alerts, Windows/Sysmon logs, IOCs and named tools.

Vestige does **not** prove attribution. It ranks groups whose documented ATT&CK tradecraft matches the
evidence. Shared tools, incomplete reporting and false flags can make the answer wrong.

The [technical documentation](docs/TECHNICAL_DOCUMENTATION.md) covers the architecture, algorithms,
operations, security boundaries and every measurement in detail.

## Strengths and limits

- **Technique mapping on real report text** is its strongest part: accepted-set F1 0.50 on 968 human-annotated
  statements from 55 real reports (CTI-HAL).
- **It names little, and has been right when it names.** On the latest gate it named 115 held-out ATT&CK groups
  and 12 of 55 real reports with no wrong name, and none of the 22 posts about actors outside ATT&CK.
- **It ranks the right group first on about half of real reports** (27 of 55 on CTI-HAL, 4 of 8 on Elastic).
- **It only knows the groups ATT&CK documents.** An actor outside ATT&CK cannot be named, and an actor
  removed from ATT&CK is occasionally named after a group that shares its malware.
- **A frontier model ranks better on new reports** (GPT-5.6 Sol 24 of 44 against Vestige's 19, mostly from
  knowledge outside ATT&CK) but names many more groups wrongly.
- **It can run without sending case data anywhere**: with no OpenRouter key it uses a local LLM and the local
  technique head on your GPU.

The benchmarks are small, so these are indications, not production accuracy (see [Results](#results)).

## Quick start

Requirements: Python 3.11–3.13, Docker, Node.js 22 with npm. An NVIDIA GPU is optional for the core app but
required for the optional local LLM (about 4 GB) and technique head (about 3.5 GB). An OpenRouter API key is
optional.

```bash
make setup
cp .env.example .env      # add OPENROUTER_API_KEY here if you have one
make up                   # local ChromaDB on port 8000
make ingest               # downloads ATT&CK (~54 MB) and ATTACK-BERT, builds the index
make web-install
make run-next             # builds the web app and serves it with the API
```

Open http://127.0.0.1:8100. Without an OpenRouter key the app uses the [local LLM](#local-llm-ollama) if
Ollama runs with the model, and otherwise runs in **local mode** (sentence-split extraction, no verifier,
CyberRE or AI review). `make api-local` forces local mode.

To run the app in a container next to ChromaDB instead: `docker compose up --build` (it reads `.env`, keeps
its data in the `app_data` volume and listens on `127.0.0.1:8100`; ingest once from the host with
`CHROMA_HOST=localhost make ingest`, or inside the container). Without [Google sign-in](#google-sign-in) there
is no authentication, so configure it or put the app behind an SSO proxy before exposing it.

## Using the app

1. **Evidence**: paste the narrative (or load the sample), optionally pick the Target OS and add alerts,
   indicators, log files and strategic context, then **Analyse evidence**. Cases are saved automatically under
   `data/cases/` after every change, including the verifier's verdicts, so reopening a case replays them.
2. **Technique review**: every candidate mapping with its gate reason and evidence; confirm, reject or
   replace, and the ranking is recomputed. Rejecting a mapping removes it; it does not count as evidence that
   the technique did not happen.
3. **Distinguish**: the first three candidates side by side (evidence only one has, documented overlaps,
   ATT&CK's stated origin and motivation), which techniques separate the shortlist, and which pending mappings
   or interview leads would sharpen or challenge the leader.
4. **Interview**: answer predicted-behaviour questions (yes / no / unknown / not visible) to separate the
   shortlist, with suggested telemetry, Windows channels and Event IDs, and optional Elasticsearch gathering.
5. **Assessment**: the stage decision with its bars, the ranking, a dossier per candidate, the optional
   independent AI review, and exports (Markdown, JSON audit, Navigator layers, Attack Flow). With
   `MODEL_OPINION_ENABLED=true` a button asks GPT-5.6 Sol for its own answer (one paid request, about $0.05)
   and shows it beside Vestige's; it never changes the ranking, the name, the case file or an export.

The Canvas, Diamond Model, live-intelligence and CyberRE views sit beside these steps. Every page shows the
case status and a **Next** hint; OpenRouter or ChromaDB errors are reported in plain language and never lose
the case. The legacy Streamlit UI (`make run`, kept on the `streamlit-stable` branch) has no sign-in and
should only be run locally.

## How it works

1. **Evidence in.** Prose is split into atomic attack-action sentences (by `openai/gpt-5.6-sol` with an
   OpenRouter key, the local LLM, or a plain sentence split). Supported Windows/Sysmon events are parsed
   locally and never sent to an LLM.
2. **Technique mapping.** Each observation is matched to ATT&CK Enterprise 20.0 techniques by dense
   (ATTACK-BERT) plus lexical (BM25) retrieval. The optional technique head, a fine-tuned 4B model, scores the
   candidates and may add ones retrieval missed. An acceptance model decides which mappings count, using
   nearest labelled public report sentences as evidence; the OpenRouter verifier, when used, sees only
   technique definitions and the evidence, never group names. Generic observations can map to nothing, and
   the case Target OS keeps techniques of other operating systems out.
3. **Group ranking.** The core is procedure text: the case's statements are compared with how ATT&CK's
   procedure examples describe each group's actions, by three members (a free-text group model, a
   behaviour-text member, and procedure text per accepted technique). Supporting members read the accepted
   technique list, named software (weighted by rarity) and the strategic context. Procedure text separates
   groups far better than technique-id lists and keeps doing so when malware names are removed (docs 21.3.24).
4. **Learned combiner.** A model fitted on ATT&CK folds and 68 development real reports orders the shortlist
   and gives the leader a calibrated probability. CTI-HAL and Elastic are never used to fit it.
5. **Naming gate.** A single group is named only when it is clearly separated from the alternatives, its
   probability is at least 0.20 and above "none of these", and the members that all read the technique list
   count as one source of evidence. Otherwise the result is a pair ATT&CK documents as the same activity (for
   example FIN7 and Carbanak), or no name, with the leader shown as the closest documented group, not an
   attribution.
6. **Around the answer.** Country and intrusion-type assessments come only from ATT&CK's documented sponsors
   of supported candidates, never from victim geography or IP addresses. Live intelligence (CISA KEV, CISA
   advisories, and ransomware.live extortion claims labelled as unverified) is matched to the case but never
   votes. The AI review (`anthropic/claude-sonnet-5`) runs blinded and named analyses of competing hypotheses
   and is advisory only.

Docs sections 8–10 hold the full member list, fusion rules and gate bars; 21.3.20 and 21.3.25 the
procedure-text members; 21.3.22 and 21.3.27 the naming rules.

## Results

Defaults of 2026-10-27 (ATT&CK 20.0, local pipeline with the technique head, no LLM verifier):

| Measure | Result | Docs |
|---|---|---|
| Technique mapping, public held-out test splits (1,391 cases), accepted-set micro F1 | 0.463 (0.417 without the technique head) | 21.3.18 |
| Technique mapping, CTI-HAL statements (968), accepted-set F1 / precision | 0.504 / 0.553 (0.378 / 0.425 without the head) | 21.3.18 |
| Attribution, ATT&CK folds (seeds 109/113/127, 465 held-out groups): top-1 | 421/465 (0.905); 115 named, 0 wrong; 2 removed actors named | 21.3.25 |
| Attribution, CTI-HAL (55 real reports, group names removed): top-1 / named | 27/55 (0.49) / 12, 0 wrong (11 after the naming rules of 2026-10-27) | 21.3.25, 21.3.27 |
| Same, malware names removed too | 20/55 (11 before procedure text became the core) | 21.3.25 |
| Elastic Security Labs (8 reports, strict) / open set (22 posts) | 4/8, 1 named, correct / 0 named | 21.3.25 |
| CTI-TAA / AthenaBench attribution, top-ranked correct | 0.54 (ChatGPT-4 0.52) / 0.23 (GPT-5 0.39) | 21.3.26 |
| 44 vendor reports newer than the model's cut-off, right group first: strict / behaviour only | Vestige 19 / 10; GPT-5.6 Sol 24 / 18 | 21.3.28, 21.3.30 |

On the post-cut-off reports the frontier model named 32 known reports with 8 wrong and 19 open-set posts;
Vestige named 8 with 1 wrong and 2 open-set posts. The ATT&CK folds measure self-consistency against
ATT&CK's own answer key, not real-world accuracy, and every real-report set is small. Earlier stages of the
work and their before/after comparisons are in docs 21.3.

## Optional services

### Local LLM (Ollama)

A small open model on your GPU can take over the LLM steps at no cost (docs 16.4a). It is on by default
(`LOCAL_LLM_ENABLED=true`) but only used when Ollama answers with the model pulled.

1. Install or upgrade [Ollama](https://ollama.com/download) (0.35 works; 0.16 cannot run Qwen 3.5). The
   official installer needs sudo:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```

   Without sudo, unpack the `ollama-linux-amd64.tar.zst` release into your home directory, start it with
   `OLLAMA_HOST=127.0.0.1:11435 ollama serve` and set `LOCAL_LLM_URL=http://127.0.0.1:11435` in `.env`.
2. Pull the model and check its speed and GPU memory. `make llm-pull` fetches `qwen3.5:4b` (about 3 GB) and
   registers `qwen3.5:4b-text`, the same model without its unused image encoder (740 MB less GPU memory):

   ```bash
   make llm-pull
   make llm-check
   ```

3. Restart the app; it looks for the model at start-up.

- **Without `OPENROUTER_API_KEY`:** the technique verifier, interview phrasing and AI review run locally, and
  report prose is rewritten sentence by sentence before mapping. Nothing leaves the computer.
- **With a key:** everything stays on OpenRouter unless a step gets a `local/` model id, for example
  `EXTRACT_MODEL=local/qwen3.5:4b-text` or `TECHNIQUE_VERIFIER_MODEL=local/qwen3.5:4b-text`.
- **GPU only:** the model never spills onto the CPU; when the GPU is too full the step says so. `make
  llm-check` also warns when a laptop has capped the GPU's power.
- **Verifier confidence:** the local verifier accepts a technique only when at least 98% sure
  (`LOCAL_LLM_VERIFIER_MIN_CONFIDENCE`); the rest stay for the analyst.
- **Off:** `LOCAL_LLM_ENABLED=false`, or `make api-local` / `make run-local` for no LLM at all.

### Technique head

The fine-tuned 4B technique model (`scripts/cribl_finetune/README.md`, docs 21.3.18) needs its own environment
(`make cribl-venv`, transformers 5) and the checkpoint under `data/models/cribl_ft/E2/step_52706`, so it runs as
a separate local service (about 3.5 GB of GPU memory, `http://localhost:8102`):

```bash
make technique-head-serve
```

The app uses it whenever the service answers with the pinned checkpoint (`TECHNIQUE_HEAD_ENABLED=true`,
`TECHNIQUE_HEAD_EXPECTED_FINGERPRINT`). When the service is down or serves another checkpoint, mapping runs
exactly as without it and the status strip says so.

### Elastic Agent Builder MCP

The connector targets Elastic Agent Builder MCP on Elasticsearch 9.2+ or Elastic Serverless (not the deprecated
standalone Elasticsearch MCP server). Create an API key restricted to the intended security indices with `read`
and `view_index_metadata`, plus Kibana `feature_agentBuilder.read`, then set:

```text
ELASTIC_MCP_URL=https://your-kibana.example/api/agent_builder/mcp   # custom space: /s/<space>/api/...
ELASTIC_MCP_API_KEY=<scoped API key>
ELASTIC_MCP_AUTO_GATHER=false
```

The connector only calls `platform.core.search`, lets the API key define the scope, caps returned records, and
imports events only after the analyst selects them.

### Google sign-in

The web app can run as a shared team workspace with Google sign-in (OpenID Connect):

1. In the Google Cloud console create an OAuth client ID of type *Web application* with the redirect URIs
   `http://127.0.0.1:5173/auth/callback` (Vite dev server), `http://127.0.0.1:8100/auth/callback` (built app)
   and your production `https://<host>/auth/callback` (with `AUTH_PUBLIC_URL=https://<host>`).
2. Set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `AUTH_SECRET_KEY` and an allowlist (`AUTH_ALLOWED_DOMAINS`
   and/or `AUTH_ALLOWED_EMAILS`) in `.env`; without an allowlist the server refuses to start in google mode.
   `AUTH_ADMIN_EMAILS` names administrators; the first user to sign in is one as well.
3. Restart the API. Every `/api` route then needs a session.

Users live in `data/users.db` (SQLite: address, name, picture, role, active flag, sign-in times). It is
personal data: git ignores it and Docker builds leave it out, so back it up separately. Administrators manage
users and can delete any case; analysts see every case, create their own and delete only their own. Without
Google credentials (`AUTH_MODE=auto` or `off`) the app runs as one local administrator, which is how the
tests run.

## Configuration

All settings are environment variables documented in `.env.example`. The ones you are most likely to change:

| Area | Setting | Default and meaning |
|---|---|---|
| Models | `EXTRACT_MODEL`, `REVIEW_MODEL` | `openai/gpt-5.6-sol`, `anthropic/claude-sonnet-5` (OpenRouter) |
| | `LOCAL_LLM_ENABLED`, `LOCAL_LLM_URL` | `true`, used only when Ollama answers with the model |
| | `TECHNIQUE_HEAD_ENABLED` | `true`, used only while the service answers with the pinned checkpoint |
| | `EMBEDDING_MODEL` | `basel/ATTACK-BERT`, pinned by `EMBEDDING_MODEL_REVISION` |
| Mapping | `TECHNIQUE_EXEMPLARS_ENABLED` | `true`; `false` for structured-telemetry deployments where narrated logs lose ranking (docs 21.3.10) |
| | `TECHNIQUE_PLATFORM_FROM_TEXT` | `true`: an observation that names an OS maps only to that OS's techniques |
| | Mapping preset (in the app) | Balanced (default), Conservative, Exploratory, Local only, Legacy (dense-only) |
| Attribution | `TEXT_GROUP_WEIGHT`, `BEHAVIOR_TEXT_WEIGHT`, `PROCEDURE_MATCH_WEIGHT` | `1.0`: the procedure-text core. They bind the combiner, so turning one off needs the matching combiner (docs 21.3.25) |
| | `ATTRIBUTION_COMBINER_PATH`, `CONFIDENCE_LABELS_PATH` | the procedure-core combiner and its labels; `ATTRIBUTION_COMBINER_ENABLED=false` turns the combiner off |
| | `ATTRIBUTION_HYPOTHESIS_MIN_PROBABILITY` | `0.20`; the floor belongs to its combiner (docs 21.3.22) |
| | `ATTRIBUTION_HYPOTHESIS_ABOVE_NULL`, `ATTRIBUTION_TECHNIQUE_LIST_ONE_FAMILY` | `true`: a name must beat "none of these", and the technique-list members count once (docs 21.3.27) |
| | `ATTRIBUTION_CLUSTER_NAMING`, `ATTRIBUTION_CLUSTER_SCOPE` | `true`, `documented`: name a pair only when ATT&CK documents it as the same activity (docs 10.11) |
| | `SOURCE_COUNTRY_WEIGHT` | `1.0`: source-country vote on (docs 9.13) |
| Off by default | `MODEL_OPINION_ENABLED` | the paid GPT-5.6 Sol second opinion on the Assessment page (docs 16.6) |
| | `GROUP_MLP_WEIGHT`, `PROFILE_FOREST_WEIGHT`, `CAMPAIGN_WEIGHT` | `0`: group network, group forest and campaign member; each made a measured result worse (docs 9.9, 9.13, 21.3.11) |
| | `STORE_TEXT_WEIGHT`, `STORE_SOFTWARE_WEIGHT` | `0`: dated knowledge-store members, no weight beat 0 (docs 12.9) |
| | `NAIVE_BAYES_BACKGROUND_ENABLED` | `false`: size correction, failed its gate (docs 21.3.33) |
| | `RERANKER_ENABLED` | `false`: local cross-encoder reranker, lowered held-out F1 (docs 21.3.6) |
| Live intel | `THREAT_INTEL_DAYS`, `INCLUDE_RANSOMWARE_CLAIMS` | `30`, `true` |

Rollback: the defaults before procedure text became the core are tagged
`rollback/pre-procedure-core-2026-10-26`; the docs (21.3.25) list the settings that restore them.

## Supported Windows events

- Sysmon: 1, 3, 7, 8, 10, 11, 12, 13, 14, 22, 23, 25
- Windows Security/System: 4624, 4625, 4688, 4698, 4720, 7045, 1102

Inputs may be a JSON array, JSON Lines, an Elastic `hits` export, or EVTX-exported nested JSON.

## Development

```bash
make api           # API on http://127.0.0.1:8100 (OpenAPI at /docs)
make web-dev       # Vite dev server on http://127.0.0.1:5173, /api proxied to 8100
make test          # LLM calls are mocked: no network or API spend
make lint
make web-check     # tsc + oxlint, before committing web changes
docker compose config --quiet
```

Every analyst operation goes through `apt_classifier/service.py`, and every API payload is produced by
`api/serializers.py`. `transformers==4.57.5` is pinned below v5 because ATTACK-BERT publishes a `.bin`
checkpoint that v5 cannot load reliably. `AGENTS.md` lists every fitting, benchmark and experiment command.

### Benchmarks

Results are composed into `data/audits/verification_results_current.json` by `make verify-results` and are
never typed by hand.

- `make benchmark-public` — **primary mapping gate**: about 1,400 held-out public cases (`tram`, `expert`,
  `derived_procedures` test splits of `tumeteor/mitre-ttp-mapping`), bootstrap intervals, with and without the
  regex cues; `make check-regression` compares against the committed baseline.
- `make benchmark-attribution` — held-out, anonymised ATT&CK procedure sentences per group through the real
  pipeline, with per-fold leak hiding and a removed-actor arm. Self-consistency, not real-world accuracy.
- `make benchmark-cti-hal` — CTI-HAL: 968 annotated statements from 55 real reports on six groups, for mapping
  and attribution with group names removed (local evaluation only; no published licence).
- `make prepare-elastic` + `make benchmark-elastic` — Elastic Security Labs posts: 8 attributed to one ATT&CK
  group plus a 22-post open set, labels pre-registered (local evaluation only; post text © Elastic).
- `scripts/attribution_benchmark.py --report-set vendor|campaigns|advisories` — the 68 development real
  reports the combiner is fitted on (diagnosis only).
- `scripts/prepare_fresh_reports.py` + `scripts/fresh_llm_attribution.py` — vendor reports newer than the
  frontier models' cut-off, Vestige against GPT-5.6 Sol, pre-registered (local only).
- `scripts/cribl_finetune/external_product_taa.py` — CTI-TAA and AthenaBench attribution (evaluation only,
  non-commercial licences).
- Diagnostics: `make benchmark-curated` (36 curated observations, the cue tables were written against them),
  `scripts/otrf_benchmark.py` (9 Sysmon narratives), `make evidenceforge-eval` (synthetic bundles; only
  `heldout-clinic` is untouched by tuning, see `docs/EVIDENCEFORGE_EVALUATION.md`).

## Data and privacy

- ATT&CK vectors and application state live locally in ChromaDB; raw logs are parsed locally.
- Free-text observations, alert text, interview context and review payloads go to OpenRouter only when those
  features run with a key. Do not use OpenRouter if your policy forbids third-party processing.
- The model second opinion (off by default) sends the case's observations, with ATT&CK group names and ids
  removed, only when the analyst clicks its button; vendor and most malware names stay in the text.
- Without an OpenRouter key (local LLM and technique head on your GPU), no case data leaves the computer; only
  the live-intelligence feeds are fetched.
- Elastic MCP searches run between this app and Kibana; the Elastic API key is never sent to OpenRouter.
  Imported events stay non-voting until the analyst selects them.
- The OpenRouter key is a Pydantic secret, never written to an audit artifact and never sent to ChromaDB.
  Review prompts and responses are kept under `data/audits/` (ignored by git).
