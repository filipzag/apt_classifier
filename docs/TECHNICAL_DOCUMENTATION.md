# Vestige — Technical Documentation

## 1. Document purpose

This document describes the current implementation of Vestige (the `apt_classifier` package and its front ends): its architecture, data flow, evidence model, retrieval and ranking algorithms, live threat-intelligence enrichment, analyst Canvas, Diamond Model intrusion analysis, interview workflow, isolated AI review, deployment, security boundaries, verification, and limitations.

The document is an implementation reference, not a claim of attribution accuracy. The application produces ranked hypotheses about overlap with documented MITRE ATT&CK tradecraft. It does not prove the identity, nationality, sponsorship, or intent of an attacker.

### 1.1 Intended audience

- Security analysts and incident responders using the web interface (React + FastAPI, 4.5; the Streamlit UI remains as a legacy front end).
- Engineers maintaining the evidence, retrieval, ranking, and review pipeline.
- Operators deploying the local ChromaDB service and application.
- Reviewers evaluating whether a result is sufficiently supported for additional collection.

### 1.2 Current implementation snapshot

The current local knowledge base and vector index contain:

| Artifact | Count / value |
|---|---:|
| ATT&CK techniques and sub-techniques | 697 |
| ATT&CK groups | 176 |
| ATT&CK campaigns | 56 |
| ATT&CK software objects | 825 |
| Technique sentence vectors | 11,614 |
| Group documents | 5,394 |
| Actor-specific behavior vectors | 5,042 |
| Embedding dimension | 768 |
| ATT&CK bundle version | 19.2 |

Counts change when the ATT&CK bundle is re-ingested. The application displays the current Chroma counts in the sidebar.

---

## 2. Product behavior and non-goals

### 2.1 What the application does

The application accepts mixed incident evidence and converts it into a structured analysis:

1. Analyst observations, alert prose, Windows/Sysmon logs, IOCs, and named tools are accepted.
2. Free-text observations and alert prose are normalized into atomic attack-action sentences through a structured OpenRouter call.
3. Windows/Sysmon logs are parsed locally and deterministically.
4. IOCs and known software aliases are extracted locally.
5. Action observations are embedded with `basel/ATTACK-BERT` and queried against local ChromaDB.
6. ATT&CK techniques are associated with evidence, including source and observation provenance.
7. Multiple deterministic classifiers produce independent group rankings.
8. Correlated classifier families are combined using family-aware Reciprocal Rank Fusion (RRF), and a learned combiner re-orders the top 20 (10.9).
9. Strategic context—target geography, target industry, motivation, objective, and optionally supported sponsor/origin—is used as a conservative reranking signal.
10. Actor-specific ATT&CK relationship descriptions provide a separate behavior signal.
11. Public live intelligence can be fetched and matched to the case as non-voting enrichment.
12. The Analyst Canvas connects evidence, techniques, behavior, tools, campaigns, candidates, countries, and intrusion type.
13. The Diamond Model view organizes adversary, capability, infrastructure, and victim vertices from current evidence with explicit uncertainty and provenance.
14. A hypothesis-driven interview predicts unobserved behaviors, recommends ATT&CK-informed collection steps, and asks prioritized follow-up questions automatically in chat form.
14. An optional four-stage isolated AI review critiques the deterministic result.
15. JSON and Markdown reports can be exported.

### 2.2 What the application does not do

The application does not:

- Geolocate an attacker from an IP address.
- Treat the victim country as the attacker country.
- Treat a named tool as proof of a group identity.
- Treat ATT&CK technique presence as a unique fingerprint.
- Treat public ransomware leak-site claims as confirmed breaches.
- Treat a classifier rank or RRF relative index as a calibrated probability.
- Automatically add live intelligence to the RRF score.
- Send bulk Windows logs to the LLM.
- Claim production-grade attribution accuracy.

A result should be read as: **“These candidates have varying degrees of documented tradecraft or contextual overlap with the supplied evidence.”**

---

## 3. High-level architecture

```text
                         +-------------------------+
                         | React web app + FastAPI |
                         | Intake / Canvas / Diamond / Intel |
                         | Distinguish / Interview |
                         | Assessment              |
                         +------------+------------+
                                      |
                                      v
                         +-------------------------+
                         | AttributionEngine       |
                         | session.py               |
                         +------------+------------+
                                      |
             +------------------------+------------------------+
             |                        |                        |
             v                        v                        v
   +----------------+       +----------------+       +----------------+
   | Evidence       |       | ATT&CK vector |       | Live public    |
   | normalization  |       | retrieval     |       | threat intel   |
   | extract/logs/  |       | Chroma +       |       | CISA /         |
   | IOCs           |       | ATTACK-BERT    |       | ransomware.live|
   +-------+--------+       +-------+--------+       +-------+--------+
           |                        |                        |
           +------------------------+------------------------+
                                    v
                         +-------------------------+
                         | ClassifierContext       |
                         | techniques / tools /    |
                         | strategic context /     |
                         | behavior scores         |
                         +------------+------------+
                                      |
                                      v
                         +-------------------------+
                         | Classifier registry     |
                         | ten deterministic       |
                         | + CyberRE-LLM          |
                         | + two forests           |
                         +------------+------------+
                                      |
                                      v
                         +-------------------------+
                         | Family-aware RRF        |
                         | + learned combiner      |
                         | correlation diagnostics |
                         | candidate eligibility   |
                         +------------+------------+
                                      |
             +------------------------+------------------------+
             |                        |                        |
             v                        v                        v
   +----------------+       +----------------+       +----------------+
   | Origin and     |       | Analyst Canvas |       | Isolated AI    |
   | intrusion type |       | graph/dossier  |       | review         |
   | assessment     |       |                |       | ACH pipeline   |
   +----------------+       +----------------+       +----------------+
```

### 3.1 Main implementation modules

| Area | Module | Responsibility |
|---|---|---|
| Web app | `web/` | React 19 + TypeScript + Vite single-page app, the primary interface since 2026-09-29 (4.5) |
| HTTP API | `api/`, `apt_classifier/service.py` | FastAPI routes, serializers and Google sign-in (4.5, 4.6); `CaseService`, the headless engine, case store and every analyst operation |
| Streamlit UI (legacy) | `app/streamlit_app.py`, `app/ui/` | Streamlit entry point; per-session engine and case store (`runtime`, `cases`), guarded steps (`guard`), shared components, sidebar and one module per page (`pages/`) |
| Orchestration | `apt_classifier/session.py` | `AttributionEngine`, analysis state, rescoring, questions, review calls, live-intel attachment |
| Configuration | `apt_classifier/config.py` | Pydantic settings and filesystem paths |
| ATT&CK download | `apt_classifier/attack/download.py` | Bundle URL selection and local download |
| ATT&CK parsing | `apt_classifier/attack/parse.py` | STIX filtering, entity parsing, relationship resolution, actor behavior extraction |
| Knowledge base | `apt_classifier/attack/kb.py` | Dataclasses and derived ATT&CK statistics |
| Embeddings | `apt_classifier/embeddings.py` | Sentence Transformer loading and normalized embedding generation |
| Vector store | `apt_classifier/vectorstore.py` | Chroma collections, document creation, upsert, query, health, counts |
| Evidence models | `apt_classifier/evidence/models.py` | Pydantic ledger, observations, provenance, statuses, strategic context |
| LLM extraction | `apt_classifier/evidence/extract.py` | Structured free-text and alert normalization |
| Windows parsing | `apt_classifier/evidence/winlogs.py` | Local Sysmon/Security/System event parsing |
| IOC parsing | `apt_classifier/evidence/iocs.py` | Hash, IP, domain, and software-alias extraction |
| Retrieval | `apt_classifier/retrieve.py` | Technique retrieval, platform filtering, aggregation, behavior retrieval, context construction |
| Classifiers | `apt_classifier/classifiers/` | Deterministic group-ranking signals plus the analyst-requested CyberRE-LLM vote |
| Fusion | `apt_classifier/fusion.py` | RRF, family budgets, correlations, contested flags |
| Attribution | `apt_classifier/separation.py`, `apt_classifier/combiner.py`, `apt_classifier/attribution.py` | Candidate-relative separation, learned combiner, stage and naming gate (10.6-10.9) |
| Top three | `apt_classifier/top3_view.py` | Display-only Top-three view (10.10) |
| Learned profiles | `apt_classifier/profile_forest.py`, `apt_classifier/torch_forest.py`, `apt_classifier/group_mlp.py` | Random-forest group and source-country models, optional group network (9.13) |
| Target OS | `apt_classifier/platforms.py` | Observation platform scope and technique filtering (5.7) |
| Diamond Model | `apt_classifier/diamond.py` | Four-vertex intrusion model with evidence-linked nodes, relationships, and caveats |
| Origin/type | `apt_classifier/origin.py` | Country-origin hypotheses and APT-like/criminal/indeterminate assessment |
| Live intel | `apt_classifier/threat_intel.py` | Public feed clients, parsers, caching data structures, case matching |
| Elastic MCP | `apt_classifier/elastic_mcp.py` | Agent Builder MCP transport, ES|QL and search flows, bounded query planning, local result normalization, staged evidence candidates |
| Canvas | `apt_classifier/canvas.py` | Graph data model and candidate dossier generation |
| Review context | `apt_classifier/review/context.py` | Compact structured reviewer payloads and sensitive-data reduction |
| Review pipeline | `apt_classifier/review/` | Blinding, ACH, devil's advocate, reconciliation, bias metrics, audit persistence |

---

## 4. Runtime lifecycle

Sections 4.1 to 4.4 describe the Streamlit application, which remains runnable (`make run`) as the legacy UI. The primary interface, the React app with its FastAPI backend, is described in 4.5 and 4.6.

### 4.1 Application startup

`app/streamlit_app.py` is a thin entry point; the UI lives in the `app/ui` package
(`runtime`, `sidebar`, `components`, `guard`, `cases`, `local_extract` and one module per
page under `app/ui/pages`). Startup:

1. Loads settings from `.env` (`pydantic-settings`) and ensures the data directories exist.
2. If `data/kb/attack_kb.json` is missing, offers one-click ingestion and stops.
3. Loads the knowledge base, ATTACK-BERT and the Chroma client **once per process**
   (`runtime.load_resources`, `st.cache_resource` keyed by KB path and modification time).
4. Builds the **per-session** `AttributionEngine` (`runtime.get_engine`, stored in
   `st.session_state`). The engine is cheap (lexical index, acceptance model, weights) and
   holds the session's own settings, so two analysts changing sliders no longer overwrite each
   other on a shared engine. It is rebuilt only when the sidebar settings change; the
   OpenRouter client and its running cost carry over.
5. Without `OPENROUTER_API_KEY` the app runs in **local mode** instead of stopping: prose is
   split into sentences locally, the verifier, CyberRE and AI review are disabled
   with a visible banner, and retrieval, gating, ranking and the interview (deterministic
   question selection) keep working.
6. Renders the sidebar (case manager, mode banner, preset, advanced settings, system info;
   Chroma health is polled at most once a minute) and the selected page.

### 4.2 Workflow navigation

`st.navigation` (top bar) with explicit URL paths:

```text
Workflow:     1 · Intake → 2 · Technique Review → 3 · Interview → 4 · Assessment
Analysis:     Canvas | Diamond Model | CyberRE-LLM
Intelligence: Live Intel
```

Only the selected page executes. Every page starts with the same status strip (case name,
observations, accepted / to-review techniques, candidates, stage badge, LLM spend) and a
**Next** hint that names the next useful step and jumps to it. Stage and candidate states
are shown as icon + label, never colour alone. Numbered pages are the analyst path; the
analysis views are read-only perspectives on the same state.

### 4.3 Analysis lifecycle

**Analyse evidence** (Intake) runs inside `st.status` with two visible steps:

```text
narrative / phase notes / alerts / IOCs / files
  -> extract_evidence()  (or local sentence split in local mode)
  -> EvidenceLedger (new case id)
  -> runtime.reanalyze(): engine.analyze() -> retrieval, gating, eligibility,
     family-aware RRF, origin/type, stage decision
  -> AnalysisState in session state, case autosaved, jump to Technique Review
```

Every later change to the ledger (technique decisions, interview answers, Elasticsearch
imports, sidebar **Re-score**) goes through `runtime.reanalyze`, which keeps what is still
valid from the previous state: live-intel matches are re-attached and CyberRE scans for
observations that still exist are kept and re-fused. The independent AI review is dropped
because it was written for the old ranking. Sidebar changes that were not applied yet are
flagged as *stale* on every page.

### 4.4 Cases, errors and production hardening

- **Cases** (`app/ui/cases.py`): the ledger, the analyst's case name, the classifier weights
  and a summary are saved as JSON under `<DATA_DIR>/cases/<case_id>.json` after every
  analysis or change (atomic write). The sidebar lists saved cases, opens one (the analysis
  is recomputed from the ledger; nothing derived is trusted from disk), renames and deletes
  (confirmation popover). Case ids are validated against path traversal.
- **Errors** (`app/ui/guard.py`): every LLM-backed or long step runs through `run_step`,
  which turns OpenRouter credit (402), key (401), rate-limit (429), timeout and Chroma
  failures into one plain-language sentence with a collapsed technical detail block and
  leaves the case state unchanged. No page can raise a raw traceback for an operational
  failure.
- **Technique review** is one `st.data_editor` table (filter by state, top-3 toggle, search)
  with a decision column; replacement ids are validated against the knowledge base and the
  editor resets after decisions are applied, so decisions cannot be applied twice.
- **Assessment** shows the stage card (maturity, margin and the candidate-relative separation of
  10.7 against their bars, the best-supported alternative, documented overlap, glossary), a ranking chart and a selectable table that opens the candidate
  dossier, evidence detail, diagnostics, the optional review and **exports that are always
  available** (Markdown report, JSON audit, Navigator layers, Attack Flow); the report says
  which steps were run.
- **Operations**: `Dockerfile` and the `app` service in `docker-compose.yml` run the FastAPI
  backend with the built React app (4.5) next to Chroma (`docker compose up`), with `/data` as a
  volume, a health check on `/api/health`, a non-root user and port 8100 bound to localhost.
  Without `GOOGLE_CLIENT_ID` sign-in is off and every request runs as a local admin (4.6), so
  configure Google sign-in before exposing it. The Streamlit UI has no authentication.
- **Tests**: `tests/test_ui_modules.py` (case store, local extraction, error mapping,
  next-step guidance) and `tests/test_app_flow.py` (AppTest through the real local pipeline:
  Intake analyses the sample and saves the case; every page renders it; skipped when the
  knowledge base is not ingested).

---

### 4.5 HTTP API and React front end

Since 2026-09-29 the primary interface is a React 19 + TypeScript + Vite + Tailwind v4 single-page
app (`web/`) served by a FastAPI backend (`api/`). The Streamlit app remains runnable (`make run`)
as the legacy UI, without the later features (Target OS, Distinguish, the Top-three view); the last
Streamlit-only revision is preserved on the `streamlit-stable` branch.

- `apt_classifier/service.py` — `CaseService`: one engine (knowledge base, embedder, vector
  store, optional OpenRouter client), the on-disk `CaseStore`, in-memory `AnalysisState` per case
  id (rebuilt from the saved ledger on first access), a per-case re-entrant lock plus one engine
  lock (the embedder and Chroma client are shared), and every analyst operation: intake
  (`IntakePayload` → LLM extraction or local sentence split → `analyze`), technique decisions,
  interview (`interview()`, `answer()`, `restart_interview()`), LLM steps (review,
  CyberRE) that raise `ServiceUnavailable` in local mode, live threat intelligence with a
  30-minute snapshot cache, Elasticsearch MCP staging, dossier/canvas/diamond views and exports.
  Re-analyses keep live-intel matches and CyberRE results for observations that still exist,
  and **verifier verdicts are saved with the case** (`verifications` in the case file, schema 2:
  observation id, technique id, verdict, reason, support span, model, tokens, cost) and replayed
  by `map_techniques(prior_verifications=...)` in both production and local mode before any
  verifier call; candidates that carry a saved verdict are never sent to the verifier again, so
  reopening a case is deterministic and costs nothing (`retrieval_diagnostics.verifier_reused`).
  `not_run` and `error` are not verdicts and are retried. Case files written before the LLM
  zero-shot vote was removed (2026-09-30) may still carry a `zero_shot` entry; it is ignored on
  load and dropped on the next save.
- `apt_classifier/report.py` — headless `STAGE_INFO`, the Markdown assessment, the JSON audit
  payload, Navigator layers and Attack Flow export (`export_document(kb, state, kind)`); the
  Streamlit components delegate to it so the two front ends cannot drift.
- `api/app.py` — `create_app(service=None, serve_web=True)`; the service is built in the lifespan
  when not injected (tests inject a stub or a local-mode service). Error mapping:
  `CaseNotFound` → 404, `ValueError` → 400, `ServiceUnavailable` → 409, anything else → 500 with a
  plain-language `detail` (402 credit exhaustion is recognised). Synchronous handlers run in the
  thread pool, so a minute-long review does not block other requests. When `web/dist` exists the
  app is served at `/` with SPA fallback; `/api/*` never falls through to it.
- `api/serializers.py` — pure reads of `AnalysisState` into JSON: observations, candidates (with
  the acceptance probability, evidence items and verifier verdict), accepted techniques, ranking
  entries with the per-channel discrimination, the decision with thresholds and gate mode,
  origin, behaviour and strategic matches, the classifier rank matrix, review,
  CyberRE, live-intel matches, and the "next step" hint. Group ids always carry the ATT&CK name.
- Endpoints: `GET /api/health`, `/api/system`, `/api/sample`, `/api/techniques?q=`; `GET|POST /api/cases`,
  `GET|PATCH|DELETE /api/cases/{id}`; `POST …/decisions`, `…/weights`, `…/platforms`; `GET …/interview`,
  `POST …/interview/answer`, `…/interview/restart`; `POST …/review`, `…/cyberre`;
  `POST …/elastic/search`, `…/elastic/import`; `GET …/dossier/{group}`, `…/distinguish`, `…/top3`, `…/canvas`, `…/diamond`,
  `…/intel`, `…/export/{markdown|json|navigator|attack-flow}`.
- Web app: routes `/` (cases), `/new` (intake), `/cases/:id` (assessment), `…/evidence`,
  `…/techniques`, `…/distinguish`, `…/interview`, `…/canvas`, `…/diamond`, `…/intel`, `…/review`,
  `/admin/users`. TanStack Query
  caches the case payload; mutations write the returned payload straight into the cache.
  Visual language: cyan for the deterministic ranking, violet for advisory AI output, amber for
  review, rose for rejection; every state carries a label, never colour alone. The canvas and
  diamond use one deterministic layered SVG layout (`LayeredGraph.tsx`), so the picture is
  identical on every render.
- Commands: `make api` (8100; 8000 is Chroma), `make api-local`, `make web-dev` (5173, proxy),
  `make web-build`, `make web-check` (tsc + oxlint), `make run-next`. Tests: `tests/test_api.py`
  (stub-service error mapping; a real local-mode flow that skips without the index).

### 4.6 Users and sign-in (2026-09-30)

`api/auth.py` installs one of two modes from the settings (`Settings.resolved_auth_mode()`):
**off** (no `GOOGLE_CLIENT_ID`): every request runs as `LOCAL_USER`, a local analyst with admin
rights, so single-user and test runs need no credentials; **google**: a signed session cookie
(Starlette `SessionMiddleware`, `AUTH_SECRET_KEY`, `AUTH_SESSION_HOURS`, SameSite=Lax,
`AUTH_COOKIE_SECURE` behind HTTPS) plus an Authlib OpenID Connect client for Google.
`/auth/login` redirects to Google, `/auth/callback` verifies the id token, checks
`email_verified`, applies the allowlist (`AUTH_ALLOWED_DOMAINS` / `AUTH_ALLOWED_EMAILS`, at
least one required or the server refuses to start) and creates or refreshes the user in
`data/users.db` (`apt_classifier/users.py`: id, address, name, picture, role, active, sign-in
times, Google subject). The store creates the file and table on first use; because it holds
personal data it is not tracked by git (untracked 2026-10-01) and `.dockerignore` keeps it out
of images. The first user and every `AUTH_ADMIN_EMAILS` address are admins.
`current_user` is the dependency behind `get_service`, so every `/api` route is authenticated;
state-changing requests must also carry `X-Requested-With: Vestige` (the SPA adds it; a
cross-site form cannot), which closes the CSRF hole a cookie session opens. `/api/me` returns
the user and the auth mode; `/api/auth` is public and tells the login page how sign-in works;
admins list and edit users (`/api/users`; an admin cannot demote or deactivate themselves, and
the last active admin cannot be removed). Cases record their creator (`owner` in the case file
and in the API payloads); everyone in the workspace can read and work on any case, deletion is
limited to the owner or an admin. Identity data is never logged. The web app wraps its routes
in `AuthGate`, which renders the sign-in page on a 401 and otherwise provides the user to the
shell (user chip, sign-out, the admin-only *Users* page, owner on case cards).

## 5. Evidence ingestion and normalization

All incoming evidence is represented in an `EvidenceLedger`.

### 5.1 Evidence statuses

`EvidenceStatus` has four values:

| Status | Meaning |
|---|---|
| `positive` | The behavior or fact is supported by supplied evidence. |
| `negative` | The analyst explicitly reports that the behavior was not observed. |
| `unknown` | The analyst does not know whether it occurred. |
| `not_visible` | The available telemetry could not observe the behavior. |

`unknown` and `not_visible` are not treated as negative evidence.

### 5.2 Provenance

Each observation has a `Provenance` record containing:

- `source`: normalized source type such as `free_text`, `alert`, `winlog`, `ioc`, or `software`.
- `source_name`: optional filename or analyst-provided source name.
- `line_numbers`: one-based source line references.
- `occurrence_count`: merged duplicate count.
- `raw_excerpt` and `raw_excerpts`: local audit excerpts.
- `event_id`: Windows/Sysmon event ID when available.
- `metadata`: parser or source metadata.

Observation identity is deterministic when an ID is not supplied. The ledger deduplicates observations by identity or equivalent content and merges provenance, line numbers, excerpts, and occurrence counts.

### 5.3 Free text and alert prose

`apt_classifier/evidence/extract.py` sends free text and alert prose to the configured extraction model using a strict JSON schema. The extraction prompt requires:

- Atomic attack-action sentences.
- Concrete actor/process and object/target where available.
- Verbatim executable names, command lines, paths, hosts, accounts, protocols, and qualifiers.
- Source line numbers.
- Exact supporting excerpts.
- No inferred identity, intent, technique, or missing detail.

The application does not send bulk Windows logs to this LLM extraction path.

### 5.4 Windows/Sysmon logs

`apt_classifier/evidence/winlogs.py` parses Windows data locally. Supported event IDs include:

- Sysmon: `1`, `3`, `7`, `8`, `10`, `11`, `12`, `13`, `14`, `22`, `23`, `25`.
- Windows Security/System: `4624`, `4625`, `4688`, `4698`, `4720`, `7045`, `1102`.

Accepted shapes include:

- JSON arrays.
- JSON Lines / NDJSON.
- Wrapped exports.
- Elastic `hits` responses.
- Nested EVTX-exported JSON.
- File-like objects, bytes, mappings, and in-memory records.

The parser generates deterministic action descriptions and retains event ID, source line, raw excerpt, and source name.

### 5.5 IOCs and software

`apt_classifier/evidence/iocs.py` extracts:

- MD5, SHA-1, SHA-256, and SHA-512 hashes.
- IPv4 addresses.
- IPv6 addresses.
- Domains.
- Named software aliases found in the ATT&CK software KB.

Software observations carry a `software_id` when an alias is resolved. IOC and software observations are not submitted as normal positive action text for technique retrieval, but software IDs are supplied to the software classifier and Canvas.

Aliases are matched case-insensitively on word boundaries, except for **ordinary-word aliases**
(2026-09-30). ATT&CK lists some software under plain English words: `Page` is an alias of
Elise, a backdoor documented for Lotus Blossom alone, and `Wiper`, `Expand`, `CALENDAR` and
`ZLib` are software names. Matched case-insensitively, the ordinary word "page" in a sentence about a
card-skimming payment page became Elise and handed Lotus Blossom decisive single-group malware
evidence in a FIN6 report.
`ambiguous_software_aliases(kb)` decides which aliases need care from the knowledge base
itself, not from a hand-written list. It reads ATT&CK's own prose: technique, group and
software descriptions and procedure sentences, with code spans, citations, link targets and
URLs removed and the software's own description excluded. An alias is ambiguous when that
prose uses its lowercase form as an ordinary word at least as often as it writes the exact
name. On ATT&CK 19.2 this flags `Page`, `Wiper`, `Expand`, `CALENDAR` and `ZLib`, while
`Mimikatz` (74 name uses, 1 lowercase), `Bumblebee` and `Responder` keep the normal match.
An ambiguous alias only matches in its exact spelling. A title-case one such as `Page` also
never matches at the start of a sentence, where a leading capital says nothing; all-caps
`CALENDAR` still does. Aliases whose canonical spelling is lowercase (`netsh`, `certutil`)
are command names and keep the case-insensitive match. The table is memoised per knowledge
base and costs about 0.15 s to build. Measured effect: 6 of the 55 CTI-HAL reports lose a
false software match (Elise in five, Wiper in one) and nothing else changes; 12 of the 5,042
ATT&CK procedure sentences change, all ordinary-word hits.

### 5.6 Strategic context

The analyst can provide optional structured context through the Intake view:

- Victim country or region.
- Targeted industry or sector.
- Suspected motivation.
- Likely end objective.
- Suspected sponsor/origin, only when independently supported.

Values are normalized into deduplicated lists by `StrategicContext`. Context is kept separate from attack-action text so words such as “energy,” “government,” or “espionage” do not accidentally become technique retrieval queries.

### 5.7 Target OS (2026-10-01)

ATT&CK records the platforms each technique applies to (131 of the 697 techniques are
Windows-only, 13 Linux-only, 18 macOS-only). The analyst can state which operating systems the
affected hosts run, the case's **Target OS**: Windows, Linux, macOS and ESXi, every one the
activity touched (a Linux DMZ server pivoting to Windows servers is Windows + Linux), empty when
unknown. A missing operating system hides its techniques (21.3.12). It is chosen in the web app's Intake (the Streamlit UI has no Target OS) and can be changed on the
Technique review page, which maps every observation again. Saved verifier verdicts are reused, so in production
mode only candidates new to the case are sent to the verifier. It is stored in the ledger's
`metadata["target_platforms"]` rather than a new ledger field, so case files stay readable by
code that predates it, and it appears in the Markdown export and the JSON audit record.

`apt_classifier/platforms.py` gives every observation a **platform scope**:

1. the platform its log source records (Windows/Sysmon event logs: Windows);
2. otherwise, when `TECHNIQUE_PLATFORM_FROM_TEXT` is on (the default), the operating systems the observation's own
   text names ("on the Linux server", Ubuntu, CentOS, macOS, ESXi, Windows, Win32/Win64). Browser
   user-agent strings and "Kali Linux" (the attacker's own distribution) are ignored, and lower-case
   "windows" is not the operating system;
3. otherwise the case's Target OS;
4. otherwise none (no filtering).

A technique is kept for a scope when ATT&CK lists none of its platforms, when it lists one of the
scope's operating systems, or when it also lists a platform the operating system says nothing
about (IaaS, SaaS, Office Suite, Identity Provider, Network Devices, Containers, PRE). So Linux rules
out Scheduled Task (T1053.005), Registry Run Keys or PowerShell, but never Valid Accounts or a cloud
technique. Event-log scopes keep their stricter earlier rule: the technique must list Windows.
The scope is applied before fusion, to dense and lexical hits, so compatible techniques take the
freed places in the pool (Cron instead of Scheduled Task), and afterwards to cue, structured-signal
and exemplar-only candidates. The retrieval diagnostics record each observation's scope and its
source (`platform_scopes`) and the number of dropped cue/signal candidates; Technique review shows
a badge on observations whose own text or log source set the scope.

The interview and the Distinguish step use the case's Target OS as well
(`shortlist_questions(platforms=...)`, which feeds the next question, the interview leads and the
Distinguish leads). A technique documented only for other operating systems is never asked about:
the hosts cannot show it, and a "no" would count against the groups that document it for a reason
that has nothing to do with them. Each lead's "scope the search" step names only the platforms still
relevant (a Linux case: Linux plus cloud, container or identity platforms), worded as places to
search ("Windows and Linux hosts, ESXi hypervisors and identity providers"). ATT&CK's `PRE` platform
(the 96 reconnaissance and resource-development techniques) is not a place on the network, so those
leads point to threat intelligence and external records (registrations, perimeter scans, tooling
recovered from hosts) instead of naming "PRE systems". The ATT&CK analytics shown as collection steps
follow the Target OS too (each analytic covers one platform, so a Linux case is no longer sent to
`WinEventLog:Security` 4688), the hand-written fallback guidance, which names Windows telemetry, is
used only when Windows is in scope, and ATT&CK's literal log-source channel "None" is not printed. In production the
question model is told which operating systems the hosts run. Per-observation text scopes do not
narrow the interview, because a case can mix hosts; only the analyst's Target OS does. Without one,
nothing changes: for "the attacker created a scheduled task that runs the payload every hour" the
local interview opens with Windows Command Shell and suggests Registry Run Keys and PowerShell;
with Target OS Linux none of the three is offered.

---

## 6. ATT&CK knowledge base construction

### 6.1 Source and object filtering

`apt_classifier/attack/download.py` downloads the selected enterprise ATT&CK STIX bundle. `apt_classifier/attack/parse.py`:

1. Loads the STIX bundle.
2. Drops revoked and deprecated objects.
3. Resolves ATT&CK external IDs.
4. Parses techniques and sub-techniques.
5. Parses intrusion-set groups.
6. Parses campaigns.
7. Parses malware and tool software.
8. Resolves `subtechnique-of` relationships.
9. Resolves group `uses` relationships to techniques and software.
10. Resolves campaign `uses` relationships to techniques.
11. Resolves campaign `attributed-to` relationships to groups.
12. Computes IDF, priors, and prominence statistics.

### 6.2 Knowledge-base records

The main records are dataclasses in `apt_classifier/attack/kb.py`:

#### `Technique`

Contains:

- ATT&CK ID and STIX ID.
- Name.
- Description.
- Detection guidance.
- Platforms.
- Tactics.
- Parent technique.
- External references.
- Sentence-level name, description, and detection fields.

#### `Group`

Contains:

- ATT&CK group ID and STIX ID.
- Name and aliases.
- Description.
- Technique IDs.
- Software IDs.
- Campaign IDs.
- `technique_behaviors`: actor-specific relationship descriptions keyed by technique ID.
- Prominence.

#### `Campaign`

Contains name, description, aliases, dates, techniques, groups, and references.

#### `Software`

Contains name, aliases, type, platforms, techniques, and linked groups.

### 6.3 Actor-specific behavior extraction

A group-to-technique ATT&CK `uses` relationship often has a description such as:

```text
The group used rundll32.exe to execute MiniDump for dumping LSASS process memory.
```

The parser preserves these relationship descriptions as `Group.technique_behaviors`. Citations are removed and descriptions are split into sentences. These sentences are more specific than a bare group→technique presence bit because they can contain:

- Executable names.
- Command fragments.
- APIs.
- Paths.
- Malware names.
- Tool combinations.
- Operational sequences.

The current index contains 5,042 vectors from these sentences.

---

## 7. Embeddings and ChromaDB

### 7.1 Embedding model

The configured model is:

```text
basel/ATTACK-BERT
revision=81e30f983a822c606825507b63ae318a6830a8a2
```

The application uses `SentenceTransformerEmbedder` with:

- Normalized sentence embeddings.
- 768 dimensions.
- CUDA when available, otherwise CPU.
- Explicit `trust_remote_code=False`.
- A project-local Hugging Face cache.

`transformers==4.57.5` is pinned because the selected model publishes a legacy `.bin` checkpoint that is not reliably loadable by Transformers v5.

### 7.2 Chroma deployment

Chroma runs as a local container from `chromadb/chroma:1.5.0`.

```yaml
ports:
  - "8000:8000"
environment:
  IS_PERSISTENT: "TRUE"
  PERSIST_DIRECTORY: /data
  ANONYMIZED_TELEMETRY: "FALSE"
volumes:
  - chroma_data:/data
```

The service exposes two collections:

| Collection | Contents |
|---|---|
| `attack_techniques` | Sentence-level technique name, description, and detection documents |
| `attack_groups` | Group technique profiles, group descriptions, and actor-specific behavior documents |

The Chroma HNSW space is cosine distance. Embeddings are supplied explicitly during upsert and query.

### 7.3 Technique documents

Technique document IDs have the form:

```text
<Technique ID>:<field>:<sentence index>
```

Fields are:

- `name`
- `description`
- `detection`

Metadata includes technique ID, name, field, sub-technique flag, parent ID, tactics, and platforms.

**Synthetic field.** `scripts/generate_synthetic_sentences.py` (`make generate-synthetic`) asks
the extraction model for about six varied observation sentences per technique (analyst
prose, alert-style one-liners, report language), grounded only in the technique's ATT&CK
description, tactics and anonymized procedure examples, with sibling sub-techniques listed
as "do not describe". Sentences are filtered (6 to 45 words, no technique IDs, no actor
names, not identical to any indexed sentence, deduplicated) and stored in
`data/kb/synthetic_sentences.json` with model, prompt version, ATT&CK version and cost.
They are indexed as the `synthetic` field (field weight 0.9, same as procedures) and are in
the lexical index too; `TECHNIQUE_INCLUDE_SYNTHETIC` (default false) includes them at
retrieval time so their effect can be measured before adoption. Benchmark text is never used for generation, so indexing
them cannot leak evaluation data. Adding the field bumped `TECHNIQUE_SCHEMA_VERSION` to 4.

### 7.4 Group documents

Group documents include:

```text
<GID>:technique-profile
<GID>:description
<GID>:actor-behavior:<TID>:<sentence index>
```

Behavior metadata includes:

- `group_id`
- `stix_id`
- `name`
- `document_type=actor_behavior`
- `technique_id`
- `sentence_index`

The technique profile is a normalized centroid of member technique vectors. It is not a concatenation of vectors.

### 7.5 Health and index readiness

The sidebar reports:

- Technique sentence count.
- Total group collection count.
- Actor behavior count.
- ATT&CK version.
- Embedding model.

If the actor-behavior count is zero, the application warns the operator to run:

```bash
make ingest
```

The current index has been rebuilt with 5,042 actor behavior documents.

---

## 8. Retrieval pipeline

`apt_classifier/retrieve.py` produces a `RetrievalResult` containing:

- `matches`: technique-level evidence matches.
- `observed_techniques`: aggregated technique confidence.
- `technique_observations`: observation IDs supporting each technique.
- `semantic_profile_scores`: group technique-profile similarities.
- `group_text_scores`: group-description similarities.
- `specific_behavior_scores`: group-specific behavior similarities.
- `specific_behavior_signals`: vector IDs supporting behavior scores.
- `specific_behavior_matches`: detailed behavior hits.
- `candidates`: typed technique candidates with verification state and analyst decisions.
- `mapping_mode`: active retrieval mode (`legacy_dense`, `hybrid`, `dense`, `lexical`).
- `parent_policy`: active hierarchy policy (`ancestor_compatible`, `exact_only`, `legacy_expand`).
- `diagnostics`: retrieval diagnostics (candidate counts, verifier status, thresholds, preset).

### 8.1 Retrieval modes

The pipeline supports four retrieval modes, selected by the mapping preset or runtime configuration:

| Mode | Dense (ATTACK-BERT) | Lexical (BM25) | Use case |
|---|:-:|:-:|---|
| `legacy_dense` | Yes | No | Original dense-only retrieval for comparison and rollback. |
| `hybrid` (default) | Yes | Yes | Fuses dense and lexical results with RRF at the technique level. |
| `dense` | Yes | No | Dense-only without legacy rollup behavior. |
| `lexical` | No | Yes | BM25-only retrieval; useful for exact-term matching. |

The lexical side is a BM25 index (`LexicalTechniqueIndex`, k1 = 1.2, b = 0.75, Lucene-style IDF) over technique `name`, `description` and anonymized `procedure` sentences (about 11,300 documents on ATT&CK 19.2). Queries use an inverted index with precomputed term frequencies and length norms (about 1 ms per query; the earlier linear scan cost over 100 ms). The actor-name anonymizer that produces procedure sentences uses one combined alternation pattern and is memoized per knowledge base, so building the index takes about half a second instead of several minutes.

In hybrid mode, dense and lexical results are fused using Reciprocal Rank Fusion at the technique level. Each technique's fused score is:

```text
fused_score = sum(1 / (60 + rank) for each retrieval source)
```

Cross-validation bonus: techniques appearing in both dense and lexical top-3 receive an additional +0.005.

### 8.2 Candidate generation and acceptance

For each positive action observation, the pipeline:

1. Splits observations longer than `LONG_OBSERVATION_WORDS` (40) words into sentences and
   retrieves per sentence; per-sentence hits are merged per technique (best similarity
   wins) and the candidate pool grows by four per extra sentence, capped at 30. Short
   observations are retrieved whole.
2. Queries the technique vector store (dense) and the lexical BM25 index. The lexical
   index covers `name`, `description` **and** the anonymized `procedure` sentences, so a
   dense procedure hit can be corroborated lexically.
3. Determines the observation's platform scope (5.7): the platform its log source records
   (Windows/Sysmon logs: Windows), the operating systems its own text names when
   `TECHNIQUE_PLATFORM_FROM_TEXT` is on, or else the case's Target OS.
4. Drops hits below `SIMILARITY_THRESHOLD` (default 0.45) and hits for techniques outside the
   platform scope; applies a relative score margin from the best dense hit. Cue and
   structured-signal candidates outside the scope are dropped after the cues are merged.
5. Fuses dense and lexical ranks per technique (RRF, k=60), adds the name-token,
   cross-validation and cue bonuses, and keeps the top `candidate_pool` (default 12).

Acceptance is decided in `finalize_retrieval_candidates` after every evidence source has
been collected. Two gates exist:

**Calibrated acceptance model (default when the artifact is present).**
`apt_classifier/acceptance.py` scores every candidate with an L2-regularised logistic
regression over its own evidence (fused score and rank, margins to the top and next
candidate, dense similarity/field/rank, lexical rank, cue and signal flags, hierarchy and
sub-technique flags, benign/sparse context, pool size, and since schema 3 the eight exemplar
features described below). Feature schema 2 (2026-09-28) adds
three **per-technique prior** slots: the log count of the technique as a label in the public
**train** splits, the log number of its ATT&CK procedure sentences and the log number of
groups documented using it. Priors are computed at fit time (`compute_technique_priors`),
stored inside the artifact and looked up by technique id at inference; unknown techniques get
zeros. The **shipped** model (`acceptance_model.json`) is fitted with `--priors-split none`,
i.e. the prior slots are zero and the model behaves as before. The priors fit is kept as
`acceptance_model_priors.json` for opt-in use (`ACCEPTANCE_MODEL_PATH`); it is better at
technique mapping (21.3.6) but shifts acceptance toward report-frequent generic techniques,
which produced a confidently wrong named attribution on a real report (21.3.4b), so it is not
the default. The model is fitted by `scripts/fit_acceptance_model.py` (`make fit-acceptance`)
on the public **dev** splits only (priors, when used, from train, never dev or test), and its
threshold is the dev operating point that maximizes accepted-set micro-F1 subject to
precision >= 0.5. Candidates at or above the threshold are `AUTO_ACCEPTED`;
several candidates per observation can be accepted. The artifact
(`data/models/acceptance/acceptance_model.json`) records the ATT&CK version, feature
schema, dev metrics and pipeline fingerprint and loads fail-closed: when it is missing,
malformed, disabled (`ACCEPTANCE_MODEL_ENABLED=false`) or built for another ATT&CK
version, the rule-based gate below is used and `retrieval.diagnostics` says so.

**Rule-based gate (fallback).** Similarity-only acceptance is reserved for the fused
rank-1 candidate: it is accepted when its dense similarity is >= `acceptance_threshold`
(0.75), or it is the best dense hit with a margin >= `relative_margin` (0.08) over the
runner-up and >= 0.60, or lexical retrieval independently ranks it in its top 5. If rank
1 is cue-supported it consumes the slot; if it fails the gate the observation abstains
from similarity-only acceptance. Earlier versions promoted the best *qualifying*
candidate instead (skipping rank 1 when it was cue-supported or when its only dense hit
was a procedure sentence); on the held-out public benchmark that promotion was worse
than accepting rank 1 (pooled F1 0.225 -> 0.240 after the change).

In both gates, cue-supported candidates (a structured signal or explicit cue plus dense
or lexical corroboration) are accepted, benign-context and sparse observations never
accept on similarity alone, and verifier verdicts override everything else.

**Cross-observation corroboration (default on, 2026-09-29).** The acceptance model judges each
observation on its own, so a technique that a multi-statement report keeps returning to can
stay in review sentence after sentence. `accept_by_corroboration` accepts a NEEDS_REVIEW
technique when at least `TECHNIQUE_CORROBORATION_ACCEPT_OBSERVATIONS` (2) *different*
observations of the case retrieved it with a calibrated probability of at least
`TECHNIQUE_CORROBORATION_ACCEPT_MIN_PROBABILITY` (0.1); the state reason says
"corroborated across N observations". Derived hierarchy parents, rejected candidates and
repeats within one observation never count, and single-observation cases (the public
statement benchmark) are unaffected by construction. Measured on the attribution benchmarks
(21.3.4b, 21.3.5): real-report accepted-technique recall 0.26 -> 0.31 at unchanged precision
(0.49), 36 of 55 reports gain techniques, top-1 unchanged, top-5 0.69 -> 0.66, 3 of 3 names
correct; ATT&CK folds technique recall 0.78 -> 0.79, precision 0.75 -> 0.71, top-1 0.53 ->
0.52, 14 of 14 names correct. A 0.2 floor barely fires because the acceptance threshold was
then 0.27 (0.21 since the exemplar refit of 2026-09-30, below).

**Ranking by acceptance probability (default on).** With `TECHNIQUE_RANK_BY_ACCEPTANCE=true`
each observation's candidates are ordered by the calibrated model's probability that the
mapping is correct instead of by the raw RRF score, after acceptance has been decided
(`rank_candidates_by_acceptance`; `fused_rank` is rewritten so the UI, the review payload and
the benchmarks all see the same order). RRF still decides which candidates enter the pool
and the accepted set is unchanged; only the order shown to the analyst and measured by
hit@k moves. The probability is stored on the candidate as `acceptance_model` evidence and
is reused by any later finalization pass (after the verifier, or in the interview) so the
features, which include the fused rank and margins, are always computed in the fusion order
the model was fitted on. Candidates the model did not score (derived hierarchy parents,
verifier-decided candidates without a stored probability) follow in fused order. Measured
on the public dev split first (hit@1 0.355 -> 0.391, MRR 0.465 -> 0.505, F1 identical) and
then through the held-out gate (21.3.6): hit@1 0.35 -> 0.40, hit@3 0.53 -> 0.58, hit@10
0.73 -> 0.75, MRR 0.47 -> 0.52, every split up, accepted-set P/R/F1 unchanged.

**Labelled-sentence exemplars (default on, 2026-09-30).** The dense and lexical indexes hold
ATT&CK's own prose. Real report text is written differently: on the CTI-HAL statements 40% of
the correct techniques were in the candidate pool but left in review, because similarity to
ATT&CK prose could not separate them from wrong neighbours. `apt_classifier/exemplars.py` adds a
second kind of evidence, *how other real report sentences that read like this one were
labelled*. The exemplar index (`make exemplars`, `scripts/build_exemplar_index.py`,
`data/models/exemplars/exemplar_index.npz`) holds the 6,360 distinct public **train** sentences of
the `tram`, `expert` and `derived_procedures` splits of `tumeteor/mitre-ttp-mapping` (CC-BY-4.0;
413 techniques), embedded with the technique embedder. It never holds the leaky `procedures`
split, dev, test, CTI-HAL or EvidenceForge text. For each observation the 64 nearest exemplars
vote with softmax weights (temperature 0.05). Any exemplar at cosine >= 0.95 is treated as a
copy of the query and never votes. Every candidate gets eight features: its vote share, a
hierarchy-relaxed share (own plus sub-technique votes), the similarity of its closest voting
exemplar relative to the best one, inverse vote rank, vote count, the best exemplar similarity,
vote x similarity, and an exemplar-only flag. Up to `TECHNIQUE_EXEMPLAR_NEW_CANDIDATES` (3)
techniques that only the exemplars suggest are appended after the retrieved pool (fused score 0,
never displacing a retrieved candidate). The acceptance model judges them on the exemplar
evidence alone. The candidate's evidence carries the nearest labelled sentence, shown in the
review table as "like: ..." with its corpus and licence. Structured telemetry is queried by its
values only: a field/value record ("command_line: whoami /all. pid: 5512. process_name: ...")
loses its lowercase schema keys and bare identifiers, because words such as "pid" and
"process_name" pull report-prose neighbours towards Process Discovery. Prose is queried unchanged
(0 of 1,391 public test texts, 0 of 968 CTI-HAL statements and 1 of 1,147 dev texts change).

Feature schema 3 appends these eight features. The shipped model is fitted with them on the
public dev split (`make fit-acceptance`; dev threshold 0.21, dev P 0.52 / R 0.42 / F1 0.47 versus
0.50 / 0.28 / 0.36 without them) and stores the exemplar index fingerprint. It refuses to run
against a different or missing index. In that case the loader uses
`ACCEPTANCE_MODEL_FALLBACK_PATH` (`acceptance_model_no_exemplars.json`, the same fit without
exemplar features, also written by `make fit-acceptance`), and diagnostics say "fallback without
exemplars". `TECHNIQUE_EXEMPLARS_ENABLED=false` therefore restores the previous calibrated gate
exactly. The index is pinned to the embedding model and revision and integrity-checked by a
fingerprint. Results are in 21.3.10: real reports and the public held-out set gain markedly, while
ATT&CK-phrased fold sentences and narrated telemetry pay a precision cost.

**The fine-tuned technique head (on by default since 2026-10-20; 2026-10-19).** The 4B technique
model of 21.3.18 (Qwen3.5-4B-Base in 4-bit with a LoRA and two cosine heads: one row per ATT&CK
family, one per technique, read at the decision slot of a one-sentence prompt) can score every
observation. It needs transformers 5, which the product's `.venv` cannot hold next to
ATTACK-BERT, so it runs as a local HTTP service (`make technique-head-serve`,
`scripts/cribl_finetune/serve.py` on `http://localhost:8102`, about 3.5 GB of GPU memory) and
`apt_classifier/technique_head.py` is the product's client. With `TECHNIQUE_HEAD_ENABLED=true`
(the default since 2026-10-20, pinned to the registered checkpoint; the pipeline runs exactly as
without it while the service is down), `retrieve_techniques` posts each observation's sentences to the service once (a long observation
is scored sentence by sentence and pooled by the largest probability) and does two things with the
answer: every candidate gets `technique_head` evidence, i.e. the head's probability for the
candidate's family and for the candidate itself, both inverse ranks, the head's confidence in its
own top family, whether the candidate is one of the head's proposals and whether it exists only
because of the head (**feature schema 4** of the acceptance model: schema 3 followed by these seven
`th_*` features; a schema-3 artifact keeps reading its own prefix); and up to
`TECHNIQUE_HEAD_NEW_CANDIDATES` (3) of the head's proposals that retrieval did not produce (its best
technique in each of the `TECHNIQUE_HEAD_TOP_FAMILIES` (3) most probable families) are appended
after the pool with fused score 0, exactly like exemplar-only candidates, platform scope applied.
Nothing retrieved is displaced and the calibrated threshold is never overridden. Prose only, for
now (product owner, 2026-10-19): observations that come from logs (the telemetry sources of the
verifier's interpreter rule in 16.4a: Windows/Sysmon events, eCAR, Elastic records, the
EvidenceForge renders) are not sent to the head and get no head evidence or proposals
(`diagnostics["technique_head_skipped_logs"]`); the head was trained on report prose. Fail closed: the
client is `None` (the pipeline runs as without it, the status strip says why) when the service does
not answer, serves another ATT&CK version or a checkpoint other than the pinned
`TECHNIQUE_HEAD_EXPECTED_FINGERPRINT` (the SHA-256 of the adapter and head files the service
reports at `/health`); an acceptance model fitted with the head active records that fingerprint in
its metadata and retrieval contract and refuses to run on another head or without one; a request
the service fails mid-run leaves its observations without head evidence and is counted in
`diagnostics["technique_head_errors"]`, and the benchmarks refuse to write such a run under
`--require-technique-head`. **Two models behind one gate.** The schema-4 model
(`ACCEPTANCE_MODEL_HEAD_PATH`, `acceptance_model_technique_head.json`) is fitted on rows that
always carry head evidence, so it must never judge a candidate without it (every `th_*` feature at
0 is a value it has not seen: on the EvidenceForge clinic bundle that cost 0.12 F1 on eCAR
windows). While the head is active the loader therefore pairs it with the shipped model in
`acceptance.HeadAwareAcceptance`: the head model judges the candidates of observations the head
scored, the shipped model (`ACCEPTANCE_MODEL_PATH`) the rest (log-derived observations, failed
requests); the pair is refused when the two were fitted on different exemplar libraries or
retrieval settings, and with the service stopped the head path is never read, so the shipped gate
runs unchanged. The pre-registered gate and its result are in 21.3.18 ("Product integration").

**Per-sentence judgement for long observations (optional, off by default).** With
`TECHNIQUE_SEGMENT_ACCEPTANCE=true` each sentence of a long observation also gets its *own*
candidate pool, fused ranks, margins and benign/sparse checks (`_build_segmented_candidates`, evidence metadata
`segment_index` / `segment_rank` / `segment_text`). Acceptance runs per sentence
(`_assign_candidate_states` groups by segment), so a technique described in the third
sentence is not judged against the first sentence's best match, and each sentence can
contribute its own accepted technique. The observation-level candidate list is the merge
of the sentence pools (best fused score per technique, `segment_support` recording how many
sentences retrieved it). This follows the behaviour-decomposition finding in the TTP
extraction literature (TTPrint: removing decomposition costs ~19 points of recall).

**Hierarchy-consistent acceptance (optional, off by default).** When two or more sub-techniques of one parent are
plausible for the same sentence (dense >= floor or lexical top-3) and none is confidently
accepted, or the only similarity-based acceptance leads its best sibling by less than
`relative_margin` in dense similarity, the evidence supports the parent but not the choice
of child: the derived parent candidate is accepted and the children stay in review
(`_apply_hierarchy_consistency`). Cue- or verifier-backed children are never replaced.
This targets the sub-technique-for-parent false positives measured on the public
benchmark (29 of 144 TRAM accepted false positives were a child of the labelled parent).
Hierarchy-derived parents are otherwise `NEEDS_REVIEW`. `TECHNIQUE_HIERARCHY_PARENT_MODE`
selects `off` (default), `swap_only` (only replace an accepted but non-separable child) or
`swap_and_fill` (also accept the parent when plausible siblings exist and none was
accepted). The fill variant collapsed held-out precision (pooled P 0.50 -> 0.27)
because it re-admitted candidates the acceptance model had rejected, and is not the default.
`TECHNIQUE_SEGMENT_ACCEPTANCE` switches the per-sentence judgement off for comparison.

**Cross-observation corroboration.** `TECHNIQUE_CORROBORATION_BONUS` (default 0.0) adds a
bonus per additional independent observation that accepted a technique (capped at three)
to that technique's observed confidence, a campaign-level vote across observations. It is
measured with the attribution benchmark before being turned on.

### 8.3 Abstention

An observation contributes no accepted technique when no candidate passes the active gate
or when a benign indicator (`_BENIGN_INDICATORS`: "self-service", "password reset",
"antivirus updated", "administrator logged in", "routine backup", "software center" ...)
or a sparse text (< 3 distinct words) blocks similarity-only acceptance. Candidates remain
visible for review in every case; abstention is about the authoritative accepted set,
not about hiding candidates.

### 8.4 Explicit technique cues

The pipeline maintains a registry of regex patterns (`_EXPLICIT_TECHNIQUE_CUES`) for common ATT&CK techniques where dense retrieval alone may not rank the correct technique first. The cues operate in two modes:

**Boost mode** (technique already retrieved by dense/lexical):
- Adds `explicit_cue` evidence to the existing candidate
- Auto-accepts the candidate
- Boosts the candidate strength by +0.1 over the dense score (bounded at 1.0)

**Injection mode** (technique NOT retrieved by dense/lexical):
- Injects the technique as a new candidate
- Auto-accepts with bounded score 0.85 (not 1.0)
- The bounded score ensures injected candidates cannot override strong dense matches

**Safety guardrails**:
- Benign indicators always block cue injection. If the observation contains benign activity phrases, no cue-based candidates are added.
- The cue evidence is visible in the candidate's evidence list for audit.
- Injected candidates have `state_reason="explicit behavior cue injected (bounded score 0.85)"`.

| Technique | Cue pattern |
|---|---|
| T1059.001 (PowerShell) | "powershell" + "encoded/enc/-enc/hidden window" |
| T1003.001 (LSASS Dump) | "lsass" + "dump/memory/credential" |
| T1003.002 (SAM Registry) | "sam" + "registry/hive/dump" |
| T1003.003 (NTDS) | "ntds" + "dit/dump/copy/snapshot" |
| T1021.002 (SMB Admin Shares) | "smb" + "admin/share" |
| T1047 (WMI) | "wmic" + "execute/remote/process/lateral" |
| T1105 (Ingress Tool Transfer) | "download/ingress/transfer" + "payload/binary/tool" |
| T1218 (System Binary Proxy) | "system binary" + "proxy/execution" |
| T1218.005 (Mshta) | "mshta" + "execute/run/payload/hta" or ".hta" |
| T1218.011 (Rundll32) | "rundll32" + "dll/load/execute" |
| T1055 (Process Injection) | "inject/injection/createremotethread" + "process/thread" |
| T1053.005 (Scheduled Task) | "scheduled task/schtasks" + "create/persistence" |
| T1113 (Screen Capture) | "screen capture/screenshot/desktop capture" |
| T1056.001 (Keylogging) | "keylog/keystroke/capture keystroke" |
| T1041 (Exfiltration Over C2) | "exfiltrate" + "c2/https/http server" |
| T1134.001 (Token Impersonation) | "token impersonation/impersonate token" |
| T1547.001 (Registry Run Key) | "registry run key/HKLM...Run" |
| T1560.001 (Archive via Utility) | "zip/7zip/winrar/tar" + "utility/compress/archive" |
| T1574.001 (DLL Side-Loading) | "side-load/side-loading" + "dll/binary/library" |
| T1687 (Defense Impairment) | "disable/turn off" + "defender/antivirus/edr" |

Explicit cues are intentionally narrow and post-retrieval: they do not replace vector retrieval but compensate for dense retrieval gaps while preserving auditability and false-positive safety.

### 8.5 Name-token matching

The `_name_token_bonus()` function boosts candidates when distinctive tokens from the ATT&CK technique name appear in the observation text:

1. Extracts tokens (>=4 chars, not stopwords) from the technique name.
2. Deduplicates tokens and requires >=2 unique tokens.
3. Computes match ratio (matched tokens / total tokens).
4. Requires >=34% match ratio for a bonus.
5. Base bonus: `0.02 * match_ratio`.
6. **Exact tool name bonus**: +0.015 for recognized tool names (powershell, mshta, rundll32, regsvr32, certutil, bitsadmin, msiexec, wmic, bash, python, lsass, schtasks, vssadmin, ntdsutil, sekurlsa).

Name-token matching also blocks abstention when the bonus exceeds 0.01, preventing false abstention on positive cases that mention specific tools.

### 8.6 Hierarchy handling

The pipeline supports three parent/sub-technique policies:

| Policy | Behavior |
|---|---|
| `ancestor_compatible` (default) | Observed sub-techniques can support groups documenting only the parent (discounted). Parent/child evidence relationships are preserved. |
| `exact_only` | Only exact technique matches count for group scoring (`hierarchy_matches_for` skips ancestor/descendant matches). The policy is copied from the retrieval result onto the classifier context by `build_classifier_context`; before that wiring existed the setting had no effect in production. |
| `legacy_expand` | Sub-technique evidence is rolled up to the parent at 95% of the child score (original behavior). |

Hierarchy expansion (`_expand_hierarchy_candidates`): when >=2 sub-techniques of the same parent are retrieved, the parent technique is auto-added as a derived candidate with `hierarchy_parent` evidence source. This ensures groups documenting only the parent can still be matched.

Non-duplicative materialization: in `_materialize_candidates`, parent techniques that were directly retrieved (dense/lexical evidence) are preserved. Parents without direct evidence that were added by hierarchy expansion are still subject to the non-duplicative filter (skipped when children exist).

### 8.7 Bounded OpenRouter verification

`AttributionEngine.map_techniques` runs the local acceptance gate (`finalize_retrieval_candidates`) **before** verification, so the verifier scope is evaluated against real candidate states:

| `TECHNIQUE_VERIFIER_MODE` | Candidates sent per observation |
|---|---|
| `ambiguous` (balanced, exploratory) | Only candidates still `NEEDS_REVIEW` after local gating. Locally accepted primaries and cue-supported mappings are trusted and not sent. |
| `all` (conservative) | Every candidate, including locally accepted ones, so the verifier can reject them. |
| `off` (local only, legacy) | None. |

The mapping preset is applied last (`AttributionEngine.__init__` calls `apply_preset`), and
every preset sets `technique_verifier_mode`, so a bare `TECHNIQUE_VERIFIER_MODE=off` in `.env`
is overridden by `TECHNIQUE_MAPPING_PRESET=balanced`. To run without the verifier use
`TECHNIQUE_MAPPING_PRESET=local_only`, which is `balanced` with the verifier off (the local
attribution benchmark does exactly this).

Hierarchy-derived parent candidates are never sent in either scope because their state is fixed to `NEEDS_REVIEW` regardless of verdict. After verification a second, idempotent finalization pass folds the verdicts into the candidate states. (Before this ordering was introduced, verification ran on unfinalized candidates, every candidate was `NEEDS_REVIEW`, and `ambiguous` was identical to `all`.)

The verifier receives only:

- The normalized observation text and source excerpt.
- The retrieved ATT&CK candidate definitions (ID, name, description).

The verifier does NOT receive group names, attribution ranking, strategic context, or live intelligence. This preserves the isolation between technique mapping and attribution.

Verification results:

| Verdict | Candidate state |
|---|---|
| `SUPPORTED` | Auto-accepted. |
| `UNSUPPORTED` | Auto-rejected. |
| `UNCERTAIN` | Needs analyst review. |
| `INVALID_SPAN` | Treated as uncertain. |
| `API_ERROR` | Non-fatal; candidate remains needs_review. |

The verifier model and effort are configurable. The default verifier model is `openai/gpt-5.6-sol` with `low` effort.

### 8.8 Analyst Technique Review

Analysts can confirm, reject, or replace technique mappings through the Technique Review UI. Decisions are applied by `apply_candidate_decisions()`:

- **Confirm**: Candidate state becomes `AUTO_ACCEPTED` with reason "analyst confirmed mapping".
- **Reject**: Candidate state becomes `AUTO_REJECTED`. Rejection removes only the mapping; it does not create negative incident evidence.
- **Replace**: Candidate is replaced with a validated ATT&CK technique. The replacement technique must exist in the KB. The original candidate and replacement rationale are preserved in audit artifacts.

After decisions are applied, downstream classifiers, Canvas, Diamond, review payloads, and exports are re-scored using the effective accepted mappings.

### 8.9 Evidence aggregation

A technique can be supported by multiple observations. The current aggregation uses the strongest three independent scores:

```text
strongest = top three scores sorted descending
mean = average(strongest)
aggregate = 0.75 * maximum + 0.25 * mean
```

The score is bounded at 1.0. Repeating one identical observation does not create an unbounded score increase because occurrence count is not multiplied directly into confidence.

### 8.10 Actor-specific behavior retrieval

Behavior retrieval queries `attack_groups` using `document_type=actor_behavior` separately for each observation. It does not query a centroid of all observations, because per-observation matching preserves behavior provenance.

For each observation:

1. Drop behavior hits below `BEHAVIOR_SIMILARITY_THRESHOLD`, default `0.55`.
2. Keep the best hit per group.
3. Require a relative margin over competing groups.
4. Reject observations that are semantically crowded across many groups.
5. Reject matches sharing only generic terms such as:
   - PowerShell.
   - LSASS.
   - Scheduled task.
   - Persistence.
   - Encoded command.
   - Process.
6. Require rare operational detail such as a distinctive executable, API, command fragment, path, artifact, or tool combination.
7. Aggregate support across independent observations, with a coverage factor so one event does not equal multiple independent behaviors.

A specific behavior match contains:

- Group ID.
- Observation ID.
- Technique ID.
- Similarity.
- ATT&CK actor-specific sentence.
- Behavior vector ID.

This classifier abstains on generic behavior rather than converting generic ATT&CK language into false attribution confidence.

### 8.11 Group profile retrieval

Two additional group queries are made from positive action text:

- `technique_profile`: semantic similarity against a group's technique centroid.
- `description`: similarity against the group's general description.

These are supporting signals and are not allowed to introduce a candidate with no technical evidence in final fusion.

### 8.12 Mapping presets

Five presets bundle retrieval, verification, hierarchy, and gating options:

| Preset | Retrieval | Verifier | Hierarchy | Threshold | Abstention |
|---|---|---|---|---|---|
| **Balanced** (default) | Hybrid | Ambiguous | Ancestor-compatible | 0.45 | Standard (0.75) |
| **Conservative** | Hybrid | All | Exact-only | 0.55 | Strict (0.70, margin 0.10) |
| **Exploratory** | Hybrid (pool 20, top-k 50) | Ambiguous | Ancestor-compatible | 0.45 | Standard |
| **Local only** | Hybrid | Off | Ancestor-compatible | 0.45 | Standard |
| **Legacy** | Dense only (`dense`, detection sentences included) | Off | Ancestor-compatible | 0.45 | Standard |

Presets are applied using Pydantic's `model_copy(update=...)` on the `Settings` model.

---

## 9. Classifiers

The registry currently creates ten deterministic classifiers, the opt-in CyberRE-LLM classifier (it votes only after an analyst runs CyberRE) and the two random-forest members of 9.13. The PyTorch group network (`group_mlp`, 9.13) is added only when `GROUP_MLP_WEIGHT` > 0 (default 0), and the free-text model (`text_group`, 9.13) only when `TEXT_GROUP_WEIGHT` > 0 (default 0). The LLM zero-shot opinion was removed from the ensemble on 2026-09-30 (9.11):

```text
bernoulli_naive_bayes
idf_tfidf_cosine
weighted_jaccard
specific_behavior
semantic_centroid
group_description
strategic_context
motivation_alignment
campaign_profile
software_tools
cyberre_llm
profile_forest
source_country
```

### 9.1 Bernoulli Naive Bayes

**What it does:** Treats each ATT&CK technique as a binary feature and computes a Bayesian log-likelihood score for every group, given the observed and negatively answered techniques. Documented techniques get likelihood `alpha` (default `0.60`); undocumented techniques get `beta` (default `0.05`); explicit negative answers penalize groups that document the negated technique at `negative_evidence_weight` (default `0.30`). The negative term is computed from two counts per group, the negated techniques it documents and those it does not, so it never depends on set iteration order and groups with equal counts receive a bit-identical negative term (their totals tie whenever their positive-evidence sums do). Until 2026-10-03 it was summed over the set in iteration order, which follows `PYTHONHASHSEED`. That could split an exact tie by one unit in the last place, differently in each process, and only cases with negative interview answers were affected (no benchmark sets them). A softmax over the log-scores produces the `bayes_posterior` used for interview context and display.

**Why it is used:** It provides a principled probabilistic baseline that combines positive and negative evidence in a single framework. Unlike pure overlap metrics, it can exploit the analyst's "no, that did not happen" answers to disconfirm candidates. It is deliberately only one ensemble voice — not the final decision — because its uniform prior and ATT&CK presence-only likelihoods favor broadly documented groups and cannot model technique rarity or operational specificity.

### 9.2 IDF/TF-IDF cosine overlap

**What it does:** Represents the evidence and each group as sparse vectors over the ATT&CK technique vocabulary, where each technique is weighted by inverse document frequency (rare techniques carry more weight than commodity ones). Computes cosine similarity between the evidence vector and each group's technique vector. The numerator sums only over the intersection; the denominator normalizes by both vector magnitudes.

**Why it is used:** It down-weights commodity tradecraft (PowerShell, scheduled tasks, LSASS access — techniques documented for dozens of groups) and up-weights distinctive tradecraft (niche sub-techniques documented for only a few groups). This makes it more discriminating than plain overlap when the evidence contains a mix of common and rare techniques. It complements Jaccard because cosine normalizes by vector direction rather than union size, so it can favor large groups whose profile aligns with the evidence direction even when much of their technique set is unmatched.

### 9.3 Weighted Jaccard

**What it does:** Computes weighted set-overlap (Ruzicka similarity) between the IDF-weighted observed technique set and each group's IDF-weighted documented technique set. The score is `Σ min(w_e, w_g) / Σ max(w_e, w_g)` over the union of both sets. Supports a Dice variant (`2 * intersection / (sum of both set totals)`) that penalizes unmatched techniques less harshly. Groups with zero intersection are omitted entirely rather than ranked at zero.

**Why it is used:** It measures the proportion of shared tradecraft weight relative to the total weight both sides cover. Unlike cosine, its union denominator explicitly penalizes groups that document many unmatched techniques — a group with 200 techniques where only 3 match scores lower than a group with 10 techniques where 3 match. This biases the ensemble toward smaller, more focused groups whose technique sets are mostly explained by the evidence, counterbalancing cosine's tendency to favor large groups with aligned profile direction.

### 9.4 Specific behavior

**What it does:** Consumes per-observation vector similarity scores against actor-specific ATT&CK relationship descriptions (e.g. "The group used rundll32.exe to execute MiniDump for dumping LSASS process memory"). Applies discrimination gates: a relative margin over competing groups, a minimum separation from the next-best group, and a specificity filter that rejects matches sharing only generic tokens (PowerShell, LSASS, scheduled task, persistence, etc.). Requires at least two non-generic shared tokens or one long distinctive token. Aggregates surviving matches with a coverage factor across independent observations.

**Why it is used:** Technique-level overlap cannot distinguish two groups that both document `T1003.001` (LSASS Memory dumping). Actor-specific behavior sentences carry operational detail — executable names, command fragments, APIs, tool combinations — that is far more discriminating. This is the strongest single signal in the ensemble (default weight `1.50`) when it fires, but it is also the most conservative: it abstains on generic observations rather than casting a weak actor vote, so it cannot inflate a candidate's rank on commodity tradecraft.

### 9.5 Semantic technique centroid

**What it does:** Uses ATTACK-BERT vector similarity between the evidence's action text and each group's technique-profile centroid — a single vector representing the average semantic direction of all techniques the group documents. The centroid is precomputed and stored in Chroma as a `technique_profile` document. The classifier reads the per-group similarity scores produced by `retrieve_techniques` and ranks groups by them.

**Why it is used:** It captures *semantic* overlap rather than set-membership overlap. Two groups may document different technique IDs but still have similar operational profiles (e.g. both focus on credential access and lateral movement) because their technique descriptions embed nearby in ATTACK-BERT's space. This can surface a candidate that exact-ID overlap misses when the evidence maps to slightly different but semantically related techniques. It is grouped in the technique-overlap family because its centroid is still derived from ATT&CK technique membership, so it shares the family RRF budget and cannot independently dominate.

### 9.6 Group description

**What it does:** Uses ATTACK-BERT vector similarity between the evidence narrative and each group's free-text ATT&CK description. The description is stored in Chroma as a `description` document and queried with the concatenated positive action text. The classifier reads the per-group similarity scores and ranks groups by them.

**Why it is used:** Group descriptions contain non-technical terms that technique IDs do not — targeting language, sector focus, geographic scope, motive, and historical context. This classifier can surface a candidate whose description semantically aligns with the evidence even when the technique-ID overlap is modest. It is downweighted (default `0.40`) and placed in the `context` family because group descriptions are broad and can match generic incident language. Final fusion gates it to technically supported candidates only, so it can rerank but never introduce an unsupported actor.

### 9.7 Strategic context

**What it does:** Compares analyst-supplied structured context (target geography, target industry, motivation, objective, and optionally supported suspected origin) against group and linked-campaign descriptions using controlled alias tables for countries, sectors, and motives. Target geography and industry are matched against sentences containing targeting cues; suspected origin requires origin-style language (`based`, `nexus`, `state-sponsored`, `attributed`, `linked`, intelligence-service). When multiple context fields are supplied, at least two distinct fields must match before a group receives a strategic-context vote.

**Why it is used:** Technical tradecraft overlap alone ignores victimology and motive, which are critical attribution factors. This classifier brings in the analyst's knowledge of *who was targeted and why* — information that technique vectors cannot encode. It is deliberately conservative: it cannot introduce a candidate lacking technical support, it requires multi-field corroboration, and it keeps victim geography separate from attacker origin to prevent circular attribution. It acts as a reranking signal within the `context` family rather than a standalone attribution vote. Motivation is scored separately by the `motivation_alignment` classifier (see 9.8) so a motive match is never discarded merely because geography, industry, or origin fail to match.

### 9.8 Motivation alignment

**What it does:** Scores groups whose ATT&CK group and linked-campaign descriptions document the analyst-supplied motive (espionage, financial gain, disruption, destruction, hacktivism, influence, intellectual property theft) using a controlled alias table. It operates only over the `motivations` field of `StrategicContext` and is gated to technically supported groups: a candidate must already have observed-technique or named-software support before motivation can add a vote. The score is the fraction of supplied motive values that match the group's profile, and it emits signal IDs of the form `motivations:espionage`.

**Why it is used:** Previously motivation was one field inside `StrategicContextClassifier`, where a valid motive match could be diluted or discarded when other context fields (geography, industry, objective, origin) did not also match, and where it competed for a single shared `context` family budget. Extracting motivation into its own classifier with its own `motivation` family budget lets a supported motive rerank a technically eligible candidate independently of unrelated context fields, without allowing motivation alone to introduce an unsupported actor. This addresses the concern that motivation matches were given too little weight while preserving the safeguard that motive is never proof of attribution: motives are shared by unrelated groups, ATT&CK motive descriptions are incomplete, and the classifier reranks rather than establishes identity.

### 9.9 Campaign profile

**What it does:** Compares the observed technique set against each ATT&CK campaign's technique set and links positive campaign overlap to the groups attributed to that campaign. A campaign with strong technique overlap but no attributed group contributes nothing; a campaign with overlap and attributed groups forwards a score to those groups.

**Why it is used:** Campaigns sometimes document a tighter, more coherent technique cluster than the parent group's full technique set, so campaign overlap can be a cleaner signal than group-level overlap. However, self-consistency evaluation showed that enabling this classifier at full or moderate weight *degraded* ensemble accuracy from 150/153 to 23/153 top-1, because campaign technique sets are small and produce noisy, high-variance votes. It is retained for inspection and potential future tuning but disabled from default RRF voting (`CAMPAIGN_WEIGHT=0.00`).

### 9.10 Software/tool links

**What it does:** Scores groups based on documented ATT&CK relationships between groups and software (malware/tools). When the evidence contains a named tool (extracted via alias matching against the ATT&CK software KB), the classifier identifies which groups are documented as using that software and scores them by the strength of the link. It abstains entirely when no named software is present or when identified software has no known group link.

**Why it is used:** Named malware families and tools (Cobalt Strike, Mimikatz, PlugX, Emotet, etc.) are among the strongest attribution signals available, because many tools are disproportionately associated with specific groups or regions. This classifier captures a signal that technique overlap cannot — two groups may both document `T1059.001` but only one may be documented as using a specific custom loader. It is weighted highly (default `1.20`) when it fires, but it is binary and abstention-based: an unknown tool produces no vote rather than a zero-score ranking of every group.

**Rarity weighting and the tool/malware distinction (2026-09-28).** Each observed software entry is weighted by its rarity across groups, `log((N + 1) / (n + 1)) + 1`. ATT&CK marks every entry as `malware` or `tool`; a *tool* is a legitimate utility (cmd, Arp, PsExec, Mimikatz) whose sparse group documentation says nothing about who used it, yet the raw formula gave Arp (documented for 4 groups) a weight of 4.6 against 5.5 for Olympic Destroyer, a malware family documented for one group. Tools are therefore capped at `SOFTWARE_TOOL_WEIGHT_CAP` (1.5) while malware keeps its full rarity weight. On the CTI-HAL Olympic Destroyer report this moved Sandworm Team from rank 2 to rank 1 inside the classifier.

**Decisive evidence boost.** Rank fusion cannot express evidence strength: with `RRF_K=20` a classifier's rank 1 is worth under five percent more than its rank 2, so a report that names a one-group implant got the same credit as one that names PsExec. `decisive_software_factor` therefore multiplies the software vote (classifier weight and family budget) by `1 + SOFTWARE_DECISIVE_BOOST` (default 2.0, i.e. ×3) when an observed *malware* entry is documented for at most two groups, and by half of that for up to four; tools never trigger it. The factor and the triggering ids are recorded on the classifier context (`decisive_software`), shown on the Assessment page and in the report. Measured (21.3.4b, 21.3.5): real-report top-1 0.36 -> 0.44, top-5 0.55 -> 0.69, MRR 0.47 -> 0.55; ATT&CK folds top-1 0.52 -> 0.53; no wrong name on either set.

### 9.11 LLM zero-shot opinion (removed 2026-09-30)

The ensemble used to ask the extraction model (`openai/gpt-5.6-sol`) to rank the deterministic
top ten, then folded that ranking into RRF as a bounded vote (`llm_zero_shot`, weight 0.50,
family `llm_opinion` 0.50). It was removed at the product owner's request so that the
ensemble is deterministic and analyses cost less. It ran on every analysis and again on every
rescore, including each interview answer, at about $0.04 to $0.09 per call on report-sized
cases. In the 10-report production measurement (21.3.4b) it changed the analyst-facing rank in
one report and moved MRR from 0.666 to 0.670; the stage decision never read it. The module,
settings, API endpoint and UI controls are gone. Saved cases that carry a vote still open, and
the vote is ignored. CyberRE-LLM (9.12) remains as an analyst-requested vote, and the
independent review remains an advisory LLM pass outside the ensemble.

### 9.12 CyberRE-LLM technique filtering and ranking

**What it does:** CyberRE-LLM is an analyst-triggered, two-stage ATT&CK labeling workflow. For one selected normalized observation, the filtering stage evaluates every technique in the local ATT&CK knowledge base independently using only the observation and the supplied technique description. The ranking stage receives only techniques accepted by filtering and ranks them with structured reasons. The result is stored per observation as `CyberREResult` and emits inferred technique scores based on rank.

**Why it is used:** It follows the CyberRE-LLM paper's technique-space reduction strategy: reduce the large ATT&CK label space first, then rank the relevant labels. This supplements ATTACK-BERT retrieval when an analyst wants a second technique-labeling opinion for a specific sentence. It is not run automatically because a full scan can require one LLM call per ATT&CK technique plus a ranking call.

**Voting behavior:** Inferred technique scores are kept separate from deterministic `observed_techniques` and are consumed by the `cyberre_llm` classifier only. The classifier has a default weight of `0.35` and a separate `cyberre_llm` family budget of `0.35`. RRF still gates group eligibility using deterministic technical classifiers, so CyberRE cannot introduce a group unsupported by the deterministic evidence.

The vote reaches only the analyst-facing ranking, `state.fusion`. When it is active, `AttributionEngine._score` fuses a second time without the `cyberre_llm` result, through the same `_fuse` path. This second call runs last, so the discrimination and combiner output it leaves on the context belong to the deterministic fusion. `state.deterministic_fusion`, that discrimination and that combiner output are therefore exactly what they would be if CyberRE had never run. Three consumers read this deterministic side:

- the stage decision and naming gate (10.6, 10.7), including the separation and the combiner's calibrated leader probability;
- the distinguish view (10.8);
- the independent AI review (16.2), which also receives no `cyberre_llm` classifier results.

A scan can re-order the ranking the analyst sees, but it cannot change the stage, the named group or its probability. The ranking table's discriminative-mass column shows the gate's shortlist, which can differ from the analyst order after a scan. Until CyberRE votes, the two fusions are the same object.

Before 2026-09-30 all three read the rescored fusion, vote included, so a scan could move the reviewer's candidate order and raise the stage. The attribution benchmarks never run CyberRE and reproduce unchanged.

**Audit and limitations:** Each scan stores the observation ID, filtered techniques, ranked techniques, model, call count, token usage, and cost. The original deterministic retrieval is preserved. Full mode is expensive, vulnerable to LLM labeling errors, and dependent on the supplied ATT&CK descriptions; the human analyst remains responsible for reviewing the resulting technique hypothesis before relying on it.

### 9.13 Random-forest group and source-country models (2026-09-30)

**What it does:** `apt_classifier/profile_forest.py` fits two random forests from the ATT&CK
knowledge base alone. The **group forest** predicts which documented group an evidence set belongs
to. The **country forest** predicts the source-country class: China, Russia, Iran, North Korea,
another state, or `unattributed`. A group's class comes from its documented origin (11.1): 44 China,
18 Iran, 13 Russia, 8 North Korea, 10 other states and 83 groups with no documented state origin.
Features are 0/1: one per ATT&CK technique, where a sub-technique also sets its parent. One more per
canonical motivation and target geography (the strategic-context vocabulary), and an "unknown"
flag for each. At inference the techniques are the accepted ones and motivation and target come
from the analyst's strategic context.

**Training data:** synthetic partial observations of every group's documented profile, 300 per
group (51,300 cases). Each keeps a random 15-70% of the group's techniques, adds false positives
drawn by how many groups document a technique (precision 45-95%), and gives the group's
motivation or one of its target geographies 40% of the time each (5% of those replaced by a random
value). No benchmark, case or CTI-HAL data is used.

**GPU training and inference:** scikit-learn's forest runs on the CPU and stores a dense class row
at every node, which exhausted memory for the 171-class model. `apt_classifier/torch_forest.py`
grows the trees in PyTorch on the GPU (CPU fallback). Every split is "feature present" versus
"absent", and a batch of trees grows level by level with integer scatter-adds, so training is
deterministic for a seed. Like scikit-learn, a node's `max_features` candidates are drawn among the
features that vary inside it. Leaves keep sparse class fractions, and inference sums them in fixed
point, so CPU and GPU predictions are identical. On the RTX 4060 the group forest (200 trees) fits
in 17 s and the country forest (300 trees, balanced classes, leaf size 3) in 15 s, with at most 2.7
GB of GPU memory. A prediction takes about 12 ms. `make fit-forest` writes the artifact
(`data/models/forest/profile_forest.npz`, 56 MB, no pickle, fingerprinted, pinned to the ATT&CK
version, loaded fail-closed) and `data/audits/profile_forest_fit.json`. `PROFILE_FOREST_DEVICE`
selects `auto` (GPU when present), `cuda` or `cpu`. When another process has filled the GPU, the
running app moves prediction to the CPU instead of failing the analysis (2026-10-01; predictions
are identical, only slower), and the system panel shows the device in use.

**Ensemble members:** `profile_forest` votes with the group probability. `source_country` votes for
every group with the probability of that group's own class, so the 83 unattributed groups are
scored like the rest, and groups of one class tie on rank. Both abstain without a loaded model or
accepted techniques. Neither can make a group eligible, because eligibility still comes from the
deterministic technical classifiers. They share one forest prediction per scoring pass, and each
has its own family budget (`learned_profile`, `origin`). The country forest's estimate is not shown
to analysts: on its own it is weaker than the origin aggregated from the ranked groups (11.1; macro-F1
0.36 against 0.80 on the ATT&CK folds, 0.18 against 0.52 on the real reports, 21.3.11).

The group forest votes at weight 0. The source-country vote is on by default since 2026-10-01
(`SOURCE_COUNTRY_WEIGHT=1.0`), together with the combiner refitted with it as a feature (feature
schema 2, `data/models/combiner/attribution_combiner_0930_fsum.json`). To turn it off, set
`SOURCE_COUNTRY_WEIGHT=0` with
`ATTRIBUTION_COMBINER_PATH=data/models/combiner/attribution_combiner_no_country_fsum.json`, a combiner
fitted without the country vote (the 2026-10-03 refit in 21.3.9). In the attribution benchmark's unknown-actor arm, the voting forest is refitted on the knowledge base without the removed actor,
because masking the actor's output would still leak its profile into the other votes. The combiner
fit's unknown lists do the same.

**Group network (2026-10-01).** `apt_classifier/group_mlp.py` is a PyTorch MLP from the same 0/1
features, plus one entry per named ATT&CK software, to the documented group. It is trained from the
knowledge base alone on synthetic report-like cases that include the group's own software, and it
adds a prominence prior at prediction. `make fit-mlp` writes `data/models/mlp/group_mlp.npz` and
`data/audits/group_mlp_eval.json`. The default fit is one network, which is as accurate on the ATT&CK
campaigns as five averaged networks (MRR 0.71 against 0.69) and takes about 20 s. The `group_mlp`
member joins the `learned_profile` family only when `GROUP_MLP_WEIGHT` > 0. At the default 0 it is
not loaded, so the fusion and the combiner fingerprint are unchanged. In the unknown-actor arm the
benchmark refits it without the removed actor. Measured on 2026-10-01 at weight 1.0 with its own
combiner (`attribution_combiner_group_mlp.json`, on earlier classifier code; refit 2026-10-03 as
`attribution_combiner_group_mlp_fsum.json`), before the anonymizer fix of 21.3.14: anonymized
CTI-HAL top-1 went from 0.455 to 0.473 and as reported stayed at 0.564, but on the ATT&CK folds
top-1 fell from 0.830 to 0.817 with one wrong name (Silence named as Kimsuky). It fails the naming
gate and is not adopted.

**Free-text model (2026-10-03).** `apt_classifier/text_group.py` reads the case narrative (the joined
positive observations) instead of the mapped techniques. Each ATT&CK group gets one TF-IDF word
1-2-gram document built from ATT&CK text only: its procedure examples, its description sentences, the
procedure examples of every malware/tool it uses and of its attributed campaigns (the last two from the
raw STIX bundle). Group names become "the adversary" (also before a hyphen) and campaign names "the
operation"; both are stop words. A case is scored by cosine to every document, z-scored, plus 0.5 x the
log prominence prior, and reported as a softmax over the groups that share a term. `make fit-text-group`
writes `data/models/text_group/text_group.json.gz` (the records and parameters, fingerprinted with the
ATT&CK version); the TF-IDF is refitted at load (about 1 s). It was chosen in a separate study
(`scripts/text_group_study/`, 21.3.16) over five ATTACK-BERT embedding models, which blur the exact
malware names and phrases this model keeps. The `text_group` member votes in the `learned_profile` family
only when `TEXT_GROUP_WEIGHT` > 0, and it is listed in `attribution.NOT_INDEPENDENT_EVIDENCE`: it ranks,
but never counts toward the naming gate's independent evidence families, because it re-reads the text the
technique, software and behaviour members already score. At weight > 0 the combiner fingerprint also binds
the text model and `text_group.py`. The benchmarks refuse to run when it fails to load, hide each known
fold's held-out sentences from it (`TextGroupModel.without`: the group's procedure examples for the
held-out techniques, and any record crediting the group or no group whose text or sentence matches a
held-out sentence, refitted exactly) and refit it without the removed actor.

---

## 10. Family-aware Reciprocal Rank Fusion

### 10.1 Standard RRF

For an active classifier ranking an entry at rank `r`:

```text
RRF contribution = classifier weight / (k + r)
```

The default `RRF_K` is `20` for the group ensemble (lowered from 60 after the attribution benchmark, section 21.3.5: top-1 0.48 -> 0.50, MRR 0.60 -> 0.64 together with the technique-family budget of 2.00). Technique-level dense/lexical fusion uses a separate `TECHNIQUE_RRF_K` (60) because the acceptance model's features were fitted against it.

### 10.2 Correlation problem

Naive Bayes, TF-IDF, Jaccard, and the semantic technique centroid all draw heavily from the same ATT&CK group→technique presence table. Treating them as four independent full votes exaggerates technique overlap and can overwhelm specific operational behavior.

The application therefore maps classifiers into families:

```text
technique_overlap:
  bernoulli_naive_bayes
  idf_tfidf_cosine
  weighted_jaccard
  semantic_centroid

context:
  group_description
  strategic_context

motivation:
  motivation_alignment

cyberre_llm:
  cyberre_llm

actor_behavior:
  specific_behavior

software:
  software_tools

campaign:
  campaign_profile

learned_profile:
  profile_forest
  group_mlp (only when GROUP_MLP_WEIGHT > 0)

origin:
  source_country
```

### 10.3 Family budgets

Default family budgets are:

| Family | Default cap |
|---|---:|
| Technique overlap | 2.00 |
| Context | 1.00 |
| Motivation | 1.50 |
| CyberRE-LLM | 0.35 |
| Actor behavior | 1.50 |
| Software | 1.20 |
| Campaign | 1.00 |
| Learned profile (group forest, group network) | 1.00 |
| Origin (source-country forest) | 1.00 |

The individual classifier weights determine how a family cap is divided. A family is scaled down only when its active member weights exceed the cap. A weak singleton classifier is not scaled up.

Effective balance with the defaults before 2026-09-22: the four technique-overlap classifiers share a single budget of 1.00 (about 0.27 each), while `specific_behavior` (1.50), `motivation_alignment` (1.50) and `software_tools` (1.20) each carry more weight than all technique evidence combined, and five of the eight family caps equal their single member's weight (no-ops). Motivation is a substring match over descriptions ("espionage" hits 50 of 176 groups, all tied at rank 1) and software links had no rarity weighting (Mimikatz is documented for 51 groups). Measured with the attribution benchmark (section 21.3.5) and adjusted on 2026-09-22: the technique-overlap budget was raised to 2.00 (top-1 0.38 -> 0.48 alone), the group RRF constant lowered to 20 (with the budget: top-1 0.50, top-5 0.83, MRR 0.64), software links are rarity-weighted (`SOFTWARE_IDF_ENABLED`), and fused-score ties are broken towards the *less* prominent group instead of the alphabetically earlier id. Lowering the software weight (1.20 -> 0.60) hurt ranking (top-1 0.37) and was not adopted. Motivation and strategic-context weights could not be measured by this benchmark (its observations carry no strategic context) and are unchanged.

**Per-classifier reliability (2026-09-28).** The attribution benchmark now records the rank each classifier gives the true group (`classifier_truth_ranks`). Hit@1 of the truth per classifier, where the classifier applies: on the 55 CTI-HAL real reports `software_tools` 0.76 (38 reports with named tooling), `specific_behavior` 0.31, Naive Bayes 0.15, group description 0.13, IDF cosine 0.04, Jaccard 0.02; on the 153 ATT&CK folds Naive Bayes 0.71, IDF 0.67, software 0.75, Jaccard 0.65, centroid 0.25, behaviour 0.13, group description 0.03. Some classifier has the truth at rank 1 in 40 of 55 reports and 143 of 153 folds, against fused top-1 of 20 and 79: fusion is the bottleneck. The two size-normalised overlaps (IDF cosine, Jaccard) penalise groups with long technique lists on real reports; per-classifier weights `NAIVE_BAYES_WEIGHT`, `IDF_OVERLAP_WEIGHT`, `JACCARD_WEIGHT` (default 1.0) allow that family to be rebalanced. A reliability-shifted configuration (technique family 1.0, IDF and Jaccard 0.3, group description 0.2, software and actor-behaviour budgets 2.0) reaches real-report top-1 0.47 / MRR 0.58 with the boost but drops the ATT&CK folds from top-1 0.53 to 0.44 (top-5 0.82 -> 0.73), so it is not the default: the folds are self-consistency over the technique table and a configuration that abandons technique overlap fails them. A milder variant (IDF and Jaccard 0.5, software and actor-behaviour budgets 2.0, technique family unchanged) gains nothing on real reports (top-1 0.44, MRR 0.56) and still costs the folds (top-1 0.53 -> 0.51). The defaults keep the family budgets and add only the tool cap and the decisive-malware boost, which improve both sets.

**Levers measured for reports without named tooling (2026-09-29).** Sixteen of the 55 CTI-HAL
reports name no tooling and depend on technique overlap alone; their accepted techniques are
generic (documented for 30-50 groups on average) and several are Carbanak reports whose ATT&CK
profile is thin, so most of their losses are knowledge-base coverage rather than mapping. Four
mechanisms were added behind settings and measured on both attribution benchmarks:

| Mechanism | Setting | Real reports (top-1 / accepted recall / precision) | Folds (top-1 / technique recall) | Default |
|---|---|---|---|---|
| Baseline (tool cap + decisive boost) | | 0.44 / 0.26 / 0.49 | 0.53 / 0.78 | |
| Soft technique evidence for the ranking (NEEDS_REVIEW candidates with acceptance probability >= 0.15 count at that probability in the technique classifiers only) | `ATTRIBUTION_SOFT_EVIDENCE_MIN_PROBABILITY`, `_WEIGHT`, `_MAX_GROUPS` | 0.40 / 0.26 (ranking-visible 0.39) / 0.49 | 0.52 / 0.78 (0.86) | off (0) |
| Group-size-free IDF cosine (`length_norm` 0) | `IDF_OVERLAP_LENGTH_NORM` | 0.45 / 0.26 / 0.49 (untooled 4 of 16 first) | 0.50 / 0.78 | 1.0 |
| Opt-in priors acceptance model | `ACCEPTANCE_MODEL_PATH=acceptance_model_priors.json` | 0.42 / 0.31 / 0.58, 1 named | 0.51 / 0.67, 10 named | not shipped |
| Priors model + group-size-free IDF | both | 0.40 / 0.31 / 0.58, 0 named | 0.48 / 0.67 | rejected |
| Cross-observation corroboration acceptance (2 observations, floor 0.1; 8.2) | `TECHNIQUE_CORROBORATION_ACCEPT_*` | 0.44 / 0.31 / 0.49 (untooled recall 0.26 -> 0.29) | 0.52 / 0.79 | **on** |

Soft evidence raises the techniques visible to the ranking but not the ranking: what the gate
leaves in review at 0.15-0.27 probability is mostly generic tradecraft that increases overlap
with many groups at once, and restricting it to distinctive techniques (documented for at most
30 groups) changes nothing. Group-size-free IDF helps the untooled real reports but fails the
folds. The priors model raises both precision and recall of the accepted set on real reports
and, under the new fusion, no longer names a wrong group, but it costs the ranking on both sets
and the naming coverage (real reports 3 -> 1); it stays opt-in.

Other default weights include:

| Classifier | Default weight |
|---|---:|
| Specific behavior | 1.50 |
| Software links | 1.20 |
| Group description | 0.40 |
| Strategic context | 0.70 |
| Motivation alignment | 1.50 |
| Semantic centroid | 0.70 |
| Campaign profile | 0.00 |
| CyberRE-LLM | 0.35 |
| Group forest (`profile_forest`) | 0.00 (not adopted, 21.3.11) |
| Source-country forest (`source_country`) | 1.00 (default since 2026-10-01, 21.3.11) |
| Group network (`group_mlp`) | 0.00 (not loaded; fails the naming gate at 1.0, 9.13) |

The effective weights are displayed in Assessment and stored in the fusion result.

### 10.4 Candidate eligibility

Final session fusion restricts candidates to groups appearing in at least one technical-evidence result. Eligibility is recomputed from scratch on every scoring pass (`AttributionEngine._score` clears `context.eligible_group_ids` before running the classifiers), so a positive interview answer can make a previously ineligible group eligible; the pool is never allowed to only shrink. The eligible set applies to fusion, not to the classifiers themselves:

- IDF overlap.
- Weighted Jaccard.
- Specific behavior.
- Campaign overlap.
- Software links.

This prevents context-only or generic description similarity from creating an apparently strong actor candidate without technical support.

### 10.5 Diagnostics

Each `FusionEntry` stores:

- Final rank.
- Raw RRF score.
- Relative ensemble index.
- Per-classifier ranks.
- Minimum and maximum classifier rank.
- Rank standard deviation.
- Contested flag.

Pairwise Kendall tau correlations are computed for active classifier rankings. The UI displays a ranking matrix and correlation matrix.

The relative ensemble index is explicitly not a probability. A small gap between the first and second candidates or a large classifier rank spread produces a warning.

### 10.6 Attribution stage

`apt_classifier/attribution.py` places every result in one of four stages above the
ranking: `INSUFFICIENT_EVIDENCE`, `UNRESOLVED_ACTIVITY_CLUSTER` (with a `CL-UNK-*` id),
`KNOWN_GROUP_SHORTLIST`, or `KNOWN_GROUP_HYPOTHESIS`. The decision uses the deterministic
fusion only and is reproducible across processes.

- Thresholds come from settings (`ATTRIBUTION_MIN_TECHNIQUES`,
  `ATTRIBUTION_MIN_INDEPENDENT_FAMILIES`, `ATTRIBUTION_MIN_MATURITY`,
  `ATTRIBUTION_HYPOTHESIS_MATURITY`, `ATTRIBUTION_HYPOTHESIS_MARGIN`) and are built once
  into `AttributionEngine.attribution_thresholds`.
- `ATTRIBUTION_HYPOTHESIS_MATURITY` defaults to 0.82 (was 0.70, then 0.78). On the ATT&CK
  self-consistency folds 0.78/0.10 named 18 known groups (1 wrong) and 2 of 25 unknown
  actors; on the 55 CTI-HAL real reports the same gate named 7 and 3 were wrong (named risk
  0.43), and the wrong ones had high maturity (0.79-0.80) and wide margins (0.25-0.28), i.e.
  they were confidently wrong. 0.82 is the lowest maturity that is clean on both sets: the
  rerun with the new default names 3 of 3 correctly on CTI-HAL and 15 of 15 on the folds,
  with 1 of 25 unknown actors named (Play, removed from the KB, named as BlackByte), at the
  cost of naming fewer cases. CTI-HAL is therefore development data for the naming gate. The
  margin gate is unchanged (0.10). Since 2026-09-29 the default gate mode is `separation`
  (10.7); this maturity/margin gate remains available as `ATTRIBUTION_GATE_MODE=maturity`.
- **Documented overlap.** `apt_classifier/attack/related.py` derives, from ATT&CK data only,
  which groups cannot be told apart: one group's description references the other (or an
  alias), or they share software used by at most three groups (commodity tooling such as
  Mimikatz or PsExec never counts). The decision carries `related_leading_groups` for the
  leader with the reason, the Assessment view prints it next to the leading candidate, and
  the attribution benchmark reports `cluster_top1` and `named_risk_cluster_aware`, which
  count a related group at rank 1 as correct. Carbanak/FIN7 (description cross-references) and Sandworm Team/APT38 (rarely
  shared software) are the pairs that occur on the benchmarks.
  Since 2026-10-02 names are matched in descriptions as proper nouns (`name_pattern`):
  - a name written in capitals (BITTER, ZINC, APT41) matches only as written;
  - any other name keeps its capital first letter, while the rest may differ in case ("Winnti
    group"). A name that starts lowercase may also start a sentence ("MenuPass").

  Before, the match ignored case, which linked Tonto Team to BITTER through "Operation Bitter
  Biscuit" in Tonto Team's description. Three independent passes checked every place on ATT&CK
  19.2 where a description names another group: 51 places, or 100 name-and-alias hits, because
  ATT&CK repeats each group's name as its first alias. The passes found that link to be the only
  pair without a real reference, so the change removes exactly that pair: 86 related pairs
  remain, 36 of them by description, and every other match is written exactly as the name is.
  Two other places name the Carbanak *malware*, but the same descriptions also name the Carbanak
  group.
  In the published artifacts the pair appears only in `leader_related_groups` of the two seed-7
  folds where Tonto Team or BITTER is both the truth and the leader. That is a display field, and
  no row has the truth and the leader or the top three across the pair, so no metric moves. A
  rerun would clear that field and change the source fingerprint. A rule that skips every match
  after "Operation" was not used, because ATT&CK operation names can be the group's own
  ("Operation Lotus Blossom").
- "Independent evidence families" counts only families whose classifier ranks the
  leading group within the top 10 (`SUPPORT_RANK_LIMIT`); classifiers that rank every
  group (Naive Bayes, group description) do not count as support by mere presence.
- The technique-specificity factor averages over the sorted first ten matched
  techniques, and the cluster id is a hash of the sorted observed technique IDs plus the
  top three candidate IDs, so two unrelated cases with the same leading group get
  different clusters.
- No separate calibration artifact exists. With the learned combiner active (the default, 10.9)
  `is_calibrated` is true and `calibrated_probability` is the combiner's leader probability;
  without it `is_calibrated` is false and the UI shows qualitative maturity only.
- The UI treats "no named group" as a first-class outcome: for every stage below
  `KNOWN_GROUP_HYPOTHESIS` the Assessment view states that no documented group meets the
  evidence bar, labels the leader as the *closest documented group (not an attribution)*,
  and shows the maturity/margin bar a named hypothesis would have to clear.

---

### 10.7 Candidate-relative separation and the separation gate (2026-09-29)

**Why.** The naming gate of 10.6 combined a maturity score with the RRF margin
`1 - rrf2/rrf1`. On the recorded artifacts the margin was the only discriminating factor
(AUC for "the leader is correct" 0.80 on real reports, 0.72 on the folds), maturity without
its margin factor was near random (0.54 / 0.56: evidence volume, independent families and
classifier agreement measure how much evidence there is, not whether it points at the right
group), and the margin itself is a rank-agreement quantity: with `RRF_K=20` a leader ranked
first by every classifier against a runner-up ranked second everywhere is 0.045 apart, below
the 0.10 bar, and classifier score gaps never widen it. 21 of 24 correct real-report leaders
and 63 of 80 correct fold leaders were held back by maturity.

**Model** (`apt_classifier/separation.py`, `discriminate`). For the top-K fused candidates
(`ATTRIBUTION_SEPARATION_TOP_K`, 5) each candidate is credited only for observed evidence
that the other top-K candidates do not explain:

- techniques: `Σ conf · hierarchy factor · x · s` over *accepted* techniques, where
  `x = 1 - (m - 1)/(K - 1)` (m = candidates documenting the technique; exclusive 1, shared
  by all 0) and `s = 1 - groups documenting it / all groups`;
- actor behaviour: per observation, the candidate's best similarity minus the best other
  candidate's (and minus the retrieval threshold), floored at 0, from the same
  `specific_behavior_matches` the actor-behaviour classifier uses;
- software: normalised rarity (tools capped, decisive-malware factor applied) times `x`;
- telemetry-validated negative answers subtract where they discriminate.

`D(g) = w_T·T + w_B·B + w_S·S - w_N·Neg` (weights
`ATTRIBUTION_SEPARATION_{TECHNIQUE,BEHAVIOR,SOFTWARE,NEGATIVE}_WEIGHT`), floored at 0;
`separation = (D(leader) - D(best alternative)) / max(D(leader), D(alternative), floor)` in
[-1, 1], the floor (`ATTRIBUTION_SEPARATION_FLOOR`) keeping "little evidence, no competitor"
from scoring 1. Shared evidence contributes nothing to any side, so documented overlap needs
no special case; the decision records the best-supported alternative and whether it is
documented-related. The score is computed inside the single fusion helper
`AttributionEngine._fuse`, the single fusion path, and stored on the classifier context.

**Gate modes** (`ATTRIBUTION_GATE_MODE`): `maturity` (10.6, unchanged), `separation`
(named iff families >= 2, `separation >= ATTRIBUTION_HYPOTHESIS_SEPARATION`,
`D(leader) >= ATTRIBUTION_HYPOTHESIS_MIN_DISCRIMINATIVE`, optional evidence specificity;
the RRF margin is reported but not required) and `both`. The lower boundaries
(insufficient evidence, unresolved cluster) keep maturity and families. Optionally
`ATTRIBUTION_RERANK_BLEND` > 0 re-orders the top-K by `rrf · (1 + β · D/max(D_max, floor))`
(the top-K set never changes; 0 = pure RRF order). The default 0.5 applies only when the learned
combiner does not run; with the combiner active, the default, the blend is skipped (10.9).

**Choosing bars offline.** The attribution benchmark now records per row the separation,
the leader's evidence specificity, the best alternative, the maturity factors and the raw
per-channel discrimination of the top-K (`topk`). `scripts/attribution_gate_sweep.py`
recombines those channels over a grid of weights, floors and bars in seconds and prints only
rules that name no wrong group on any known arm and at most one removed actor, ordered by
correct names, with the guard band (distance of the highest-scoring wrong leader to the
bar). Rules are chosen by slack, not by knife-edge coverage.

**Measured and adopted (2026-09-29).** Separation AUC for "leader is correct": folds 0.89 (margin
0.72), real reports 0.82 (margin 0.80). The sweep over both artifacts chose behaviour weight
2.0, software weight 1.0, K = 5, `separation >= 0.5`, `D(leader) >= 3.0`, no margin
requirement (guard band 0.135 on both sets). Gate only (ranking untouched): folds 35 named
hypotheses, all correct (was 14), 0 of 25 removed actors named (was 1); real reports 12 named,
all correct (was 3). With the re-ranking blend at 0.5 (0.25 and 1.0 were measured too: 1.0 is
best on the folds, 0.62, but slightly worse on real reports): folds top-1 0.52 -> 0.59,
cluster-aware 0.60, MRR 0.65 -> 0.71, top-5 unchanged; real reports top-1 0.44 ->
0.45, cluster-aware 0.56 -> 0.60, MRR 0.55 -> 0.57; named risk 0 on both, no removed
actor named. `ATTRIBUTION_GATE_MODE=maturity` restores the 10.6 behaviour.

### 10.8 Distinguishing the shortlist (2026-09-29)

A step after the ensemble has picked its shortlist, with three parts.

1. **Contrastive re-ranking over a longer shortlist** (`ATTRIBUTION_RERANK_TOP_K`, default 10).
   The discriminative re-ranking of 10.7 used the same five candidates the naming gate scores.
   The re-ranking now contrasts the top-K of the ensemble order for its own K while the gate keeps
   scoring its five, so the gate's tuned bars and their measured naming behaviour are unchanged
   (`_fuse` computes a second `discriminate` when the two sizes differ; 0 means "same as the gate").
   Adoption numbers are in 21.3.8. Like the blend, it is skipped while the learned combiner runs
   (10.9).
2. **Contrastive near misses** (`ATTRIBUTION_CONTRAST_MIN_PROBABILITY`, default 0 = off).
   NEEDS_REVIEW candidates whose calibrated acceptance probability clears the floor enter a
   separate `soft` channel of the discrimination (`DiscriminationWeights.soft`,
   `ATTRIBUTION_SEPARATION_SOFT_WEIGHT`) with the same exclusivity × specificity weighting as
   accepted techniques, and never reach a classifier. A generic near miss shared by the shortlist
   still counts for nobody; a distinctive one sharpens the separation. Measured and not adopted
   (21.3.8): no ranking gain and one fewer correct name on the folds. The channel is recorded per
   row (`topk[].soft`) so `scripts/attribution_gate_sweep.py --soft-grid` can replay it offline.
3. **Leader consistency guard.** The separation is scored for the ensemble's rank-1 candidate.
   If a re-ranking places another group first, `assess_attribution` refuses to name it on that
   borrowed separation and the rationale says so. With K equal to the gate's this cannot happen
   (a candidate overtakes only when it has more discriminative mass, which makes the scored
   separation negative); with a longer re-ranking shortlist it can, so the guard is what keeps
   the naming gate's promise.

The analyst sees the step as **Distinguish** (web app step 3; `GET /api/cases/{id}/distinguish`,
`CaseService.distinguish`): the technique × shortlist matrix with each technique's exclusivity,
documented specificity and weight (`separation.distinguishing_techniques`), the current
separation and discriminative mass against their bars, and two counterfactual lists produced by
replaying `discriminate` on the decision's shortlist order: pending candidates ranked by expected
change (|Δ separation| × acceptance probability) with confirm/reject in place, and interview
leads with the effect of a yes and of a no, flagging any change that would put another group in
front. Confirming or answering records evidence through the normal paths; nothing in the view
changes weights by itself.

### 10.9 Learned attribution combiner (2026-09-30)

`apt_classifier/combiner.py`. Reciprocal-rank fusion weights every classifier the same way in
every case, but the evidence regime is not the same: on real reports technique recall is low and
the technique-profile classifiers collapse (CTI-HAL top-1: Naive Bayes 0.13, IDF 0.04) while
software (0.86 when it applies) and actor behaviour (0.61 when it applies) decide the answer. At
least one classifier already ranks the true group first in 93% of the ATT&CK folds and 73% of
the real reports against 60% and 51% for the fused ranking, so the combiner is the largest lever.

Feature schema 2 (2026-09-30) adds the random-forest `source_country` vote (9.13) to the
per-classifier features, and the fit's unknown-actor lists refit the country forest without the
removed actor. An artifact lists the features it was fitted on and is scored on those columns, so
a schema-2 artifact (fitted with the country vote at 1.0) and a schema-1 artifact (the original
no-country combiner) both load; a
newer schema, unknown feature names or mismatched vectors are still refused.

- **Model.** A conditional logit (listwise softmax) over the top-N fused candidates (N = 20):
  every candidate gets a utility from *features only*, never from its identity, so a removed or
  never-seen group needs no special case. Fitted by damped Newton's method in numpy, the same
  conventions as the acceptance model (JSON artifact with kind, feature schema version, feature
  names, means/scales, ATT&CK version, pipeline fingerprint; `from_dict` rejects mismatches;
  `load_attribution_combiner` fails closed). An optional **null option** (utility from the
  evidence-regime features, trained on unknown-actor lists) gives P(no documented group).
- **Features** (76 + 8 null in feature schema 2, 73 + 8 in schema 1): per deterministic classifier `present`, `inv_rank`, `score_rel`;
  RRF `rel`, `inv_rank`, `support`, `rank_std`, `contested`; the discrimination channels of 10.7
  (technique, behaviour, software, soft, negative, total relative to the shortlist maximum);
  candidate facts (`matched_share`, `matched_distinctive_share`, `exclusive_software`,
  `log_group_techniques`, `log_prominence`); and interactions of nine channel ranks with three
  regime variables (`distinctive_share`, `software_rarity_max`, `log_accepted`). Regime main
  effects cancel in a softmax and enter only the null utility. Inputs are exactly what RRF
  consumes (`fusion.filter_eligible_results`).
- **Training data** (`scripts/fit_attribution_combiner.py`, `make fit-combiner`): the fold
  machinery of `scripts/attribution_benchmark.py` with training seeds 11, 23, 37 (seed 7 stays
  evaluation-only); per fold one clean list plus five randomised regimes built by re-scoring a
  perturbed context (a 30-80% subset of accepted techniques, false-positive techniques injected
  from an empirical pool to a precision of 30-80%, software dropped with p 0.3, behaviour matches
  thinned with p 0.3); an unknown-actor list for every group. Group-level five-fold
  cross-validation reports out-of-fold top-1/top-5/MRR against the RRF order of the same lists
  and the AUC/Brier of P(leader); L2 is chosen on the mixed MRR; the artifact records the CV.
  CTI-HAL is never training data.
- **Integration** (`AttributionEngine._fuse`): RRF -> discrimination over the top-N -> combiner
  probabilities -> `apply_combiner_order` (top-N reordered by `(1 - blend) * rrf_rel + blend * P`,
  scores mapped into the head's RRF range so the top-N set and the tail never change) ->
  discrimination recomputed for the gate over the **union of the RRF top-K and the combiner
  top-K** with the new leader first (a competitor the re-ordering pushed down must still count as
  shared evidence: with the gate scored on the combiner's own top-5 alone, Carbanak dropped out of
  FIN7's window on one real report and FIN7 was named on evidence the two share) -> stage. The
  discriminative blend of 10.7 is skipped when the combiner runs (its channels are features).
  The shipped blend is 0.5: a full blend (pure combiner order) lost top-5 and MRR on the real
  reports, the half blend keeps the fold gains and restores them (21.3.9). `context.combiner` carries the
  probabilities, the RRF ranks of the shortlist and the status; the decision exposes the leader
  probability as `calibrated_probability` (the naming gate is unchanged). Settings:
  `ATTRIBUTION_COMBINER_ENABLED`, `_PATH`, `_TOP_N`, `_BLEND`. Benchmark rows record `rrf_rank`,
  `combiner_leader_probability`, `combiner_null_probability`, `combiner_status`; the summary adds
  `rrf_top1`, `leader_probability_auc` and top-1 with/without software.
- **Adoption rule.** Group-CV out-of-fold top-1 and MRR at or above RRF on the same lists; the
  seed-7 folds and CTI-HAL (run once on the frozen artifact) with no wrong named group, at most
  1/25 removed actors named and no top-1/MRR regression. Results in 21.3.9.

### 10.10 Distinguishing the top three (2026-10-01)

**Why.** The right group is often in the top three but not first: on the seed-7 folds the truth is
first in 127 of 153 rows and in the top three in 142; on the anonymized CTI-HAL reports first in
25 of 55 and in the top three in 31 (29 after the anonymizer fix later that day, 21.3.14). Two kinds of miss dominate. In the first, the combiner's
leader has clearly less discriminative mass (10.7) than another top-three member. In the second,
the three include a documented-overlap pair (Carbanak/FIN7, FIN6/Cobalt Group) whose evidence
cannot be separated.

**A reorder rule, pre-registered and not adopted** (`apt_classifier/top3.py`, results in 21.3.13).
The rule was written down and committed before it was run on any fresh data:

- Within the first three places, take the member with the largest gate-window D.
- Move it to first place when its D exceeds the leader's by at least 1.25 and the leader's largest
  weighted single-channel advantage over it is at most 0.75.
- A promoted leader would never be named in the same analysis.

On the data it was found on it looked clean: seed-7 folds +5/-0, anonymized CTI-HAL +2/-0. On the
fresh fold seeds 41 and 53 it gave +1/-1 and +1/-0. It fails the pre-registered bar (net +2 per
seed, a slack of 0.1 to the largest gap a correct leader faced), and raising the margin, the only
change the protocol allowed, removes the wins with the losses. Nothing in the engine reorders the
top three. `decide_promotion` stays as a pure function for the replay
(`scripts/top3_replay.py`) and for the view below.

**What did generalise: a warning, not an answer.** When another top-three member holds clearly
more exclusive evidence than the leader, the leader is rarely right: 1 of 8 such rows on the fresh
seeds, against about 87% leader accuracy overall. The exclusive-evidence favourite is right only
2 of 8 times, and in the other 5 the truth is elsewhere. Such a leader always has a negative
separation, so it is never named already. The analyst view reports the state, not a winner.

**Top-three view** (`apt_classifier/top3_view.py`, `CaseService.top3`,
`GET /api/cases/{id}/top3`, the light `top3` summary in every case payload). It is display only:
it reads the decision's recorded gate discrimination, the combiner's raw probabilities and RRF
ranks, and the stage, and the engine never imports it. For each of the first three groups it shows:

- the exclusive evidence by channel (stacked with the gate's weights), the raw combiner
  probability (labelled as fitted on ATT&CK folds, never renormalised over three) and the RRF
  place before the combiner;
- the documented profile size, with a caution for the largest tenth;
- the named software;
- **ATT&CK's own documentation**: the stated origin (with the sentence that states it), the
  threat type from the state-sponsorship and criminal markers (`origin.documented_threat_type`)
  and the motivations the motivation classifier's vocabulary finds. These are labelled "ATT&CK
  documents (not inferred from this case)".

For each pair it shows the separation computed as the gate computes it, the documented-overlap
reason and profile containment, and the accepted techniques, named software and actor-behaviour
wins that only one side documents. A state banner leads, first match wins:

| state | condition |
|---|---|
| named | the decision named the leader |
| overlap | the leader and another top-three group are a documented twin under the cluster scope (`twin_link`, 10.11: by default a same-activity link or nesting profiles, and nesting or a pair separation below 0.5) |
| split | another top-three member leads the leader's D by at least 1.25 (the summary adds when one channel favours the leader) |
| thin | the largest D among the three is at most 1.5 |
| not separated | the leader is ahead but below the naming bar |

The thresholds are display labels found on the development data, not gates. The React app shows
the view as the default "Top three" tab of Distinguish, and the Assessment page shows a badge that
links to it. Streamlit is unchanged.

**Distinguish replays the gate's own window.** `CaseService.distinguish` used to default to the
top 10 in the combined order, while the decision scores the gate window: the combiner's top five
plus the RRF top five, 5 to 8 groups. The two separations therefore differed (one test case:
0.891 in Distinguish, 0.875 in the decision). The default is now the recorded window in its
recorded order, returned as `gate_top_k`, so both pages show the same number. An explicit `top_k`
still works.

**Benchmark instrumentation.** Each attribution benchmark row now records:

- for the first three places, the combiner probability, RRF rank and classifier ranks;
- the accepted techniques with their confidences;
- the gate window's actor-behaviour matches (ids and scores, never text);
- decisive software and negative answers;
- whether the leader-consistency guard fired.

From these inputs `discriminate` replays every recorded channel to within 0.0003 (rounding of
the recorded inputs). Before, the technique channel replayed at 0.77-0.86x and behaviour could
not be replayed at all. The summary adds `top1_given_top3`, `pairwise_12` (the truth first when
it is first or second) and `leader_guard_fired`, and `fingerprint_inputs` records the
discrimination weights. A run refuses to start or to write when the learned combiner is enabled
but did not load. A fingerprint mismatch would otherwise fall back to the RRF order silently
(fold top-1 0.83 -> about 0.60); `--allow-inactive-combiner` measures that fallback on purpose.

### 10.11 Documented clusters, confidence labels and software questions (2026-10-02)

Three additions to the top three, each pre-registered and checked on the fresh fold seeds 43 and
47 (21.3.15). Confidence labels are on by default and software questions ship off as a pilot.
Cluster naming passed with its first rule. It was then narrowed to same-activity links, and the
narrowed rule passed its own pre-registered check on the unused seeds 59 and 61. It is on by
default since 2026-10-02 (21.3.15 A′).

**Documented-overlap clusters** (`apt_classifier/overlap_cluster.py`, stage
`known_cluster_hypothesis`, `ATTRIBUTION_CLUSTER_NAMING`, `ATTRIBUTION_CLUSTER_SCOPE=documented`).
Some ATT&CK groups describe the same activity under two names: Carbanak and FIN7, APT38 and
Lazarus Group. The evidence cannot separate them, so the single-name gate holds back even when the
pair as a whole is clearly ahead. The cluster step names the pair, for example "FIN7 / Carbanak
(documented overlap)":

- **Twin.** A gate-window member that ATT&CK links to the leader as the same activity, and that
  either nests (profile containment >= 0.8) or does not separate from the leader (|pair
  separation| < 0.5). "Same activity" (`twin_link`, scope `documented`) has two routes. In the
  first, ATT&CK's own descriptions document the pair as the same or overlapping activity, as
  recorded in the reviewed table `attack/same_activity.py`. In the second, the profiles nest, the
  review has no verdict against the pair (distinct, unclear or false match), and no unreviewed
  description link could call them distinct.
- **Not twins.** Two groups that only share one rarely used malware family are different actors
  and are never named together: Sandworm Team and APT38 both used KillDisk, Wizard Spider and
  TA505 both used TrickBot. A plain mention is not enough either, because many descriptions name
  another group to say it is different. Examples: Ember Bear and Saint Bear are "distinct
  clusters", APT28 and Sandworm Team are different GRU units, and ShinyHunters and Scattered
  Spider are members of one broader collective. Such pairs still appear in the Top-three view
  ("shares rare malware", "ATT&CK-linked") and in an "ATT&CK-linked groups" note on the
  Assessment page, but never as one cluster.
- **Selection.** The twin with the largest gate-window mass among the qualifying members wins;
  pairs only.
- **The reviewed table (ATT&CK 19.2, 2026-10-02).** At review time 87 group pairs were related:
  37 through a description reference and 50 only through shared rare malware. Since the
  proper-noun matching (10.6), the false match below is gone, leaving 86 and 36. Five independent
  classification passes read the sentences behind the 37 description links and the alias lists,
  and a judge decided the 3 pairs they split on. 33 of the 37 were unanimous. The labels:
  - 13 same activity, which are the only ones that count: FIN7 / Carbanak, Carbanak / Cobalt
    Group, Deep Panda / APT19, Molerats / WIRTE, Lotus Blossom / Thrip, Lazarus Group with APT37,
    APT38, Kimsuky, Andariel and AppleJeus, Winnti Group / APT41, menuPass / MirrorFace, Mustang
    Panda / LuminousMoth;
  - 18 distinct;
  - 5 unclear;
  - 1 false match: "Bitter" inside the name Operation Bitter Biscuit.

  Any verdict against a pair also blocks the nesting route. For example, APT30 / Naikon nest
  fully but "do not appear to be exact matches". A false match is no ATT&CK link at all, and it
  is not shown as one; the proper-noun matching no longer produces it.

  The table fails closed. A description link on a newer ATT&CK version is neither the same
  activity nor a nesting twin until it is reviewed (`unreviewed_description_pairs`). A listed
  pair counts only while ATT&CK still links it.
- **Other scopes.** Scope `full` is the first, pre-registered rule (any relation, shared malware
  included), and `nested` allows nesting pairs only.
- **Evidence.** The gate-window discrimination is recomputed with the pair merged into one unit
  (`unit_totals`), so evidence that only the two members share now counts as exclusive.
  Hierarchy factors take the larger member, behaviour the better member per observation, and
  software counts when either member documents it. Singleton units reproduce the gate's own
  totals to within 0.0003.
- **Naming.** It runs only when the single name failed, or when a single name has a nested twin
  in the window. The leader must pass the lower stage checks and the leader-consistency guard
  must not have fired. Three bars apply: cluster separation >= 0.5, cluster mass >= 3.5 (>= 3.0
  for a nested pair), and the members' combiner probabilities summing to >= 0.8. A single name
  whose nested twin is in the window becomes the cluster when it clears, and is held at shortlist
  otherwise.
- **Evidence-volume guard (2026-10-19).** The three bars were set and validated on cases with 3
  to 23 accepted techniques. A commodity malware write-up that enumerates one family's whole
  capability list (50 to 95 accepted techniques) clears them on mass alone: two giant profiles
  contain most of any long list, the merged mass reaches 20 and the combined probability
  saturates. With `ATTRIBUTION_CLUSTER_MAX_TECHNIQUES` (30) a cluster on more accepted techniques
  than that is held at shortlist unless decisive software supports the leader
  (`decisive_software.factor > 1`, group-exclusive malware named in the report); the held
  candidate records `volume_guard` with the count and the bar. The guard is a hold, never a
  name; the single-name gate is untouched (it named no open-set post). The bar comes from the
  design sets (vendor reports and the design fold seeds, where correct names carry at most 20
  techniques) and is inert on every design and test set except the Elastic open set, where it
  removes both wrongly named commodity posts (21.3.18, "Cluster gate").
- **Scoring.** A benchmark counts a cluster name as correct when the truth is one of its
  members and as wrong otherwise (`AttributionPrediction.named_members`).
  `named_risk_single` is the single-name risk; `named_cluster_recovered` counts clusters whose
  leader alone was wrong.
- **Display.** The decision records `named_cluster` and the cluster's separation, mass,
  probability, reason ("ATT&CK link: ...") and runner-up. The candidate also records its link kind:
  `same_activity`, `nesting`, `description` (a mention that is not the same activity) or
  `shared_software`. Under the default scope only the first two can be named. Report, API,
  Streamlit and React show the pair
  with a "documented overlap" label. The Top-three view calls a pair "documented overlap · one
  cluster" under the same rule and scope as the gate.

The step needs the learned combiner and is off in maturity mode. The settings live outside the
combiner fingerprint, so no refit is needed.

**Calibrated confidence labels** (`apt_classifier/confidence_labels.py`,
`CONFIDENCE_LABELS_ENABLED=true`). Display only: they never change the stage, the ranking or an
export, and the engine never imports the module. Three logistic regressions read what the
decision already records:

- inputs: the combiner probabilities of places 1-3, its "none of these" probability, the gate
  separation, the number of accepted techniques and whether named software supports the leader;
- outputs: P(leader right), P(truth in the top two) and P(truth in the top three).

The Top-three view shows one of three labels:

- "strong single answer" when P(leader right) >= 0.88;
- "one of these 2" or "one of these 3" when the smallest set reaching 0.865 has two or three
  members;
- "open" otherwise, and always at insufficient evidence.

A named cluster is labelled as a two-group set, never "strong single answer". A name from the
gate always takes the headline.

The model is fitted on the design seeds 41 and 53 (`make fit-confidence-labels`), with removed
actors counted as wrong leaders and the thresholds picked on out-of-fold predictions with
one-sided Clopper-Pearson bounds. The artifact is pinned to the ATT&CK version and the exact
combiner artifact; when either differs the labels are hidden. Refit after `make fit-combiner`.
On real reports the labels run overconfident: on anonymized CTI-HAL 21 of 24 "strong" leaders
were right, against 184 of 192 on the fresh folds. The view says so ("Calibrated on ATT&CK test
folds; real reports run overconfident").

**Software questions (pilot, off by default)** (`apt_classifier/software_questions.py`,
`INTERVIEW_SOFTWARE_QUESTIONS=false`). While a case is not named, the interview first asks about
malware that ATT&CK documents for exactly one of the decision's top three. Its users must all sit
inside the gate window, and by default it has only one user (`INTERVIEW_SOFTWARE_MAX_USERS=1`).
The question reads: "Did you find <malware> (<S-id>) or its known artefacts?". The wording never
names a group, and malware named like a group is skipped. Limits: at most 3 questions per case and
one per group, ranked by how much a "found" would change the separation.

The analyst answers:

- **Found** adds the software as evidence.
- **Not found** can only lower the leader's separation (by 0.3 x rarity x exclusivity).
- **Not checked** records nothing.

Two rules keep a name from resting on the interview alone. First, the leader's or cluster's
separation without the interview software must be >= 0. Second, the "not found" filter runs last
in the gate and can only move a hypothesis or cluster back to shortlist.

Each answer stores a pilot record (`metadata["pilot"]`: before/after stage and separation and the
settings in force; no owner, user, e-mail, address or note text). The record is readable per case
(`GET /api/cases/{id}/interview/pilot-log`) and as an admin-only aggregate
(`GET /api/pilot/software-questions.jsonl`).

In the pre-registered simulation the questions add many correct names when analysts answer
reliably. They also add about 2.4 wrong or removed-actor names per fold seed when 5% of "found"
answers are wrong, more than the bar of 1.0 (21.3.15 C). Turning them on is a product-owner
decision for a supervised pilot.

## 11. Origin and intrusion-type assessment

`apt_classifier/origin.py` performs a separate post-fusion assessment.

### 11.1 Country-origin evidence

Country hypotheses are extracted only from documented group or linked-campaign descriptions containing origin-style language. Examples include:

- `China-nexus`.
- `Chinese state-sponsored`.
- `based in Iran`.
- `attributed to Russia`.
- `North Korean state-sponsored`.
- `support of Iranian interests`.

A sentence such as “targeted Chinese government organizations” does not qualify as origin evidence by itself.

Since 2026-09-30 more of ATT&CK's own origin phrasings count, which raises the groups with exactly
one documented source country from 53 to 91 of 176 without losing any:

- a country adjective in the group's own self-description, anchored to the sentence form "<name
  or alias> is a …" (for example "APT1 is a Chinese threat group", "Magic Hound is an
  Iranian-sponsored threat group", "MirrorFace is a People's Republic of China (PRC)-aligned
  cyberespionage actor");
- "operating out of China", "operate from the Guandong Province of China", "likely Chinese
  origins", "a suspected nexus to the Iranian government", "associated with the Chinese
  government".

The anchoring is what keeps victims and other groups out: "SideCopy's name comes from …
Sidewinder, a suspected Indian threat group" does not make SideCopy Indian. Language never counts
("Russian-speaking"), and "South American" is not the United States. Citations are stripped
before matching, and the table is built once per knowledge base in about 0.3 s. The 38 groups this
adds were each checked against their ATT&CK description. Group5 and POLONIUM still state two
countries and stay ambiguous for the origin assessment.

### 11.2 Candidate weighting

For the top fusion candidates, candidate contribution is weighted using both normalized fusion score and rank:

```text
raw candidate weight = (candidate score / leader score)^2 / candidate rank
```

Weights are normalized before country support is aggregated. The result includes:

- Country hypothesis.
- Relative documented support.
- Supporting candidate IDs.
- Supporting description excerpts.
- Origin-bearing candidate coverage.
- Confidence.
- Caveats.

Confidence is reduced or marked insufficient when the ranked candidate set has weak separation, low origin coverage, or no origin evidence.

### 11.3 Intrusion type

The predictor classifies profile evidence into:

- `APT-like state/espionage activity`.
- `APT-like with mixed state and criminal objectives`.
- `Likely financially motivated criminal intrusion`.
- `Indeterminate / mixed`.

APT-like support requires documented state sponsorship, espionage, government backing, military/intelligence links, or similar language. Criminal support requires documented financial motivation, ransomware, fraud, extortion, or e-crime language.

Dual-mission groups can receive the mixed APT-like label when both state/espionage and financial-operation evidence are documented.

The assessment is not a geopolitical attribution oracle. It is a structured summary of public ATT&CK profile language for the current candidate set.

---

## 12. Public live threat intelligence

Live intelligence is deliberately separate from the deterministic attribution score.

### 12.1 Sources

The application uses hardcoded, HTTPS-only public endpoints:

| Source | Endpoint | Classification |
|---|---|---|
| CISA KEV | `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json` | Authoritative active-exploitation catalog |
| CISA Alerts & Advisories | `https://www.cisa.gov/cybersecurity-advisories/all.xml` | Authoritative alerts/advisories and recent campaign reporting |
| ransomware.live | `https://api.ransomware.live/v2/recentvictims` | Unverified extortion-site victim claims |

ThreatFox is not included in the default integration because its current API requires an Auth-Key and its terms distinguish community use from commercial use.

### 12.2 Fetching behavior

The Live Intel view:

- Uses a configurable lookback from 1–90 days.
- Limits records per source.
- Caches snapshots for 1,800 seconds.
- Supports explicit refresh.
- Records per-source health and errors.
- Applies a 20 MB response limit.
- Retries transient request failures once.
- Continues if one source fails.
- Accepts only safe HTTPS links on expected source hosts.
- Stores fetch time and source status.

Each normalized `ThreatIntelItem` includes:

- Stable item ID.
- Source.
- Record kind.
- Title and summary.
- Publication/discovery time.
- Source URL.
- Verification classification.
- Actors.
- Countries.
- Industries.
- CVEs.
- Products.
- Tags.

### 12.3 Record types

Current record kinds include:

- `active_exploitation` for CISA KEV records.
- `campaign_advisory` for campaign-oriented CISA records.
- `security_advisory` for other CISA advisories.
- `claimed_breach` for ransomware.live records.

Ransomware claims are labeled `unverified_extortion_claim`. They are not silently promoted to confirmed incidents.

### 12.4 Case matching

A live record is matched to the current case through explicit overlap:

| Match signal | Relative matching weight |
|---|---:|
| CVE overlap | 5.0 |
| Ranked ATT&CK group name/alias | 4.0 |
| Named software/tool overlap | 3.0 |
| Product overlap | 2.0 |
| Target industry overlap | 1.0 |
| Target geography overlap | 0.5 |

A `ThreatIntelMatch` records:

- Item ID.
- Match score.
- Reasons.
- Matched group IDs.
- Matched observation IDs.
- Verification status.

The matching layer does not alter RRF, Bayes posterior, or classifier output. It is non-voting enrichment intended to help the analyst decide what to investigate next.

### 12.5 Live-intel privacy and bias boundary

Live feed data is not passed into the isolated AI reviewer by default. This prevents recency, source promotion, and unverified victim claims from becoming hidden review priors. The analyst can independently add corroborated evidence through the normal Intake workflow.

---

## 13. Analyst Canvas

`apt_classifier/canvas.py` builds a serializable graph model:

```text
CanvasNode(id, label, kind, score, details)
CanvasEdge(source, target, relationship, weight, details)
AnalystCanvas(nodes, edges)
```

### 13.1 Node layers

The Canvas can display:

- Evidence.
- Strategic context.
- ATT&CK techniques.
- Tools/software.
- Campaigns.
- Live threat intelligence.
- Candidate groups.
- Country hypotheses.
- Threat type assessment.

### 13.2 Connection types

Connections include:

| Relationship | Meaning |
|---|---|
| `maps_to_technique` | An observation produced a technique retrieval match |
| `documented_technique` | ATT&CK documents the technique for the group |
| `specific_behavior` | An observation matched an actor-specific ATT&CK sentence |
| `identified_tool` | A tool was identified in the evidence |
| `documented_tool` | ATT&CK links the tool to the group |
| `campaign_link` | A campaign profile links to the group |
| `context_match` | Supplied strategic context matched group reporting |
| `documented_origin` | Public ATT&CK reporting supports an origin hypothesis |
| `supports_apt_like` | Group profile contains state/espionage evidence |
| `supports_criminal` | Group profile contains criminal-motivation evidence |
| `live_actor_reference` | Live record explicitly references a ranked actor |
| `live_indicator_overlap` | Live record shares a CVE/product/tool signal with evidence |

### 13.3 Canvas layout

Both front ends position layers in the same four columns (Streamlit: a Vega-Lite chart; web app: the layered SVG of `LayeredGraph.tsx`, 4.5):

```text
Column 0: evidence and strategic context
Column 1: techniques, tools, campaigns, live intel
Column 2: candidate groups
Column 3: countries and threat type
```

Edges are drawn between nodes (rules in Vega-Lite, curves in the SVG). Hovering over nodes and edges shows details, supporting text, publication time, verification, and match reasons.

### 13.4 Canvas controls

The analyst can:

- Select candidate count.
- Select evidence count.
- Enable/disable node layers.
- Enable/disable connection types.
- Inspect layer counts.
- View unresolved observations that have no displayed connection.
- Select a candidate for the Hypothesis Inspector.

### 13.5 Hypothesis Inspector

The selected candidate dossier shows:

- Candidate rank and relative index.
- Contested status.
- Per-classifier ranks.
- ATT&CK group description.
- Specific behavior matches.
- Matched techniques and supporting observation IDs.
- Unmatched observed techniques.
- Contradicting negative analyst answers.
- Named tools.
- Country-origin excerpts.
- Structured classifier signals.

The dossier is evidence-oriented and does not convert missing ATT&CK documentation into proof of absence.

---

## 14. Diamond Model intrusion analysis

`apt_classifier/diamond.py` builds a separate four-vertex model from the existing `AnalysisState`:

```text
DiamondNode(id, label, vertex, score, status, details)
DiamondEdge(source, target, relationship, weight, details)
DiamondModel(nodes, edges, caveats)
```

### 14.1 Vertices

| Vertex | Current data source | Interpretation |
|---|---|---|
| Adversary | Top fused group hypotheses | Ranked hypotheses only; not proof of attribution |
| Capability | Observed ATT&CK techniques and named software/tools | Evidence-linked TTP and tooling capability |
| Infrastructure | Explicit IOC observations and matched public-intelligence records | Observed or externally referenced infrastructure, with source/verification status |
| Victim | Analyst-supplied target geography and industry | Victimology only; never treated as attacker origin |

Capabilities connect to adversary hypotheses only when ATT&CK documents the relationship. Victimology connects to an adversary only when the strategic-context classifier matched the corresponding public group reporting. Public-intelligence infrastructure records retain their verification label and remain non-voting enrichment. If victim context is absent, the model displays an explicit unknown victim node rather than inferring one.

### 14.2 UI behavior and limitations

The Diamond Model workflow view provides controls for the number of adversary hypotheses, capability nodes, and infrastructure nodes. It renders a four-column relationship graph, per-vertex tables, relationship details, and caveats. Scores are display-oriented evidence strength or ensemble-relative support; they are not probabilities.

The model does not manufacture missing infrastructure, infer a victim from an IP address, or convert a ranked adversary into an attribution claim. Shared capabilities, reused infrastructure, incomplete ATT&CK reporting, false flags, and unverified public claims can connect unrelated intrusions. Analysts should use the model to structure and challenge hypotheses, then validate relationships with independent telemetry and reporting.

---

## 15. Hypothesis-driven investigation interview

The Interview view is an automatic chat workflow that predicts behaviors which may be present but have not yet been observed. These predictions are collection leads derived from current candidate profiles, not incident findings.

### 15.1 Ensemble hypothesis weights

Question generation uses the top ten fused candidates rather than the standalone Naive Bayes posterior. Each candidate receives a normalized investigation weight:

```text
raw weight(group) = (RRF relative score / 100)²
hypothesis weight(group) = raw weight(group) / Σ raw weights
```

Squaring emphasizes better-ranked candidates while retaining alternatives. The UI labels these as hypothesis weights, not attribution probabilities.

### 15.2 Predicting unobserved behavior

The engine considers ATT&CK techniques documented for the current weighted candidate set and excludes techniques that are already observed or have already been answered. For each remaining technique it computes:

```text
profile support = Σ hypothesis weight(group)
                  for groups that document the technique

P(yes | group) = alpha if group documents technique else beta
model-estimated yes rate = Σ hypothesis weight(group) × P(yes | group)
```

`alpha` and `beta` use the configured Bernoulli values, defaulting to `0.60` and `0.05`. The estimated yes rate is a model-derived prioritization signal and is not a calibrated probability that the behavior occurred.

The interview priority combines likely discovery value and attribution discrimination:

```text
investigation priority = expected information gain + 0.25 × estimated yes rate
```

This allows the interview to ask about both:

- Behaviors likely across the current candidate profiles, even when they do not sharply separate candidates.
- Behaviors that would strongly distinguish competing hypotheses, even when their expected occurrence is lower.

The five highest-priority unobserved leads are shown before the current chat question with technique ID, profile support, number of supporting profiles, estimated yes rate, information gain, and the first collection step.

### 15.3 Investigation guidance

Every lead contains deterministic collection guidance derived from the selected ATT&CK technique:

- Technique-specific Windows channels, Event IDs, fields, and correlation pivots where a reliable mapping exists.
- Relevant platforms, such as Windows, Linux, macOS, network devices, identity systems, or cloud control planes.
- Tactic-specific telemetry recommendations for initial access, execution, persistence, privilege escalation, defense evasion, credential access, discovery, lateral movement, collection, command and control, exfiltration, or impact.
- The first concise sentence from the technique's ATT&CK detection guidance, when available.

Current deterministic Windows playbooks include:

| Behavior family | Primary collection pivots |
|---|---|
| Process and command execution | Security `4688`; Sysmon `1`; full command line, parent process, user, elevation, hashes, signer |
| PowerShell | `Microsoft-Windows-PowerShell/Operational` `4103` and `4104`; Security `4688`; Sysmon `1`; AMSI/Defender/EDR script telemetry |
| Scheduled tasks | Security `4698`–`4702`; `Microsoft-Windows-TaskScheduler/Operational` `106`, `140`, `141`, `200`, `201`; process trees for `schtasks.exe` and task hosts |
| Windows services | System `7045`, `7036`, `7040`; Security `4697`; Security `4688`; Sysmon `1` |
| Registry modification and Run Keys | Sysmon `12`–`14`; Security `4657` where registry auditing is enabled; process and value-data correlation |
| Process injection | Sysmon `8`, `10`, `25`, plus `1` and `7`; EDR cross-process memory and module telemetry |
| LSASS credential access | Sysmon `10`; Security `4688`; Sysmon `1`; EDR process-handle/memory-read and dump-file evidence |
| SAM/NTDS credential access | Security `4656`, `4663`, or domain-controller `4662` where required auditing/SACLs are enabled; process execution and shadow-copy evidence |
| Valid accounts | Security `4624`, `4625`, `4648`, `4672`, `4768`, `4769`, `4776`; identity-provider sign-in, MFA, device, IP/ASN, and token logs |
| Account creation | Security `4720`, `4722`, `4724`, `4728`, `4732`, `4756`; cloud identity and role-assignment audit |
| WMI | `Microsoft-Windows-WMI-Activity/Operational` `5857`–`5861`; Security `4688`; Sysmon `1`; `wmiprvse.exe` child activity |
| Network/C2/DNS | Sysmon `3` and `22`; Security `5156` where enabled; DNS, proxy, firewall, TLS, NDR, prevalence, and beacon timing |
| Log clearing | Security `1102`; System `104`; Security `4688`; Sysmon `1`; `wevtutil` and `Clear-EventLog` execution |
| File deletion | Sysmon `23` and `26`; Security `4663` where object auditing is enabled; EDR, USN journal, and MFT evidence |
| Tool transfer | Sysmon `3`, `11`, `22`; Security `4688`; proxy/web gateway; created-file hash and immediate-execution correlation |
| Archive creation | Security `4688`; Sysmon `1` and `11`; EDR/DLP file-read and archive-to-egress correlation |
| Security-control impairment | Defender Operational `5001` and `5007`; Security `4719`; System `7036` and `7040`; process execution and EDR tamper events |
| Encryption/recovery impact | Security `4663`; Sysmon `1`, `11`, `23`, `26`; backup/storage audit; mass file-write and recovery-tool execution |

Event IDs are investigation pivots, not proof that an ATT&CK technique occurred. They are generated only when the relevant provider, advanced audit policy, SACL, PowerShell logging, Sysmon configuration, or product integration was enabled. The analyst must correlate host, user, logon ID, timestamp, process lineage, command line, destination, file hash, and surrounding behavior. Missing events must not be treated as negative evidence when the source was not collected.

The Windows log upload parser automatically normalizes only the Event IDs listed in the Supported Windows Events section. Additional IDs in this table guide external SIEM/EDR hunting; the analyst can record the result with the interview answer or add a concise finding through Intake.

The configured question model selects one lead from the deterministic shortlist and phrases a neutral yes/no question. The model cannot add a technique outside that shortlist or alter the deterministic collection instructions. Candidate group names are omitted from the question prompt and question text to reduce confirmation bias.

### 15.4 Startup and progression

When the analyst selects Interview after analysis:

1. The engine derives hypothesis weights from the fused ranking.
2. It creates and displays predicted unobserved behavior leads.
3. It attaches deterministic platform, tactic, and ATT&CK detection guidance.
4. The configured question model selects one lead and phrases a neutral question.
5. The UI marks the lead as an unconfirmed investigation hypothesis.
6. The analyst checks the suggested telemetry.
7. The analyst selects a quick reply or types a normalized response.
8. The answer and investigation metadata are added to the evidence ledger.
9. Classifiers, RRF, origin/type hypotheses, and live-intel matches are rescored locally.
10. The next question is selected automatically from the updated candidate set.
11. The process continues until the stopping rule is reached.

There is no normal per-question “Generate next question” click.

With `INTERVIEW_SOFTWARE_QUESTIONS=true` (off by default, a pilot), up to three software questions
come first while the case is not named. Each asks about malware that only one of the top three
documents, with found / not found / not checked answers (10.11).

### 15.5 Answer semantics

The parser accepts variants such as:

- `yes`, `y`, `observed`, `present`, `confirmed`.
- `no`, `n`, `not observed`, `absent`.
- `unknown`, `unsure`, `I don't know`, `not sure`.
- `not visible`, `no visibility`, `cannot see`.

Answer effects are:

| Answer | Scoring effect |
|---|---|
| Yes | Adds the technique as positive evidence with confidence 1.0 and rescales all classifiers. |
| No | Adds explicit negative technique evidence for the Bernoulli model and rescales the ensemble. |
| Unknown | Does not change positive or negative technique evidence; prevents the same question from repeating. |
| Not visible | Does not treat lack of telemetry as negative evidence; prevents the same question from repeating. |

Typed response text is retained in the question-answer record. Unrecognized replies are rejected without changing the ledger.

Each answer also retains the information gain, estimated yes rate, profile support, investigation priority, uncertainty statement, and collection steps that existed when the question was asked.

### 15.6 Stopping rule

The interview stops when any of these conditions applies:

- The configured maximum question count is reached.
- No unanswered candidate technique remains.
- After at least one question, the best lead's investigation priority is below `0.08`.
- After at least one question, a hypothesis already exceeds `0.60` weight and the best remaining lead has both information gain below `0.02` bits and estimated yes rate below `0.20`.

If at least one technique is supported by the current weighted candidates, actionability thresholds cannot suppress the first question. Techniques belonging only to groups outside the weighted candidate set are removed before ranking, so the prediction list cannot be populated with unrelated zero-support ATT&CK entries.

A strong leader alone no longer stops the interview. If the candidate profiles still predict an actionable hidden behavior, the workflow continues so the analyst can seek confirming or disconfirming evidence. `interview_complete` prevents repeated calls after stopping. The state stores a specific stop reason (`question_limit`, `no_candidates`, `low_priority`, `concentrated_low_value`, or `selection_failed`) and the UI reports that reason rather than presenting every stop as a question-limit condition.

The Interview view provides **Restart interview**, which clears interview-derived positive/negative answers and reconstructs the context from the original retrieval result. Interview state also carries a revision number; when question semantics change after an application update, an older in-memory interview is reset once instead of remaining incorrectly completed across a hot reload.

### 15.7 Analytical safeguards

- Predictions are always displayed as unconfirmed leads, never as observed facts.
- Candidate-profile support means ATT&CK documents the technique for weighted candidates; it does not mean the behavior occurred in this intrusion.
- Actor names are withheld from investigation questions to reduce confirmation bias.
- `Unknown` and `Not visible` are not converted into negative evidence.
- Questions cannot introduce techniques outside the deterministic ATT&CK shortlist.
- Collection guidance is derived from ATT&CK metadata and a fixed tactic-to-telemetry map, not invented by the question model.
- The interview is capped and low-priority leads are suppressed to avoid open-ended hypothesis fishing.

### 15.8 Optional Elasticsearch evidence gathering through MCP

The Interview can gather candidate events for the current lead through Elastic Agent Builder's MCP endpoint:

```text
{KIBANA_URL}/api/agent_builder/mcp
{KIBANA_URL}/s/{SPACE_NAME}/api/agent_builder/mcp
```

This is the supported Elastic Agent Builder endpoint available in Elastic Serverless and recent Elastic Stack releases. The application does not target the deprecated standalone `mcp-server-elasticsearch` service.

The connector uses the official Python MCP SDK over Streamable HTTP. It discovers the server's available tools and selects the best supported flow:

**Preferred ES|QL flow** (when `platform.core.generate_esql` and `platform.core.execute_esql` are both available and read-only):

1. Initialize the MCP session and discover tools.
2. If `platform.core.list_indices` is available, call it to discover security-relevant indices and their mappings.
3. Build a natural-language search request from the selected interview technique and deterministic collection instructions.
4. Call `platform.core.generate_esql` to produce a concrete ES|QL query string.
5. Display the generated ES|QL query in the UI for analyst audit.
6. Call `platform.core.execute_esql` with the generated query to retrieve tabular results.
7. Normalize structured tabular results locally.
8. Stage normalized candidates for analyst review (Streamlit session state, or `CaseService` behind the API).
9. Import only records explicitly selected by the analyst, then rerun vector retrieval, all classifiers, RRF, origin/type assessment, and live-intel matching.

**Fallback search flow** (when ES|QL tools are not available but `platform.core.search` is):

1. Initialize the MCP session and discover tools.
2. Reject `platform.core.search` if it explicitly declares itself non-read-only.
3. Build a natural-language search request from the selected interview technique and deterministic collection instructions.
4. Call `platform.core.search` with only the `query` argument; no index argument is supplied.
5. Normalize structured tabular data, Elasticsearch `hits`, event arrays, or bounded text records locally.
6. Stage normalized candidates and proceed as above.

If neither flow's required tools are available, the connector returns a staged error listing the expected tool names.

Because no index is specified, the Elastic API key and Agent Builder space define the complete data scope. This was selected intentionally to let Elastic choose among only the indices authorized to the connection. The key should grant `read` and `view_index_metadata` only on intended security data, plus Kibana `feature_agentBuilder.read`; it should have an expiration date and no write privileges.

The generated search request requires exact available fields such as timestamp, index, document ID, event code, host, user, source/destination, process image, parent process, command line, file path, hash, and original message. It explicitly forbids inference of missing fields, attacker identity, or whether the ATT&CK technique occurred.

#### Manual collection

- **Search this lead** (Streamlit) or `POST /api/cases/{id}/elastic/search` performs an explicit one-time request; the web app has no search button.
- There is no automatic gathering: `ELASTIC_MCP_AUTO_GATHER` is defined in the settings but neither front end reads it.
- A search only gathers and stages candidate records; it does not import evidence or change attribution.
- The analyst must select candidates and click **Import selected and re-rank** (Streamlit) or call `POST …/elastic/import`.

#### Local normalization and privacy

MCP results are converted to `Observation(kind="elastic_mcp")` locally without sending the raw result to OpenRouter. Accepted observations retain:

- `source=elasticsearch_mcp`.
- MCP tool name.
- Query ID.
- Queried technique ID.
- Event ID where present.
- Timestamp.
- Elasticsearch index and document ID where present.
- A bounded raw source excerpt.
- `review_status=analyst_accepted`.

The Elastic API key is a Pydantic secret, used only in the MCP `Authorization: ApiKey ...` header, excluded from batches and observations, redacted from errors, and never sent to Chroma or OpenRouter. Accepted event text becomes normal case evidence and can therefore be included later in a review payload if the analyst explicitly runs the AI review.

#### Failure handling

- Missing credentials disable the connector without affecting the normal interview.
- Non-HTTPS remote endpoints are rejected; HTTP is allowed only for localhost development.
- Redirects are disabled.
- Connection and tool calls use a configurable timeout.
- Result count, fields, text length, raw excerpt length, and query length are bounded.
- Missing or non-read-only search/ES|QL tools produce a staged error rather than changing analysis state.
- Empty results are not interpreted as negative evidence because telemetry may be disabled, incomplete, delayed, or outside retention.

---

## 16. Isolated AI review

The AI review is not part of the deterministic RRF decision. It is an independent critique stage launched from Assessment in Streamlit and from the Review page in the web app.

### 16.1 Four review calls

The pipeline performs four fresh structured calls:

1. Blinded ACH review.
2. Unblinded ACH review.
3. Devil’s-advocate review against the ensemble leader.
4. Advisory reconciliation; it summarizes conflicts and records a proposed review ordering without changing RRF.

No prior LLM message history is passed between calls.

### 16.2 Blinding

The blinded payload removes or replaces identity-bearing terms including:

- Group names.
- Group aliases.
- Campaign names.
- Software names.
- Citation strings and related identity terms.

The reviewer receives structured candidate IDs, evidence, ranking records, and reference data rather than persuasive classifier prose.

The reviewer receives no LLM output. `AttributionEngine.run_review` passes `state.deterministic_fusion`, which is fused from the deterministic classifiers only, together with classifier results that exclude `cyberre_llm`.

Until an analyst runs CyberRE, `state.fusion` and `state.deterministic_fusion` are the same object. After a scan, `state.fusion` carries CyberRE's bounded vote for the analyst-facing ranking. The deterministic fusion is recomputed without the vote (9.12), so the reviewer's candidates and their order are the ones the stage decision reads.

### 16.3 Review payload reduction

The review context intentionally excludes:

- Raw log excerpts.
- Large external-reference objects.
- Full unbounded software records.
- Full fusion entries outside the reviewed candidate set.
- Repeated copies of the same per-classifier evidence.

Technique matches and actor-specific behavior references are retained in compact structured form.

### 16.4 OpenRouter structured response handling

`OpenRouterClient` supports:

- Provider-native parsed fields.
- String JSON.
- JSON content blocks.
- Markdown-fenced JSON.
- Tool-call argument JSON.
- JSON embedded in explanatory text.
- Required top-level type/field checks.
- One automatic structured-response retry.
- Safe diagnostics including stage, finish reason, and content length without printing secrets or response content.

For Anthropic schema-constrained calls, configured `max/xhigh` effort is mapped to medium structured-output reasoning so hidden reasoning cannot consume the entire visible JSON budget.

### 16.4a Local LLM (Ollama), hybrid routing

Added 2026-10-02 and on by default (`LOCAL_LLM_ENABLED=true`, product owner's decision): the
local model is used only when the Ollama server at `LOCAL_LLM_URL` answers with
`LOCAL_LLM_FALLBACK_MODEL` pulled. `OpenRouterClient` routes by model id: ids prefixed `local/`
(for example `TECHNIQUE_VERIFIER_MODEL=local/qwen3.5:4b`) go to Ollama; every other id goes to
OpenRouter. Without an OpenRouter key a remote id is answered by `LOCAL_LLM_FALLBACK_MODEL`, so
with a key nothing changes unless a model id starts with `local/`, and without one the
technique verifier, the prose normaliser, interview phrasing and the review run locally instead
of being switched off. Benchmarks are unaffected: their local pipeline builds no LLM client and
their production pipeline is run with a key. The system endpoint reports `llm_provider`
(`openrouter`/`local`/`hybrid`/`none`), the model answering each task (`llm_models`) and
`prose_normalisation` (`extract`/`sentences`/`split`).

- **Setting it up:** a current Ollama (0.16 cannot load Qwen3.5; 0.35 works), `make llm-pull`
  (pulls `qwen3.5:4b` and registers `qwen3.5:4b-text`, see below), `make llm-check`, then restart
  the app, which checks for the model at start-up.
  A server on another address (for example a user-level install without sudo on port 11435) is
  set with `LOCAL_LLM_URL`. With an OpenRouter key, `EXTRACT_MODEL=local/qwen3.5:4b` moves the
  prose normaliser and interview phrasing to the local model and
  `TECHNIQUE_VERIFIER_MODEL=local/qwen3.5:4b` the verifier. Step by step: README, "Optional local
  LLM (Ollama)".
- **Model choice (research 2026-10-02):** Qwen3.5 (Apache-2.0, March 2026; vendor IFEval 91.5 for
  the 9B, 89.8 for the 4B). Its hybrid attention (Gated DeltaNet:full = 3:1) keeps the KV cache for
  whole reports small. The default `LOCAL_LLM_FALLBACK_MODEL` is the **4B**, the only size that
  fits the 8 GB RTX 4060 Laptop next to the app's own GPU models and a benchmark, as
  `qwen3.5:4b-text`: Ollama's build bundles the vision encoder in the model file and loads it onto
  the GPU although Vestige sends text only, so `scripts/ollama_text_only.py` (run by
  `make llm-pull`) registers a copy without the `v.*` tensors and `qwen35.vision.*` keys, every
  other byte and the Modelfile unchanged. Measured 2026-10-03: identical verifier answers on
  recorded inputs, GPU memory at a 32k context 4,547 -> 3,807 MiB. The 9B (Ollama build 6.1 GiB with its vision projector) loads only when nothing else
  holds GPU memory. Mixture-of-experts models (Qwen3.6-35B-A3B, Gemma 4 26B-A4B) need 17-22 GB.
- **GPU only (`LOCAL_LLM_GPU_ONLY=true`):** every request sends `num_gpu=999`, so Ollama either puts
  every layer on the GPU or refuses to load (measured: a CUDA out-of-memory error, no CPU spill);
  the client reports this once, without retries, as "does not fit in the free GPU memory".
  Without it Ollama silently ran 45% of the 9B on the CPU (one verifier call 13 minutes) and slowed
  the whole machine. `LOCAL_LLM_NUM_THREAD=2` bounds the CPU threads, `LOCAL_LLM_KEEP_ALIVE=5m`
  frees the GPU memory soon after the last call, and `LOCAL_LLM_TIMEOUT=300` ends a stuck call.
- **Context size:** `LOCAL_LLM_NUM_CTX=32768` (whole reports). The cache grows with it, so in
  GPU-only mode a model that does not load can often be made to fit by lowering it.
- **Laptop power cap:** the laptop's power and thermal management can lower the GPU limit from
  55 W to 15 W and hold it in P8 (210 MHz) even under load; the 4B then generates 0.4 tokens/s
  instead of 50. `make llm-check` prints the GPU state and warns when the limit is below half its
  default.
- **Native API:** the client calls Ollama's `/api/chat`, not its OpenAI-compatible `/v1`, because
  only the native API sets `num_ctx` per request; Ollama's default context silently truncates a
  whole report. `format` carries the JSON schema (grammar-constrained decoding), the same parser
  and single retry as OpenRouter apply, and the cost is recorded as 0.
- **Sampling:** Qwen's recommended values (non-thinking 0.7/0.8/20, thinking 1.0/0.95/20).
  `presence_penalty` is not sent, so the model's own Ollama default applies: 1.5 for qwen3.5:4b,
  and every measurement in this section ran with it (an earlier version of this note said the
  penalty was off; Ollama merges the model's defaults into each request). Thinking is on only
  for the efforts in `LOCAL_LLM_THINK_EFFORTS` (`high,xhigh,max`, i.e. the review), with
  `LOCAL_LLM_THINK_TOKENS` added to the output budget. Seeds are always sent (0 when the caller
  passes none), so benchmark runs repeat.
- **Technique verifier, measured 2026-10-02 (4B, thinking off):** with the production prompt the
  4B agreed with almost everything (branch-office: 19 supported, 0 correct; clinic micro-F1 below
  no verifier in every mode). Local models therefore get their own prompt
  (`cyberre._verifier_prompt`): it names the usual false positive (a shared word, tool or
  category) and puts `required_behavior` and `support_span` before `verdict` in the schema, so
  the verdict follows from the evidence. Developed on the branch-office and manufacturing bundles
  (F1 up or equal in all six mode comparisons; one-candidate-per-call and thinking were slower
  and not better; thinking took 2.5-4 min per call), then confirmed once on the held-out clinic
  bundle, micro-F1 no verifier -> 4B -> GPT-5.6: narrative 0.357 -> 0.378 -> 0.526, windows
  0.600 -> 0.571 -> 0.571, ecar 0.545 -> 0.571 -> 0.615. Most remaining false accepts are
  interpreter techniques (T1059.x) for commands whose labelled purpose is something else.
- **Verifier confidence and interpreter rule, measured 2026-10-03 (4B text-only, thinking off):**
  the product owner chose the cheap levers over trying other models (a researched shortlist of
  Qwen3.5-9B at 3 bits, Granite 4.2 3B and Qwen3-4B-Instruct-2507 was not run).
  - *Confidence.* Local verifier calls request the top-20 token alternatives; at each verdict
    `_verdict_confidences` renormalises P("supported") over the tokens that can start an allowed
    verdict (Ollama reports the distribution before the JSON grammar, where "No" or "false" lead)
    and stores it in `TechniqueVerification.confidence`. A local "supported" below
    `LOCAL_LLM_VERIFIER_MIN_CONFIDENCE` (0.98) becomes "uncertain" and stays for the analyst;
    OpenRouter verdicts carry no probability and are never gated. The threshold was fitted by a
    rule written before the results were read (best mean F1 over the six dev bundle/modes, ties
    to the lower threshold, at least as many wrong as correct accepts removed). It is fragile: the
    model is confident even when wrong (wrong accepts at 0.89-0.999), and the fit gave 0.999 for
    the plain prompt but 0.98 with the rule, where 0.999 would cost F1.
  - *Interpreter rule.* For log-derived observations (provenance `winlog*`, `ecar*`, `sysmon*`,
    `elasticsearch*`, `evidenceforge_*`) the local prompt adds: map what a wrapped command is
    for, and support T1059.x only when the interpreter itself is the point (a script, an
    encoded or obfuscated command, a chain of commands). Dev bundles (branch-office +
    manufacturing, 1,108 candidates): correct accepts 6 -> 13, wrong 11 -> 15, F1 up in all six
    bundle/modes (mean 0.516 -> 0.638; 0.652 with the threshold); it worked by supporting the
    purpose technique, not by dropping T1059 (4 wrong T1059 accepts either way). On report prose
    it backfired (80 CTI-HAL statements with a T1059 candidate, 1,330 candidates): wrong accepts
    10 -> 24 with 7 correct either way, so prose, alerts and interview answers keep the plain
    prompt. The threshold helped on prose too (plain prompt: precision 0.41 -> 0.46 at 0.98).
    The 4B verifier recovers few labelled techniques from prose (7 of 78).
  - *Held-out clinic, run once with the final setup*, micro-F1 no verifier -> 4B before -> 4B
    final -> GPT-5.6: narrative 0.357 -> 0.378 -> **0.556** -> 0.526, windows 0.600 -> 0.571 ->
    **0.462** -> 0.571, ecar 0.545 -> 0.571 -> **0.643** -> 0.615; mean 0.501 -> 0.507 -> 0.554 ->
    0.571. Narrative and ecar now match or beat GPT-5.6; windows regressed (three wrong accepts,
    none correct, on a small mode).
- **Prose normaliser, measured 2026-10-02 (CTI-HAL statements joined into prose, deterministic
  mapping, no verifier):** the 4B rewriting whole reports with the production extraction prompt
  lowered technique F1 from 0.431 to 0.370 on 14 reports because it leaves actions out. Splitting
  first and letting the 4B rewrite each sentence 1:1 (thinking off) gained on those 14 reports
  (0.431 -> 0.503), not on the other 41 (0.470 -> 0.460), and over all 55 F1 0.459 -> 0.472
  (precision 0.552 -> 0.564, recall 0.393 -> 0.405; 22 reports better, 20 worse). The product
  owner adopted it as the local prose normaliser: `prose_normalisation` sends prose to OpenRouter's
  whole-report extraction when OpenRouter answers the extract model, to
  `evidence/standardise.py` when a local model does, and to the plain split without an LLM. The
  split decides coverage: an omitted sentence, a rewrite under three words ("None.") and a failed
  call keep the split sentence; a rewritten observation keeps its line provenance and the
  original sentence in `provenance.metadata["original_sentence"]`; at most 60 sentences go into
  one call. Alerts stay on the plain split (not measured), and the case diagnostics record
  `normalization: local_llm_sentences` with rewritten/kept/failed counts.
- **Availability:** `llm_available` needs an OpenRouter key or the local server answering
  `/api/tags` with the fallback model pulled (cached 30 s per URL), so an old Ollama without the
  model leaves the app in local mode. The API decides once at start-up. `make api-local` /
  `make run-local` force no-LLM mode, and the test suite switches the local LLM off in
  `tests/conftest.py`.

### 16.5 Review limitations

The review can critique evidence and ranking but cannot add reliable evidence. A reviewer’s confidence is not an attribution probability. The deterministic RRF ranking remains authoritative; reconciliation stores any alternative ordering under `advisory_ranking` and records differences under `rank_changes` without applying them. The report retains the counter-case, limitations, disconfirmation tests, and fame-bias measurements.

---

## 17. Data and state model

### 17.1 `EvidenceLedger`

```text
observations: list[Observation]
technique_matches: list[TechniqueMatch]
question_answers: list[QuestionAnswer]
strategic_context: StrategicContext
case_id: optional string
metadata: dictionary
```

### 17.2 `AnalysisState`

```text
ledger: EvidenceLedger
retrieval: RetrievalResult
context: ClassifierContext
classifier_results: tuple[ClassifierResult, ...]
fusion: FusionResult
deterministic_fusion: optional FusionResult
bayes_posterior: dict[group_id, probability]
origin_assessment: OriginAndThreatAssessment
weights: dict[classifier, weight]
current_question: optional SelectedQuestion
interview_complete: bool
interview_stop_reason: optional string
interview_revision: int
threat_intel_snapshot: optional ThreatIntelSnapshot
threat_intel_matches: tuple[ThreatIntelMatch, ...]
cyberre_results: map[observation_id, CyberREResult]
technique_verifications: map[key, TechniqueCandidateVerificationResult]
review: optional ReviewOutcome
telemetry_profile: optional
attribution_decision: optional AttributionDecision
```

### 17.3 `ClassifierResult`

A classifier result contains:

- Classifier name.
- Applicability flag.
- Ordered `ClassifierEvidence` entries.
- Input count.

Each entry stores:

- Group ID.
- Score.
- Rank.
- Matched technique contributions.
- Unmatched observed techniques.
- Signal IDs.
- Applicability.

### 17.4 Audit representation

Audit records are JSON serializations of prompts, model responses, parsed stages, usage, fusion, ledger, and review result. Secrets are not serialized. Compact review payload logic avoids copying raw log excerpts into reviewer prompts.

---

## 18. Configuration

Settings are read from `.env`; `.env.example` documents the available variables.

### 18.1 Models and Chroma

```text
OPENROUTER_API_KEY=
CHROMA_HOST=localhost
CHROMA_PORT=8000
CHROMA_SSL=false
EMBEDDING_MODEL=basel/ATTACK-BERT
EMBEDDING_MODEL_REVISION=81e30f983a822c606825507b63ae318a6830a8a2
EXTRACT_MODEL=openai/gpt-5.6-sol
REVIEW_MODEL=anthropic/claude-sonnet-5
```

### 18.2 Retrieval and technique mapping

```text
SIMILARITY_THRESHOLD=0.45
RETRIEVAL_TOP_K=30
BEHAVIOR_SIMILARITY_THRESHOLD=0.55
BEHAVIOR_RETRIEVAL_TOP_K=20
TECHNIQUE_MAPPING_PRESET=balanced
TECHNIQUE_RETRIEVAL_MODE=hybrid
TECHNIQUE_CANDIDATE_POOL=12
TECHNIQUE_INCLUDE_DETECTION=false
TECHNIQUE_PARENT_POLICY=ancestor_compatible
TECHNIQUE_ACCEPTANCE_THRESHOLD=0.75
TECHNIQUE_RELATIVE_MARGIN=0.08
ACCEPTANCE_MODEL_ENABLED=true
ACCEPTANCE_MODEL_PATH=data/models/acceptance/acceptance_model.json
TECHNIQUE_CORROBORATION_BONUS=0.0
TECHNIQUE_INCLUDE_SYNTHETIC=false
TECHNIQUE_SEGMENT_ACCEPTANCE=false
TECHNIQUE_HIERARCHY_PARENT_MODE=off
TECHNIQUE_RANK_BY_ACCEPTANCE=true
TECHNIQUE_EXEMPLARS_ENABLED=true
TECHNIQUE_PLATFORM_FROM_TEXT=true
TECHNIQUE_CORROBORATION_ACCEPT_OBSERVATIONS=2
TECHNIQUE_CORROBORATION_ACCEPT_MIN_PROBABILITY=0.1
TECHNIQUE_VERIFIER_MODE=ambiguous
TECHNIQUE_VERIFIER_MODEL=openai/gpt-5.6-sol
TECHNIQUE_VERIFIER_EFFORT=low
RERANKER_ENABLED=false
RERANKER_MODEL=
RERANKER_MODEL_REVISION=
RERANKER_MAX_CANDIDATES=20
TECHNIQUE_HEAD_ENABLED=true
TECHNIQUE_HEAD_URL=http://localhost:8102
TECHNIQUE_HEAD_EXPECTED_FINGERPRINT=5b5d015d8766dd27cdeceaf0ea1aaa0ea7c613f397d2bf0fecb07eea41fcea01
TECHNIQUE_HEAD_TOP_FAMILIES=3
TECHNIQUE_HEAD_NEW_CANDIDATES=3
TECHNIQUE_HEAD_TIMEOUT=120
ACCEPTANCE_MODEL_HEAD_PATH=data/models/acceptance/acceptance_model_technique_head.json
```

The `TECHNIQUE_HEAD_*` settings drive the fine-tuned technique head's local service (8.2,
21.3.18): the URL of `make technique-head-serve`, the checkpoint fingerprint the client requires
(empty accepts any), how many of the head's top families propose a candidate and how many
proposals may join the pool. `ACCEPTANCE_MODEL_HEAD_PATH` is the acceptance model fitted with the
head's features; it is read only while the head is active and then judges the observations the
head scored, `ACCEPTANCE_MODEL_PATH` the rest.

### 18.3 Interview and fusion

```text
MAX_QUESTIONS=10
RRF_K=20
TECHNIQUE_RRF_K=60
TECHNIQUE_MAPPING_PRESET=balanced
TECHNIQUE_FAMILY_WEIGHT=2.00
CONTEXT_FAMILY_WEIGHT=1.00
MOTIVATION_FAMILY_WEIGHT=1.50
ACTOR_BEHAVIOR_FAMILY_WEIGHT=1.50
SOFTWARE_FAMILY_WEIGHT=1.20
CAMPAIGN_FAMILY_WEIGHT=1.00
SPECIFIC_BEHAVIOR_WEIGHT=1.50
CAMPAIGN_WEIGHT=0.00
GROUP_DESCRIPTION_WEIGHT=0.40
SEMANTIC_CENTROID_WEIGHT=0.70
STRATEGIC_CONTEXT_WEIGHT=0.70
MOTIVATION_WEIGHT=1.50
SOFTWARE_WEIGHT=1.20
SOFTWARE_IDF_ENABLED=true
SOFTWARE_TOOL_WEIGHT_CAP=1.5
SOFTWARE_DECISIVE_BOOST=2.0
NAIVE_BAYES_WEIGHT=1.00
IDF_OVERLAP_WEIGHT=1.00
JACCARD_WEIGHT=1.00
IDF_OVERLAP_LENGTH_NORM=1.0
ATTRIBUTION_SOFT_EVIDENCE_MIN_PROBABILITY=0.0
ATTRIBUTION_SOFT_EVIDENCE_WEIGHT=1.0
ATTRIBUTION_SOFT_EVIDENCE_MAX_GROUPS=0
ATTRIBUTION_HYPOTHESIS_MATURITY=0.82
ATTRIBUTION_HYPOTHESIS_MARGIN=0.10
ATTRIBUTION_GATE_MODE=separation
ATTRIBUTION_SEPARATION_TOP_K=5
ATTRIBUTION_SEPARATION_TECHNIQUE_WEIGHT=1.0
ATTRIBUTION_SEPARATION_BEHAVIOR_WEIGHT=2.0
ATTRIBUTION_SEPARATION_SOFTWARE_WEIGHT=1.0
ATTRIBUTION_SEPARATION_NEGATIVE_WEIGHT=0.3
ATTRIBUTION_SEPARATION_FLOOR=1.0
ATTRIBUTION_HYPOTHESIS_SEPARATION=0.5
ATTRIBUTION_HYPOTHESIS_MIN_DISCRIMINATIVE=3.0
ATTRIBUTION_HYPOTHESIS_SPECIFICITY=0.0
ATTRIBUTION_RERANK_BLEND=0.5
ATTRIBUTION_RERANK_TOP_K=10
ATTRIBUTION_COMBINER_ENABLED=true
ATTRIBUTION_COMBINER_BLEND=0.5
PROFILE_FOREST_WEIGHT=0.0
SOURCE_COUNTRY_WEIGHT=1.0
GROUP_MLP_WEIGHT=0.0
NB_ALPHA=0.60
NB_BETA=0.05
NEGATIVE_EVIDENCE_WEIGHT=0.30
POSTERIOR_TEMPERATURE=2.0
```

### 18.4 Live intelligence

```text
THREAT_INTEL_DAYS=30
THREAT_INTEL_TIMEOUT=20.0
THREAT_INTEL_MAX_ITEMS_PER_SOURCE=100
INCLUDE_RANSOMWARE_CLAIMS=true
```

### 18.5 Elastic Agent Builder MCP

```text
ELASTIC_MCP_URL=
ELASTIC_MCP_API_KEY=
ELASTIC_MCP_TIMEOUT=60.0
ELASTIC_MCP_LOOKBACK_HOURS=72
ELASTIC_MCP_MAX_RESULTS=20
ELASTIC_MCP_AUTO_GATHER=false
```

`ELASTIC_MCP_URL` must end with `/api/agent_builder/mcp`, including `/s/{SPACE_NAME}` before that suffix for a custom Kibana space. Remote endpoints must use HTTPS. `ELASTIC_MCP_API_KEY` is stored as `SecretStr`. `ELASTIC_MCP_AUTO_GATHER` is not read by either front end; every search is started by the analyst (15.8).

`CAMPAIGN_WEIGHT=0.00` means campaign results remain inspectable but do not vote in default RRF. The Live Intel layer is non-voting regardless of feed matches.

---

## 19. Deployment and operations

### 19.1 Environment setup

The project uses a local `.venv` and the pinned requirements file.

```bash
make setup
```

The current Makefile creates the environment with `--system-site-packages`. This can expose unrelated host-package diagnostics through `pip check`; use a clean environment without inherited system packages when strict isolation is required.

### 19.2 Start ChromaDB

```bash
make up
```

Verify the service:

```bash
curl http://127.0.0.1:8000/api/v2/heartbeat
```

Expected response is a JSON heartbeat object.

### 19.3 Ingest or rebuild the ATT&CK index

```bash
make ingest
```

This downloads or reuses the configured ATT&CK bundle, parses group-specific behavior descriptions, saves the KB, and upserts explicit embeddings into Chroma.

Current ingestion output includes:

```text
techniques=697
 groups=176
campaigns=56
software=825
sentences=11614
group_documents=5394
actor_behaviors=5042
```

### 19.4 Start the web app (or Streamlit)

The primary interface is the React app (4.5). After `make web-install`, `make run-next` builds `web/` and serves it with the API at `http://127.0.0.1:8100`; for development run `make api` plus `make web-dev` (Vite on 5173), and `make api-local` runs without OpenRouter. The Streamlit app remains runnable:

```bash
make run
```

The Makefile binds Streamlit to localhost:

```text
http://127.0.0.1:8501
```

The app should not be exposed directly to an untrusted network without an appropriate authenticated reverse proxy and organizational data-handling review.

### 19.5 Live Intel operation

1. Complete an analysis in Intake.
2. Open **Live Intel**.
3. Select the lookback window.
4. Decide whether to include ransomware victim claims.
5. Click **Refresh feeds** when a fresh snapshot is needed.
6. Inspect source health and verification labels.
7. Review current-case matches.
8. Open Canvas to see matched live records connected to evidence or candidate groups.

A source failure does not discard healthy source records. The source-status table identifies unavailable feeds.

### 19.6 Streamlit health

```bash
curl http://127.0.0.1:8501/_stcore/health
```

Expected response:

```text
ok
```

---

## 20. Security and privacy

### 20.1 Secret handling

- `OPENROUTER_API_KEY`, `ELASTIC_MCP_API_KEY`, `GOOGLE_CLIENT_SECRET` and `AUTH_SECRET_KEY` are loaded as Pydantic `SecretStr` values.
- `.env` is ignored by Git.
- `.env` should be mode `600`.
- The key must not be printed, placed in an audit, or uploaded to a remote secrets service without explicit authorization.
- Error messages avoid dumping provider payloads that could contain evidence or secrets.

### 20.2 External processing boundaries

The following may be sent to OpenRouter when used:

- Free-text observations.
- EDR/SIEM alert prose.
- Compact normalized review payloads.
- Adaptive question context.
- Review prompts and structured results.

The following are processed locally:

- Windows/Sysmon log parsing.
- IOC extraction.
- Software alias resolution.
- ATT&CK parsing.
- Embeddings.
- Chroma queries.
- Deterministic classifier scoring.
- RRF fusion.
- Origin/type aggregation.
- Live feed fetching and matching.
- Elastic MCP transport, search-result parsing, staging, and analyst selection.

Elastic MCP search requests and results travel between the application and the configured Kibana endpoint. They are not sent to OpenRouter during gathering or local normalization. After an analyst imports a result, its normalized text becomes case evidence and may be sent to OpenRouter later if extraction-dependent workflows or AI review are explicitly run.

The review payload excludes bulk raw log excerpts and large citation objects. Organizational policy should determine whether any incident text may be sent to a third-party model provider.

### 20.3 Public-feed trust boundaries

CISA data is labeled authoritative for what CISA reports; this does not mean every CISA record proves a specific actor was responsible for a case.

ransomware.live records are public operator or extortion-site claims. They may be false, incomplete, duplicated, stale, or misdated. They are never converted into `confirmed breach` status by this application.

### 20.4 Input and prompt considerations

Incident text and live-feed summaries are untrusted data. Structured schemas, bounded payloads, explicit evidence instructions, candidate allowlists, and fresh review sessions reduce prompt-injection and anchoring risks. They do not eliminate them.

---

## 21. Testing and verification

### 21.1 Standard checks

```bash
make test
make lint
make web-check
.venv/bin/python -m compileall -q apt_classifier api app scripts tests
docker compose config --quiet
```

`make test` runs the suite (about 460 test functions on 2026-10-01; `gates.pytest` in `data/audits/verification_results_current.json` records the return code and run time). Coverage includes:

- ATT&CK parsing and relationship resolution.
- Revoked/deprecated object filtering.
- Actor behavior extraction.
- Vector document determinism.
- Platform-aware retrieval.
- Evidence aggregation.
- Classifier abstention.
- Campaign partial abstention.
- Specific behavior discrimination.
- Strategic context normalization and ranking.
- Country-origin extraction and target-country exclusion.
- APT-like/criminal/indeterminate classification.
- RRF eligibility and family budgets.
- Live-feed parser normalization.
- Partial feed failures.
- Non-voting live-intel case matching.
- Canvas graph relationships and dossier gaps.
- Review payload reduction and identity blinding.
- OpenRouter structured-response variants and retry handling.
- Automatic interview answer parsing and stopping state.
- Deterministic PowerShell and scheduled-task channel/Event ID guidance.
- Elastic MCP endpoint validation, read-only tool enforcement, ES|QL generation/execution flow, search fallback, bounded query planning, structured-result normalization, result caps, secret redaction, and accepted-evidence reanalysis.
- Benchmark tooling: stratified case loading, leaky-split refusal, indexed-text filtering, regression comparison, verification-artifact composition, and the `cues_enabled=False` retrieval path.

### 21.2 Live feed smoke test

A seven-day public-feed smoke test returned records from all configured sources:

| Source | Smoke-test records |
|---|---:|
| CISA KEV | 10 |
| CISA Alerts & Advisories | 10 |
| ransomware.live | 10 |

Feed records are capped for the smoke test and should not be interpreted as a complete current-threat census.

### 21.3 Retrieval and ensemble benchmarks

The repository includes benchmark scripts under `scripts/` for the held-out public
technique-mapping gate, curated retrieval, OTRF-style retrieval, EvidenceForge synthetic
bundles, held-out ATT&CK group self-consistency, attribution (text in, group out) on the ATT&CK
folds, and CTI-HAL real-report mapping and attribution. All local results are composed into
`data/audits/verification_results_current.json` by `scripts/collect_verification.py`
(`make verify-results`); that file is never edited by hand and records the path, mtime and
hash of every source artifact.

Important interpretation:

- The **public held-out benchmark** is the primary gate for changes to retrieval, gating
  and reranking. It is the only benchmark with enough cases (about 1,400) for confidence
  intervals narrower than the effects being measured.
- The curated and OTRF sets are development diagnostics. The regex cue tables
  (`_EXPLICIT_TECHNIQUE_CUES`, `_BENIGN_INDICATORS`, `technique_signals.SIGNALS`) were
  written against them, so their numbers are reported with and without cues.
- Held-out group evaluation tests reproduction of the ATT&CK documentation table. It is
  not a real-world actor-attribution benchmark, and high self-consistency can coexist with
  poor field attribution because ATT&CK group technique data is presence-only and
  reporting-biased.
- The attribution benchmarks (real reports in 21.3.4b, ATT&CK folds in 21.3.5) measure group
  ranking and the naming gate; the other benchmarks below measure technique mapping.

#### 21.3.1 Public held-out benchmark

`scripts/benchmark_mapping.py` (`make benchmark-public`) runs the `tram`, `expert` and
`derived_procedures` test splits of the CC-BY-4.0 `tumeteor/mitre-ttp-mapping` corpus
(prepared by `scripts/prepare_public_benchmarks.py`, pinned by blob SHA in
`data/benchmarks/public/manifest.json`) through `retrieve_techniques` with the production
settings and no LLM verifier.

- The `procedures` split is refused by default: about 22% of its texts are byte-identical
  to the ATT&CK procedure sentences embedded in the technique index. Any case in any
  split whose normalized text equals an indexed procedure sentence is dropped and counted.
- Sampling, when requested, is stratified per dataset with a fixed seed; the script never
  takes the first N rows.
- The artifact stores per-dataset and pooled metrics with case-level bootstrap 95%
  intervals, a per-case row (text, expected, top-10 ranking, accepted set and the state,
  reason and evidence scores of each candidate) for error analysis, a case-set hash, and a
  pipeline fingerprint that covers the ATT&CK version, embedding model, retrieval settings
  and the source of `retrieve.py`, `technique_signals.py`, `presets.py` and
  `vectorstore.py`.
- `--ablate-cues` runs every mode twice, with and without the regex cue tables.
- `scripts/check_regression.py` (`make check-regression`) fails when pooled micro-F1 drops
  by more than 0.02 or hit@10 by more than 0.03 against
  `data/audits/public_heldout_baseline.json`. It displays parent-aware F1 but never gates
  on it.
- Parent-aware precision, recall and F1 give one-to-one credit when an accepted id equals
  an expected id or is its parent or sub-technique, counted as a maximum matching between
  the accepted and expected sets of each case. Before the fix on 2026-10-02 (artifacts up to
  and including that day's runs) the pairing was first-fit over Python sets, so the same
  predictions could score differently under another `PYTHONHASHSEED`. On the current public artifact only the `expert` slice with cues moved
  (parent-aware precision 0.7265 to 0.7350, F1 0.2665 to 0.2696, pooled F1 0.4505 to
  0.4511). The maximum matching always gives the top of that range, which is also the value
  the published artifacts hold, so no published number changes.

Label characteristics: `tram` rows are single sentences (median 15 words, about 1.1
labels per case), `expert` rows are report paragraphs (median 62 words, about 3.3 labels
per case), `derived_procedures` rows are paraphrased report paragraphs (median 49 words).
All test labels occur in the training splits, which makes supervised reranking feasible.

#### 21.3.2 Curated retrieval benchmark

`scripts/curated_retrieval_benchmark.py` runs 36 analyst-style observations (30 positive
including multi-label and hierarchy cases, 6 hard negatives) through the live ChromaDB
technique index. Top-1/3/10 are computed over all visible candidates (accepted and
needs-review), so they describe retrieval order; the `accepted_set` block describes the
authoritative AUTO_ACCEPTED mappings. Hard negatives are judged by `accepted_fp_cases`: a
hard negative that accepts any technique is a false-positive case. Candidates remain
visible for review on hard negatives by design, so the older "abstention" wording (which
counted cases with no candidate at all) is no longer used.

#### 21.3.3 OTRF Sysmon benchmark

`scripts/otrf_benchmark.py` runs 9 OTRF-style Sysmon log narratives through the live index
and reports the rank of the expected technique. Every expected technique in this set has a
dedicated regex cue, so 9/9 top-1 measures cue firing as much as retrieval.

#### 21.3.4 EvidenceForge synthetic bundles

`scripts/evidenceforge_eval.py` (`make evidenceforge-eval-all`) evaluates three
EvidenceForge bundles; see `docs/EVIDENCEFORGE_EVALUATION.md`. Only `heldout-clinic` is
untouched by tuning. Case counts per mode are small (2 to 21).

#### 21.3.4b CTI-HAL real-report benchmarks

CTI-HAL (dessertlab, 2025) pairs human-annotated statements from real threat reports with
ATT&CK techniques and, through its layout, with the reporting subject (six groups with
data: Carbanak, FIN6, FIN7, OilRig, Sandworm, Wizard Spider; the APT29 folder is empty).
`scripts/prepare_cti_hal.py` converts a clone into `data/benchmarks/cti_hal/` (968
statement-level cases, 55 reports; labels use the annotated sub-technique when present).
The repository publishes no licence, so the derived files are local evaluation data and are
not redistributed. Two benchmarks use them:

- statement-level technique mapping through `benchmark_mapping.py --root
  data/benchmarks/cti_hal --datasets cti_hal` (report sentences, none indexed);
- real-report attribution through `attribution_benchmark.py --cti-hal`: each report's
  annotated statements are the observations, the expected answer is the report's group,
  nothing is hidden from the store (the group's documented behaviour is legitimate evidence
  for a real report), and there is no unknown-actor arm. Since 2026-09-30 the headline
  replaces every ATT&CK group name and alias in the statements with "the adversary"
  (`--anonymize-actors`, below).

**Actor names in the statements (measured 2026-09-30).** Until then the real-report run read
each report's statements as reported (the first 30 with at least three words). In 27 of the 55
reports they name the report's own group or one of its ATT&CK aliases: 92 names in 87 of 846
statements (FIN6 32, FIN7 24, Carbanak 23, APT34 4, OilRig 4, Anunak 2, ITG08 2, UNC1878 1).
Name-sensitive signals can then score by reading the label instead of the tradecraft.
`--anonymize-actors` applies the held-out folds' replacement (`_actor_name_pattern` and
`_sanitize_procedure` over all 599 group names and aliases, each replaced by "the adversary")
after the statements are selected; a statement without a name stays byte-identical, so both
runs read the same statements. The artifact records `parameters.actor_names` (`as_reported` or
`anonymized`), the totals under `leak_control.actor_name_replacement` and a per-report
`actor_name_replacement` record on each row. Eighteen ATT&CK 19.2 group names or aliases are
also software names (for example Carbanak and Anunak for S0030, Sofacy, Snake, Machete, Play).
On CTI-HAL only Carbanak and Anunak occur, and all 25 of those mentions refer to the malware, so
the replacement also removes 25 software matches and drops S0030 from the detected software of
six reports.

| CTI-HAL, 55 reports, local pipeline | Top-1 [95% CI] | Top-5 | MRR | Cluster top-1 | Named (wrong) |
|---|---|---|---|---|---|
| statements as reported (headline until 2026-09-30) | 0.55 [0.42, 0.69] | 0.75 | 0.63 | 0.67 | 13 (0) |
| **`--anonymize-actors` (headline; 2026-09-30 figures, current in 21.3.14)** | **0.44** [0.31, 0.58] | 0.65 | 0.53 | 0.55 | 8 (0) |

The 28 reports without a name give identical results in both runs. On the 27 that name their
group, top-1 falls from 16 to 10, MRR from 0.71 to 0.50 and named hypotheses from 7 to 2; six
reports lose the correct top answer and none gains one (two-sided sign test p = 0.03). Neither
run names a wrong group. Per classifier (truth rank from `classifier_truth_ranks`: top-1 count out
of 55 / MRR; a classifier that does not apply counts as a miss):

| Classifier | As reported | Anonymized | MRR on the 27 name-bearing reports |
|---|---|---|---|
| `group_description` | 7 / 0.19 | 4 / 0.10 | 0.21 -> 0.03 |
| `specific_behavior` (actor-behaviour retrieval) | 17 / 0.35 | 12 / 0.29 | 0.49 -> 0.36 |
| `software_tools` | 31 / 0.60 | 25 / 0.49 | 0.70 -> 0.48 |
| `campaign_profile` (weight 0) | 5 / 0.12 | 3 / 0.10 | 0.09 -> 0.05 |
| `semantic_centroid` | 1 / 0.06 | 0 / 0.05 | 0.08 -> 0.06 |
| `bernoulli_naive_bayes` | 6 / 0.23 | 6 / 0.24 | 0.23 -> 0.24 |
| `idf_tfidf_cosine` | 0 / 0.07 | 1 / 0.07 | 0.08 -> 0.09 |
| `weighted_jaccard` | 0 / 0.03 | 0 / 0.04 | 0.04 -> 0.05 |

Two things cause the drop. The first is name matching: on the name-bearing reports
group-description similarity was almost entirely the name and actor-behaviour retrieval partly,
and the three named FIN6 hypotheses exist only while "FIN6" is in the text (the leader stays
correct; the naming gate stops firing). The second is lost malware evidence: four of the six
lost top-1 answers are FIN7 reports whose only software match was the Carbanak malware, removed
with the name. A diagnostic run that keeps the 18 software-name aliases (not a script option)
measured top-1 0.51, top-5 0.71, MRR 0.60 and 10 named, all correct, but it still leaks the
group name where the malware carries it (group description ranks Carbanak first on one Carbanak
report only while "CARBANAK" is in the text). The technique-set classifiers and accepted-technique
recall (0.40 -> 0.41) do not move. The anonymized figure is therefore a lower bound for
attribution from tradecraft alone and the as-reported figure an upper bound that includes
reading the label.

Decision (2026-09-30, product owner): the anonymized number is the headline real-report
metric and the as-reported number is quoted beside it. It is measured the way the ATT&CK folds
are, incident evidence rarely carries the actor's name, and the naming promise holds under it
(8 of 8 named groups correct). `make benchmark-cti-hal` writes the headline to
`data/audits/attribution_cti_hal_current.json` and the as-reported run to
`attribution_cti_hal_as_reported.json`; `make benchmark-cti-hal-production` anonymizes as well.
Without `--output` the script uses the same names (an as-reported production run writes
`attribution_cti_hal_as_reported_production.json`), so no run replaces another mode's artifact;
before this change a real-report run without `--output` overwrote the ATT&CK folds'
`attribution_heldout_current.json`.
`collect_verification.py` reports both (`cti_hal_attribution`, `cti_hal_attribution_as_reported`)
and warns when the headline artifact was not anonymized (artifacts written before this change
carry no `parameters.actor_names` and read the statements as reported). Naming-gate sweeps
(`attribution_gate_sweep.py`) over `attribution_cti_hal_current.json` now read the anonymized
rows. Every earlier CTI-HAL attribution number in this document, including the production
measurement of 2026-09-28, was measured as reported; its before/after comparison stays valid
because both sides read the same text. On 2026-10-01 a leak in the anonymization was closed: a
name that ended a sentence survived. The headline on the same code moved from top-3 31 to 29,
top-5 0.636 to 0.600 and MRR 0.530 to 0.512, with top-1 unchanged at 0.455 (21.3.14).

#### 21.3.4c Elastic Security Labs real-report benchmark

A second real-report source, independent of CTI-HAL and untouched by any tuning: Elastic
Security Labs posts in which Elastic attributes observed activity to one ATT&CK 19.2 group.
Unlike CTI-HAL's annotated snippets, each post is read **the way the product's no-LLM mode reads
a pasted report**: every prose sentence through `local_observations`, no curation. This set is the
only clean real-report holdout; nothing may be tuned on it.

**Provenance and licence.** The Security Labs RSS feed carries all 302 posts (May 2022 to
September 2026) with their full HTML, so no scraping is needed (a sitemap check found one missing
post, on log4j). `scripts/prepare_elastic_reports.py` (`make prepare-elastic`) downloads it once to
`data/benchmarks/elastic/feed.xml`. The post text is (c) Elastic: the derived files stay under the
gitignored `data/benchmarks/elastic/` for local evaluation and are not redistributed. Only slugs,
ATT&CK ids and our own paraphrases are committed (`data/elastic_security_labs/labels.json`).

**Selection (pre-registered 2026-10-01, commit 520ca28).** All 302 posts were screened for
attribution to an ATT&CK group, 25 candidates were adversarially verified and a completeness pass
looked for misses. Eight posts remain (six groups): Elastic's own attribution of real observed
activity with enough behaviour prose. Contagious Interview (G1052); KANDYKORN and RUSTBUCKET
(Lazarus Group G0032, APT38 G0082 also accepted, since other vendors file both under BlueNoroff);
SPECTRALVIPER and PHOREAL (APT32 G0050); LOBSHOT (TA505 G0092); Gamaredon Group (G0047); and the
Malaysian lure (Leviathan G0065). Fourteen posts are excluded with reasons in the labels file, for
example operation-bleeding-bear (the label follows from WhisperGate being Ember Bear-only, the
same mapping the software classifier reads), dynowiper (contested attribution), emulations and
product demos. An **open set** of 22 posts on actors that are in no ATT&CK group (Elastic REF
clusters and vendor names outside ATT&CK; 7 APT-like, 15 commodity) was checked by two
independent refuters each; the correct answer there is "no named group", and any name is
unsupported rather than proven wrong. Labels, accept sets, rules, metrics and arms were committed
with the sha256 of the prepared files before the first run.

**Reading rule (label-blind, the same for every post).** `<pre>` code and code pasted between
paragraphs, markdown pipe tables and fences, image and "Figure N" lines and headings themselves
are dropped, as are whole non-narrative sections matched by exact heading (MITRE ATT&CK lists,
detection and hunting rules, YARA, observations and indicators, references, appendix, diamond
model). The 8 posts give 848 statements (44 to 191 each), the open set 3,619 (67 to 407).
Report-level technique labels come from every `attack.mitre.org/techniques/...` link in a post,
with revoked ids followed to their successors (six of the eight posts carry links). Leak audit: no
statement equals an ATT&CK procedure sentence or an exemplar sentence, and no ATT&CK object cites
any of the eight posts.

**Anonymization levels** (`scripts/report_anonymizer.py`, `--anonymize LEVEL`):

- `names`: every ATT&CK group name and alias -> "the adversary" (the CTI-HAL headline level,
  byte-identical to before).
- `strict` (**headline**): additionally the report's own attribution prose (a sentence that
  refers to an actor *and* argues attribution, a paragraph with two such sentences, a section
  about attribution: heading rule plus curated headings fixed in the labels), vendor cluster
  names (REF/UNC/Storm numbers, family-suffix names, a short list such as DangerousPassword),
  ATT&CK object ids, and a case-sensitive country / nationality / sponsor / region / language
  gazetteer (origin compounds such as "DPRK-aligned" are deleted, nouns become "an undisclosed
  country"). ALL-CAPS codenames (CUBA, KREMLIN) and software names (Cuba, S0625) are protected,
  ordinary-word aliases ("Play", "Chromium") are replaced only in their ATT&CK spelling. Victim
  geography goes too: strict is a no-victimology regime. On the eight posts it removed 88
  attribution statements (51 in attribution sections, 34 sentences, 3 paragraph neighbours),
  9 group names, 23 cluster names and 30 place or sponsor words (12 origin, 18 victim or other),
  leaving 760 statements; the residual audit (boundary-free search for every alias, short form,
  place and cluster id left) finds nothing on them, and on the open set lists only the Chromium
  browser, defanged domains and a malware marker string.
- `tradecraft`: strict plus every ATT&CK *malware* name -> "the malware" (tools stay).

**Results (2026-10-01, local pipeline, no credit; `make benchmark-elastic`).**

| Elastic, 8 reports, 6 groups | Top-1 (k/8) [Wilson 95%] | Top-3 | Top-5 | MRR | Group-macro top-1 | Named (wrong) |
|---|---|---|---|---|---|---|
| **strict (headline)** | **4/8 = 0.50** [0.22, 0.78] | 0.50 | 0.63 | 0.54 | 0.50 | 1 (0) |
| names | 5/8 = 0.63 [0.31, 0.86] | 0.63 | 0.75 | 0.66 | 0.67 | 1 (0) |
| tradecraft | 1/8 = 0.13 [0.02, 0.47] | 0.25 | 0.25 | 0.21 | 0.17 | 0 (0) |
| as reported | 5/8 = 0.63 [0.31, 0.86] | 0.63 | 0.75 | 0.66 | 0.67 | 0 (0) |
| strict, 30 random sentences (seeds 1-5, mean) | 0.40 | 0.48 | 0.53 | 0.46 | | 0-2 per seed (0) |

Lenient scoring (APT38 accepted on the two Lazarus reports) changes nothing: both miss either
way. Cluster-aware top-1 equals top-1. On the 22 open-set posts **no group was named** (all
reached a shortlist), so the combined named risk over labelled and open-set reports is 0; no
group leads the open-set rankings often (Kimsuky three times, five groups twice, Lazarus once,
the other labelled groups never), so the hits below are not a default leader showing up
everywhere. Since documented-cluster naming became the default (10.11), a cluster name counts as a
name given on an open-set report too (open-set `named`, `named_clusters`, combined named risk);
on a labelled report it is wrong when the truth is not a member. Lenient correct names still count
single-group names only.

| Report | Truth | Rank strict / names / tradecraft / as reported | Strict stage, leader | Group-only ATT&CK malware left at strict |
|---|---|---|---|---|
| contagious-interview | G1052 | 1 / 1 / 7 / 1 | shortlist, G1052 | BeaverTail S1246 |
| kandykorn | G0032 | 4 / 4 / 3 / 4 | shortlist, APT3 | none (APT3's OSInfo matched the field name "osinfo") |
| rustbucket | G0032 | 52 / 36 / 52 / 36 | shortlist, Mustang Panda | none |
| spectralviper | G0050 | 1 / 1 / 11 / 1 | shortlist, G0050 | PHOREAL S0158 |
| phoreal | G0050 | 1 / 1 / 18 / 1 | named, G0050 | PHOREAL S0158 |
| lobshot | G0092 | 52 / 1 / 52 / 1 | shortlist, Gamaredon | none (Get2 sat in the dropped attribution paragraph) |
| gamaredon | G0047 | 1 / 1 / 1 / 1 | shortlist, G0047 | Pteranodon S0147 |
| malaysian (APT40) | G0065 | 108 / 107 / 108 / 108 | shortlist, Patchwork | none |

What the numbers say. (1) **The hits are mostly malware lookups.** Three of the four strict hits
rest on malware only the truth group uses in ATT&CK, and masking it (tradecraft) leaves one hit:
Gamaredon, the only report recognised from behaviour (actor-behaviour channel 0.83 against 0.39
for the next group). Top-1 is 0.80 on the five reports where any ATT&CK software is detected and
0.00 on the three without. (2) **Elastic's
attribution prose carries evidence, not only claims:** removing it costs LOBSHOT, whose only
mention of the TA505-only Get2 loader is in the attribution paragraph (rank 1 -> 52). Group names
alone moved almost nothing: names and as-reported rank the truth the same on seven of the eight
reports (the Malaysian lure 107 against 108). (3) **The
naming promise holds**: one name in eight labelled reports, correct, and none in 22 open-set
reports, at every arm and seed. (4) **Reading the whole report helps**: 30 random sentences give
top-1 0.40 on average against 0.50 for all of them. (5) Two failure causes are visible: a false
software match ("osinfo" as a record field matched ATT&CK's OSInfo, an APT3-only malware, and the
decisive-malware boost put APT3 first on KANDYKORN; the same class as the "Page" alias in 5.5),
and reports whose behaviour is generic for their group (RUSTBUCKET and the Malaysian lure have no
group-specific malware or procedures in ATT&CK that the text matches).

**Registered run, review and corrected run.** The first run used the rules as committed with the
labels. A read-only code review afterwards (three lenses, each finding reproduced by an
independent skeptic) confirmed 18 defects; six changed what the model read: acronym compounds and
prefixes kept ("CIS-region", "pro-DPRK"), "The DPRK continues" lost its subject, the cue "known as"
dropped technique sentences ("a technique known as reflective binary loading"), "U.S." and spaced
"state sponsored" were missed, and three heading families (`Techniques (sub-techniques)`,
`Detection rules for X`, an escaped `ATT\&CK`) kept ATT&CK names and rule names in three
open-set posts. Latent gaps were closed too (plural demonyms, cluster names after "/", group
names in dash compounds), and the artifact now records the rule and input hashes. The fixes
implement the registered rules as written; no label, threshold or metric changed. The corrected
run (the artifacts above) gives the same strict, names and as-reported ranks on every report and
the same open-set result as the registered run; tradecraft MRR moves 0.23 -> 0.21 (KANDYKORN rank
2 -> 3) and the volume diagnostic 0.33 -> 0.40 (two more statements change the samples). Still
open: the newest post renders its tables as runs of `<p>` cells, which are read as prose (one
open-set post, about 56 statements), and cities other than capitals are not masked.

Caveats: n = 8 (one report moves top-1 by 0.125; no significance claim is made, and none against
CTI-HAL, whose groups and reading regime differ); several labels are moderate-confidence or rest
on overlap with other vendors' reporting; spectralviper and phoreal share a victim environment and
the two Lazarus posts cross-reference each other; the Elastic technique lists are short author
lists (mean accepted-technique recall 0.76 over the six labelled posts, precision 0.06 to 0.39).

Running the open set exposed a product bug: retrieval sent every sentence of a report to ChromaDB
in one query, and the server refuses about 32k result rows (queries x results) in one call, so a
pasted report of several hundred sentences crashed analysis in local mode. `vectorstore._query`
now batches the queries under 16k rows per call; each query is independent, so the hits are
identical and shorter reports still make a single call.

Artifacts (`make benchmark-elastic`): `attribution_elastic_current.json` (strict headline with the
open set under `open_set` and `metrics.open_set`), `_names.json`, `_tradecraft.json`,
`_as_reported.json`; the volume diagnostic is `attribution_benchmark.py --report-set elastic
--cti-hal data/benchmarks/elastic/reports.jsonl --anonymize strict --report-observations 30
--report-sample-seed S`. `default_output` names the headline level of each report set
`*_current` (CTI-HAL: names; Elastic: strict). `collect_verification.py` reports
`elastic_attribution` and `elastic_attribution_as_reported` and warns when the Elastic headline
is not strict. `make benchmark-elastic-production` (technique verifier) is built but not run: at
about $0.02 and 11 s per sentence it would cost roughly $15-20 and 2.5 h for the eight posts.

#### 21.3.5 Attribution benchmark (text in, group out)

`scripts/attribution_benchmark.py` (`make benchmark-attribution`) is the attribution gate.
For every group with at least 5 documented techniques and 5 procedure sentences it holds
out 40% of the group's (technique, sentence) behaviours, anonymizes actor names to "the
adversary", and runs the sentences as free-text observations through
`AttributionEngine.analyze`: retrieval, acceptance, eligibility, family weights and the
stage decision run exactly as in the product. `--pipeline local` (default, the adoption
gate) switches the verifier off, so no OpenRouter call is made; `--pipeline production`
(`make benchmark-attribution-production`, `make benchmark-cti-hal-production`) runs the
shipped pipeline with the configured technique verifier on the ambiguous candidates and
records verifier calls, errors and cost per fold. `rank` is the analyst-facing ranking and
`deterministic_rank` the ranking the stage decision reads; they differed only while the LLM
zero-shot vote existed (removed 2026-09-30) and are both kept so older artifacts compare. `--sample N --seed S` runs a
group-balanced random subset of real reports (to fit a credit budget); the run aborts if every
LLM call of the first fold fails and stops, keeping the measured folds and recording
`truncated_at`, if the API goes away mid-run. A per-fold store proxy
hides the group's own evidence (technique-index hits identical to a held-out sentence and
the group's actor-behaviour documents for the held-out techniques), and since 2026-10-03 the
BM25 lexical index is rebuilt per fold without the documents whose text is a held-out
sentence (`fold_lexical_index`; document frequencies and average length recomputed). Before
that, only the dense index was filtered and a held-out sentence was the top BM25 hit for
1,268 of 1,272 seed-7 fold observations; fold artifacts, design/confirmation seeds and the
combiner lists made before then carry that leak (`parameters.leak_control` is `dense+lexical`
from then on, and the combiner records `lists_leak_control`). Not hidden: the fold group's
one-word variants of a held sentence filed under other techniques (about 1.4% of queries have
one in the BM25 top 3) and, with `TECHNIQUE_INCLUDE_SYNTHETIC` on, synthetic paraphrases of
procedure examples. Measured on 2026-10-03 with the shipped combiner
(`data/audits/bm25_fold_hiding_2026-10-03/`): seed 7 technique recall 0.723 -> 0.604, top-1 0.830 ->
0.824, MRR 0.880 -> 0.867, named 32 -> 26 (0 wrong either way, 0/25 removed actors named); seed 41
recall 0.740 -> 0.600, top-1 0.863 -> 0.824, named 37 -> 33. 153 (seed 7) and 190 (seed 41) correctly
accepted techniques existed only through the leak. A combiner refitted on leak-closed lists did not
beat the shipped one (seed 7 top-1 0.804, seed 41 0.817; CTI-HAL as reported 0.564 -> 0.545), so the
shipped combiner stays; the refit is kept as `attribution_combiner_bm25_clean_rejected.json`. Fold
numbers quoted elsewhere in this document before 2026-10-03 were measured with the leak. An *unknown-actor* arm
removes sampled groups from the knowledge base and hides all their documents, so the
correct answer is "no named group"; `unknown_recall`, `false_named_attribution_rate`,
`named_coverage` and `named_risk` come from `evaluation.evaluate_attribution_predictions`.
The artifact (`data/audits/attribution_heldout_current.json`) stores per-fold ranks,
stages, accepted techniques and the technique recall of the mapping step.

Interpretation: group technique lists, technique-profile centroids and descriptions still
contain the held-out techniques, so this is self-consistency over the ATT&CK answer key
with the text leak removed (fully only since 2026-10-03, see above), not field attribution
accuracy. It and the CTI-HAL
real-report run (21.3.4b) measure the stage machinery; only this one has an unknown-actor arm.

`scripts/ensemble_benchmark.py` is the older technique-ID version (perfect mapping
assumed, three voting classifiers, no eligibility gate, no stage). It remains as a
diagnostic; about 19% of a group's held-out techniques re-enter it through campaigns.

#### 21.3.6 Benchmark results

Numbers below were recorded between 2026-09-22 and 2026-09-29 with ATT&CK 19.2,
`basel/ATTACK-BERT`, the balanced preset and no LLM verifier unless stated. "Current" and
"shipped" here mean at that date; later subsections (21.3.8 onward) supersede several of them,
and the current figures are in `data/audits/verification_results_current.json`.

**Curated retrieval (30 positive cases, 6 hard negatives;
`data/audits/curated_hybrid_current.json`)**

| Mode | Cues | Top-1 | Top-3 | Top-10 | Accepted-set P / R / F1 | Hard-negative accepted-FP cases |
|---|---|---|---|---|---|---|
| legacy_dense | on | 26/30 | 29/30 | 30/30 | 0.85 / 0.83 / 0.84 | 0/6 |
| legacy_dense | off | 15/30 | 24/30 | 28/30 | 0.46 / 0.34 / 0.39 | 0/6 |
| **hybrid (default)** | on | 25/30 | 29/30 | 30/30 | 0.79 / 0.77 / 0.78 | 0/6 |
| hybrid | off | 15/30 | 24/30 | 29/30 | 0.52 / 0.43 / 0.47 | 0/6 |
| lexical | on | 26/30 | 26/30 | 26/30 | 1.00 / 0.63 / 0.77 | 0/6 |
| lexical | off | 12/30 | 22/30 | 25/30 | 0.00 / 0.00 / 0.00 | 0/6 |

Ten of the 25 hybrid top-1 hits depend on the regex cue tables, which cover 19 of the 23
distinct expected technique IDs in this set. The accepted-set F1 rose from 0.63 to 0.78 with
the P2 gating changes and the acceptance model; top-1 moved 26 -> 25.

**OTRF Sysmon benchmark (9 cases)**

| Case | Expected | Rank |
|---|---|---|
| PowerShell encoded listener | T1059.001 | 1 |
| LSASS Dumpert | T1003.001 | 1 |
| MSHTA execution | T1218.005 | 1 |
| Windows logon script | T1037.001 | 1 |
| Remote-thread injection | T1055 | 1 |
| WMI lateral movement | T1047 | 1 |
| Registry Run key | T1547.001 | 1 |
| Credential dumping via ntds.dit | T1003.003 | 1 |
| PowerShell downloaded cradle | T1059.001 | 1 |

**Public held-out benchmark (hybrid; `public_heldout_baseline.json` vs `public_heldout_current.json`)**

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
| **shipped 2026-09-28 (schema 2, no priors, threshold 0.27; superseded by exemplars, 21.3.10)** | 1391 | 0.32 [0.30, 0.34] | 0.49 | 0.24 | 0.40 | 0.58 | 0.75 |
| opt-in `acceptance_model_priors.json` (schema 2, train priors) | 1391 | 0.36 [0.34, 0.39] | 0.51 | 0.28 | 0.44 | 0.64 | 0.76 |

The priors model (2026-09-28) passes the mapping gate on every split (tram F1 0.43 -> 0.48,
expert 0.18 -> 0.20, derived_procedures 0.26 -> 0.28; pooled MRR 0.52 -> 0.55; CTI-HAL
statements F1 0.31 -> 0.35, hit@1 0.33 -> 0.37) but is **not shipped**: with it the
attribution benchmarks lose technique recall on ATT&CK procedure sentences (folds 0.78 ->
0.67, top-5 0.82 -> 0.81) and, decisively, the naming gate issued a confidently wrong name on
a real report (Olympic Destroyer, Sandworm Team, named as BlackByte at maturity 0.93 on
three newly accepted generic techniques; CTI-HAL named risk 0.00 -> 0.50). No maturity or
margin bar separates that case from the correct names, so the shipped model is the schema-2
fit without priors (identical behaviour to before, threshold 0.28 -> 0.27). Schema-1 models
are kept under `data/models/acceptance/superseded_schema1/` and no longer load.

Per split with probability ranking (2026-09-28): tram hit@1 0.43 -> 0.46, expert 0.38 -> 0.44,
derived_procedures 0.24 -> 0.30; MRR pooled 0.47 -> 0.52. The accepted set, and therefore
P/R/F1 and every attribution metric, is identical (the attribution artifacts were not rerun).

Baseline = pre-P2 pipeline (`data/audits/public_heldout_baseline.json`). Current = rank-1-only rule gate, procedure sentences in the lexical index, sentence-split retrieval for long observations, `RETRIEVAL_TOP_K=30`, and (last row) the calibrated acceptance model fitted on the dev splits (threshold 0.28; dev P/R/F1 0.51/0.28/0.36). `make check-regression` compares later runs against the baseline.

**EvidenceForge (local pipeline)**: see `docs/EVIDENCEFORGE_EVALUATION.md`. Held-out
clinic narrative micro-F1 0.23 [0.00, 0.45], hit@1 0.22; two of the three benign red
herrings are accepted (`cln-rh-001` as T1560.001, `cln-rh-002` as T1036.005), zero-label FPR 0.67.
With the OpenRouter verifier (`--pipeline production`, ambiguous scope) the clinic narrative
micro-F1 is 0.53 [0.34, 0.69] (recall 0.56); the red herrings are unchanged because locally
accepted mappings are not re-verified in that scope.

**Per-sentence acceptance and hierarchy consistency: ablation (public held-out, hybrid,
acceptance model fitted on merged pools; `check_regression.py` against the 0.321 artifact)**

| Variant | pooled micro-F1 | P | R | expert F1 | derived F1 | parent-aware F1 |
|---|---|---|---|---|---|---|
| reference (current artifact) | 0.321 | 0.50 | 0.24 | 0.178 | 0.258 | 0.391 |
| both off (reproduction) | 0.319 | 0.46 | 0.24 | 0.185 | 0.256 | 0.386 |
| per-sentence acceptance only | 0.322 | 0.38 | 0.28 | **0.244** | 0.257 | 0.388 |
| hierarchy `swap_and_fill` only | **0.268** | 0.27 | 0.26 | 0.194 | 0.178 | 0.373 |
| per-sentence + hierarchy `swap_only` | 0.317 | 0.37 | 0.28 | 0.241 | 0.249 | 0.391 |

| per-sentence + `swap_only`, acceptance model **refitted on per-sentence features** | 0.304 | 0.48 | 0.22 | 0.164 | 0.234 | 0.372 |

Filling the parent whenever two siblings look plausible re-admits candidates the acceptance
model had rejected and is rejected. Per-sentence acceptance lifts the long-paragraph split
with the merged-pool model but costs 12 points of precision; refitting the acceptance model
on per-sentence features (`scripts/fit_acceptance_model.py` extracts features through the
same `acceptance_views` the runtime judges) restored precision but the dev-selected
threshold then suppressed the paragraph recall (expert F1 0.164). Neither variant improves
held-out F1, so both switches default to off and the default pipeline is the measured 0.321
configuration; the per-sentence model is kept as `superseded_schema1/acceptance_model_per_sentence.json` for
opt-in use. The corroboration bonus had no measurable effect on the attribution benchmark
(identical metrics at 0.0 and 0.1) and stays off.

**Acceptance model refit with CTI-HAL statements (2026-09-24; not adopted).** The acceptance
model was refitted on the public dev splits plus the 968 CTI-HAL statements (2,115 cases,
31,443 candidate rows; `scripts/fit_acceptance_model.py --extra-jsonl`). The dev operating
point moved from threshold 0.28 to 0.30 and the model trades recall for precision everywhere:
public held-out pooled F1 0.320 -> 0.313 (P 0.50 -> 0.52, R 0.24 -> 0.22, hit@1/hit@10
unchanged; tram flat, expert -0.015, derived_procedures -0.013) and CTI-HAL statement F1
0.310 -> 0.297 even though those statements were in its training pool. The regression gate
passes (drop < 0.02) but nothing improves, so the merged-pool model stays active and CTI-HAL
remains unseen by the shipped acceptance model. The rejected fit is kept as
`data/models/acceptance/superseded_schema1/acceptance_model_public_dev_plus_ctihal.json`.

**CTI-HAL real-report benchmarks (`cti_hal_mapping_current.json`,
`attribution_cti_hal_current.json`; 2026-09-23)**

Statement-level technique mapping over 968 human-annotated report statements: micro-F1
0.31 [0.28, 0.34], P 0.44, R 0.24, in line with the public held-out set (0.32), so the
public gate transfers to real report text. Ranking by acceptance probability (8.2, adopted
2026-09-28) lifts hit@1 0.27 -> 0.33, hit@3 0.49 -> 0.53, hit@10 0.67 -> 0.69 and MRR
0.41 -> 0.46 on these statements with the accepted set unchanged, the same effect as on
the public set. The opt-in priors model (21.3.6) would lift the accepted set itself (F1
0.31 -> 0.35, hit@1 0.33 -> 0.37) but is not shipped because of the wrong name it produces
on the report-level benchmark below.

Real-report attribution over 55 reports (each report's annotated statements as
observations, no evidence hidden, no unknown-actor arm):

| Metric | Value |
|---|---|
| Top-1 | 0.45 [0.33, 0.58] |
| Top-5 | 0.65 |
| MRR | 0.57 |
| Median rank of the true group | 2 |
| Statement technique recall (accepted vs annotated) | 0.31 |
| Cluster-aware top-1 (a documented related group at rank 1 counts) | 0.60 |
| Named coverage (separation gate, 10.7) | 0.22 (12 of 55) |
| **Named risk** (separation gate) | **0.00** (12 of 12 correct; maturity gate 0.82: 3 of 3; at maturity 0.78 before the 2026-09-28 fusion changes: 7 named, 3 wrong, 0.43) |

With the opt-in priors acceptance model (21.3.6) this table changes to top-1 0.35, top-5
0.56, statement recall 0.31, 2 named of which 1 wrong (Olympic Destroyer -> BlackByte,
maturity 0.93, margin 0.14): the reason that model is not the default.

| Group | Reports | Top-1 | Top-5 | Not ranked |
|---|---|---|---|---|
| Carbanak (G0008) | 10 | 1 | 5 | 3 |
| Sandworm Team (G0034) | 5 | 3 | 5 | 0 |
| FIN6 (G0037) | 11 | 7 | 9 | 1 |
| FIN7 (G0046) | 18 | 8 | 11 | 0 |
| OilRig (G0049) | 8 | 4 | 6 | 0 |
| Wizard Spider (G0102) | 3 | 1 | 2 | 0 |

At the previous naming bar (maturity 0.78) named risk on real reports was 0.43, far above
the self-consistency figure at the same bar (0.15): the two wrong FIN7/Carbanak attributions
pointed at each other and the third named APT29 for a FIN7 report. Raising the bar to 0.82
leaves 3 named reports, all correct. Carbanak reports are mostly attributed to FIN7 (three of
the ten have FIN7 at rank 1, one FIN7 report has Carbanak at rank 1), which ATT&CK itself
documents as overlapping; one Sandworm Team report has APT38 at rank 1 (rarely shared
software). Counting those related leaders as correct gives cluster-aware top-1 0.45 against
0.36 exact. This is the first attribution number on real text and should be treated as the
reference for the naming gate rather than the held-out ATT&CK folds.

**Production pipeline on real reports (`attribution_cti_hal_production.json`; 2026-09-28).**
`make benchmark-cti-hal-production` with `--sample 10 --seed 7` ran a group-balanced sample
of 10 of the 55 reports (two per group, except one each for OilRig and Wizard Spider)
through the shipped pipeline as it was then: OpenRouter verifier (`openai/gpt-5.6-sol`, low
effort) on the candidates still ambiguous after local gating and the bounded zero-shot vote,
since removed. 165 verifier
calls, 0 errors, 10 votes, $3.37 (about $0.34 per report; the full set would cost about $19).
Paired comparison with the local run on the same 10 reports:

| Metric (10 reports) | Local | Production |
|---|---|---|
| Technique recall of the mapping step (accepted vs annotated) | 0.23 | 0.48 |
| Technique micro-precision of the accepted set | 0.49 | 0.40 |
| Accepted techniques / true positives | 82 / 40 | 189 / 76 |
| Top-1 | 0.50 | 0.50 |
| Top-5 | 0.80 | 0.80 |
| MRR | 0.64 | 0.67 (0.67 before the vote) |
| Named hypotheses (maturity 0.82) | 2, both correct | 2, both correct |

The verifier is what changes: it roughly doubles technique recall on real report statements
for about nine points of precision, which is the trade the product makes (the local gate is
tuned for precision because it has no second opinion). The group ranking barely moves
(two reports gain one rank, one Carbanak report 7 -> 6 with the vote, OilRig 28 -> 33) and
the stage decisions are identical, as designed: the stage is decided on the deterministic
fusion and the vote only reorders. Ten reports give a wide interval (top-1 CI [0.2, 0.8]); the
number to carry is the paired recall change, not the ranking. The production artifact is
reported in the verification summary as `cti_hal_attribution_production` and is never the
adoption gate.

**Synthetic sentences (partial).** `make generate-synthetic` produced 300 sentences for 50
of the 575 sparsely documented target techniques (about $0.17; 0 rejected by the filters)
before OpenRouter returned `402 Payment Required`; the account credit is exhausted and the
run resumes from the saved state once it is restored. With this coverage the field does not
help: public held-out pooled F1 0.320 -> 0.315 (precision -0.01, hit@10 unchanged), CTI-HAL
mapping F1 0.308 -> 0.310, and among the 23 statements labelled with a covered technique no
top-1 changed. The field therefore stays indexed but excluded by default
(`TECHNIQUE_INCLUDE_SYNTHETIC=false`) until the full set exists and is re-measured.

**Local reranker (cross-encoder) evaluation**

A first cross-encoder (ATTACK-BERT base, one epoch, 26,321 pairs with 18,888
retrieval-mined negatives, `data/models/technique-reranker`) was trained on 2026-09-22 and
evaluated with the pipeline above:

| Configuration | pooled micro-F1 | P | R | hit@1 | hit@10 | clinic narrative F1 / hit@1 / zero-label FPR |
|---|---|---|---|---|---|---|
| without reranker | 0.32 | 0.50 | 0.24 | 0.35 | 0.73 | 0.23 / 0.22 / 0.67 |
| with reranker v1 | 0.26 | 0.20 | 0.37 | 0.24 | 0.73 | 0.14 / 0.06 / 1.00 |

It was **not enabled**. Three defects were found and fixed as a result: (1) the threshold
calibration preferred full decision coverage, collapsing promote and demote to one
threshold (0.42) so the reranker decided every candidate; it now selects a
precision-floored abstention band (promote precision >= 0.85, demote precision >= 0.90 on
dev). (2) Reranker promotion bypassed the benign-context and sparse-text guards and
demotion could override cue-corroborated mappings; both are now blocked. (3) The
reranker's order replaced the fused order for all candidates; it now moves only
candidates above the promote threshold to the front and leaves the rest in fused order.
A two-epoch model trained with these fixes (`data/models/technique-reranker-v2`; dev band
promote 0.89 / demote 0.09, promote precision 0.875 on 63 promotions) was evaluated on the
clinic bundle: narrative F1 0.23 -> 0.25 with one red herring correctly demoted, windows
unchanged, eCAR F1 0.55 -> 0.38 with hit@1 0.50 -> 0.14. Its public held-out run was stopped
before completion to prioritise the gating work below. Both models stay **disabled**; the
artifacts and pair cache are kept so a retrain with more or synthetic data costs 40 minutes.

**Attribution benchmark results (text in, group out; 153 known folds, 25 unknown-actor
folds; `attribution_heldout_baseline.json` vs `attribution_heldout_current.json`)**

| Metric | Before rebalancing | After (2026-09-29 defaults; superseded, 21.3.8 onward) |
|---|---|---|
| Top-1 (true group ranked first) | 0.38 [0.30, 0.46] | 0.59 [0.52, 0.67] |
| Cluster-aware top-1 (a documented related group at rank 1 counts) | n/a | 0.60 |
| Top-5 | 0.71 | 0.80 |
| MRR | 0.52 | 0.71 |
| Median rank of the true group | 3 | 1 |
| Technique recall of the mapping step | 0.77 | 0.79 |
| Unknown-actor recall (not named) | 0.84 | 1.00 |
| False named attribution rate (unknown actors named) | 0.16 | 0.00 |
| Named coverage (cases reaching a named hypothesis) | 0.23 | 0.20 |
| **Named risk (named hypotheses that are wrong)** | **0.56** | **0.00** |
| Named risk, cluster-aware | n/a | 0.00 |

"Before" is the ensemble as shipped (technique family budget 1.00, `RRF_K=60`, plain
software ratio, alphabetical tie-break, hypothesis maturity 0.70). "After" is the 2026-09-29
default (budget 2.00, `RRF_K=20`, rarity-weighted software links with the tool cap and the decisive-malware boost of 2026-09-28, the separation gate and 0.5 discriminative re-ranking of 2026-09-29 (10.7), prominence tie-break,
hypothesis maturity 0.82). Stage distribution after: known {'known_group_hypothesis': 35, 'known_group_shortlist': 104, 'unresolved_activity_cluster': 14};
unknown {'insufficient_evidence': 2, 'known_group_shortlist': 19, 'unresolved_activity_cluster': 4}. The rarity weighting alone is neutral here
(top-5 0.82 with vs 0.82 without); it is kept because commodity tools should not count as
much as bespoke implants. The named hypotheses are 35 known groups (all correct) and no removed actor (the separation gate of 10.7 also stopped the Play -> BlackByte false name).
Treat the named-risk figure as a lower bound for field data.

**Ensemble self-consistency (legacy technique-ID diagnostic; 153 groups, 40% coverage, seed 7; measured before the 2026-09-22 fusion rebalancing)**

| Configuration | RRF top-1 | RRF top-5 | MRR |
|---|---|---|---|
| with campaign voting | 101/153 (66%) | 153/153 (100%) | 0.807 |
| without campaign voting | 147/153 (96%) | 153/153 (100%) | 0.977 |

Campaign voting at a nonzero weight degrades the self-consistency benchmark and is
disabled by default.

#### 21.3.7 Improvement progression (curated set, cues enabled)

| Stage | Hybrid top-1 | Hybrid top-10 | Key change |
|---|---|---|---|
| Original | 4/18 (22%) | 8/18 (44%) | Dense-only with no abstention |
| After hybrid fixes | 6/18 (33%) | 14/18 (78%) | Lexical acceptance, abstention threshold, hierarchy expansion |
| After explicit cues | 16/18 (89%) | 18/18 (100%) | 12 explicit technique cue patterns, name-token matching |
| After benign detection | 18/18 (100%) | 18/18 (100%) | Benign indicator detection, exact tool name matching, SAM/MSHTA cues |
| After conservative refactor | 26/30 (87%) | 30/30 (100%) | Bounded cue scores (0.85 not 1.0), benign indicators override cues, extended test set (30 cases) |
| Remediation pass (2025-09-15) | 28/30 (93%) | 30/30 (100%) | Structured signals, candidate-specific gating, 3x dense over-fetch |
| Current (2026-09-22) | 26/30 (87%) | 30/30 (100%) | Top-1 regressed from the remediation peak; cause not recorded |

The progression was driven by the curated set itself, which is why most of the gain came
from cue tables that do not transfer to the public held-out data (hybrid top-1 without
cues: 15/30).

---

#### 21.3.8 Distinguishing the shortlist: contrastive re-ranking and near misses (2026-09-29)

Both attribution benchmarks, local pipeline, shipped acceptance model. "Named" counts known
rows that reached `known_group_hypothesis`; every named row was correct in every run and no
removed actor was named (0/25). Leader consistency: 0 named rows in any run had a re-ranked
leader different from the scored one.

| variant | folds top-1 | folds top-5 | folds MRR | folds named | CTI-HAL top-1 | CTI-HAL top-5 | CTI-HAL MRR | CTI-HAL named |
|---|---|---|---|---|---|---|---|---|
| before (gate K=5, re-rank K=5) | 0.595 | 0.804 | 0.710 | 35 | 0.455 | 0.655 | 0.568 | 12 |
| gate and re-rank K=10 | 0.601 | 0.915 | 0.737 | 27 | 0.509 | 0.745 | 0.617 | 8 |
| contrastive near misses (p ≥ 0.2), K=5 | 0.601 | 0.804 | 0.713 | 34 | 0.436 | 0.655 | 0.556 | 12 |
| near misses + K=10 | 0.601 | 0.915 | 0.739 | 26 | 0.509 | 0.745 | 0.613 | 7 |
| **re-rank K=10, gate K=5 (adopted)** | **0.601** | **0.915** | **0.737** | **35** | **0.509** | **0.745** | **0.617** | **12** |
| re-rank K=15, gate K=5 | 0.608 | 0.922 | 0.738 | 35 | 0.509 | 0.745 | 0.619 | 12 |

Reading: contrasting ten candidates instead of five is what moves the ranking (fold top-5
+0.11, real-report top-1 +0.05), but scoring the *gate* over ten shrinks every exclusivity
weight and drops named coverage (35 → 27, 12 → 8) because the bars were tuned for five; an
offline sweep over the K=10 rows found clean rules only with a guard band of 0.02–0.04, so the
gate keeps its five. Near misses sharpen the separation AUC slightly on real reports (0.789 →
0.812) but cost ranking (top-1 0.455 → 0.436) and one fold name; they stay off
(`ATTRIBUTION_CONTRAST_MIN_PROBABILITY=0`). Re-ranking over fifteen is within noise of ten;
ten is shipped (`ATTRIBUTION_RERANK_TOP_K=10`). The `separation_auc` diagnostic falls with the
longer re-ranking (real 0.789 → 0.719) because it scores the gate's rank-1 candidate while
"correct" now refers to the re-ranked leader; the naming gate is unaffected by construction
(10.8, leader consistency guard).

#### 21.3.9 Learned attribution combiner (2026-09-30)

Original artifact `data/models/combiner/attribution_combiner.json` (fit on seeds 11/23/37, 2,609 known
lists of which 458 clean and 2,151 randomised, 153 unknown-actor lists; L2 0.3 chosen on the
mixed out-of-fold MRR). Group-level five-fold cross-validation, out of fold, against the RRF order
of the same lists:

| lists | combiner top-1 | RRF top-1 | combiner top-5 | RRF top-5 | combiner MRR | RRF MRR |
|---|---|---|---|---|---|---|
| clean (458) | 0.878 | 0.544 | 0.983 | 0.812 | 0.923 | 0.661 |
| randomised (2,151) | 0.714 | 0.403 | 0.932 | 0.724 | 0.809 | 0.549 |
| mixed | 0.742 | 0.428 | 0.941 | 0.740 | 0.829 | 0.569 |

Leader-probability AUC 0.863 (Brier 0.13); the null option separates unknown-actor lists from
known ones with AUC 0.77 (mean null probability 0.19 vs 0.05). Largest weights: weighted-Jaccard
relative score (+2.3), software relative score (+1.4), centroid relative score (+1.3), documented
profile size (+0.8), the discrimination's software and behaviour channels (+0.7, +0.5).

Benchmarks (seed-7 folds never trained on; CTI-HAL run once per configuration):

| configuration | folds top-1 | folds top-5 | folds MRR | folds named | unknown named | CTI-HAL top-1 | CTI-HAL top-5 | CTI-HAL MRR | CTI-HAL named |
|---|---|---|---|---|---|---|---|---|---|
| before (RRF + discriminative re-rank) | 0.601 | 0.915 | 0.737 | 35 (all correct) | 0/25 | 0.509 | 0.745 | 0.617 | 12 (all correct) |
| combiner, full blend, gate on its own top-5 | 0.830 | 0.974 | 0.891 | 44 (all correct) | 1/25 | 0.527 | 0.655 | 0.587 | 17, **1 wrong** (Carbanak named FIN7) |
| combiner, full blend, union gate | 0.830 | 0.974 | 0.891 | 31 (all correct) | 0/25 | 0.527 | 0.655 | 0.587 | 14 (all correct) |
| combiner, half blend, gate on its own top-5 | 0.830 | 0.954 | 0.880 | 37 (all correct) | 0/25 | 0.527 | 0.745 | 0.623 | 14 (all correct) |
| **combiner, half blend, union gate (adopted)** | **0.830** | **0.954** | **0.880** | **36 (all correct)** | **0/25** | **0.527** | **0.745** | **0.623** | **14 (all correct)** |

Reading: the combiner transfers to real reports far less than the cross-validation suggests
(top-1 +0.02 instead of +0.31), and a pure combiner order costs top-5 and MRR there: ten reports
got worse against five better, mostly reports with two to six accepted techniques and the
Carbanak/FIN7 pair. The half blend keeps RRF's say and restores those metrics; the union gate
keeps the naming promise (the wrong name had leader probability 0.87, so a probability bar could
not have caught it). Real reports with software: top-1 0.667 (was 0.641); without: 0.188
(unchanged). Leader-probability AUC on the folds 0.952, on real reports 0.82; it is shown on the
assessment page as the calibrated probability and is informational (the gate is unchanged). A
harsher randomisation (recall 15-60%, precision 25-60%, generic false positives, software dropped
40%) was fitted as a second artifact (`attribution_combiner_v2.json`, kept on disk, not shipped):
out-of-fold mixed top-1 0.727 vs RRF 0.404 on its own lists, but on the seed-7 folds 0.771 top-1 /
0.837 MRR and on CTI-HAL 0.509 / 0.612 with the same half blend and union gate (paired: 3 reports
better, 4 worse), 36 and 15 names all correct. Making the training regime harsher did not close the
transfer gap; the gap sits in technique recall on real text, which the labelled-sentence
exemplars (21.3.10) then targeted.

**Relabel for the counted Naive Bayes negative term (2026-10-03).** The pipeline fingerprint
hashes every classifier source file, so counting the Naive Bayes negative term (9.1) moved it.
Training lists never carry negative answers, so the combiner's inputs did not change. A refit
from the cached lists (`--reuse-cache`) reproduced bitwise-identical weights, and with it the
seed-7 ATT&CK folds (153 known and 25 removed-actor rows) and both CTI-HAL runs (55 + 55 reports)
matched the pre-fix runs row for row. The default, no-country and group-network artifacts are
therefore relabelled copies (`*_nb_count.json`, with a `metadata.fingerprint_refreshed` entry)
written next to the originals. The originals keep loading on checkouts without the change, which
share `data/models`. The Top-three confidence labels (on `frontend-next`) pin the combiner's
fingerprint and file hash, so after a merge they need `make fit-confidence-labels` on attribution
artifacts rerun with the relabelled combiner.

**Refit for exactly rounded classifier sums (2026-10-03).** Every order-dependent float sum in the
classifiers (Naive Bayes, IDF cosine, Jaccard, campaign, software, CyberRE) is now `math.fsum`,
so no score depends on set iteration order or on the Python version's `sum()`, and the ATT&CK
parent-child map is built once per
technique table (a scoring pass 0.41 s -> 0.15 s, outputs bit-identical). Naive Bayes scores move
by a few units in the last place, so the combiner was refitted (`make fit-combiner`, seeds
11/23/37, today's pipeline), not relabelled: `attribution_combiner_fsum.json` with its lists in
`training_lists_fsum.jsonl.gz`, both next to the earlier files. The code change alone is
outcome-neutral: with the previous weights stamped for the new code, the seed-7 ATT&CK folds and
both CTI-HAL runs had 0 rank changes. The refit's differences therefore come from retraining on
today's lists, which have drifted since 2026-09-30 (only 757 of 2,594 shared lists keep their
shortlist order; RRF mixed top-1 on them 0.425 -> 0.420).

| Set | Before (2026-09-30 weights) | Refit |
|---|---|---|
| ATT&CK folds, seed 7 | top-1 0.830, top-5 0.961, MRR 0.880, 32 named, all correct, 0/25 removed named | top-1 0.817 (-2 folds), top-5 0.961, MRR 0.871, 32 named, all correct, 0/25 removed named |
| CTI-HAL anonymized | top-1 0.455, top-5 0.600, MRR 0.512, 10 named, all correct | top-1 0.473 (+1 report), top-5 0.600, MRR 0.521, 10 named, all correct |
| CTI-HAL as reported | top-1 0.564, top-5 0.727, MRR 0.636, 15 named, all correct | unchanged |
| Elastic strict (open set) | top-1 4/8, 1 named, open set 0 named | unchanged |
| Cross-validation, mixed | top-1 0.722, MRR 0.812 (2,565 lists, L2 0.3) | top-1 0.711, MRR 0.804 (2,536 lists, L2 3.0) |

Cross-validation top-1 is 0.0014 short of the "within 0.01" bar used for combiner refits; on
today's lists the 2026-09-30 weights score top-1 0.730 and the refit 0.731, so the gap is the
list set, not the fit. Adopted by the product owner on 2026-10-03 with that caveat.

The two opt-in combiners were refitted the same way (`--set source_country_weight=0`,
`--set group_mlp_weight=1.0`) as `attribution_combiner_no_country_fsum.json` and
`attribution_combiner_group_mlp_fsum.json`; without a refit those settings would run with no
combiner at all. Against the previous code with its previous artifact, with the same setting:

| Setting | ATT&CK folds, seed 7 | CTI-HAL anonymized | CTI-HAL as reported |
|---|---|---|---|
| `SOURCE_COUNTRY_WEIGHT=0` (CV mixed top-1 0.709) | top-1 128 -> 124 of 153, 33 -> 32 named, all correct, 0/25 removed named | top-1 24 -> 26 of 55, 8 -> 10 named, all correct | top-1 30 -> 31, 13 -> 14 named, all correct |
| `GROUP_MLP_WEIGHT=1.0` (CV mixed top-1 0.722) | top-1 125 -> 124, still one wrong name (Silence named as Kimsuky, as before) | unchanged top-1 26, 10 named, all correct | top-1 31 -> 31, 15 named, all correct |

The no-country fold drop (4 folds) exceeds the 0.02 bar used for the default; its previous artifact
was the schema-1 combiner fitted before the country vote existed. The group-network setting keeps
the wrong name that failed its naming gate on 2026-10-01 and stays opt-in only. The free-text
group model's trial combiner (`attribution_combiner_text_group.json`, 21.3.16) was fitted for the
earlier classifier code and needs the same refit before `TEXT_GROUP_WEIGHT` > 0 is used.

**Leak-free check and decision (2026-10-04).** After the merge with the BM25 fold hiding (21.3.5) the
refit was checked on the leak-free fold benchmark. Against the previous code with the 2026-09-30
combiner it lost 5 folds (seed 7 top-1 126 -> 121 of 153, MRR 0.867 -> 0.848, 26 named, all correct),
more than the 2 measured before the hiding, and gained one CTI-HAL anonymized report (25 -> 26; as
reported unchanged). The 2026-09-30 weights stamped for the merged code reproduced the previous code
exactly (0 rank changes on those folds and both CTI-HAL runs), so the product owner chose them:
the default is `attribution_combiner_0930_fsum.json` (a `metadata.fingerprint_refreshed` entry records
the stamp) and the refit stays on disk unused. The confidence labels were re-pinned to it
(`confidence_labels_fsum.json`, `provenance.repinned`): a fresh fit on the leak-free design seeds 41/53
was refused by its own rule, because only 6 rows reached the "set" label (0 misses, upper bound
0.39 against the 0.20 bar; the adopted fit had 24). The no-country and group-network refits above
remain the opt-in artifacts.

#### 21.3.10 Labelled-sentence exemplars: technique recall on real text (2026-09-30)

Diagnosis. On the 55 CTI-HAL reports the median share of a report's annotated techniques that
reached the accepted set was 0.31. Of the annotated statement labels, 25% were accepted, 40% were
in the candidate pool but left in review (median rank 3) and 20% never entered the pool; the
measurement itself (statements shown per report) cost almost nothing. The loss was therefore the acceptance
decision on real prose, which 8.2's exemplar evidence targets. An offline prototype over saved
candidate features (18 configurations of K, temperature and duplicate cut-off, including the
strict 0.95 cut-off shipped) moved the report median from 0.26 to 0.39-0.41 with precision up in
every configuration. The prototype printed CTI-HAL numbers next to the public ones, so the
configuration was not chosen blind; the gain was stable across all 18, which limits the selection
effect. Leak checks: 5 CTI-HAL statements are exact copies of an exemplar and 4.6% have one at
cosine >= 0.95; both are excluded at query time by the duplicate rule.

Same-code comparison. "Before" is a schema-3 model fitted without exemplars on the same code.
It reproduces the committed CTI-HAL mapping and clinic artifacts exactly, so it equals the
shipped behaviour and is what `TECHNIQUE_EXEMPLARS_ENABLED=false` restores (8.2).

| benchmark | before | with exemplars |
|---|---|---|
| public held-out pooled F1 / P / R | 0.318 / 0.485 / 0.237 | **0.416 / 0.515 / 0.349** |
| public held-out hit@1 / hit@10 / MRR | 0.400 / 0.748 / 0.517 | **0.500 / 0.804 / 0.605** |
| public held-out F1 by split (tram / expert / derived) | 0.421 / 0.177 / 0.257 | 0.556 / 0.262 / 0.309 |
| public held-out pooled F1 without cues | n/a | 0.410 |
| CTI-HAL statements F1 [95% CI] | 0.317 [0.285, 0.352] | **0.379 [0.350, 0.411]** |
| CTI-HAL statements P / R | 0.440 / 0.248 | 0.426 / 0.342 |
| CTI-HAL statements hit@1 / hit@10 / MRR | 0.333 / 0.688 / 0.453 | 0.383 / 0.741 / 0.508 |
| CTI-HAL reports: median accepted-technique recall | 0.308 | **0.417** (36 reports up, 3 down) |
| CTI-HAL attribution top-1 / top-5 / MRR | 0.527 / 0.745 / 0.623 | 0.545 / 0.745 / 0.632 |
| CTI-HAL named hypotheses | 14, all correct | 13, all correct |
| ATT&CK folds top-1 / top-5 / MRR | 0.830 / 0.954 / 0.880 | 0.837 / 0.967 / 0.883 |
| ATT&CK folds named / removed actors named | 36 correct / 0 of 25 | 33 correct / 0 of 25 |
| ATT&CK folds accepted-technique recall / precision | 0.792 / 0.714 | 0.723 / 0.637 |

The public held-out gain is partly within-collection: the exemplars are the train splits of the
same three datasets, whose phrasing the test splits share. The CTI-HAL gain (a different source,
never used for fitting) is the cleaner estimate of transfer to real reports. The accepted-set
loss on the folds is real, not a labelling artefact: only 31 of the 144 lost correct techniques
were replaced by a sibling in the same family. ATT&CK's procedure sentences are close to the
index's own prose, where dense similarity is the most reliable signal, and the model now leans on
report exemplars instead. Attribution ranking on the folds did not suffer.

A wrong name surfaced and was fixed at its source. With exemplars the FIN6 report
"AnotherVictim" gained two accepted techniques and cleared the insufficient-evidence boundary.
Its only software match, however, was Elise via ATT&CK's alias "Page", matched on the word
"page" in a sentence about a payment page, and Elise is documented for Lotus Blossom alone. The gate named Lotus Blossom.
The alias matcher now treats ordinary-word aliases case-sensitively (5.5), which removes Elise
from five reports and Wiper from one. The final run above includes that fix: no wrong name on
either benchmark.

EvidenceForge (narrated synthetic telemetry). The first version queried the exemplars with the
raw record text and lost ranking on the development bundles. On manufacturing narrative hit@1
fell from 0.235 to 0.059, and in four of five lost top-1s Process Discovery (T1057) took first
place, with 82-94% of the exemplar vote. The record's field names ("pid", "process_name") were
pulling the neighbours. Value-only record queries (8.2) were designed on the two development
bundles; the clinic bundle was then run once:

| bundle, mode (cases) | F1 before -> after | hit@1 before -> after | red herrings with an acceptance |
|---|---|---|---|
| clinic narrative (21) | 0.231 -> **0.357** | 0.222 -> **0.333** | 2 -> 3 of 3 |
| clinic windows (7) | 0.600 -> 0.600 | 0.500 -> 0.500 | 1 -> 1 |
| clinic eCAR (17) | 0.545 -> 0.545 | 0.571 -> 0.500 | 1 -> 2 |
| branch-office narrative (11, dev) | 0.500 -> 0.444 | 0.400 -> 0.200 | 0 -> 0 |
| manufacturing narrative (21, dev) | 0.320 -> 0.333 | 0.235 -> 0.176 | 0 -> 1 |
| manufacturing eCAR (13, dev) | 0.526 -> 0.500 | 0.417 -> 0.333 | 0 -> 1 |

Reading: on narrated telemetry the exemplars trade precision for recall, and in four of the nine
bundle/mode combinations one more benign red herring is accepted (clinic `cln-rh-002#1` as
T1070.003, manufacturing `mfg-rh-001#0` as T1074). Nothing was tuned on the clinic bundle. The
change is adopted because the primary gate, the real-report target and both attribution gates
pass. Deployments whose input is mostly structured telemetry can set
`TECHNIQUE_EXEMPLARS_ENABLED=false` to get the previous gate back.

#### 21.3.11 Random-forest group and source-country models (2026-09-30)

Standalone, on 153 validation folds (seed 11: the same groups as the seed-7 evaluation folds,
different held-out sentences) using the mapped techniques the pipeline accepted:

| model | result | reference |
|---|---|---|
| group forest (scikit-learn, best of 16 settings) | top-1 0.451, MRR 0.575 | Naive Bayes 0.588 / 0.724 |
| group forest (PyTorch GPU, shipped settings) | top-1 0.418-0.444, MRR 0.555-0.571 | |
| country forest (balanced, leaf 3) | accuracy 0.62-0.66 | country-level Naive Bayes 0.47 (always "unattributed") |

On the test sets (seed-7 folds, 153 cases; CTI-HAL, 55 reports), standalone and against simple
references (one correct answer per case, so micro-F1 equals accuracy):

| measure | ATT&CK folds | real reports |
|---|---|---|
| group forest accuracy / macro-F1 | 0.36 / 0.27 | 0.00 / 0.00 |
| Naive Bayes top-1 | 0.46 | 0.11 |
| country forest accuracy / macro-F1 | 0.52 / 0.36 | 0.47 / 0.18 |
| country of the ensemble's top-ranked group, accuracy / macro-F1 | 0.87 / 0.80 | 0.85 / 0.52 |
| always "unattributed", accuracy | 0.45 | 0.71 |

Country-forest F1 per class on the folds: China 0.46, Iran 0.27, North Korea 0.22, Russia 0.15,
another state 0.40, unattributed 0.64 (real reports: Iran 0.40, Russia 0.00, unattributed 0.65). As
predictors the forests are weak, which is why the country estimate is not shown in the UI; the
country forest's value, where it has any, is as a vote that shifts close calls.

Weights were chosen on the validation folds with the learned combiner off on both sides, so the
comparison is between rank fusions:

| group forest / country forest weight | top-1 | top-5 | MRR | named (all correct) | folds better / worse |
|---|---|---|---|---|---|
| 0 / 0 | 0.601 | 0.869 | 0.716 | 27 | |
| 0.5 / 0 | 0.667 | 0.980 | 0.788 | 29 | 40 / 3 |
| 1.0 / 0 | 0.725 | 0.980 | 0.840 | 30 | 48 / 4 |
| 0 / 0.5 | 0.608 | 0.882 | 0.723 | 27 | 13 / 2 |
| 0 / 1.0 | 0.621 | 0.915 | 0.739 | 28 | 26 / 2 |
| 0.5 / 0.5 | 0.693 | 0.980 | 0.810 | 28 | 45 / 2 |

The real-report check (55 CTI-HAL reports with the statements as reported, the protocol before the
anonymized headline of 21.3.4b; combiner off on both sides; every look is listed) decided the
adoption:

| group forest / country forest weight | top-1 | top-5 | MRR | named | wrong names | reports better / worse |
|---|---|---|---|---|---|---|
| 0 / 0 | 0.491 | 0.764 | 0.611 | 11 | 1 (FIN7 as Kimsuky) | |
| 0.5 / 0 | 0.491 | 0.764 | 0.608 | 11 | 1 | 2 / 10 |
| 1.0 / 0 | 0.455 | 0.727 | 0.575 | 9 | 0 | 2 / 14 |
| **0 / 1.0** | **0.545** | **0.764** | **0.646** | **12** | **0** | **14 / 2** |
| 0.5 / 0.5 | 0.491 | 0.764 | 0.614 | 12 | 0 | 6 / 8 |

Reading: the group forest's fold gains do not transfer. On the folds a median 92% of the accepted
techniques lie in the true group's ATT&CK profile, on the real reports 65%, and the extra techniques
of real reports are genuine tradecraft that resembles other groups' documented profiles, which the
forest amplifies. It stays wired at weight 0 (its votes are visible in the classifier matrix). The
country forest transfers: on the real reports all 14 improvements are criminal groups whose evidence
it recognises as having no documented state origin (Carbanak, FIN6, FIN7), OilRig loses two reports
and the Russian groups are unchanged; on the validation folds every class improves slightly. The
source-country vote at weight 1.0 was carried to the full pipeline: the combiner was refitted
with it as a feature (seeds 11/23/37, 2,718 lists; out of fold clean top-1 0.860 against RRF 0.540,
mixed MRR 0.812 against 0.570), and the unknown-actor lists refitted the country forest without the
removed actor. Final comparison with the shipped pipeline (combiner on in both):

| benchmark | top-1 | top-5 | MRR | named (all correct) | unknown named | better / worse |
|---|---|---|---|---|---|---|
| ATT&CK folds, seed 7 | 0.837 -> 0.830 | 0.967 -> 0.961 | 0.883 -> 0.880 | 33 -> 32 | 0 / 25 | 5 / 9 |
| CTI-HAL, 55 reports (as reported) | 0.545 -> 0.564 | 0.745 -> 0.727 | 0.632 -> 0.636 | 13 -> 15 | | 14 / 3 |
| CTI-HAL, 55 reports (anonymized headline, 21.3.4b) | 0.436 -> 0.455 | 0.655 -> 0.636 | 0.527 -> 0.530 | 8 -> 10 | | 17 / 2 |

Real reports improve (as reported: cluster-aware top-1 0.673 -> 0.709, reports without software
0.222 -> 0.278; anonymized: cluster-aware top-1 0.545 -> 0.582, without software 0.125 -> 0.167), so the
gain does not come from actor names, and no wrong name appears anywhere, but the folds lose one top-1 case and 0.003 MRR. Seven of the nine
worse folds were already fine in the rank fusion and worsened only in the refitted combiner's order.
Under the no-regression rule for both attribution benchmarks this was first kept as an opt-in. On
2026-10-01 the product owner made it the default, accepting the one-fold loss on the ATT&CK folds for
the real-report gains, which hold on the anonymized headline. The previous configuration remains as
an opt-out (9.13); the published attribution artifacts were regenerated with the new default.
The anonymized top-5, MRR and cluster-aware top-1 were superseded on 2026-10-01 by the anonymizer
fix (0.600, 0.512 and 0.564, 21.3.14).

#### 21.3.12 Target OS and operating systems named in the text (2026-10-01)

Measured against a same-code baseline (5.7). With no Target OS and text detection off, all 1,391
public held-out cases and all nine EvidenceForge bundle/mode runs are identical to the baseline, so
the change is inert unless used.

**Operating systems named in the observation (`TECHNIQUE_PLATFORM_FROM_TEXT`).** Before measuring,
the detector was checked on the public train and dev splits only: 786 of 7,610 sentences name an
operating system (704 Windows, 61 Linux, 48 macOS, 6 ESXi), and the rule would remove 9 of their 983
gold labels (user-agent strings, multi-OS paragraphs). User agents and "Kali Linux" are ignored since.

| benchmark | without | with | changed |
|---|---|---|---|
| Public held-out, pooled (1,391) | F1 0.416, P 0.515, hit@1 0.500, hit@10 0.804 | F1 0.417, P 0.516, hit@1 0.501, hit@10 0.805 | 78 cases, 6 better / 3 worse |
| Public held-out, expert (157) | P 0.700, hit@1 0.580 | P 0.718, hit@1 0.592 | |
| CTI-HAL statements (968) | F1 0.379, hit@1 0.383 | F1 0.378, hit@1 0.383 | 14 statements, one correct technique lost |
| ATT&CK folds, attribution | top-1 0.830, MRR 0.880, 32 named, 0/25 | identical | 1 fold, one extra technique |
| CTI-HAL attribution, anonymized / as reported | 0.455 / 0.564, 10 / 15 named | identical | 1 report |
| EvidenceForge, 3 bundles × 3 modes | | identical | one ranking below top-1 |

Neutral everywhere: these texts rarely describe non-Windows hosts, and their "Windows" mentions only
remove Linux and macOS techniques that were not competing. Two of the three worse public cases are
user-agent discussions the detector does not recognise ("Mozilla v5.1 (Windows NT …)", "MacOS X
User-Agents"); the pattern was not tuned on them. One CTI-HAL statement lost Regsvr32 although it is a
Windows technique: removing other candidates changes the acceptance model's pool features. Adopted as
the default for what it does on Linux, macOS and ESXi narratives, which the benchmarks cannot show.

**Case Target OS (analyst setting).** EvidenceForge with the Target OS given for every observation:

| bundle (hosts) | setting | narrative F1 / hit@1 | eCAR F1 / hit@1 | Windows-log F1 |
|---|---|---|---|---|
| branch office (Windows) | none -> Windows | 0.44 / 0.20 -> 0.44 / 0.20 | 0.67 / 0.50 -> 0.67 / 0.67 | 0.67 -> 1.00 |
| manufacturing (Windows) | none -> Windows | 0.33 / 0.18 -> unchanged | 0.50 / 0.33 -> unchanged, red herring no longer accepted | unchanged |
| clinic, held-out (Ubuntu DMZ + Windows) | none -> Windows | 0.36 / 0.33 -> 0.29 / 0.22 | 0.55 / 0.50 -> 0.42 / 0.36 | unchanged |
| clinic, held-out | none -> Windows + Linux | unchanged | unchanged | unchanged |

A correct Target OS helps or is neutral. A wrong one hurts: the clinic intrusion starts on a Linux
portal (cron persistence, `/etc/shadow`, `chmod`), and "Windows" removed exactly those techniques. The
analyst guidance is therefore to pick every operating system the activity touched and to leave the
field empty when unknown; the setting is never inferred from the evidence.

#### 21.3.13 Distinguishing the top three: a pre-registered reorder that did not replicate (2026-10-01)

**Design data, every look.** About 80 configurations of the rule (margins, the guard, separation
triggers, K=3 recomputed discrimination, a learned pairwise model, the group network as a judge,
victimology, Condorcet, size-normalised mass) were replayed offline over the seed-7 folds and both
CTI-HAL variants (`top3`/`topk` rows). The best was the rule of 10.10 at margin 1.25 with
tolerance 0.75: folds 127 -> 132 (+5/-0), anonymized CTI-HAL 25 -> 27 (+2/-0), as reported
31 -> 31. Any margin from 1.0 to 1.5 gave the same result. Because these sets chose the rule, the
rule and its acceptance bar were committed before any fresh run (10.10).

**Fresh fold seeds 41 and 53** (153 known rows and 25 removed actors each, combiner active on every
row, stage off, replayed with `scripts/top3_replay.py`):

| seed | top-1 | top-5 | MRR | named (all known correct) | removed actors named | top-1 given top-3 | rule at 1.25 |
|---|---|---|---|---|---|---|---|
| 41 | 0.863 | 0.948 | 0.899 | 37 | 1 of 25 (G1053 as G0027) | 0.936 | +1/-1 |
| 53 | 0.876 | 0.935 | 0.909 | 35 | 0 of 25 | 0.944 | +1/-0 |

Pre-registered bar: at most one loss pooled (met), net +2 per seed (not met: 0 and +1) and a
largest correct-leader gap of at most 1.15 (not met: 1.54 on seed 41). The allowed fallback,
raising the margin to the smallest value with no loss and 0.1 slack (1.75), leaves seed 41 at
+0/-0 and seed 53 at +1/-0, so the rule is **not adopted**. The exploratory sweep on the same
seeds fails everywhere (wins/losses on seeds 7 | 41 | 53):

| variant | seed 7 | seed 41 | seed 53 |
|---|---|---|---|
| margin 0.75 | +6/-2 | +3/-2 | +1/-3 |
| margin 1.0 | +5/-0 | +2/-2 | +1/-0 |
| margin 1.25 (pre-registered) | +5/-0 | +1/-1 | +1/-0 |
| margin 2.0 | +3/-0 | +0/-0 | +1/-0 |
| leader separation < -0.5, promote max D | +5/-2 | +2/-3 | +1/-3 |
| leader separation < -0.3, promote max D | +7/-8 | +3/-8 | +2/-8 |

Reading: the seed-7 gain was mostly a selection effect, and the fresh seeds show what
pre-registration is for. Where the condition holds (another top-three member leads D by at least
1.25), the leader was right in 1 of 8 fresh rows and the favourite in 2. The leader is unreliable
there, and promoting the favourite does not fix it. The analyst view reports this as "evidence
pulls both ways" (10.10), and naming is unaffected (such leaders have negative separation).

**Same-code baseline and parity.** The main checkout's `*_current.json` artifacts were regenerated
on 2026-10-01 by the target-OS change (`platform_from_text`, 21.3.12, since merged). They differ
from the code measured here in one technique on one row per set, so the before artifacts here are a same-code
run at a5de814. Seed 7: top-1 0.830, MRR 0.880, 32 named. CTI-HAL anonymized: 0.455 and 10 named.
As reported: 0.564 and 15 named. With the instrumentation added, every previously recorded field
is identical row for row on seed 7 (153 rows) and on anonymized CTI-HAL (55). The new summary
fields read: seed 7 top-1 given top-3 0.894, pairwise #1-vs-#2 0.948; CTI-HAL anonymized 0.806 and
0.862 (0.862 and 0.926 after the anonymizer fix, 21.3.14).

#### 21.3.14 Shortlist recall on real reports (2026-10-01)

**Diagnosis** (anonymized CTI-HAL headline, 24 reports with the truth outside the top three). The
software matcher misses no ATT&CK software in any of them. The malware these reports name is
either not in ATT&CK (Bateleur, TONEDEAF, TinyMet, Ammyy) or not linked to the true group, so
better software matching recovers nothing. The primary causes:

- Carbanak's profile is 9 techniques, all inside FIN7's (5 reports, not fixable).
- The anonymizer removes the Carbanak malware name, which is also the group name (3 reports,
  measurement policy).
- The technique scorers favour small profiles over groups that explain all of the evidence
  (7 reports).
- Thin, generic evidence (4 reports).
- Shared software favours another group (2 reports).
- The country vote gives no vote to groups whose class got probability 0 (1 report).

**Measurement fix.** The anonymizer kept a name that ends a sentence: 3 CTI-HAL statements and 3
seed-7 folds read their group's name. It is closed in the benchmark (commit 550cce9); results
after the fix are in this section.

**Pre-registered design protocol** (written before any run). The variants, all at the rank-fusion
level with the learned combiner off:

1. `NAIVE_BAYES_WEIGHT` 2.0;
2. `NAIVE_BAYES_WEIGHT` 3.0;
3. `IDF_OVERLAP_LENGTH_NORM` 0.5;
4. variants 1 and 3 together;
5. `SOURCE_COUNTRY_SCORE_ALL`;
6. the best weight variant combined with variant 5.

Each runs on fold seeds 41 and 53, known rows only, against the same-code baseline.

Selection over the 306 pooled known rows:

- **Eligible:** top-1 at most 2 rows below the baseline, MRR at most 0.005 below it, and no wrong
  named group.
- **Choice:** the most top-5 rows, with MRR breaking ties.
- **Bar to continue:** at least +3 top-5 rows over the baseline. A variant that clears it is
  carried to a combiner refit and evaluated once on the seed-7 folds, the anonymized and
  as-reported CTI-HAL runs, the public mapping gate and the EvidenceForge development bundles.
- **Below the bar:** nothing is refit, and the result is recorded here.

**Result of the design sweep: nothing clears the bar, nothing is adopted.** Pooled over the 306
known rows of seeds 41 and 53, rank fusion, combiner off:

| variant | top-1 | top-3 | top-5 | MRR | named | wrong names | top-5 better / worse |
|---|---|---|---|---|---|---|---|
| baseline | 194 | 254 | 272 | 0.747 | 61 | 0 | |
| Naive Bayes weight 2.0 | 190 | 250 | 273 | 0.735 | 60 | 1 | +5 / -4 |
| Naive Bayes weight 3.0 | 188 | 245 | 269 | 0.724 | 61 | 1 | +3 / -6 |
| IDF length norm 0.5 | 189 | 251 | 275 | 0.734 | 61 | 0 | +5 / -2 |
| both | 186 | 245 | 271 | 0.723 | 59 | 0 | +4 / -5 |
| source country scores every group | 193 | 254 | 271 | 0.745 | 61 | 0 | +0 / -1 |

Reading: weighting coverage more, the change the real-report diagnosis pointed at, costs first
places on the ATT&CK folds, and the Naive Bayes variants name a wrong group. The best shortlist
gain (+3 top-5 rows with the length norm) is below the bar and costs five first places. The
real-report benefit would come largely from CTI-HAL's large, prominent truths, which the
diagnosis warned about. Variant 6 was not run, because no weight variant was eligible. As
pre-registered, CTI-HAL was not consulted for these variants, and the source-country option was
removed again so the combiner artifact stays valid.

**The anonymizer fix, same code** (frontend-next engine; only the benchmark's evidence
anonymization differs):

| set | top-1 | top-3 | top-5 | MRR | named (all correct) |
|---|---|---|---|---|---|
| ATT&CK folds, seed 7 (153 + 25 removed actors) | 127 -> 127 | 142 -> 142 | 147 -> 147 | 0.880 -> 0.880 | 32 -> 32; removed actors named 0 of 25 |
| CTI-HAL anonymized headline (55) | 25 -> 25 | 31 -> **29** | 35 -> **33** | 0.530 -> **0.512** | 10 -> 10 |
| CTI-HAL as reported (55) | 31 -> 31 | 37 -> 37 | 40 -> 40 | 0.636 -> 0.636 | 15 -> 15 |

The two Carbanak reports that read "... known as Carbanak." lose the S0030 match and fall from
rank 2 to 71 and out of the list. The folds' three leaking statements changed no outcome.

**Cluster-aware top-3** (`cluster_top3`, a group ATT&CK documents as overlapping with the truth is
in the top three; always reported beside the strict figure):

| set | strict top-3 | cluster-aware top-3 |
|---|---|---|
| CTI-HAL anonymized | 29 (0.527) | 36 (0.655) |
| CTI-HAL as reported | 37 (0.673) | 41 (0.745) |
| ATT&CK folds, seed 7 | 142 (0.928) | 142 (0.928) |

The Top-three view now lists, for each of the three, the groups ATT&CK documents as overlapping
with it and their rank in the case. FIN7 ranked first shows "Carbanak #8".

**What remains.** About 8 to 10 of the 24 reports outside the top three before the anonymizer fix (26 after it)
cannot be reached by a
leak-safe pipeline change. In five Carbanak reports ATT&CK documents almost none of the report's
techniques for Carbanak. Three come back only through the Carbanak malware, whose name is the
group's. Two or three carry only generic evidence. The rest need evidence ATT&CK does not link to
the group: malware not in ATT&CK, such as Bateleur, TONEDEAF and TinyMet, or tools documented for
other groups.

#### 21.3.15 Clusters, confidence labels, software questions and window recall: pre-registration (2026-10-02)

Product-owner decisions of 2026-10-02:

- name a documented-overlap pair as a cluster, counted as correct when the truth is in it and as
  wrong otherwise;
- show calibrated confidence labels;
- build interview questions about malware that only one top-three group uses, recording found /
  not found / not checked;
- work on real-report recall.

This entry fixes every parameter and acceptance bar **before** the fresh fold seeds 43 and 47 are
opened. Their stage-off baselines were run on 2026-10-01 on frontend-next and have not been read
beyond row counts and combiner status.

**What was already seen.** The designs used seeds 41 and 53. The cluster and window designs also
replayed seed 7 and CTI-HAL, so those sets are disclosed as already seen for those two tracks and
are run once at the end as reports, not as tests. The Elastic set (8 anonymized reports,
data/benchmarks/elastic, local evaluation only) is a report-only check. Nothing is ever fitted or
tuned on seeds 7, 41, 43, 47, 53, 59 or 61, or on CTI-HAL. The combiner trains only on seeds
11/23/37.

**Order.** Instrumentation, then cluster naming (A), then confidence labels (B), software questions
(C) and window recall (D). Each is measured on its own.

**A. Documented-cluster naming**

- **Settings.** `ATTRIBUTION_CLUSTER_NAMING`, default off.
- **Members.** Pairs only: the gate-window leader plus one twin.
- **Twin.** A gate-window member that ATT&CK relates to the leader (`attack/related.py`: a
  description reference, or non-tool software used by at most three groups) and that also either
  nests (profile containment >= 0.8) or does not separate from the leader (|pair separation| <
  0.5). If several qualify, the twin with the largest gate-window mass is chosen.
- **Cluster evidence.** Discrimination is recomputed over the gate window with the pair merged
  into one unit.
- **A cluster is named when all of these hold:**
  - the leader passes the lower stage checks;
  - the leader-consistency guard did not fire;
  - cluster separation >= 0.5;
  - cluster mass >= 3.5, or >= 3.0 for a nested pair;
  - the members' combiner probabilities sum to >= 0.8.
- **Nested twin.** A single name whose nested twin is in the window becomes the cluster, or is held
  at shortlist when the cluster fails.
- **Bars** on seeds 43+47 pooled:
  - A1: zero wrong cluster names on known rows and zero removed actors named through a cluster. One
    failure means it is not adopted, with no retuning.
  - A2: single names identical to the off run, apart from correct nested upgrades.
  - A3: at least 2 correct cluster names.
  - A4: the on run equals the offline replay on every row.
  - A5: if 2 or more removed actors clear two of the three cluster bars, only the nested scope may
    ship.

**B. Confidence labels (display only)**

- **Settings.** `CONFIDENCE_LABELS_ENABLED`, default off.
- **Model.** Logistic calibrators for "leader correct" and "truth in the top 2 / top 3". Inputs:
  logit combiner P, gate separation, logit null P, log accepted count, leader software vote.
- **Fit.** Seeds 41+53, removed actors counted as wrong leaders, GroupKFold(5) by group.
- **Thresholds.** Strong at >= 0.88 (one-sided 95% Clopper-Pearson upper bound on the leader error
  <= 5%). "One of these 2/3" at >= 0.865. Insufficient evidence is open; that rule was chosen after
  seeing seed 7.
- **When a cluster is named,** the label describes the cluster as a two-group set and never says
  "strong single answer".
- **Bars** on seeds 43+47 (known + removed):
  - B1: strong leader error <= 5.0% pooled and <= 8% per seed.
  - B2: ECE of the calibrated leader probability <= 0.06 pooled.
  - B3: "one of these" covers the truth in >= 80% of its rows pooled.
  - B4: decisions, rankings and exports are identical with the labels on and off.

**C. Software questions (pilot-ready)**

- **Settings.** `INTERVIEW_SOFTWARE_QUESTIONS`, default off.
- **Questions.** For each top-three group, malware that only it documents in ATT&CK and whose only
  user is inside the gate window. At most 3 questions per case, asked only while the case is not
  named.
- **Answers.** Found adds the software as evidence. Not found may only lower the leader's
  separation. Not checked records nothing.
- **Guard.** A name must not rest on interview answers alone: the leader's separation without them
  must be >= 0.
- **Order in the gate.** Single gate, then the cluster step on the evidence before any "not found"
  answer, then the "not found" filter. The filter can only move hypothesis or cluster to
  shortlist.
- **Bars** per fresh seed, with the simulated answer model q in {0.1, 0.3, 1.0} and a wrong-"found"
  rate in {0, 0.05}:
  - C1: at a false rate of 0, no wrong name and no removed actor named.
  - C2: at least 8 names gained at q = 0.3 and at least 30 at q = 1.0.
  - C3: no correct name lost.
  - C4: all answers "not found" leaves the named count unchanged.
  - C5: at a 5% false rate, at most 1.0 wrong or removed-actor names per seed. This needs product
    owner sign-off.
  - C6: replay and pipeline agree within ±2 names.
  - C7: monotone: a "not found" never adds or changes a name.
  - C8: C1 and C5 still hold with A on.

**D. Window recall (refit per experiment, defaults unchanged)**

- **D2: combiner window 40.** Refit with `--top-n 40` on seeds 11/23/37.
  - Fold bars per fresh seed: top-1 and top-3 >= baseline - 1, named correct >= baseline - 2, no
    wrong name, removed actors named <= baseline.
  - Cross-validation bar on the lists common to both windows: randomised top-1 >= 0.682, top-5 >=
    0.905, AUC >= 0.85.
  - Reported once: CTI-HAL anonymized strict top-3 >= 31/55, truth in the window >= 40/55, no wrong
    name; as reported top-3 >= 37 and >= 15 names, all correct.
- **D1: strict-twin admission.** Description reference or malware with exactly two users; combiner
  schema 3; an admitted ineligible group is never named alone.
  - Same fold bars, plus no removed actor's twin named.
  - Reported once: CTI-HAL anonymized strict top-3 >= 29, truth in the window >= 42/55,
    cluster-aware top-3 >= baseline + 3, no wrong name.

**Expectation stated up front.** Strict top-3 on anonymized CTI-HAL is expected to move by about +0
to +3 of 55. Five Carbanak reports cannot be ranked above FIN7 on technique evidence, and three more
lose the Carbanak malware to anonymization. Most of the visible gain will come from counting
documented clusters, a choice the product owner has made explicitly.

**Results (2026-10-02).** Every track was measured once on seeds 43 and 47 with 153 known folds
and 25 removed actors each, on the same combiner artifact (pipeline fingerprint
`fb4404a2819581fefb7b14b5`). Source fingerprints match the committed code. B is adopted.
A passed but was then narrowed by the product owner (amendment A′ below), and it stays off until
A′ passes. C ships off as a pilot. D1 and D2 are not adopted.

| track | verdict | what the fresh seeds showed |
|---|---|---|
| A clusters | passed; narrowed (A′), off until re-confirmed | 4 correct cluster names, 0 wrong, 0 removed actors; single names identical |
| B confidence labels | adopted (default on) | "strong" wrong 8 of 192 (4.2%), calibration error 0.041, "one of these" 14 of 14 |
| C software questions | pilot, default off | many correct names when answers are reliable; about 2.4 extra wrong names per seed at 5% wrong "found" |
| D1 twin admission | not adopted | seed 43 top-1 137 -> 134, seed 47 top-3 144 -> 142 |
| D2 window 40 | not adopted | cross-validated randomised top-1 0.630, bar 0.682 |

**A. Clusters: all bars pass.**

| seed | top-1 | top-3 | single names (correct) | cluster names (correct) | removed actors named |
|---|---|---|---|---|---|
| 43 off | 137/153 | 145 | 39 (39) | 0 | 1/25 |
| 43 on | 137/153 | 145 | 39 (39) | 2 (2) | 1/25 |
| 47 off | 126/153 | 144 | 36 (36) | 0 | 0/25 |
| 47 on | 126/153 | 144 | 36 (36) | 2 (2) | 0/25 |

- A1: no wrong cluster names and no removed actor named through a cluster.
- A2: the single names are identical in both arms.
- A3: four correct cluster names, against a bar of 2:
  - seed 43, Aquatic Panda / APT41, truth APT41, separation 0.736. This one recovers a case
    whose single leader was wrong.
  - seed 43, GOLD SOUTHFIELD / FIN7, truth GOLD SOUTHFIELD, separation 0.727.
  - seed 47, LuminousMoth / Mustang Panda, truth LuminousMoth, separation 0.727.
  - seed 47, ShinyHunters / Scattered Spider, truth ShinyHunters, separation 0.512.
- A4: the on-runs equal the offline replay (`scripts/cluster_replay.py`) on every row.
- A5: no removed actor cleared two of the three cluster bars, so the full scope ships.

The removed actor named on seed 43 (OilRig, named as FIN6, separation 0.591) is named with
clusters off too. It is a single name from the existing gate, not from this track; the same kind
of error appeared on design seed 41 (Storm-0501 named as Threat Group-3390). It is listed under
open questions below.

**B. Confidence labels: all bars pass.** The frozen artifact was evaluated on seeds 43 and 47
(known plus removed actors, 356 rows, `confidence_labels_confirm.json`):

| set | strong (right) | one of these (right) | open | leader calibration error |
|---|---|---|---|---|
| seed 43 | 99 (96) | 10 (10) | 69 | 0.050 |
| seed 47 | 93 (88) | 4 (4) | 81 | 0.063 |
| pooled | 192 (184) | 14 (14) | 150 | 0.041 |

The bars:

- B1: "strong" wrong 4.2% pooled (upper 95% bound 7.4%), 3.0% and 5.4% per seed, against 5% and
  8%.
- B2: calibration error 0.041, against 0.06.
- B3: "one of these" covers 14 of 14, against 80%.
- B4: decisions, rankings and exports are identical with the labels on and off
  (`tests/test_confidence_labels.py`).

Half of the 8 "strong" errors are removed actors.

On real reports the labels run overconfident, so the view says so:

| set | strong (right) | leader calibration error |
|---|---|---|
| CTI-HAL anonymized | 24 (21) | 0.19 |
| CTI-HAL as reported | 30 (23) | 0.24 |

**C. Software questions: pilot only, off by default.** The answers to every question were
enumerated through the real pipeline: found with probability q when the truth documents the
malware, and a wrong "found" at rate fp otherwise. Expected names per seed:

| seed | baseline names | q = 0.3, fp = 0 | q = 1.0, fp = 0 | wrong or removed added at fp = 0.05 (q = 0.3) |
|---|---|---|---|---|
| 43 | 40 | 54.3 (+15.3) | 90 (+51) | 2.37 |
| 47 | 36 | 47.4 (+11.4) | 74 (+38) | 2.35 |

These bars pass:

- C1: no wrong name and no removed actor named when every "found" is true.
- C2: gains of 15.3/51 and 11.4/38, against bars of 8 and 30.
- C3: no correct name lost.
- C4: all "not found" changes no name.
- C7: no "not found" adds or changes a name.

These fail:

- **C5** (at most 1.0 extra wrong or removed-actor name per seed at a 5% wrong-"found" rate): 2.37
  and 2.35 per seed. One false "found" on malware that only one group uses is enough to name
  that group.
- **C6:** the fixed-window offline replay undercounts the pipeline's gains by 6 to 9 names at
  q = 1. That is a limit of the replay method, not of the feature: the pipeline re-runs retrieval
  and the window after an answer, and the replay cannot.
- **C8** (C1 and C5 with clusters on, full scope): C1 holds on both seeds. C5 fails as it does
  with clusters off: 2.47 and 2.35 at q = 0.3.

The feature therefore ships off. A supervised pilot needs the product owner to accept the C5 risk
or to require a second confirmation for "found".

**D. Window recall: neither variant adopted.**

- **D2, a 40-candidate combiner window.** Refitted on seeds 11/23/37. Randomised top-1 was 0.630
  over all lists, and at most 0.656 on the lists common to both windows, against a bar of 0.682.
  The fold benchmarks were not run, as the protocol allows once the cross-validation bar fails.
- **D1, strict-twin admission** (branch `twin-admission`, combiner schema 3, refitted on
  11/23/37):

  | seed | top-1 | top-3 | names |
  |---|---|---|---|
  | 43 off | 137 | 145 | 39 |
  | 43 on | 134 | 145 | 38 |
  | 47 off | 126 | 144 | 36 |
  | 47 on | 127 | 142 | 36 |

  Seed 43 top-1 and seed 47 top-3 fall by more than the bar of 1. There were no wrong names and no
  removed actor's twin was named. Only an unrelated fix from the branch was taken: the
  related-group memo now keys on the user limit (38653e4).

The 53% strict top-3 on anonymized real reports is therefore not fixable by widening the window
without losing folds. In most misses the true group is outside the first 20 candidates. The gain
that does reach the analyst is the cluster count, which 21.3.14 measured at a strict 0.527 against
a cluster-aware 0.655.

**Report-only runs with the full scope (2026-10-02).** These used seed 7 and both CTI-HAL
variants, the same combiner and the code at 793e640. Every row equals the published clusters-off
artifact except the rows that became a cluster:

| set | top-1 | strict / cluster-aware top-3 | single names | cluster names (truth inside) | removed actors named |
|---|---|---|---|---|---|
| seed 7 folds | 127/153 | 142 / 142 | 32 (32) | 3 (3) | 0/25 |
| CTI-HAL anonymized | 25/55 | 29 / 36 | 10 (10) | 5 (5) | n/a |
| CTI-HAL as reported | 31/55 | 37 / 41 | 13 (13) | 10 (10) | n/a |

The seed-7 clusters were APT38 / Lazarus Group, GOLD SOUTHFIELD / FIN7 and ShinyHunters /
Scattered Spider. All five anonymized CTI-HAL clusters were pairs linked only through one rare
malware family:

- FIN6 / Wizard Spider twice (Ryuk);
- Cobalt Group / FIN6 (More_eggs);
- Sandworm Team / APT38 (KillDisk);
- Wizard Spider / TA505 (TrickBot).

As reported, 6 of the 10 were FIN7 / Carbanak. Two of those were correct single FIN7 names before
(the nested-twin rule). The other 4 were the shared-malware pairs above.

**Amendment A′: same-activity links only (product-owner decision, 2026-10-02).** Every cluster above
contains the true group, but a pair such as "Sandworm Team / APT38" reads as one actor, and the two
are different actors that both used one wiper. The product owner decided that a cluster is named
only for a same-activity link. This amendment is written down before the unused fold seeds 59 and
61 are opened.

- **Rule.** The twin must satisfy the first rule (an ATT&CK relation, and nesting or |pair
  separation| < 0.5). It must also qualify through one of two routes:
  - the reviewed table documents the pair as the same or overlapping activity (13 pairs on ATT&CK
    19.2, 10.11);
  - the profiles nest (containment >= 0.8), the review has no verdict against the pair (distinct,
    unclear or false match), and no unreviewed description link exists for it.

  Neither a shared rare malware family nor a description mention is enough on its own.
- **How the rule was fixed.** A first draft counted any description mention. The review workflow
  of the draft found that many mentions say the groups are different (Ember Bear / Saint Bear
  "distinct clusters", Magic Hound / APT42 "distinct entities"), and one is a false match (Tonto
  Team / BITTER). The 37 description links were therefore classified from ATT&CK's own sentences
  before any fresh seed was opened. The classification read only the knowledge base, never a
  benchmark row.
- **Unchanged.** The bars (cluster separation 0.5, mass 3.5 or 3.0 nested, combined combiner
  probability 0.8), the guard, the nested-twin hold and the scoring stay as they were.
- **Selection.** The heaviest qualifying member under the narrowed rule wins, so a different,
  lighter twin than before can be chosen.
- **Settings.** `ATTRIBUTION_CLUSTER_SCOPE=documented` is the default; `full` keeps the first rule.
  `ATTRIBUTION_CLUSTER_NAMING` defaults off until A′ passes.
- **Already seen.** The narrowing was chosen after seeds 43/47, seed 7 and CTI-HAL were read, so
  those sets are reported, not tests. From the recorded rows, the narrowed rule keeps at most:
  - seeds 43/47: 1 of 4 clusters (LuminousMoth / Mustang Panda);
  - seed 7: 1 of 3 (APT38 / Lazarus Group);
  - CTI-HAL anonymized: 0 of 5;
  - CTI-HAL as reported: 6 of 10 (FIN7 / Carbanak).

  These are upper bounds where a lighter twin could replace a dropped one.
- **Confirmation on seeds 59 and 61.** Each seed has 153 known folds and 25 removed actors. Each
  gets an off run (default, cluster naming off) and an on run (`--set
  attribution_cluster_naming=true`), all on the same combiner artifact. Bars, pooled:
  - A′1: zero wrong cluster names and zero removed actors named through a cluster. A failure keeps
    cluster naming off, with no retuning.
  - A′2: the single names are identical in both arms apart from correct nested upgrades and nested
    holds.
  - A′3: every named cluster's link is `same_activity` or `nesting`, never `shared_software` or a
    plain `description` mention.
  - A′4: the on run equals `scripts/cluster_replay.py` over the off run on every row.
  - Reported without a bar: the number of correct cluster names, which may be 0 because the rule
    was narrowed for meaning, not for count; removed actors that clear two of the three cluster
    bars.
- **If A′1, A′2, A′3 and A′4 pass,** cluster naming defaults on with the documented scope. Seed 7
  and both CTI-HAL variants are then run once more, report-only.

**A′ results (2026-10-02, seeds 59 and 61, code fe511df, one combiner artifact): all four bars pass.**

| seed | top-1 | single names (correct) | cluster names (correct) | removed actors named through a cluster |
|---|---|---|---|---|
| 59 off | 132/153 | 34 (34) | 0 | 0/25 |
| 59 on | 132/153 | 34 (34) | 1 (1) | 0/25 |
| 61 off | 125/153 | 33 (33) | 0 | 0/25 |
| 61 on | 125/153 | 33 (33) | 0 | 0/25 |

- A′1: no wrong cluster name and no removed actor named through a cluster.
- A′2: single names identical in both arms (34/34 and 33/33).
- A′3: the one cluster, LuminousMoth / Mustang Panda (truth LuminousMoth), is a `same_activity`
  link. Only 3 of 178 seed-59 rows had a cluster candidate at all.
- A′4: the on runs equal `scripts/cluster_replay.py` over the off runs on every row.
- No removed actor cleared two of the three cluster bars.
- Every row ran with the combiner active, and off and on share one source fingerprint.

Cluster naming is now on by default with the documented scope. The value is small and honest: a
few correct pair names where the single gate holds back.

**Report-only reruns with the adopted defaults (code 2da2c4e).** These are the published
artifacts now. Every row equals the clusters-off artifact except the rows listed:

| set | top-1 | strict / related-aware / same-activity-aware top-3 | single names | cluster names (truth inside) |
|---|---|---|---|---|
| seed 7 folds | 127/153 | 142 / 142 / 142 | 32 (32) | 1 (1): APT38 / Lazarus Group |
| CTI-HAL anonymized | 25/55 | 29 / 36 / 32 | 10 (10) | 0 |
| CTI-HAL as reported | 31/55 | 37 / 41 / 38 | 13 (13) | 6 (6): FIN7 / Carbanak |

As reported, three Carbanak reports where FIN7 led now name FIN7 / Carbanak. Two correct single
FIN7 names became FIN7 / Carbanak through the nested-twin rule: Carbanak's profile lies inside
FIN7's, and ATT&CK documents the link. No removed actor is named on seed 7.

The related-aware top-3 counts any ATT&CK relation, shared rare malware included, as 21.3.14 did.
The same-activity-aware top-3 counts only the links the cluster rule accepts, and it is the honest
measure of "the right actor or its documented alias is in the top three": 32/55 (0.582) on the
anonymized headline, against a strict 29/55 (0.527).

**The single-name gate names removed actors on some seeds (not part of this work).** Every fold
seed run in this programme used clusters off for its single names. Across seven seeds, the
existing gate named 4 of 175 removed actors as a known group:

| seed | removed actors named | named as | separation |
|---|---|---|---|
| 7 | 0/25 | | |
| 41 | 1/25 | Storm-0501 as Threat Group-3390 | 0.600 |
| 43 | 1/25 | OilRig as FIN6 | 0.591 |
| 47 | 0/25 | | |
| 53 | 0/25 | | |
| 59 | 2/25 | Blue Mockingbird as APT41; Gamaredon Group as TA2541 | 0.511; 0.508 |
| 61 | 0/25 | | |

All four sit just above the 0.5 separation bar. The published "0/25 removed actors named" is
seed 7 alone; across seeds the rate is about 2% (4/175, Clopper-Pearson 95% 0.6-5.7%). Raising
the bar is a gate decision for the product owner. It needs a pre-registered check on seeds that
have not been used, because the four examples were found on these.

**Open questions for the product owner.**

- The C5 risk for a software-question pilot.
- The removed-actor single names on seeds 41 and 43 from the existing gate.
- Whether the gate's safety margin should be revisited (decision 2 of the research programme).

#### 21.3.16 Free-text group model: study and gate trial (2026-10-02/03)

**Study.** Training text was ATT&CK only (17,396 anonymised records); model selection used the 23
single-group ATT&CK campaigns (near-copies of their sentences removed from training); CTI-HAL (names) and
Elastic (strict) were scored once. Six families were tuned (sentence kNN, sentence classifier, pooled
MLP, attention MIL, prototypes on ATTACK-BERT embeddings; TF-IDF words). The TF-IDF group document won on
the campaigns and on the test sets. Standalone on CTI-HAL it reached top-1 0.491 / MRR 0.584 (pipeline
0.455 / 0.512, not significant); combined with the group network (9.13) MRR 0.628. Audits found no name
leak, but 12 of its 27 CTI-HAL hits rest on ATT&CK wording that cites the same vendor reports CTI-HAL
quotes; malware-name lookups drive 9. `scripts/text_group_study/README.md` has the protocol and numbers.

**Gate trial** (pre-registered, `data/audits/text_group_gate_trial_2026-10-03/`): weight 1.0 with a
combiner refitted on seeds 11/23/37 (`data/models/combiner/attribution_combiner_text_group.json`)
against weight 0 with the shipped combiner, from one frozen code copy; the baselines reproduced the
published artifacts (folds 153/153 rows; CTI-HAL metrics).

| Set | Before | After | Wrong names |
| --- | --- | --- | --- |
| ATT&CK folds seed 7: top-1 / top-5 / MRR / named | 0.830 / 0.961 / 0.880 / 32 | 0.824 / 0.935 / 0.880 / 29 | 0 -> 0; removed actors 0/25 -> 0/25 |
| CTI-HAL anonymized: top-1 / top-3 / top-5 / MRR | 0.455 / 0.527 / 0.600 / 0.512 | 0.491 / 0.618 / 0.655 / 0.572 | 0 -> 0 (10 named) |
| CTI-HAL as reported: top-1 / top-3 / top-5 / MRR | 0.564 / 0.673 / 0.727 / 0.636 | 0.564 / 0.745 / 0.764 / 0.666 | 0 -> 0 (13 named) |
| Elastic strict (diagnostic): top-1 / top-3 / MRR | 4/8 / 0.500 / 0.537 | 4/8 / 0.625 / 0.553 | 0 -> 0; open set 0/22 named |
| Elastic tradecraft (diagnostic): top-1 / MRR | 1/8 / 0.209 | 2/8 / 0.346 | 0 -> 0 |

The technique-mapping gate is unchanged by construction (retrieval never reads the setting). The fold
rule "top-1 not below baseline" fails by one case, so the default stays 0. Checks: the fold change is
noise (5 gained / 6 lost, exact p = 1.0; top-5 1/5, p = 0.22) and survives closing the BM25 leak of the
fold benchmark (the lexical index is not hidden by `FoldStore`; with it hidden both arms drop, trial
0.817 vs 0.824). All six losses are thinly documented groups: the fully blinded text model votes for a
rival while the technique-list members still see the held-out techniques in the group lists. On CTI-HAL
the MRR gain is 24 better / 2 worse (p = 1e-5; group sign-flip one-sided p = 0.031, the minimum for six
groups); about 28% of it depends on same-source ATT&CK records, 42% partly. The Elastic gains (4/0 strict,
7/0 tradecraft) do not: ATT&CK cites none of the eight posts.

#### 21.3.17 Exemplars in Chroma and retrieval tuning without the LLM verifier (2026-10-02)

Pre-registered in `docs/preregistration/chroma_exemplars_2026-10-02.json` (it names this section
21.3.15; the number moved when the Top-three and free-text sections were merged first). Goal:
move the labelled-sentence exemplars (21.3.10) into Chroma, tune what is retrieved and how the
votes combine, and see how much of the LLM verifier's recall local acceptance can recover.

**Migration (adopted as tooling, default unchanged).** `scripts/sync_exemplars_to_chroma.py`
copies the 6,360 public-train exemplar vectors into the collection
`technique_exemplars-0f8d2e19a59967c63b03ea71-s1-m32-efc400`. Chroma re-normalises float32
vectors on insert (5,208 of 6,360 rows bitwise identical, largest component change 1.5e-8), so the
Chroma copy is its own library (fingerprint `d8fbb0565e9addeaa3cb048c`) verified within a 1e-6
tolerance, and an acceptance model must be fitted on the library it serves with
(`TECHNIQUE_EXEMPLAR_BACKEND=npz|chroma|auto`, the model stores the library fingerprint and fails
closed otherwise). With the shipped model on the Chroma library, every accepted set and candidate
row on public dev and test, CTI-HAL and both EvidenceForge development bundles was unchanged
(largest feature difference 9.4e-7). Copies of `attack_techniques` and `attack_groups` with
other HNSW settings exist for measurement; the live collections were never modified. The default
backend stays `npz`.

**Tuning (not adopted).** `scripts/tune_retrieval.py` swept 27 variants in 9 families on public dev
only (grouped 5-fold CV, seeds 7/11/13, paired cluster bootstrap): exemplar bandwidth (k, tau),
exemplar-only candidates, vote normalisation, per-segment votes, exact vs approximate dense search,
dense depth, relative margin, candidate pool and L2. Every family's best variant was within +/-0.003
micro-F1 of the reference with a 95% interval spanning zero, and none replicated on seeds 41/53.
Search settings are not the bottleneck: about 80% of the expected techniques reach the candidate
pool, but only about 40% are accepted.

**Gates for the aligned refit (T1: Chroma library, fit settings equal to serving).** Public test
micro-F1 0.417 -> 0.414 with cues (precision +1.7 points, recall -1.1) and 0.411 -> 0.408 without;
CTI-HAL statements 0.378 -> 0.380. ATT&CK folds seed 7 top-1 0.830 -> 0.837, all names correct
(fold harness before the 2026-10-03 BM25 fix, 21.3.5). On the CTI-HAL reports as written T1 named one Carbanak
report as FIN7, a hard failure, so the clinic bundle and Elastic were not run and T1 is not adopted.

**Against the verifier** (the 10 CTI-HAL reports of the 2026-09-28 production run, same sentences):

| | Precision | Recall | F1 |
|---|---|---|---|
| production, LLM verifier | 0.40 | 0.51 | 0.45 |
| local, shipped model | 0.48 | 0.38 | 0.43 |
| local, T1 | 0.51 | 0.38 | 0.44 |

Removing the verifier needs a better decision on which pooled candidates to accept, not better
search.

#### 21.3.18 Fine-tuning cribl-decision on the Jev C1 technique question and a family head (2026-10-11/13)

An experiment, not wired into the product. `cribl-ai/cribl-decision-1.0` is an open LoRA (r 16) on
`Qwen/Qwen3.5-4B-Base`, a sibling of TypeSafe Jev. It was asked the Jev C1 question: one report
sentence, the pipeline's top-10 candidates plus `none_of_these`, three option orders. Answers came from
a lettered readout (fp32 hidden state × `lm_head` letter rows) in the chat format chosen on dev. The
question text is byte-identical to Jev's (cache hashes 1977/1977 dev, 2904/2904 CTI-HAL) and the metric
code replays Jev C1 exactly.

**Arms.**
- **B0**: the published adapter.
- **B1**: that adapter trained further.
- **B2**: a fresh LoRA on the plain base.

**Training.** B1 and B2 were trained with 4-bit QLoRA for 2 epochs on 6,308 public train sentences
(tram/expert/derived). Train candidates came from the pipeline with the exemplar index off, because the
index holds the train sentences. Settings: lr 1e-4, effective batch 16. Checkpoints were taken every
half epoch and selected on dev only.

**Pre-registration.** `docs/preregistration/cribl_finetune_2026-10-11.json` has amendments 1–5.
B2's second epoch and all pod scoring ran on a rented RTX 4090. For the pod's CTI-HAL runs, the owner
allowed a one-off upload, deleted with the pod.

**Registered rule (the answer with the highest probability, `none_of_these` = no pick = miss).** The
trained arms learned to abstain: 27.6% of training targets were `none_of_these`, against 16.2% on dev.
They said it on 22–23% of dev, 33% of test and 35% of CTI-HAL statements, and only about 40% of those
had no correct candidate. Every registered success criterion on test and CTI-HAL failed except CTI-HAL
calibration.

**Never-abstain rule (amendment 4).** The pick is the most probable of the ten techniques, and
P(`none_of_these`) stays as a flag. `rescore.py` applies it offline to the stored predictions. It was
chosen on dev after the test result was seen, then confirmed once on CTI-HAL, which no arm had been
scored on before.

| Exact match (parent-aware) | dev 659 | public test 1,391 | CTI-HAL 968 |
|---|---|---|---|
| Pipeline rank-1 | 0.589 (0.669) | 0.501 (0.567) | 0.383 (0.475) |
| Jev C1, as reported / never-abstain | 0.514 / 0.520 | n/a | 0.460 / 0.478 (0.596) |
| B0, registered / never-abstain | 0.480 / 0.486 | 0.416 / 0.420 | 0.409 / 0.415 |
| B1 step_9462, registered | 0.618 | 0.466 | 0.347 |
| **B1 step_9462, never-abstain** | **0.687 (0.744)** | **0.562 (0.614)** | **0.431 (0.528)** |
| B2 step_12616, never-abstain | 0.680 | 0.552 | 0.428 |

**Amendment 4 outcome on CTI-HAL.**
- **Against the pipeline**: B1 minus pipeline rank-1 is +0.048, 95% CI [+0.019, +0.075]. This part of
  the headline passes. On test (not confirmatory) it is +0.061, CI [+0.038, +0.085].
- **Against Jev C1**: B1 is −0.029 versus Jev as reported, CI [−0.059, −0.003], and −0.048 versus Jev
  under the same rule. **The headline fails.**
- **Calibration**: the ECE of P(pick) is 0.256, against Jev's 0.283 under the same rule. This criterion
  passes.
- **Cribl's adapter against a fresh LoRA**: B1 is +0.003 above B2 on CTI-HAL, CI [−0.011, +0.019]. Not
  shown.
- **The `none_of_these` flag** separates "gold not among the candidates" with AUROC 0.80 on test and
  0.73 on CTI-HAL.

**Conclusion.** Under the never-abstain rule, a small local judge trained on public sentences beats the
pipeline's rank-1 on all three sets. The gain shrinks from public data to real reports (+0.10 dev,
+0.06 test, +0.05 CTI-HAL), and it stays below hosted Jev C1 on CTI-HAL. Cribl's starting adapter adds
nothing measurable over a fresh LoRA.

**Cost.** Training took 6.7 h for B1 on the laptop. B2 took 3.1 h on the laptop plus 35 min on the pod.
Peak GPU memory was 4.1 GB. The pod sessions cost about $1.60 in total.

Predictions and metrics are in `data/audits/cribl_ft/`: `<arm>_<split>.json(l)` and, for the
never-abstain rule, `*_forced.json`. Adapters are in `data/models/cribl_ft/`. CTI-HAL rows are local
only.

**B3: real-report sentences (2026-10-13, prereg `cribl_annoctr_2026-10-13.json`, amendments 1–4).**
B3 continues B1 step_9462 on 1,772 AnnoCTR training sentences (CC-BY-SA 4.0; Lange et al.,
LREC-COLING 2024), each paired with a public replay sentence. AnnoCTR links real vendor-report
sentences to ATT&CK techniques, almost only at parent level, so its rows use family targets.

Training ran for 4 epochs on a community A100, a run the owner judged needed more than the planned 2
epochs. Checkpoints were taken every half epoch. The 2-epoch checkpoint was chosen on AnnoCTR dev
(right family 0.580, against B1's 0.370), and dev levelled off after it.

Results, never-abstain rule:

| | CTI-HAL exact (right family) | AnnoCTR test right family (exact) | public test exact |
|---|---|---|---|
| Pipeline rank-1 | 0.383 (0.475) | 0.358 (0.227) | 0.501 |
| Jev C1 as reported / never-abstain | 0.460 (0.567) / 0.478 (0.596) | n/a | n/a |
| B1 | 0.431 (0.528) | 0.429 (0.285) | 0.562 |
| **B3** | **0.441 (0.576)** | **0.580 (0.290)** | 0.537 |

**Pre-registered outcome.**
- **Headline: fails.** On CTI-HAL exact, B3 minus B1 is +0.010, CI [−0.013, +0.036].
- **AnnoCTR test, right family: passes.** B3 minus B1 is +0.151, CI [+0.125, +0.178]. The gain is
  largest on implicit mentions (0.375 → 0.551; explicit 0.523 → 0.630).
- **CTI-HAL against Jev C1: fails.** B3's exact 0.441 is below 0.460. The gap narrowed, to −0.019 with
  CI [−0.048, +0.007].
- **Public test, forgetting check: fails.** B3 minus B1 is −0.025, against the −0.02 limit (TRAM −0.049,
  expert −0.076).

**Reading.** Real-report training teaches the technique family but not the sub-technique. On CTI-HAL,
B3 gains +0.049 in right-family accuracy over B1 (CI [+0.025, +0.074]) and reaches Jev as reported
(+0.009, CI [−0.015, +0.035]). Exact-id accuracy barely moves, because AnnoCTR's parent-level labels
carry no signal for choosing between sub-techniques.

Calibration improved: CTI-HAL ECE 0.181, against 0.256 for B1. The `none_of_these` flag's AUROC rose
from 0.73 to 0.74 on CTI-HAL and from 0.68 to 0.78 on AnnoCTR test.

**Cost.** Training took 2.4 h on the A100, and the pods cost about $5.20 in total. That includes about
$1 of aborted starts: an A40 run stopped for the epoch change, and a CPU-bound H100 NVL that needed
Triton 3.7.1 for flash-linear-attention on Hopper.

**AnnoCTR text fix (2026-10-13, after B3).** `prepare_annoctr.py` built each sentence from AnnoCTR's
`context_left + mention + context_right`. Those fields are stripped, so the mention was glued to its
neighbours ("installsransomware", "theBATfile"), and the retriever never saw the word that names the
technique. The script now uses `_context_left`/`_context_right`, which keep the spaces. It also strips
markdown links that carry a title, and images. A sentence with several annotated mention spans is now
one case: train 1,500, dev 336 and test 738 scorable sentences, against 1,884, 443 and 959 before.
The candidate files were rebuilt with `--max-candidate-rows 40`. B3's own inputs are kept in
`data/audits/cribl_ft/b3_inputs/`.

On AnnoCTR dev, the gold family is now among the pipeline's top 10 for 0.875 of sentences, against
0.722 before. B3 scores 0.693 right-family on the clean dev set, against 0.580 on the glued one, B1
scores 0.527 and pipeline rank-1 0.470. Offering 10 distinct families instead of 10 techniques adds
only 0.006–0.016, because the retrieval pool holds just 16–19 techniques.

**Family head without retrieval (2026-10-13, prereg `cribl_family_head_2026-10-13.json`, amendments
1–2).** The C1 arms can only pick among the pipeline's 10 candidates, so a family the retriever misses
is lost: 15% of CTI-HAL statements and 12% of public test cases. `family_head.py` asks the model a
family question with no options. The prompt holds only the sentence. The decision-slot state then
goes through a scaled-cosine head with one row per family: 222 rows, one for each active parent
technique of the C1 catalogue (`catalog.json`). Each row starts as the mean decision-slot state of
that family's own ATT&CK texts (the parent and its sub-techniques, label and cut description), so
the untrained head, F0, is a nearest-description classifier. F0's test rows were scored on the laptop
from a locally rebuilt step 0 (`--epochs 0`, batch 8). Its dev scores are within 0.008 of the pod's
step 0: public dev 0.443 against 0.451, AnnoCTR dev equal.

F1 trains Cribl's adapter and the head together with cross-entropy over the gold families.
- Data: public train (6,308 cases) plus clean AnnoCTR train (1,453).
- Optimiser: LoRA learning rate 1e-4, head 1e-3, 16 examples per step, 3 epochs.
- Checkpoint: selected on the mean family top-1 over public dev and AnnoCTR dev.

Two amendments changed the compute, not the method:
1. Amendment 1 moved training to a RunPod secure RTX 4090. CTI-HAL was not uploaded; the selected
   checkpoint was downloaded and scored on CTI-HAL on the laptop.
2. Amendment 2 batches 16 prompts per forward pass. Batch 1 was CPU-bound on the pod: the GPU was
   11–14% busy and each example took 0.95 s. Batched, training took 0.11 s per example, 43 minutes in
   total.

| Dev family top-1 | step 0 (F0) | ½ | 1 | 1½ | **2** | 2½ | 3 epochs |
|---|---|---|---|---|---|---|---|
| public dev | 0.451 | 0.677 | 0.715 | 0.728 | **0.734** | 0.748 | 0.741 |
| AnnoCTR dev | 0.220 | 0.658 | 0.705 | 0.688 | **0.720** | 0.696 | 0.696 |

Family top-1 on the confirmation sets. The family question has no candidates, so every row is scored.
The C1 arms are under never-abstain.

| Family top-1 (top-5) | CTI-HAL (968) | public test (1,391) | AnnoCTR test, clean (738) |
|---|---|---|---|
| Pipeline rank-1 | 0.475 | 0.567 | 0.497 |
| Jev C1 as reported / never-abstain | 0.567 / 0.596 | n/a | n/a |
| B1 (C1) | 0.528 | 0.614 | 0.509 |
| B3 (C1) | 0.576 | 0.609 | 0.688 |
| F0, untrained head | 0.360 (0.660) | 0.365 (0.634) | 0.335 (0.611) |
| **F1, family head** | **0.612 (0.882)** | **0.609 (0.854)** | **0.717 (0.900)** |

Against the pre-registered criteria (paired bootstrap, 1000 samples, seed 7):
- **Headline, CTI-HAL F1 minus B3: passes.** +0.035, CI [+0.007, +0.060].
- **CTI-HAL F1 minus pipeline rank-1: passes.** +0.136, CI [+0.106, +0.169].
- **AnnoCTR test F1 minus B3: fails narrowly.** +0.028, CI [−0.003, +0.058]. F1 minus B1 is +0.207.
- **Public test F1 minus B1: fails.** −0.005, CI [−0.027, +0.017]: a tie. By source, TRAM is −0.017,
  expert −0.013 and derived procedures +0.013, all with intervals spanning 0.
- **CTI-HAL F1 top-5 at least the pipeline's top-10 family rate: passes.** 0.882, against 0.852.

Not registered:
- F1 also beats Jev C1 as reported on CTI-HAL, +0.044, CI [+0.017, +0.073].
- It ties Jev under never-abstain, +0.015, CI [−0.013, +0.045].

**Reading.** Without any retrieval, the 4B model names the family better than every
retrieve-then-pick arm on CTI-HAL and on real-report text. It ties them on the public test.

Part of the gain comes from statements the retriever missed. On the 143 CTI-HAL statements whose
family is not among the pipeline's top 10, F1 is right for 15.4%; a C1 arm cannot score on those. On
the 825 where the family is offered, F1 scores 0.691, against B3's 0.676.

On AnnoCTR test, F1 is ahead most clearly on explicit mentions: 0.863 against B3's 0.798. On
implicit mentions it scores 0.615 against 0.599.

F1's confidence is well calibrated: the ECE of P(pick) against a family hit is 0.045–0.057. F1 names only the family:
sub-techniques need a second step within the predicted family, which is not built yet.

**Several families per sentence (amendment 3, reported only).** `multi_family.py` names the top
family plus any other top-3 family with probability ≥ 0.25. The threshold was chosen on dev, where
mean F1 rose from 0.670 to 0.677. Family micro-F1 on the confirmation sets, one family → up to three:

| Set | One family | Up to three |
|---|---|---|
| CTI-HAL | 0.588 (P 0.612, R 0.567) | 0.598 (P 0.590, R 0.606) |
| public test | 0.523 | 0.532 |
| AnnoCTR test | 0.662 | 0.659 |

The rule names about 1.1 families per case. The head's probabilities concentrate on its first
answer, so most of the F1 gap is wrong first answers rather than missing second families.

**With ATT&CK's procedure examples (F2, 2026-10-13, prereg `cribl_family_attack_2026-10-13.json`).**
`export_attack_procedures.py` turns ATT&CK's active `uses` relationships into 17,033 distinct
sentences. It drops any that overlap a held-out text of public dev or test, CTI-HAL, or AnnoCTR dev
or test, which leaves 16,273 sentences covering 213 families:
- 378 near duplicates;
- 243 sentences that contain a held-out text of 6 or more tokens;
- 132 sentences that share a 10-token run with one;
- 7 repeats.

Of the dropped, 613 matched TRAM dev or test sentences and 6 matched CTI-HAL statements. F2 is F1's
recipe from the same starting point, with these rows added: 24,034 rows, 2 epochs, a checkpoint
every quarter epoch. It ran on a RunPod secure RTX 4090 at 0.08 s per example, 62 minutes in all.
The checkpoint at 1.5 epochs was selected, with dev mean 0.743 against F1's 0.727.

| Family top-1 | CTI-HAL | public test | AnnoCTR test |
|---|---|---|---|
| F1 | 0.612 | 0.609 | 0.717 |
| **F2** | **0.626** | **0.618** | **0.729** |
| F2 minus F1, paired 95% CI | +0.014 [−0.006, +0.036] | +0.009 [−0.008, +0.025] | +0.012 [−0.008, +0.033] |

- **Headline (CTI-HAL F2 > F1): fails.** The same holds for both secondary criteria; every delta is
  positive, but no interval excludes 0.
- **Contamination check: no sign of leakage.** On the public test, the gain without TRAM is +0.010
  (CI [−0.015, +0.034]), against +0.008 on TRAM itself.
- **Where it helps: rare families.** On the 69 CTI-HAL statements whose gold family had fewer than
  20 rows in F1's data, family top-1 rises from 0.232 to 0.333. On the other 899 it moves from 0.641
  to 0.648.
- **Against the other methods (not registered).** On CTI-HAL, F2 is +0.050 over B3 and +0.059 over
  Jev C1 as reported. It is +0.030 over Jev under never-abstain, CI [+0.002, +0.057].
- **Family micro-F1, one family → up to three.** CTI-HAL 0.602 → 0.607; public test 0.531 →
  0.537; AnnoCTR test 0.673 → 0.684.
- **Calibration is worse.** CTI-HAL ECE is 0.099, against F1's 0.057.

The pod cost about $1.20 (80 minutes at $0.89/h).

Quirk fixed after the run: the half-epoch boundary of the last epoch fell 3 examples before the end
(23,280 of 23,283). Two near-identical final checkpoints were saved and scored; the selection was
not affected. Checkpoints now sit at `round(k × rows / 2)`.

**Cost.** The pod cost about $1.20: 81 minutes at $0.89/h, including about 20 idle minutes while the
batching fix was made. No OpenRouter credit was used.

**ATT&CK definitions and a rival margin (F3, 2026-10-13, prereg `cribl_family_margin_2026-10-13.json`).**
F3 targets F2's look-alike confusions. In F2's 260 dev errors the gold family was second in 38%, and
the top two were within 0.3 in 147. F3 is F2's recipe with two changes:
- **Definitions.** `export_attack_procedures.py --descriptions` adds one row per listed technique,
  "Name: description" cut at a sentence end within 1,500 characters. The same overlap filter drops 4
  rows (a shared 10-token run), leaving 693 rows over 221 families.
- **Rival margin.** `--margin 1 1` adds `1.0 × max(0, 1 − (log P(gold families) − log P(strongest
  non-gold family)))` to each example's cross-entropy (`losses.rival_margin`; λ and m fixed, not tuned).

The first run, on an RTX A6000, was stopped by RunPod after 30,919 of 49,454 examples (amendment 2).
It was discarded. Its dev curve (public dev / AnnoCTR dev family top-1, checkpoints 6,192 to 30,919:
0.693/0.646, 0.707/0.682, 0.718/0.723, 0.725/0.720, 0.739/0.723) was seen, and nothing was chosen
from it.

F3 was retrained from scratch on a secure RTX 4090, the GPU of F2. Its untrained checkpoint
reproduces F2's to the last digit. Training took 75 minutes (0.09 s per example). The selected
checkpoint is the last one, at 2 epochs, with dev mean 0.739 against F2's 0.743:

| Checkpoint (epochs) | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 1.75 | **2** |
|---|---|---|---|---|---|---|---|---|
| public dev | 0.687 | 0.709 | 0.721 | 0.721 | 0.716 | 0.728 | 0.730 | **0.734** |
| AnnoCTR dev | 0.625 | 0.685 | 0.735 | 0.735 | 0.744 | 0.747 | 0.738 | **0.744** |

| Family top-1 | CTI-HAL | public test | AnnoCTR test |
|---|---|---|---|
| F2 | 0.626 | 0.618 | 0.729 |
| **F3** | **0.620** | **0.618** | **0.732** |
| F3 minus F2, paired 95% CI | −0.006 [−0.024, +0.011] | −0.001 [−0.015, +0.014] | +0.003 [−0.015, +0.019] |

- **Headline (CTI-HAL F3 > F2): fails.** Both secondary criteria fail too. F3 equals F2 within
  ±0.6 points on all three sets.
- **Look-alike subset (gold family in T1598, T1566, T1486, T1560, T1140, T1027, T1059, T1105,
  T1003, T1555): no gain.**
  - CTI-HAL 0.748 → 0.735, −0.013 [−0.037, +0.010].
  - Public test 0.620 → 0.594, −0.027 [−0.051, −0.002].
  - AnnoCTR test 0.803 → 0.798, −0.005 [−0.029, +0.019].
- **Pair mix-ups are unchanged.** On CTI-HAL, Command and Scripting Interpreter ↔ Ingress Tool
  Transfer stays at 16 and Deobfuscate ↔ Obfuscate at 5. Many of these are label ambiguity, not model
  error: statements that describe downloading and running a script carry only one of the two ids, and
  sometimes the less obvious one. A single-answer margin cannot win both. Cases whose gold family was
  second dropped slightly: CTI-HAL 143 → 137, public test 184 → 182, AnnoCTR test 71 → 66. Total
  errors did not drop.
- **Family micro-F1, one family → up to three (threshold 0.25, not re-tuned).**
  - CTI-HAL 0.596 → 0.602 (F2 0.602 → 0.607).
  - Public test 0.530 → 0.538.
  - AnnoCTR test 0.675 → 0.679.
- **Calibration is mixed.** ECE is 0.041 on CTI-HAL (F2 0.099), 0.088 on the public test (F2 0.071)
  and 0.118 on AnnoCTR test (F2 0.046).
- **Against the other methods (not registered).** On CTI-HAL, F3 is +0.043 over B3 [+0.017, +0.070],
  +0.053 over Jev C1 as reported, and +0.024 over Jev under never-abstain [−0.004, +0.052].

F2 remains the best family classifier. Definitions and the margin neither help nor hurt measurably.
The look-alike errors left are mostly pairs where the data itself names one of two valid families.
The up-to-three rule already credits these.

**Cost.** About $2.20 in all: about $0.95 for the stopped A6000 run (93 minutes at $0.59/h) and about
$1.25 for the RTX 4090 rerun (85 minutes at $0.89/h). No OpenRouter credit was used.

**Exact techniques without retrieval (E1, 2026-10-14, prereg `cribl_technique_head_2026-10-14.json`).**
E1 is F2 with a second head. `family_head.py --technique-head` adds a cosine head over the 697
listed techniques, each row starting from the technique's own text. The head is read as
P(technique | family), a softmax within the family's parent and sub-techniques (`hierarchy.py`):
- **Parent as an option.** A family's parent is its own option: the "family only" answer.
- **The exact answer.** The most probable technique by P(family) × P(technique | family) among the
  three most probable families. The owner chose three families over a hard limit to one: on F2's
  dev, the top family is right on 0.73–0.76 and the top three on 0.87–0.91.
- **Training.** As F2: same 24,034 rows, 2 epochs, Cribl adapter, seed 7. It adds a within-family
  cross-entropy on the 22,581 rows with exact labels. AnnoCTR rows are left out of it, because their
  links are parent-level by convention.
- **Selection.** Mean of public dev exact top-1 and AnnoCTR dev family top-1. The final checkpoint was
  selected: dev exact 0.653, AnnoCTR dev family 0.753.

| Checkpoint (epochs) | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 1.75 | **2** |
|---|---|---|---|---|---|---|---|---|---|
| public dev exact | 0.322 | 0.528 | 0.589 | 0.584 | 0.627 | 0.639 | 0.648 | 0.654 | **0.653** |
| public dev family | 0.451 | 0.669 | 0.719 | 0.713 | 0.727 | 0.727 | 0.730 | 0.731 | **0.728** |
| AnnoCTR dev family | 0.220 | 0.619 | 0.690 | 0.729 | 0.705 | 0.732 | 0.738 | 0.747 | **0.753** |

| Exact top-1 | CTI-HAL | public test |
|---|---|---|
| pipeline rank-1 | 0.383 | 0.501 |
| B1 (never-abstain) | 0.431 | 0.562 |
| B3 (never-abstain) | 0.441 | 0.537 |
| Jev C1 as reported / never-abstain | 0.460 / 0.478 | n/a |
| **E1** | **0.504** | **0.560** |

- **Headline (CTI-HAL exact, E1 > B3): passes.** +0.063, CI [+0.034, +0.094].
- **Secondary.**
  - CTI-HAL against Jev C1 as reported: passes, +0.044 [+0.011, +0.077].
  - Public test against B1: fails, −0.002 [−0.027, +0.022]. B1 was trained and selected on the same
    public sentences. E1 leads on derived procedures (+0.050 [+0.010, +0.088]) and trails on TRAM
    (−0.034 [−0.069, 0.000]).
- **Not registered.**
  - Against Jev under never-abstain: +0.026 [−0.008, +0.060].
  - Against the pipeline: +0.121 [+0.089, +0.153] on CTI-HAL and +0.059 on the public test.
  - The retrieve-then-pick models and Jev cannot name a technique outside the pipeline's top 10,
    which holds the exact gold on 74% of CTI-HAL.
- **The three-family rule.**
  - CTI-HAL: 0.504 with the top family only 0.496 and with every family 0.504.
  - Public test: 0.560, 0.551 and 0.559.
  - Exact top-3/5 on CTI-HAL: 0.723 / 0.803.
- **By label kind (CTI-HAL).** 0.578 on the 431 statements labelled with a sub-technique, 0.445 on
  the 537 labelled with a family only. Of the 131 misses where the family is right, 111 are a
  granularity mismatch rather than a wrong sibling:
  - a sub-technique where the label names only the family (50);
  - the family where the label names a sub-technique (61).

  Only 20 are wrong siblings. The other 349 misses are wrong families.
- **Family accuracy is not lost.** Family top-1 against F2: CTI-HAL 0.631 (+0.005 [−0.011,
  +0.023]), public test 0.625 (+0.007), AnnoCTR test 0.730 (+0.001). On AnnoCTR test it is 0.042
  over B3 [+0.014, +0.075].
- **Calibration of the exact answer.** ECE is 0.101 on CTI-HAL and 0.053 on the public test.
  AnnoCTR's 0.154 is not meaningful, because its labels are parent-level.

E1 is the first model here to name the exact technique more often than Jev on CTI-HAL. It also
reads the sentence without the pipeline's retrieval. Its remaining errors are mostly the wrong family
(73%) or a granularity mismatch with the label (23%).

**Cost.** The pod cost about $1.10 (73 minutes at $0.89/h). Training took 62 minutes at 0.08 s per
example. CTI-HAL was scored on the laptop. No OpenRouter credit was used.

**More and cleaner training data for the family (E2, 2026-10-14, prereg
`cribl_family_data_2026-10-14.json`).** E2 is E1's recipe with the training rows changed. 178
CTI-HAL statements have the right family outside E1's top three, and no answer rule can reach them.
The question was whether more real-report text and cleaner rows move them.
- **Joint government advisories** (`prepare_advisories.py`).
  - Source: the FBI IC3 copies of the joint cybersecurity advisories from 2021 to 2026; 213 of 218
    PDFs were downloaded. CISA's own pages refuse automated clients, so they were not used.
  - Rows: each sentence carrying the authors' bracketed ATT&CK tag, labelled with that tag. Tags,
    tactic headings and update markers are removed.
  - Advisories kept: 59. Of the rest, 141 have fewer than five tags (guidance notes, FBI-only
    alerts). 12 name a CTI-HAL group, an alias or the group's own software. 1 overlaps a held-out
    text.
  - Yield: 765 rows over 155 families, far fewer than the 2–5k first estimated.
- **Atomic Red Team** (`prepare_atomic.py`, MIT): 1,874 test descriptions ("name. description",
  without commands), labelled with the test's technique, 151 families.
- **Cleaning** (`clean_cases.py`, training rows only; dev and test unchanged). It drops fragments,
  figure captions, markdown table rows, text naming a technique id and hash tables:
  - public train: 38 of 6,308 dropped;
  - AnnoCTR train: 282 of 1,453 dropped;
  - ATT&CK procedures: none dropped.
- **Training.** 26,353 rows (E1: 24,034), 2 epochs, selection as E1. The final checkpoint was
  selected: dev exact 0.660, AnnoCTR dev family 0.741. Training took 71 minutes.

| Checkpoint (epochs) | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 1.75 | **2** |
|---|---|---|---|---|---|---|---|---|---|
| public dev family | 0.449 | 0.693 | 0.713 | 0.728 | 0.733 | 0.725 | 0.730 | 0.733 | **0.734** |
| AnnoCTR dev family | 0.223 | 0.604 | 0.604 | 0.690 | 0.688 | 0.735 | 0.723 | 0.735 | **0.741** |
| public dev exact | 0.320 | 0.528 | 0.557 | 0.622 | 0.602 | 0.636 | 0.657 | 0.656 | **0.660** |

| E2 − E1 (paired, 95% CI) | E1 | E2 | difference |
|---|---|---|---|
| CTI-HAL family top-1 (headline) | 0.631 | 0.642 | +0.010 [−0.007, +0.029] |
| CTI-HAL exact top-1 | 0.504 | 0.514 | +0.010 [−0.007, +0.030] |
| AnnoCTR test family top-1 | 0.730 | 0.717 | −0.014 [−0.035, +0.005] |
| public test family top-1 | 0.625 | 0.633 | +0.008 [−0.005, +0.022] |

- **Headline: fails.** The CI includes 0. Every secondary also fails.
- **Reported.**
  - CTI-HAL right family in the top three: 0.824 (E1 0.816). It is outside the top three on 170
    statements (E1: 178). E2 alone is right on 39 statements and E1 alone on 29.
  - CTI-HAL exact top-3/5: 0.743 / 0.809 (E1 0.723 / 0.803).
  - The expert test subset, the text closest to advisories: family 0.682 vs 0.669
    (+0.013 [−0.032, +0.057]).
  - Exact misses on CTI-HAL: 347 wrong family, 131 right family.
- **Not registered.** CTI-HAL exact against Jev C1 under never-abstain: +0.036 [+0.003, +0.067],
  the first CI above 0 against that rule (E1: +0.026 [−0.008, +0.060]). The difference from E1 is
  itself not significant, so this is not evidence that the new data helped.

About 2,300 rows of new sources moved no result beyond noise. The advisories were too few: most
IC3 items carry no tags. Atomic tests describe lab procedures more than report prose. Cleaning
removed broken rows without a measurable effect. E1 stays the reference; E2 is no worse anywhere
and is slightly ahead on CTI-HAL.

**Cost.** The pod cost about $1.20 (82 minutes at $0.89/h). CTI-HAL was scored on the laptop. The
downloads were free. No OpenRouter credit was used.

**A yes/no probability per family and up to three answers (M1, 2026-10-14, prereg
`cribl_yesno_2026-10-14.json`).** The owner asked for up to three families or techniques per
sentence when the model is confident enough. A sentence often describes two or three techniques
but carries one label.
- **Why the one-answer model resists this.** It shares one unit of probability among the families,
  and training pushes it onto the single label. On E2's dev the second family scores 0.05 (median),
  and a second labelled family only 0.03.
- **What M1 changes.** M1 gives every family its own sigmoid: "is this family in the sentence?".
  - Loss: `losses.yes_no` scores the labelled families as "yes" and every other family as a "no"
    with weight 0.1, so a missing label is weak evidence.
  - Starting bias: minus the median second-highest logit over 64 training sentences.
  - Everything else is E2's: data, recipe, the technique head and the selection rule. The final
    checkpoint was selected.
- **Answers and scoring** (`multi_answer.py`).
  - Answers: the top family, plus the second and third when they reach a threshold. Each answer
    family names its most probable technique.
  - The threshold is set per model from its own probabilities, without labels, to give 1.3 answers
    per sentence. Both models are therefore compared with the same number of answers.
  - A hit is a labelled family (or technique) among the answers.

| Checkpoint (epochs), M1 | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 1.75 | **2** |
|---|---|---|---|---|---|---|---|---|---|
| public dev family top-1 | 0.451 | 0.663 | 0.680 | 0.695 | 0.718 | 0.712 | 0.712 | 0.721 | **0.724** |
| AnnoCTR dev family top-1 | 0.220 | 0.548 | 0.607 | 0.685 | 0.673 | 0.714 | 0.690 | 0.717 | **0.717** |
| public dev exact hit, 1.3 answers | 0.326 | 0.524 | 0.563 | 0.616 | 0.643 | 0.654 | 0.675 | 0.680 | **0.689** |
| (E2, same rule) | 0.341 | 0.564 | 0.607 | 0.657 | 0.649 | 0.680 | 0.713 | 0.712 | 0.715 |

| CTI-HAL, M1 − E2 (paired, 95% CI) | E2 | M1 | difference |
|---|---|---|---|
| family hit, 1.3 answers (headline) | 0.716 | 0.691 | −0.025 [−0.044, −0.007] |
| exact hit, 1.3 answers | 0.568 | 0.549 | −0.020 [−0.039, −0.001] |
| family top-1, one answer (non-inferiority −0.02) | 0.642 | 0.623 | −0.019 [−0.037, −0.001] |
| public test family hit, 1.3 answers | 0.702 | 0.688 | −0.014 [−0.028, +0.001] |
| AnnoCTR test family hit, 1.3 answers | 0.790 | 0.778 | −0.012 [−0.033, +0.008] |

- **Every registered criterion fails.** Scored against the labels, M1 is slightly worse than E2 at
  every budget (1.1: −0.025; 1.5: −0.018).
- **The number of answers follows the sentence a little more.**
  - On the 66 CTI-HAL statements with several labelled families, M1 gives 1.38 answers against 1.29
    on the others; E2 gives 1.32 and 1.30.
  - All labelled families are found on 12% of those statements (E2: 8%).
- **At a fixed threshold of 0.5** (the "clears a confidence threshold" reading), M1 gives 1.81
  answers per sentence. Family hit is 0.758, against 0.773 for E2 with the same number of answers.
  M1's probabilities are not calibrated as single-answer confidences: one-answer ECE is 0.29, by
  design of the weak "no".
- **Blind check** (reported, not a gate).
  - Method: 99 extra answers on CTI-HAL at 1.3 (50 per model, one merged). Each was shown with the
    sentence and technique name only, with model and label hidden. Claude judged each "supported by
    the sentence" or not; one AI reader.
  - Results: supported for 64% of M1's extra answers (32/50) and 46% of E2's (23/50). Among extras
    that are not a labelled family, 61% (23/38) for M1 and 34% (12/35) for E2. With 50 per model the
    difference is uncertain: roughly ±0.19 at 95%, unpaired.
  - Labelled-family extras were judged supported only 73–75% of the time. Some labels or judgments
    disagree.

By the labels, a yes/no head does not beat the one-answer head with a threshold. By a blind reading,
its extra answers are more often real techniques the annotator did not write down. The data cannot
settle this without more independent judgments of every answer, including the first. E2 stays the
reference for single answers. M1 is kept as the candidate for multi-answer output.

**Cost.** The pod cost about $1.25 (83 minutes at $0.89/h). CTI-HAL and the blind check ran on the
laptop. No OpenRouter credit was used.

**Expert-labelled paragraphs and reading them sentence by sentence (2026-10-14, prereg
`cribl_sentence_split_2026-10-14.json`, no training).**
- **The expert set read whole.** The public test's `expert` rows are 157 report paragraphs (median
  62 words). They carry 2.7 labelled families and 3.3 techniques each, so extra answers can be
  graded there.
  - One answer: no model is clearly ahead. Family / exact: B1 0.688 / 0.675, E2 0.682 / 0.650,
    E1 0.669 / 0.643, M1 0.656 / 0.605, pipeline 0.631 / 0.592.
  - Same number of answers per paragraph: E2 and M1 are level at every count. With two answers,
    56% and 55% of answers are labelled families.
  - So M1's advantage in the blind check does not show on complete labels.
  - Recall is the limit: three answers find about half of a paragraph's labelled families.
- **The split.** Every row had been read whole (83% of expert rows hold 2+ sentences, 48% of
  derived ones). `sentence_split_eval.py` splits the expert and derived_procedures paragraphs with
  the product's no-LLM splitter (`local_observations`), giving 521 and 1,621 sentences.
  - Merging: the model reads each sentence. A family's paragraph score is its highest probability
    in any sentence.
  - Grading: per paragraph against the paragraph's labels, with the same number of answers for
    both readings.

| E2, labelled families found (paired CI, split − whole) | whole | split | difference |
|---|---|---|---|
| expert, 3 answers (headline) | 0.488 | 0.465 | −0.042 [−0.084, +0.001] |
| expert, 1 / 2 / 5 answers | 0.251 / 0.409 / 0.615 | 0.237 / 0.380 / 0.613 | all CIs span 0 |
| expert, one answer per sentence (2.5 per paragraph) | 0.460 | 0.437 | −0.035 [−0.074, +0.005] |
| derived_procedures, 3 answers | 0.633 | 0.604 | −0.023 [−0.051, +0.003] |
| derived_procedures, one answer per sentence | 0.564 | 0.520 | −0.041 [−0.067, −0.014] |

- **Headline fails.** Splitting finds no more labelled techniques at the same number of answers.
  - Expert: it is level or slightly below. At three answers, 33 labelled families are found only by
    the split and 43 only by the whole paragraph.
  - Derived procedures: it is 2–5 points below, and significantly so at most counts and for the
    exact technique.
  - M1 behaves the same: expert −0.014 to +0.012, derived −0.017 to −0.074.
- **Why.** The models were trained on such paragraphs too (half of the expert and derived training
  rows hold several sentences), and a paragraph's labels describe the whole of it. A split sentence
  loses its references: "Because of this, we conclude that DeathStalker is a cyber-mercenary
  organization." Splitting does surface more distinct techniques when every sentence answers. At an
  equal answer count, though, the whole-paragraph ranking already holds them.

**Multi-label rows weighted 4x (M2, 2026-10-14, prereg `cribl_multilabel_2026-10-14.json`).**
- **Why.** Training never limited the model to one technique. A row with several labels shares the
  target among them under softmax, and every one is a "yes" under the yes/no loss. But only 4.4% of
  training rows (1,167 of 26,353) carry two or more families. Expert training rows carry several in
  19% of cases; the expert test paragraphs in 70%.
- **What.** M2 is M1 (yes/no head, E2's data) with the whole loss of those 1,167 rows multiplied by
  4, about 16% of the gradient instead of 4%. The final checkpoint was selected.

| Checkpoint (epochs), M2 | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 1.75 | **2** |
|---|---|---|---|---|---|---|---|---|---|
| public dev family top-1 | 0.451 | 0.666 | 0.675 | 0.697 | 0.718 | 0.718 | 0.721 | 0.716 | **0.715** |
| AnnoCTR dev family top-1 | 0.220 | 0.577 | 0.586 | 0.685 | 0.649 | 0.696 | 0.685 | 0.711 | **0.708** |
| public dev exact hit, 1.3 answers | 0.326 | 0.522 | 0.552 | 0.590 | 0.612 | 0.610 | 0.631 | 0.636 | **0.642** |

| Share of labelled families found per text (paired CI) | E2 | M1 | M2 | M2 − E2 |
|---|---|---|---|---|
| expert paragraphs, 2 answers (headline) | 0.495 | 0.480 | 0.496 | +0.000 [−0.042, +0.050] |
| expert paragraphs, 1.3 answers | 0.382 | 0.361 | 0.393 | +0.011 [−0.028, +0.052] |
| expert paragraphs, 3 answers | 0.574 | 0.584 | 0.591 | +0.016 [−0.022, +0.061] |
| public test (all), 1.3 answers | — | — | — | −0.018 [−0.032, −0.004] |
| CTI-HAL family hit, 1.3 answers (non-inferiority −0.02) | 0.716 | 0.691 | 0.699 | −0.017 [−0.035, +0.001] |

- **Headline fails.** On the expert paragraphs M2 finds as many labelled families as E2 at every
  answer count.
- **Against M1, the weighting helps a little where several labels are common.**
  - Expert family recall: +0.016 at 2 answers [−0.020, +0.055] and +0.032 at 1.3 [−0.003, +0.071].
  - Exact recall at 1.3: +0.038 [+0.005, +0.073].
  - One-answer expert family top-1 recovers to 0.688 (M1 0.656, E2 0.682).
- **It costs on one-label text.**
  - Public test exact top-1 is 0.531, against E2's 0.561: −0.030 [−0.050, −0.012].
  - AnnoCTR test exact is 0.421, against 0.451.
  - CTI-HAL's non-inferiority bound fails.
- **The number of answers still does not follow the paragraph.** At 2 answers, paragraphs with
  several labelled families get 1.94 answers against 2.15 for one-label ones (E2: 1.96 vs 2.09).
- **Fixed threshold of 0.5.** 2.20 answers per paragraph and family F1 0.472 (E2 at 2 answers:
  0.478).

The 1,167 multi-label rows are too few and too varied for a 4x weight to teach the model how many
techniques a text holds. They shift it a little toward multi-label text and away from one-label
text. More multi-label training data, not a reweighting, is what this needs. E2 stays the reference.

**Cost.** The pod cost about $1.30 (87 minutes at $0.89/h). CTI-HAL was scored on the laptop. No
OpenRouter credit was used.

**More multi-label data: CTI-Rationale paragraphs (E3, 2026-10-15, prereg `cribl_cti_rationale_2026-10-15.json`).**
- **Search.** Released ATT&CK-labelled text with several techniques per unit is scarce.
  - Text2TTP (QCRI, GPL-3.0): its public `sentences.csv` holds only the hand-labelled (2,017) and TRAM
    (1,522) rows; the CISA and WeLiveSecurity parts of the paper are not released. After the overlap
    and CTI-HAL-group checks, 25 new sentences carry two or more families, so it was not used.
  - The DFIR Report sentence set (arXiv 2606.18166, 40% multi-label) and CTI2TTP-Bench / TRAM-Clean
    (arXiv 2605.25836) are not released.
  - CTI-Rationale (Hugging Face `EbruResul/CTI-Rationale`, gated, accepted by the owner) is used:
    1,394 paragraphs of 100 reports from 8 publishers, two analysts each with adjudicated gold rows,
    ATT&CK v18.1; annotations CC BY 4.0, paragraph text the publishers' (local training only).
- **Data** (`prepare_cti_rationale.py`). The dataset's report split is kept: train + validation
  reports become 1,011 training paragraphs (174 with 2+ families, 121 families), test reports a
  209-paragraph test set (33 with 2+ families).
  - 10 reports are dropped whole. 8 name a CTI-HAL group, an alias or its own software (SUNBURST,
    NOBELIUM, Ryuk, Conti, Industroyer, NotPetya and others). 2 are attributed to one without naming
    it, found by reading the kept report list: CISA's Infamous Chisel report (Sandworm) and Mandiant's
    CEELOADER report (APT29).
  - Paragraphs overlapping held-out, Elastic or vendor-set text are dropped (19 train, 3 test), as are
    8 repeats of existing training rows and 16 test paragraphs overlapping any training text.
- **What.** E3 is E2's recipe with these rows added (both losses): 27,364 rows, 1,341 with 2+ families
  (E2: 1,167). The pre-registered rule selected step 41,060 (1.5 epochs).

| Checkpoint (epochs), E3 | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | **1.5** | 1.75 | 2 |
|---|---|---|---|---|---|---|---|---|---|
| public dev family top-1 | 0.449 | 0.709 | 0.718 | 0.722 | 0.715 | 0.728 | **0.731** | 0.734 | 0.737 |
| AnnoCTR dev family top-1 | 0.223 | 0.643 | 0.667 | 0.699 | 0.676 | 0.699 | **0.729** | 0.714 | 0.714 |
| public dev exact top-1 | 0.320 | 0.543 | 0.566 | 0.595 | 0.602 | 0.631 | **0.637** | 0.648 | 0.648 |

| Paired, E3 − E2 | E2 | E3 | E3 − E2 [95% CI] |
|---|---|---|---|
| expert paragraphs (157), 2 answers: share of labelled families found (headline) | 0.495 | 0.482 | −0.013 [−0.046, +0.019] |
| expert paragraphs, 1.3 / 3 answers | 0.382 / 0.574 | 0.393 / 0.583 | +0.011 [−0.012, +0.033] / +0.008 [−0.021, +0.041] |
| CTI-Rationale test (209), 1.3 answers: share found (in-domain) | 0.559 | 0.586 | +0.027 [−0.013, +0.070] |
| CTI-Rationale test, one answer: family top-1 | 0.517 | 0.560 | +0.043 [+0.005, +0.086] |
| CTI-Rationale test, 33 multi-family paragraphs, 2 answers: share found | 0.518 | 0.528 | +0.010 [−0.058, +0.066] |
| CTI-HAL family top-1 (non-inferiority −0.02) | 0.642 | 0.636 | −0.005 [−0.023, +0.011] |
| CTI-HAL exact top-1 / family top-3 | 0.514 / 0.824 | 0.513 / 0.841 | −0.001 [−0.018, +0.014] / — |
| public test family / exact top-1 | 0.633 / 0.561 | 0.625 / 0.559 | −0.008 [−0.020, +0.005] / −0.003 [−0.017, +0.012] |
| AnnoCTR test family / exact top-1 | 0.717 / 0.451 | 0.726 / 0.440 | +0.009 [−0.008, +0.027] / −0.011 [−0.030, +0.008] |

- **Headline fails.** On the expert paragraphs E3 finds as many labelled families as E2 at every
  answer count.
- **The new rows help on their own kind of text, not elsewhere.** On the CTI-Rationale test reports
  (same publishers and annotators) the one-answer family top-1 rises 4.3 points (exact, top family
  only, 0.421 → 0.464), and at two answers E3 gives multi-family paragraphs more answers than
  one-family ones (2.30 vs 1.94; E2 2.03 vs 1.99). On expert paragraphs the answer count still does not
  follow the label count (1.98 vs 2.04; E2 1.96 vs 2.09).
- **CTI-HAL.** No real change (−0.5 point), but the non-inferiority bound misses by 0.003.

About 170 more multi-label rows do not teach the model how many techniques an expert paragraph
holds. They help only on text like their own. E2 stays the reference.

**Cost.** The pod cost about $1.32 (89 minutes at $0.89/h). E2 on the CTI-Rationale test and E3 on
CTI-HAL were scored on the laptop. No OpenRouter credit was used.

**Parent-level scoring, a recipe sweep and a second base model (2026-10-15, prereg
`cribl_recipe_base_2026-10-15.json`).** The owner's three questions after the data runs: score
without the annotators' parent-or-sub-technique habit; vary the one recipe ever used; try a second
base model.

- **Parent-level accuracy** (`relabel_parents.py`, `parent_rescore.py`): the answered technique's
  parent is among the gold ids' parents. Every evaluation split has a parent-relabelled copy
  (`cases_<split>_parents.jsonl`); `family_head.py evaluate` accepts them and reports
  `parent_level` beside exact. On CTI-HAL E2 scores family 0.642, parent level 0.639, exact 0.514:
  the exact answer's family is right as often as the first family, so "exact" was 12 points lower
  only through granularity (111 statements) and about 20 wrong siblings. No earlier run beats E2 at
  parent level either (E1 0.639, E3 0.637, F2 0.626; the yes/no heads M1/M2 0.597/0.593, their exact
  pick often leaving the first family). Training labels were not relabelled.
- **Selection set widened.** `prepare_cti_rationale.py` now keeps the dataset's 15 validation
  reports as `cases_cti_rationale_validation.jsonl` (221 paragraphs; E3 had merged them into its
  training rows, `--merge-validation`). The sweep selects on the mean parent-level accuracy over
  public dev, AnnoCTR dev and these paragraphs (`--selection parent`), checkpoints every half epoch.
- **Runs**, all on E2's 26,353 training rows (names in docs only):
  - R1: E2's recipe for 3 epochs (selected: 2 epochs).
  - R2: 3 epochs, LoRA lr 2e-4 (selected: 2.5 epochs; the best selection score, 0.710).
  - R3: 3 epochs, a fresh LoRA of rank 64 / alpha 128 (`--lora-r`; selected: 2 epochs).
  - G1: `google/gemma-4-E4B@411aa17b` (Apache 2.0, 4.5B effective / 8B with embeddings) with a
    fresh LoRA r 16 on the attention projections (`--base gemma-4-e4b`, `decision_model.BASES`;
    the family prompt in Gemma's turn format, same body), E2's recipe for 2 epochs (selected: the
    last). 132 LoRA modules, 724 4-bit linear layers, both embedding tables left in bf16; 17 GB peak
    on the pod. Untrained, the nearest-description head scores 0.04 on dev (Qwen 0.45): Gemma's
    decision-slot states carry little before training, and the gap closes within half an epoch.
  - Each run ~100 min on a secure RTX 4090; the four pods about $6.85 in all. CTI-HAL was scored on
    the laptop, Gemma included: on an 8 GB GPU `decision_model` keeps Gemma's 5.2 GB per-layer
    embedding table in system RAM and looks rows up on the CPU (`BaseSpec.cpu_modules`,
    `keep_on_cpu`; 4.4 GB of GPU memory, 0.23 s per statement).

| Parent level, run − E2 (paired 95% CI) | CTI-HAL (968) | public test (1,391) | AnnoCTR test (738) | CTI-Rationale test (209) |
|---|---|---|---|---|
| E2 (reference) | 0.639 | 0.636 | 0.714 | 0.517 |
| R1, 3 epochs | 0.630 (−0.009 [−0.026, +0.007]) | 0.635 (−0.001) | 0.729 (+0.015 [−0.005, +0.035]) | 0.531 (+0.014) |
| **R2, lr 2e-4 (best on dev, headline)** | **0.605 (−0.034 [−0.057, −0.011])** | 0.649 (+0.013 [−0.005, +0.030]) | 0.702 (−0.012) | 0.531 (+0.014) |
| R3, rank 64 | 0.624 (−0.015 [−0.037, +0.006]) | 0.636 (+0.000) | 0.725 (+0.011) | 0.517 (+0.000) |
| **G1, Gemma 4 E4B (headline)** | **0.636 (−0.003 [−0.025, +0.020])** | 0.627 (−0.009 [−0.028, +0.011]) | 0.701 (−0.014) | 0.512 (−0.005) |

- **Both headlines fail.** The best-on-dev recipe (R2) is 3.4 points *worse* on CTI-HAL, and the
  interval excludes 0; Gemma ties E2 (−0.3).
- **Strict exact on CTI-HAL:** E2 0.514, R1 0.503, R2 0.469 (−0.045 [−0.067, −0.022]), R3 0.483
  (−0.031 [−0.054, −0.008]), G1 0.527 (+0.012 [−0.011, +0.037]). Family top-3: E2 0.824, G1 0.827,
  R2 0.811, R3 0.803.
- **Where the higher learning rate "won":** public dev exact rose to 0.71 (E2 0.66), and that
  carried to the public test's TRAM split (parent level +0.036 [+0.013, +0.059]; exact +0.032
  [+0.012, +0.051] pooled), not to real reports: derived procedures −0.013, expert −0.006, CTI-HAL
  −0.034. Rank 64 behaves the same (TRAM +0.035, derived −0.036 [−0.067, −0.004]). Both are worse
  calibrated (CTI-HAL family ECE 0.144 and 0.100 against E2's 0.067; Gemma 0.060). The wider
  selection set did not catch this: the three dev sets agreed on R2.
- **Non-inferiority on public test family (−0.02):** R2 and R3 pass; R1 (−0.010 [−0.023, +0.003])
  and G1 (−0.016 [−0.033, +0.004]) miss the bound by a fraction of a point.
- **Rare families** (gold family under 100 training rows; CTI-HAL 89 statements, public test 209):
  no run moves them (CTI-HAL E2 0.393, R1 0.382, R2 0.382, R3 0.416, G1 0.404, every interval
  spanning 0).
- **Longer training alone** (R1) changes nothing: the selection rule stopped at 2 epochs, exactly
  E2's recipe, and the three dev curves are flat from 1.5 epochs on.

Reading: at parent level the model sits at about 64% on CTI-HAL under every recipe and both base
models, which is the agreement rate of CTI-HAL's own annotators. A faster learning rate or more
LoRA capacity buys TRAM-style sentences at the cost of real-report statements and calibration.
Gemma 4 E4B reaches the same point from a fresh LoRA with no Cribl adapter, so the adapter and the
base family are not what limits this; the labels are. E2 stays the reference.

**Cost.** Four pods, about $6.85 (R1 111 min, R2 124 min, R3 118 min, G1 110 min at $0.89/h). The
laptop scored CTI-HAL for all four and ran the two pilots. No OpenRouter credit was used.

**Amendment 1: a security-pretrained 8B base (S1, 2026-10-15).** After the four results above the
owner asked for one larger base. S1 is `fdtn-ai/Foundation-Sec-8B@9e6bb1f7` (Cisco: Llama 3.1 8B
continued on 5.1B tokens of security text; Apache 2.0; text-only, so `decision_model` loads it
through the causal-LM class; the family prompt in Llama 3's header format, same body), a fresh LoRA
r 16 on the attention projections (128 modules, 13.6M parameters), E2's recipe for 2 epochs with
the widened selection set. Registered as amendment 1 with the same bar before training. Training
took 74 min at 0.084 s per example (13.7 GB peak), the same speed as the 4B models; the pod cost
about $1.35. On the laptop the 1 GB token table and the unused 1 GB output head stay in RAM
(`cpu_modules`), leaving 7.0 GB of GPU use for scoring at batch 4.

| Parent level, S1 − E2 (paired 95% CI) | CTI-HAL | public test | AnnoCTR test | CTI-Rationale test |
|---|---|---|---|---|
| E2 (reference) | 0.639 | 0.636 | 0.714 | 0.517 |
| **S1, Foundation-Sec-8B (headline)** | **0.627 (−0.012 [−0.034, +0.009])** | 0.647 (+0.011 [−0.006, +0.028]) | 0.736 (+0.022 [−0.003, +0.045]) | 0.565 (+0.048 [−0.010, +0.105]) |
| strict exact, S1 − E2 | 0.503 (−0.011) | 0.580 (+0.019 [0.000, +0.037]) | 0.466 (+0.015) | 0.469 (+0.048) |

- **Headline fails.** On CTI-HAL the 8B security model is 1.2 points below E2 at parent level and
  1.1 below on exact; family top-3 0.822 against 0.824; the 89 rare-family statements are at the
  same 0.393. Its calibration matches E2's (family ECE 0.064).
- **Elsewhere it is the first run slightly ahead of E2 on every set**, each within noise: public
  test exact +1.9 (TRAM +4.3 [+1.5, +6.9], derived procedures −1.2, expert +0.6), AnnoCTR +2.2,
  CTI-Rationale +4.8. The untrained head starts at 0.21 on dev (Qwen 0.45, Gemma 0.04) and the dev
  curves end above every Qwen run (public dev parent level 0.762, exact 0.709).
- **Sub-techniques:** within the right family S1 picks the exact id 80% of the time (487 of 607),
  as E2 does (498 of 619), with 13 wrong siblings against 18.

Reading: twice the parameters and 5 billion tokens of security pretraining buy a better fit to the
public sentences and nothing on CTI-HAL. Three base models of two sizes now sit at 63–64% there.
E2 stays the reference; S1 is the strongest alternative when the public sets matter and the memory
is available.

**Cost.** One pod, about $1.35 (91 minutes at $0.89/h). CTI-HAL was scored on the laptop. No
OpenRouter credit was used.

**A group head: the threat group from the extracted techniques (A1/A2, 2026-10-15, prereg
`cribl_group_head_2026-10-15.json`).** The owner asked for a model built the same way (base LLM,
fresh LoRA, cosine head) that reads only the techniques previously extracted from a report and names
the group. `export_group_cases.py` builds a 153-group catalogue (groups with 5+ documented
techniques, each its own family, rows initialised from "Name (Gxxxx): description") and the case
files; `family_head.py --task group` trains it with the family-head code unchanged (the prompt
asks for the group; the state is the technique list, "Txxxx Parent name: name" entries joined by
"; ", no prose, no actor or tool names). Training rows: 45,900 synthetic draws from the ATT&CK
profiles, mirroring the group network's generator without software and context (3–20 true
techniques, false positives by popularity at precision 0.4–0.95, a sub-technique swapped for its
parent with p 0.2). Selection on the 20 single-group ATT&CK campaigns and the 34 vendor reports
(those carry no technique labels: their input is what E2 names on their statements). Baselines on
the same inputs (`group_baseline.py`): the product's group network with the software block empty,
and a Jaccard match to the documented profiles. A1 = Qwen3.5-4B-Base, A2 = Foundation-Sec-8B,
fresh LoRA r 16, one epoch, a checkpoint every quarter; the technique lists are five times longer
than sentences, so a run took ~2.7 h (0.18–0.20 s per example) and the two pods about $5.05
instead of the planned $2.10.

| Group top-1 (top-5), 153 groups ranked | CTI-HAL extracted (55, headline) | CTI-HAL gold (55) | CTI-HAL accepted (54) | Elastic (6) | campaigns dev (20) | vendor dev (34) |
|---|---|---|---|---|---|---|
| group network, techniques only | **0.127** (0.273) | **0.182** (0.364) | **0.111** (0.333) | 0 (0.167) | 0.300 (0.500) | 0.294 (0.382) |
| Jaccard to profile | 0.018 (0.018) | 0.000 (0.036) | 0.000 (0.019) | 0 (0) | 0.150 (0.200) | 0.294 (0.471) |
| A1, Qwen3.5-4B group head | 0.018 (0.073) | 0.073 (0.091) | 0.000 (0.074) | 0 (0.333) | 0.150 (0.250) | 0.265 (0.471) |
| A2, Foundation-Sec-8B group head | 0.036 (0.145) | 0.018 (0.145) | 0.037 (0.093) | 0 (0) | 0.100 (0.250) | 0.235 (0.412) |

- **Both headlines fail**, and not narrowly: A1 is −0.109 [−0.200, −0.036] against the group
  network on the extracted input, A2 −0.091 [−0.182, 0.000]. The right group's median rank is
  40–48 of 153 for the heads against 17–23 for the network. Sandworm is found (A2 gold: ranks 1–7
  on all five reports); the financially motivated groups, whose technique sets are generic, are not.
- **On the dev sets the heads come close** (vendor: A1 0.265 vs 0.294) and were still rising
  (A1 vendor 0.15 → 0.26 → 0.24 across the quarter-epoch checkpoints), so one epoch over one draw
  of the synthetic rows is too little for 153 classes; the network sees 400 draws per group
  redrawn every 4 of its 20 epochs, and adds a prominence prior at prediction, which favours the
  well-known CTI-HAL groups.
- **The product's full attribution stage** (techniques plus software, the combiner) reaches 0.455
  on the same reports, quoted for context: the group signal is in the named tools, not in the
  technique mix, for every model here.

Reading: a language model reading a bag of technique names is a worse group classifier than a
small network over the same bag, at 50 times the compute. The technique list is not a fingerprint,
and the LLM's strength (reading prose) is unused on it. If the group question is to be pursued, the
input should be the report text (the free-text group study, docs 21.3.16, is the comparison), not
the extracted list. The product's attribution stage is unchanged.

**Cost.** Two pods, about $5.05 (A1 175 min, A2 165 min at $0.89/h). The vendor extraction, the
baselines and all test scoring ran on the laptop. No OpenRouter credit was used.

**Refitting the group network for extracted inputs (2026-10-16, prereg
`cribl_group_mlp_refit_2026-10-16.json`).** The owner asked whether the network itself can be made
better. Two changes through the fitter's own options (`refit_group_mlp.py`):
- **Measured draws.** How E2 actually extracts, measured on the 85 CTI-Rationale train and
  validation reports: about 10 techniques per report (mean 10.1), precision 0.62 exact and 0.71 at
  parent level (10th–90th percentile 0.50–1.00), recall 0.50; of the wrong ones 67% are unrelated
  techniques, 24% a parent or sub-technique of a labelled one, 9% a sibling. The derived draw
  range is 3–11 true techniques at precision 0.50–1.00 (the hand-set range: 3–20 at 0.40–0.95).
- **Real cases.** The 34 vendor reports with the techniques E2 names on them, repeated
  `real_weight` times per draw (0, 3, 10, 30).
- Grid of 8 fits (~40 s each on the laptop GPU), selected on the 20 single-group ATT&CK campaigns
  with techniques only (MRR): the vendor reports help on dev (MRR 0.413 → 0.487 at weight 30,
  top-1 0.30 → 0.40); the measured draws do not (0.411 at weight 0). Selected: current draws with
  the vendor reports at weight 30, written to `data/models/mlp/group_mlp_extracted.npz` (the
  product's artifact is untouched).

| Group top-1 (top-5), software block empty | CTI-HAL extracted (55, headline) | CTI-HAL gold (55) | CTI-HAL accepted (54) | Elastic (6) |
|---|---|---|---|---|
| current artifact | 0.127 (0.273) | 0.182 (0.364) | 0.111 (0.333) | 0 (0.167) |
| refit (vendor reports, weight 30) | 0.127 (0.345) | 0.145 (0.309) | 0.130 (0.370) | 0 (0.333) |
| refit − current, top-1 (paired CI) | +0.000 [−0.091, +0.109] | −0.036 [−0.109, +0.036] | +0.019 [−0.037, +0.074] | 0 |

- **Headline fails:** the same 7 of 55 reports right at top-1. Top-5 rises on the extracted input
  (+0.073, CI [−0.018, +0.164]) and MRR from 0.224 to 0.232, within noise.
- The dev gain (ATT&CK campaigns) did not carry to CTI-HAL: the vendor reports cover 34 groups
  with one report each and none of the 6 test groups' habits.

Reading: the network's limit is its input, not its training. From technique sets alone, 13–18% is
where three different models (this network, its refit, the LLM heads) land on these reports, and
the group signal the product uses comes from the named tools. No adoption; the attribution stage is
unchanged.

**Cost.** None: eight fits and all scoring on the laptop.

**An actor head: the group from the report's sentences (T1, 2026-10-16, prereg
`cribl_actor_head_2026-10-16.json`).** The owner's pick after the technique-list results: let the
LLM read the statements, since the group signal is in how a technique is done. `export_actor_cases.py`
builds the training texts from `text_group.build_corpus` (ATT&CK's per-group procedure examples and
descriptions, single-group software and campaign procedure examples; group names → "the adversary",
campaign names → "the operation"; 12,675 sentences over the 153 catalogue groups, 892 with 2–3
groups) minus 49 that overlap a CTI-HAL, Elastic or vendor statement, and the report sets as one row
per anonymised statement (CTI-HAL 922, Elastic 848, vendor 3,821; 850 of the vendor ones as the
selection set). `family_head.py --task actor` trains the same head (Qwen3.5-4B-Base, fresh LoRA
r 16, 2 epochs, 21 min, ~$0.45); `group_text_eval.py` averages each group's probability over a
report's statements and scores the product's TF-IDF text-group model on the same statements.

| Group top-1 (top-3 / top-5), 153 groups | CTI-HAL (55, headline) | Elastic (8) | vendor (34, dev) |
|---|---|---|---|
| TF-IDF text-group model (`text_group.py`) | **0.491** (0.618 / 0.691) | 0.125 (0.500 / 0.625) | 0.853 (0.882 / 0.912) |
| T1, actor head, per-report mean probability | 0.255 (0.436 / 0.491) | 0.375 (0.500 / 0.625) | 0.353 (0.441 / 0.500) |
| group network, E2's extracted techniques | 0.127 (0.236 / 0.273) | 0 (0.167 / 0.167) | — |
| T1 − TF-IDF, paired 95% CI | top-1 −0.236 [−0.382, −0.109]; MRR −0.211 [−0.323, −0.102] | +0.25 [−0.25, +0.75] | −0.50 [−0.68, −0.29] |
| T1 − group network | top-1 +0.127 [0.000, +0.255]; top-5 +0.218 [+0.073, +0.382] | | |

- **Headline fails:** the TF-IDF model is 24 points ahead on CTI-HAL (median gold rank 2 against
  7). Per group (T1 / TF-IDF right of n): Carbanak 0/2 of 10, Sandworm 4/4 of 5, FIN6 3/6 of 11,
  FIN7 2/8 of 18, OilRig 2/4 of 8, Wizard Spider 3/3 of 3.
- **Reading text beats reading technique lists:** the same head doubles the group network's
  top-1 on the same reports (+12.7 points, CI touching 0) and nearly doubles its top-5.
- **One caveat for the TF-IDF side:** its corpus is unfiltered, and ATT&CK paraphrases some of
  these reports; on the vendor reports, which ATT&CK cites heavily, it scores 85% where the
  overlap-filtered head scores 35%. The CTI-HAL gap is too wide to be that alone. On Elastic
  (reports after the ATT&CK snapshot) the head leads 3 to 1 of 8.
- Statement-level top-1 is 11–13% everywhere: most single sentences carry no group identity;
  the report-level average does the work.

Reading: given the same anonymised text, a bag of words matched against the groups' documented
procedures still beats a 4B model with a cosine head trained on 12,675 sentences. The LLM's gain
over technique lists is real, but it does not reach the simple text model, which is already in the
product (weight 0, docs 21.3.16). No adoption.

**Cost.** One pod, about $0.45 (30 minutes at $0.89/h). Scoring on the laptop. No OpenRouter credit.

**CTIBench and AthenaBench (2026-10-19, prereg `cribl_external_bench_2026-10-19.json`, evaluation
only).** The owner asked for the public CTI benchmarks. Both are non-commercial (CTIBench CC BY-NC-SA
4.0; AthenaBench's academic research licence): local evaluation, nothing trained on them, raw files
under `data/benchmarks/` (gitignored). Only their extraction and attribution tasks fit models that
do not generate text; the knowledge, CVE→CWE, CVSS and mitigation tasks were not run.
`external_bench.py` prepares the cases, runs the heads and scores under the benchmarks' rules.

*Technique extraction.* CTI-ATE (CTIBench): 60 ATT&CK software descriptions (47 Enterprise, 13
Mobile), 6.6 parent techniques each, F1. E2 reads the sentences the product's splitter makes (398)
and answers one technique per sentence (or up to three at 1.3 per sentence); the union at parent
level is scored. AthenaBench ATE: 500 GPT-written scenarios, one answer each, accuracy at parent
level (the benchmark's rule; 352 answers are sub-techniques).

| Technique extraction | E2 (this work) | published models (same metric) |
|---|---|---|
| CTI-ATE, all 60 texts, F1 (micro / mean per text) | 0.575 / 0.583; up to 3 per sentence 0.578 / 0.599 | ChatGPT-4 0.639, Llama3-70B 0.472, Gemini-1.5 0.461, ChatGPT-3.5 0.311, Llama3-8B 0.156 (macro-F1 as the paper labels it) |
| CTI-ATE, the 47 Enterprise texts | 0.746 / 0.745; up to 3: 0.755 / 0.765 | not reported separately |
| AthenaBench ATE, accuracy (top-1 at parent level) | **0.736** (top-3 0.910, top-5 0.942; exact id 0.600) | Gemini-2.5-Pro 76.2, GPT-5 76.0, GPT-4o 51.6, Gemini-2.5-Flash 51.6, GPT-4 35.8, Llama-Primus 33.8, Llama 3.3-70B 30.4, Qwen3-14B 19.4, Qwen3-8B 11.8, Qwen3-4B 5.6 |

- On AthenaBench the 4B head sits 2.5 points under the two frontier models and far above every open
  model the authors ran, including its own base family (Qwen3-4B 5.6%, which had to generate an id
  from a prompt). Scenarios whose answer is a sub-technique score 0.719, parent-only ones 0.777.
- On CTI-ATE the catalogue is Enterprise only, so the 13 Mobile texts score 0 and pull the
  all-rows F1 to 0.58, second to ChatGPT-4's 0.64; on the Enterprise rows it reaches 0.75. These
  texts are ATT&CK's software descriptions and E2 trained on ATT&CK's procedure sentences about the
  same software, so the CTI-ATE numbers are flattering; the AthenaBench ones are not.

*Threat-actor attribution.* CTI-TAA: 50 anonymised reports (answers from the authors' key);
AthenaBench TAA: 100 bullet profiles. Both models here name an ATT&CK group: the actor head
(mean probability over the report's sentences or bullets) and the product's TF-IDF text-group
model (whole text). Scored with AthenaBench's rule and lists: correct when alias-connected to the
answer, plausible when connected through related groups, combined 1 / 0.5 / 0.

| Attribution | actor head (T1) | TF-IDF text-group model | published |
|---|---|---|---|
| CTI-TAA (50; 48 answers map to an ATT&CK group): correct / plausible / combined | 0.38 / 0.96 / 0.67 | 0.38 / 0.98 / 0.68 | correct / plausible: ChatGPT-4 52 / 86, Llama3-70B 52 / 80, ChatGPT-3.5 44 / 62, Gemini-1.5 38 / 74, Llama3-8B 28 / 36 |
| AthenaBench TAA (100; 75 map to an ATT&CK group): correct / plausible / combined | 0.16 / 0.84 / 0.50 | 0.23 / 0.79 / 0.51 | accuracy: GPT-5 39, GPT-4o 35, GPT-4 31, Gemini-2.5-Pro 31, Llama 3.1-8B 24, Qwen3-14B 17, Qwen3-4B 15 |

- Correct attributions match Gemini-1.5 on CTI-TAA and sit below the frontier models on both;
  a quarter of AthenaBench's answers are actors ATT&CK does not list (TraderTraitor, WageMole,
  Cleaver), which no model here can name. The two models agree on 13 of 50 and 18 of 100 picks
  and are right together on 12 and 11; either is right on 26 and 28. The actor head leans on a few
  prominent groups (Lazarus on 17 of 50 CTI-TAA reports).
- "Plausible" follows the authors' transitive related-group walk, which connects most known
  actors; the CTIBench paper's plausible numbers came from the authors' own notebook and may be
  stricter, so compare the correct column.

**Cost.** None: all scoring on the laptop (about 12 minutes).

**Product integration: the technique head as mapping evidence (2026-10-19).** Pre-registered
in `docs/preregistration/cribl_product_integration_2026-10-19.json` before any run; the wiring
is described in 8.2. The reference head (E2, `data/models/cribl_ft/E2/step_52706`, checkpoint
fingerprint `5b5d015d…`) runs as a local service (`make technique-head-serve`); with
`TECHNIQUE_HEAD_ENABLED=true` every prose observation is scored once, each candidate gets the
head's probabilities as evidence (acceptance feature schema 4) and up to three head proposals that
retrieval missed join the pool with fused score 0. Log-derived observations are not scored (prose
only, for now). Three arms, each one run per test set, the acceptance model of the registered arm
refitted once on public dev with the head active (`acceptance_model_technique_head.json`, schema
4, threshold 0.23, dev F1 0.488 against 0.468 for the shipped model; it records the head's
fingerprint and runs only on that head). Accepted-set metrics, one statement per case:

| Arm | Public test (1,391): P / R / F1 / parent-aware F1 / hit@1 | CTI-HAL (968): P / R / F1 / parent-aware F1 / hit@1 |
|---|---|---|
| off: the shipped pipeline rerun under this code (reproduces the committed artifacts exactly) | 0.516 / 0.349 / 0.417 / 0.451 / 0.501 | 0.425 / 0.341 / 0.378 / 0.425 / 0.383 |
| proposer: head on, shipped schema-3 model | 0.518 / 0.349 / 0.417 / 0.452 / 0.501 | 0.428 / 0.341 / 0.379 / 0.426 / 0.383 |
| **head**: head on, schema-4 model (registered) | **0.553 / 0.398 / 0.463 / 0.500 / 0.556** | **0.553 / 0.463 / 0.504 / 0.564 / 0.509** |

- **Headline (CTI-HAL, head minus off, paired case bootstrap, 1,000 resamples, seed 7):** F1
  **+0.126 [+0.101, +0.149]**, precision +0.128 [+0.098, +0.156], recall +0.122 [+0.099, +0.146].
  Public test: F1 +0.046 [+0.031, +0.061], precision +0.037 [+0.016, +0.057], recall +0.049
  [+0.035, +0.063]; every dataset gains (tram 0.555 → 0.597, derived procedures 0.311 → 0.372,
  expert 0.263 → 0.308). The precision constraints (no drop beyond 0.02) are met with precision
  rising on both sets.
- **Where it comes from (CTI-HAL).** Accepted techniques 840 → 877, of which right 357 → 485;
  statements with nothing accepted 305 → 198. Of the 305 statements the shipped pipeline accepted
  nothing for, 75 now get a right technique (92 a wrong one, 138 still nothing); of the 312 it
  accepted only wrong techniques for, 73 now get a right one and 54 nothing; of the 351 it got
  right, 330 stay right (15 slip to wrong only, 6 to nothing). Multi-technique acceptances fall
  from 164 to 102 statements: the model trades spread for one confident answer. The head's own
  proposals matter little: within the stored top-12 rows the head arm accepts 41 head-only
  candidates on CTI-HAL, 16 of them right (33 / 15 on the public test); the gain is the
  re-scoring of retrieved candidates with the head's opinion.
- **The proposer arm** (head on, shipped model) changes almost nothing (+0.001 F1 on both sets):
  the shipped model judges head-only candidates on no retrieval evidence, so they stay in review.
  This is the arm a user gets by starting the service without the refit.
- **Weights (standardised).** `th_family_p` +0.37, `th_technique_rank` +0.31, `th_technique_p`
  +0.23, `th_top_p` −0.32 (a head confident about *its* family lowers every other candidate),
  `th_proposed` +0.08, `th_only` +0.07; the exemplar vote features keep the largest weights.
- **Runtime.** One request per statement: +0.17 s per public case (546 s against 316 s for 1,391)
  and +0.07 s per CTI-HAL statement; the service holds about 3.6 GB of GPU memory. No API credit.
- **Honesty.** The head's checkpoint was selected on public dev + AnnoCTR dev and the acceptance
  model is fitted on public dev, so the dev operating point is flattering; the public test splits
  and CTI-HAL were never used to train or choose anything on either side. 55% precision is still
  the ceiling the CTI-HAL labels allow a one-answer gate (21.3.18 above).

*Two models behind one gate (amendment 2).* The first clinic check ran the schema-4 model on
every candidate, including those of log-derived observations the head never scores: with every
`th_*` feature at 0, a value absent from the fit, it lost 0.12 F1 on the eCAR windows and 0.07 on
the narrative mode (`clinic_head_schema4_only.json`). The registered arm therefore pairs the two
models (`acceptance.HeadAwareAcceptance`, `ACCEPTANCE_MODEL_HEAD_PATH`, 8.2): the schema-4 model
judges the observations the head scored, the shipped model the rest. Rerun under the pair, the
public and CTI-HAL numbers above are unchanged to the last digit (every statement is prose) and
the checks below are the pair's.

*No-regression checks (head on vs off, same code, one run each).*

| Check | Off | Head on | Reading |
|---|---|---|---|
| EvidenceForge clinic bundle, local pipeline (narrative / windows / eCAR F1) | 0.357 / 0.600 / 0.545 | identical, bit for bit | the head reads no log-derived observation and the pair leaves them to the shipped model |
| `check-regression` against the committed public baseline | pass | pass (F1 0.225 → 0.463) | |
| CTI-HAL attribution, anonymised headline (55 reports): top-1 / top-5 / MRR / named correct / wrong names / technique recall | 0.455 / 0.600 / 0.512 / 10 / 0 / 0.405 | 0.473 / 0.636 / 0.540 / 9 / 0 / 0.491 | one more report right, one fewer reaches the naming gate, no wrong name |
| ATT&CK folds, seed 7 (178 cases, 153 known): top-1 / top-5 / named correct / wrong names / technique recall | 0.824 / 0.941 / 26 / 0 / 0.604 | 0.837 / 0.974 / 38 / 0 / 0.778 | **not evidence of a gain**: E2 trained on ATT&CK's procedure sentences, which are the fold cases (the fold hiding keeps them out of the indexes, not out of the head); read as "nothing broke, no wrong name" only. The off numbers are the leak-free baseline under this code, below the committed artifact of 2026-10-02 that predates the BM25 fold hiding (21.3.5) |
| Elastic real reports, strict anonymisation (amendment 3; 8 attributed posts + 22-post open set): top-1 / named correct / wrong names / technique recall; open-set posts given a cluster name | 4/8 / 1 / 0 / 0.759; 1 of 22 | 4/8 / 1 / 0 / 0.833; **2 of 22** | the attributed posts are unchanged and clean, but the head's extra accepted techniques push a second commodity post (the GoSar RAT write-up) through the cluster gate as "Lazarus Group / APT38" at probability 0.99, next to the banking-extension post the shipped pipeline already names "Kimsuky / Lazarus Group"; the benchmark's combined named risk rises from 0.50 to 0.67 |

**Outcome and decision.** The registered headline holds with room to spare (CTI-HAL accepted-set
F1 +0.126, precision up), the public test gains, the clinic bundle and the regression check are
clean, and real-report attribution ranks slightly better. The naming side is not clean: one more
open-set post is named, and the fold naming check is uninformative because the head saw the fold
sentences in training. Under the project's rule that a default flip needs a clean naming gate, the
head stays **off by default**: everything is in place for an opt-in (`make technique-head-serve`,
`TECHNIQUE_HEAD_ENABLED=true`, the pinned fingerprint and `ACCEPTANCE_MODEL_HEAD_PATH` with the
shipped model as the partner), and the open-set naming is the one thing to settle before a default
flip, for instance by holding the cluster gate to its own separation floor on head-boosted cases,
measured on a fresh fold seed, never on the Elastic set. Artifacts: `data/audits/cribl_ft/
integration/` (gitignored; `integration_report.py` composes `integration_report.json`).

**Multi-answer acceptance (2026-10-19, rejected).** Pre-registered in
`docs/preregistration/cribl_multi_answer_acceptance_2026-10-19.json`. Motivation: multi-technique
statements are under-recovered (CTI-HAL: 68 statements with two or more labels, 45 get one right,
6 get all; of the 96 missed labels 53 sit in the top-12 rows at a median acceptance probability of
0.05, and for 40% of them the head gives the label's family at least a quarter of its top family's
probability). Feature schema 5 adds the head's second answer as relative evidence
(`th_rel_family`, `th_rel_technique`, a `th_second` flag for the head's second or third family at
≥ 25% of the top one, the statement length and their product); one refit on public dev
(`--feature-schema 5`; dev F1 0.489 against 0.488, threshold 0.25), one run per set through the
head-aware pair. Result against the schema-4 pair (paired, 1,000 resamples): CTI-HAL F1 −0.004
[−0.014, +0.005] with precision +0.014 and recall −0.016; public test F1 −0.006 [−0.012, −0.001];
multi-label recall 0.351 → 0.338 (CTI-HAL) and 0.207 → 0.200 (public); the expert split is
unchanged (0.308 → 0.306). The fit used the new features to raise the threshold, not to admit second
answers: the relative family probability took a negative weight that offsets the absolute one. The
headline rule (no loss) fails on the point estimate and the multi-label constraint fails, so
schema 4 stays the `ACCEPTANCE_MODEL_HEAD_PATH` model; schema 5 remains loadable.

What the breakdown says instead: acceptances beyond the first on a statement are almost always
wrong. With the schema-4 pair, CTI-HAL has 102 statements with more than one accepted technique
and the extra acceptances are 7 right / 100 wrong; on the public test 17 right / 235 wrong. From the
stored rows, keeping only the most probable acceptance per statement would read as precision 0.553
→ 0.62 and F1 0.504 → about 0.53 on CTI-HAL (0.463 → about 0.49 on the public test). That is an
estimate from test rows, not a result: a single-answer cap (or a much higher threshold for a second
acceptance) is the next pre-registrable change, with the choice made on public dev.

**Single-answer cap (2026-10-19, declined on dev).** Pre-registered in
`docs/preregistration/cribl_single_answer_cap_2026-10-19.json`: at most one probability-only
acceptance per observation, a second one only above a bar, with the bar chosen on public dev by
the fit's out-of-fold runtime rule from {none, 0.5 … 1.0} (`fit_acceptance_model.py
--single-answer-cap`; the artifact carries the choice as `metadata.gate`, the runtime follows it
through `TECHNIQUE_SINGLE_ANSWER_CAP=model`, `off` ignores it and a number forces a bar for an
ablation). Both fits declined the cap: on dev the uncapped gate scores out-of-fold F1 0.435
(shipped recipe) and 0.488 (head recipe), every bar 0.429–0.431 and 0.472–0.476, i.e. +2.5 points
of precision for −4 of recall. The dev split's second acceptances are right often enough to keep;
the CTI-HAL rows that motivated the cap (7 right / 100 wrong extras) are not representative of
dev. The "cap" artifacts are therefore uncapped refits and serve as same-code controls: the
head-free refit (threshold 0.23 against the shipped 0.21) reads F1 −0.003 [−0.008, +0.002] on the
public test and +0.002 [−0.006, +0.008] on CTI-HAL against the shipped model; the head-recipe
refit reproduces the schema-4 pair to the last digit. Forced-cap diagnostics (report only, since
dev chose against them) are listed at the end of this subsection.

**Agreement filter: keep only what both the head and the pipeline pick (2026-10-19).**
Product owner's request; pre-registered in `docs/preregistration/cribl_head_agreement_2026-10-19.json`.
A post-gate rule (`TECHNIQUE_HEAD_AGREEMENT`, off by default): an accepted technique of an
observation the head scored stays accepted only when the head picks it too, at one of three
levels (its most probable technique; its most probable family; one of its three most probable
families); verifier verdicts and observations the head did not score are left alone. No refit.
Arms on the shipped gate (the head present only for the filter) and on the schema-4 pair:

| Arm | Public test: P / R / F1 | CTI-HAL: P / R / F1 | Paired F1 vs its baseline: public; CTI-HAL |
|---|---|---|---|
| off (shipped) | 0.516 / 0.349 / 0.417 | 0.425 / 0.341 / 0.378 | |
| shipped gate ∩ head technique | 0.738 / 0.263 / 0.388 | 0.675 / 0.276 / 0.392 | −0.029 [−0.044, −0.015]; +0.013 [−0.006, +0.030] |
| **shipped gate ∩ head family (registered)** | 0.677 / 0.277 / 0.393 | 0.640 / 0.299 / 0.407 | −0.024 [−0.038, −0.011]; **+0.029 [+0.014, +0.043]** |
| shipped gate ∩ head top-3 families | 0.586 / 0.328 / 0.421 | 0.501 / 0.333 / 0.400 | +0.004 [−0.004, +0.011]; +0.022 [+0.015, +0.028] |
| head pair (schema 4) | 0.553 / 0.398 / 0.463 | 0.553 / 0.463 / 0.504 | |
| head pair ∩ head family | 0.603 / 0.344 / 0.438 | 0.601 / 0.435 / 0.505 | vs pair −0.025 [−0.035, −0.016]; +0.001 [−0.010, +0.010] |

- The registered arm passes its CTI-HAL headline (+0.029) but fails the public-test constraint
  (−0.024 beyond its interval): on the public sentences the shipped gate's acceptances that the
  head disagrees with are right far more often than on CTI-HAL. It is a precision/recall trade
  (precision +0.16 / +0.22, recall −0.07 / −0.04), not adopted.
- Top-3 agreement is the one level that costs nothing on the public test (+0.004 n.s.) while
  gaining +0.022 on CTI-HAL with +0.07 precision on both; it is a secondary arm here, so it is a
  candidate for its own registration (as an opt-in precision setting), not an adoption.
- Filtering the pair with the head's family adds precision (+0.05) at a recall cost that leaves
  CTI-HAL flat and the public test lower: the head's evidence is already in the gate, and the
  second vote only removes recall.
- Against the shipped pipeline the pair alone remains the best extraction setting by F1 on both
  sets; the filters are for a user who wants fewer, surer techniques.

*Attribution with the filters (local pipeline, one run each).*

| Arm | CTI-HAL (55): top-1 / top-5 / MRR / named correct / wrong / technique recall | Elastic strict (8 + 22): top-1 / named / wrong / technique recall; open-set named |
|---|---|---|
| off (shipped) | 0.455 / 0.600 / 0.512 / 10 / 0 / 0.405 | 4/8 / 1 / 0 / 0.759; 1 |
| head-free refit control | 0.455 / 0.600 / 0.515 / 10 / 0 / 0.394 | 4/8 / 1 / 0 / 0.745; 1 |
| head pair | 0.473 / 0.636 / 0.540 / 9 / 0 / 0.491 | 4/8 / 1 / 0 / 0.832; 2 |
| shipped gate ∩ head family (registered) | 0.473 / 0.618 / 0.520 / **15** / 0 / 0.363 | 4/8 / 1 / 0 / 0.722; 2 |
| head pair ∩ head family | 0.473 / 0.618 / 0.538 / **15** / 0 / 0.467 | 4/8 / 1 / 0 / 0.832; 2 |

- Fewer, surer techniques help the naming gate more than more techniques do: with the family
  filter 15 of the 55 CTI-HAL reports are named correctly (coverage 27%) against 10 shipped and 9
  with the pair, with no wrong name, although technique recall falls to 0.36. The pair with the
  filter keeps most of the recall (0.47) and the same 15 names.
- The open-set finding of the parent registration repeats: every arm that cleans the accepted set
  names the GoSar commodity post as "Lazarus Group / APT38" (cluster probability 0.99 with 51 to 60
  accepted techniques), the shipped pipeline does not. It is not a head artifact but a cluster-gate
  weakness on a cleaner technique set; the registered arm fails its open-set constraint on it, as
  the pair did. The fix belongs in the cluster gate (8.2, 10.11), designed on the vendor set and the
  campaigns, never on Elastic.
- Decision: nothing adopted. The filters stay opt-in (`TECHNIQUE_HEAD_AGREEMENT`), the head stays
  off by default, and the one lever both experiments point at is the cluster gate's behaviour on
  commodity reports with clean technique sets.

*Forced-cap diagnostics (report only; dev chose against the cap).* `TECHNIQUE_SINGLE_ANSWER_CAP=1.0`
(strictly one probability-only acceptance per statement; cue-supported ones are never capped) on
the shipped gate and on the head pair, one run per set:

| Gate | Public test: P / R / F1 (statements with > 1 accepted) | CTI-HAL: P / R / F1 (statements with > 1 accepted) |
|---|---|---|
| shipped | 0.516 / 0.349 / 0.417 (256) | 0.425 / 0.341 / 0.378 (164) |
| shipped, strict cap | 0.562 / 0.322 / 0.410 (73) | 0.467 / 0.307 / 0.371 (27) |
| head pair | 0.553 / 0.398 / 0.463 (230) | 0.553 / 0.463 / 0.504 (102) |
| head pair, strict cap | 0.585 / 0.369 / 0.452 (69) | 0.580 / 0.434 / 0.497 (19) |
| head pair, second acceptance above 0.5 | 0.586 / 0.373 / 0.456 | 0.582 / 0.439 / 0.501 |

The cap buys 3 to 4.5 points of precision for 2.7 to 3 points of recall and loses F1 everywhere,
on test exactly as on dev. The earlier estimate from the stored rows (+2 to +3 F1) was wrong: it
counted the extra acceptances on statements whose first acceptance was already wrong as pure
noise, while the cap keeps that wrong first one and drops the right second one. The registered
procedure (choose on dev) got this right; the lesson for the record is that a test-row estimate
is not a dev result.

**Cluster gate: the evidence-volume guard for commodity write-ups (2026-10-19).** Pre-registered
in `docs/preregistration/cribl_cluster_volume_guard_2026-10-19.json`; the rule is in 10.11. The
analysis behind it: the two wrongly named Elastic open-set posts carry 51 to 95 accepted
techniques, every correctly named report in the design sets at most 20 (vendor set 7 to 18 among
the 16 correct names, design fold seeds 41/53 up to 17); profile coverage of the named unit
(0.68 to 0.76) and the number of rare matched techniques do not separate them (the long lists
hold more rare matches, not fewer), evidence volume does. The bar of 30 accepted techniques was
fixed from the design sets before any run. An offline replay over every existing artifact finds
the guard inert on the design seeds, the vendor set, the seed-7 folds and CTI-HAL (no row above
30 without decisive software) and holding exactly the two wrong clusters; applied to single
names it would hold none anywhere. Confirmation runs, guard on against the existing arms:

| Set | Head off: without → with the guard | Head on: without → with the guard |
|---|---|---|
| Elastic attributed (8): top-1 / named correct / wrong | 4/8 / 1 / 0 → unchanged | 4/8 / 1 / 0 → unchanged |
| Elastic open set (22): cluster names | 1 → **0** | 2 → **0** |
| CTI-HAL (55): top-1 / named correct / wrong | 0.455 / 10 / 0 → identical row for row | 0.473 / 9 / 0 → identical |
| Vendor set (34, dev): top-1 / named correct / wrong | 0.706 / 16 / 2 → identical | not run |
| ATT&CK folds, seed 7 (178): top-1 / named / wrong / cluster names | 0.824 / 26 / 0 / 1 correct → identical row for row | not run (the head saw the fold sentences) |

The headline holds: no correct name is lost anywhere, the Elastic attributed posts are unchanged,
and the open set names nothing with the head off or on. **Adopted**: `ATTRIBUTION_CLUSTER_MAX_TECHNIQUES`
defaults to 30 (commit of 2026-10-19); the current attribution artifacts on the main branch
predate it and are regenerated after the merge (they cannot change on the folds or CTI-HAL, only
the Elastic open-set count). The guard holds, it never names; a long APT report with group-exclusive
malware named passes as before, and a long one without it is shown as a shortlist, which is what
the Elastic attributed posts above 30 techniques already were. The open set has 7 APT-like posts
and 15 commodity ones; the guard does not look at that label, only at the evidence volume.

**The head on by default: re-evaluation of 2026-10-20.** Product owner's decision after the
cluster guard settled the open-set naming: `TECHNIQUE_HEAD_ENABLED=true`, pinned to the registered
checkpoint, the head-aware acceptance pair, the cluster volume guard at 30. Every local benchmark
was rerun under these defaults with the service up (the benchmarks refuse to write a run the head
did not score throughout), and the four other technique sets were run head off and on:

| Benchmark (local pipeline) | Before (committed artifact or the same-code off arm) | New default |
|---|---|---|
| Public held-out, pooled (1,391): P / R / F1 / parent-aware F1 / hit@1 / hit@10 | 0.516 / 0.349 / 0.417 / 0.451 / 0.501 / 0.805 | **0.553 / 0.398 / 0.463 / 0.500 / 0.556 / 0.851**; without cues 0.467 (the cues are now a hair net-negative); `check-regression` passes |
| CTI-HAL statements (968): P / R / F1 / parent-aware F1 | 0.425 / 0.341 / 0.378 / 0.425 | **0.553 / 0.463 / 0.504 / 0.564** |
| CTI-HAL attribution, anonymised (55): top-1 / top-5 / MRR / named correct / wrong / technique recall | 0.455 / 0.600 / 0.512 / 10 / 0 / 0.405 | 0.473 / 0.636 / 0.540 / 9 / 0 / 0.491 |
| CTI-HAL attribution, as reported | 0.564 / 0.727 / 0.636 / 13 / 0 / 0.399 | 0.564 / 0.764 / 0.650 / 12 / 0 / 0.488 |
| Elastic strict (8 + 22 open): top-1 / named / wrong / technique recall; open-set names | 4/8 / 1 / 0 / 0.759; 1 (same code) | 4/8 / 1 / 0 / 0.832; **0** |
| Elastic names / tradecraft / as reported: top-1, top-5, named | 5/8, 0.75, 1 / 1/8, 0.25, 0 / 5/8, 0.75, 0 | 5/8, 0.75, 1 / 1/8, 0.375, 0 / 5/8, 0.75, 1 |
| ATT&CK folds, seed 7 (178): top-1 / top-5 / named / wrong / technique recall | 0.824 / 0.941 / 26 / 0 / 0.604 (same code) | 0.837 / 0.974 / 38 / 0 / 0.778 (optimistic: the head trained on the procedure sentences; no wrong name is the only reading) |
| EvidenceForge clinic (logs) | 0.357 / 0.600 / 0.545 | identical (logs are not scored) |
| Curated retrieval diagnostic | 0.765 | identical (the script calls retrieval without the gate or the head) |

The other technique sets (head off → on, same code; `scripts/cribl_finetune/export_eval_sets.py`
puts them in the mapping benchmark's format under `data/benchmarks/extra/`):

| Set | Rule | Off | Head on |
|---|---|---|---|
| AnnoCTR test (738 sentences, parent-level labels) | parent-aware F1 / exact F1 / hit@1 | 0.414 / 0.272 / 0.331 | **0.611** / 0.392 / 0.453 |
| CTI-Rationale test (209) | exact P / R / F1, hit@1 | 0.394 / 0.287 / 0.332, 0.325 | 0.447 / 0.357 / **0.397**, 0.416 |
| AthenaBench ATE (500 scenarios, parent labels; evaluation only) | top-ranked technique right at parent level / top-3 / accepted set holds the gold family / parent-aware F1 | 0.498 / 0.758 / 0.348 / 0.390 | **0.682** / 0.860 / 0.616 / 0.603 (the head alone 0.736; frontier models 0.76) |
| CTIBench ATE (60 texts, accepted techniques pooled per text, parent level; evaluation only) | P / R / F1 | 0.533 / 0.385 / 0.447 | 0.644 / 0.506 / **0.567** (the head alone 0.575) |

| CyberThreat-Eval TTP extraction (xse/CyberThreat-Eval, MIT; 186 expert-labelled CTI digests read as pasted reports, 7,612 sentences; prereg `cribl_cyberthreat_eval_2026-10-20.json`) | the authors' rule: article-level exact ids, micro P / R / F1; parent level beside | 0.119 / 0.244 / 0.160; parent 0.229 | 0.139 / 0.310 / **0.192** (+0.032 [+0.021, +0.043]); parent 0.261 |

Every set moves the same way: precision and recall both up, hit@1 up 10 to 18 points, no wrong
name anywhere, the open set clean. CyberThreat-Eval is the one set where both arms score low in
absolute terms: its labels are a curated set of about 8 techniques per article (median 681 words,
39 sentences), while the pipeline reads every sentence and accepts 17 to 19 techniques per
article, so precision is the limit (0.14) and recall only reaches 0.31; the head adds three
points of F1 at both levels. That is the task formulation (an analyst's digest set against an
exhaustive statement-level read), not the per-statement quality measured above; a
precision-oriented setting was tried on it: the family agreement filter on the head pair
(`TECHNIQUE_HEAD_AGREEMENT=family`) scores 0.141 / 0.295 / 0.190, i.e. it removes about one
accepted technique per article and changes nothing, because the surplus acceptances are
techniques the head agrees the sentences describe; they are simply not in the curated article
set. The paper's own LLM baselines are not in the dataset card and are not quoted. The gains are largest where the shipped pipeline was weakest
(AnnoCTR, CTI-HAL) and smallest on CTI-Rationale, whose labels are rationale-level and partly
sub-technique. The committed `*_current.json` artifacts on the main branch predate the default
and are regenerated after the merge (the numbers here are the worktree's run of the same
targets).

## 22. Failure modes and troubleshooting

### 22.1 “ATT&CK knowledge base has not been ingested”

Run:

```bash
make up
make ingest
```

Then restart the API (`make api` or `make run-next`; it loads the knowledge base at startup) or
reload Streamlit.

### 22.2 Actor behavior index shows zero

The existing group collection predates actor-specific behavior vectors. Run:

```bash
make ingest
```

`GET /api/system` (`index.actor_behavior_documents`) or the Streamlit sidebar should report a
nonzero actor behavior count.

### 22.3 Chroma unavailable

Check:

```bash
curl http://127.0.0.1:8000/api/v2/heartbeat
docker compose config --quiet
```

Then inspect the local container runtime. The application requires the Chroma HTTP service at the configured host and port.

### 22.4 No candidates after analysis

This is an intentional safe outcome when no candidate has a documented technical link. Add:

- More concrete attack behavior.
- Tool names and command fragments.
- Relevant Windows/Sysmon logs.
- Verified IOCs.
- Additional independent observations.

Do not lower thresholds blindly; lower thresholds increase unrelated retrieval.

### 22.5 Live source unavailable

The Live Intel layer continues with healthy sources and displays the failed source status. Retry later. Do not treat absence from a failed feed as negative evidence.

### 22.6 Live ransomware claim appears

Treat it as an unverified claim. Corroborate with the victim, regulator, law enforcement, a trusted incident report, or independent reputable reporting before adding it as case evidence.

### 22.7 OpenRouter structured-response error

The client automatically handles common content formats and retries once. If the final error includes a stage name, use it to identify whether the failure occurred during:

- Blinded ACH.
- Unblinded ACH.
- Devil’s advocate.
- Final reconciliation.

Check API access, model availability, provider limits, and the reported finish reason. The application does not print response contents in the error because they may contain sensitive incident data.

---

## 23. Known limitations and analytical risks

### 23.1 ATT&CK presence is not uniqueness

A group documenting a technique means the group has been reported using it. It does not mean:

- The group is the only actor using it.
- The supplied event was performed by that group.
- The reporting is complete.
- The reporting is independent.

### 23.2 Reporting and publication bias

Groups with more public reporting have more descriptions, software links, campaigns, and citations. This can make them easier to match than less-documented groups.

### 23.3 Shared tooling and false flags

PowerShell, scheduled tasks, LSASS access, RDP, common loaders, and commodity malware are widely shared. Deliberate false flags can imitate another actor’s visible tradecraft.

### 23.4 Country inference

Country hypotheses come from public sponsor/origin language associated with ranked ATT&CK groups. They are not network geolocation. Infrastructure may be rented, proxied, compromised, or deliberately misleading. Country confidence is reduced when candidates are close or origin coverage is incomplete.

### 23.5 APT terminology

“APT-like” is used here for documented state-linked or espionage-style profile evidence. A financially motivated group can be technically advanced and persistent without being state-sponsored. “Not APT-like” does not mean low severity.

### 23.6 Live-intel staleness

Live feed snapshots are cached. A cached record is not evidence that activity is still active at the moment of analysis. The fetch timestamp must be retained with reports.

### 23.7 Public breach claims

Ransomware leak-site records are claims. Victim names, sectors, countries, discovery dates, and descriptions may be inaccurate or generated from incomplete information.

### 23.8 Incomplete negative evidence

Failure to observe a technique or failure to find a live report is not proof that it did not occur.

### 23.9 Model and vector limitations

ATTACK-BERT is optimized for ATT&CK attack-action similarity, not calibrated actor attribution. Similarity thresholds and behavior discrimination margins require evaluation against representative internal data.

---

## 24. Extension points

### 24.1 Add a new live feed

Implement in `apt_classifier/threat_intel.py`:

1. Add a hardcoded HTTPS endpoint.
2. Add a parser returning `ThreatIntelItem`.
3. Normalize dates and cap record sizes.
4. Set an explicit verification classification.
5. Add source health status.
6. Add mocked parser and partial-failure tests.
7. Add the source to the Live Intel UI and documentation.

Do not allow arbitrary analyst-provided URLs to become server-side fetch targets.

### 24.2 Add a new classifier

1. Create a classifier implementing the `Classifier` protocol.
2. Return `ClassifierResult` and structured `ClassifierEvidence`.
3. Abstain when its required signal is absent.
4. Avoid ranking zero-score candidates.
5. Add the classifier to `default_classifiers()`.
6. Add a default weight and, if appropriate, a family mapping.
7. Add contribution, unmatched evidence, and correlation tests.
8. Verify it does not duplicate an existing evidence family.
9. Refit the learned combiner (`make fit-combiner`; add the classifier to `CLASSIFIERS` in
   `apt_classifier/combiner.py` to give it features) and rerun both attribution benchmarks. Any
   change under `apt_classifier/classifiers/` changes the pipeline fingerprint, and the combiner
   fails closed until it is refitted.

### 24.3 Add a new Canvas layer

1. Add a `CanvasNode.kind` and relationship types in `apt_classifier/canvas.py`.
2. Add deterministic IDs and deduplication.
3. Include supporting details and provenance.
4. Add layer labels, colors, and column placement in `web/src/pages/Canvas.tsx` (`KIND_LABELS`,
   `COLUMN_BY_KIND`, `COLOR_BY_KIND`) and in `_canvas_spec()` (`app/ui/pages/canvas.py`, Streamlit).
5. Add an inspector section if the layer is candidate-specific.
6. Add graph-model tests.

### 24.4 Replace or supplement RRF

Any alternative fusion method must preserve:

- Classifier abstention.
- Candidate eligibility.
- Correlation diagnostics.
- Effective-weight auditability.
- Separation/contested warnings.
- The distinction between ranking index and probability.

### 24.5 Add calibrated evaluation

A future attribution evaluation should use time-split, independently labeled cases with:

- Actor labels.
- Victimology labels.
- Tool and infrastructure provenance.
- Unknown/unattributed cases.
- Negative and missing evidence.
- Confidence calibration metrics.
- Abstention quality.
- Feed availability timestamps.

The current ATT&CK self-consistency benchmarks are not sufficient for this purpose.

---

## 25. Source map

| Implementation reference | Key symbols |
|---|---|
| `api/app.py`, `api/serializers.py` | `create_app`, routes and error mapping, `AnalysisState` → JSON |
| `apt_classifier/service.py` | `CaseService`: engine, case store and every analyst operation behind the API |
| `web/src/` | React pages (`pages/`), API client and types (`api/`) |
| `app/streamlit_app.py`, `app/ui/` | Streamlit UI: `runtime.get_engine`, one `render()` per page in `pages/` |
| `apt_classifier/session.py` | `AnalysisState`, `AttributionEngine.analyze`, `rescore`, `attach_threat_intel`, `next_question` |
| `apt_classifier/evidence/models.py` | `EvidenceStatus`, `Provenance`, `Observation`, `TechniqueMatch`, `QuestionAnswer`, `StrategicContext`, `EvidenceLedger` |
| `apt_classifier/evidence/extract.py` | `extract_attack_actions`, `extract_evidence` |
| `apt_classifier/evidence/winlogs.py` | `parse_winlogs` |
| `apt_classifier/evidence/iocs.py` | IOC and software alias extraction |
| `apt_classifier/attack/parse.py` | `parse_attack_bundle`, relationship resolution, actor behavior preservation |
| `apt_classifier/attack/kb.py` | `Technique`, `Group`, `Campaign`, `Software`, `AttackKnowledgeBase` |
| `apt_classifier/vectorstore.py` | `technique_documents`, `group_documents`, `ChromaVectorStore` |
| `apt_classifier/retrieve.py` | `retrieve_techniques`, `apply_candidate_decisions`, `LexicalTechniqueIndex`, hybrid fusion, abstention, explicit cues, name-token matching, hierarchy expansion, context construction |
| `apt_classifier/presets.py` | `MappingPreset`, `apply_preset`, preset definitions (balanced, conservative, exploratory, local_only, legacy) |
| `apt_classifier/reranker.py` | `LocalReranker`, `RerankerConfig`, optional cross-encoder reranker adapter |
| `apt_classifier/classifiers/registry.py` | `default_classifiers`, classifier order |
| `apt_classifier/fusion.py` | `reciprocal_rank_fusion`, effective family weights, Kendall tau |
| `apt_classifier/origin.py` | `assess_origin_and_threat_type` |
| `apt_classifier/threat_intel.py` | public feed client, parsers, matching, verification states |
| `apt_classifier/elastic_mcp.py` | `ElasticMCPClient`, `StreamableHTTPMCPTransport`, ES|QL flow, search fallback, bounded query planning, staged evidence normalization |
| `apt_classifier/canvas.py` | `build_analyst_canvas`, `build_candidate_dossier` |
| `apt_classifier/questions.py` | information gain, chat answer parsing, stopping rule |
| `apt_classifier/review/context.py` | compact review payload and raw-data reduction |
| `apt_classifier/review/pipeline.py` | four-stage isolated review and audit persistence |
| `apt_classifier/llm.py` | OpenRouter client, structured response parsing and retry |
| `scripts/curated_retrieval_benchmark.py` | 36-case curated retrieval benchmark with mode comparison |
| `scripts/otrf_benchmark.py` | 9-case OTRF Sysmon retrieval benchmark |
| `scripts/ensemble_benchmark.py` | held-out group self-consistency benchmark |
| `tests/test_retrieve.py` | retrieval, hierarchy, abstention, explicit cue, and candidate decision tests |
| `tests/test_presets.py` | mapping preset tests |
| `tests/test_reranker.py` | local reranker adapter tests |
| `tests/test_benchmarks.py` | benchmark case structure validation tests |
| `scripts/cribl_finetune/` | cribl-decision experiment (21.3.18): `common.py` (Jev C1 question, `forced_pick`, metrics), `decision_model.py` (`Readout`), `train.py`, `evaluate.py`, `rescore.py`, `prepare_annoctr.py`, `family_head.py` (family head without retrieval), `multi_family.py`, `export_attack_procedures.py`, `runpod/` |

---

## 26. Quick-start checklist

```bash
# 1. Create the environment
make setup

# 2. Configure the local secret file
cp .env.example .env
# Set OPENROUTER_API_KEY in .env without committing the file.

# 3. Start ChromaDB
make up

# 4. Build/update ATT&CK and actor-behavior vectors
make ingest

# 5. Start the local application (React + FastAPI on http://127.0.0.1:8100)
make web-install   # once
make run-next
# legacy Streamlit UI: make run

# 6. Verify the project
make test
make lint
docker compose config --quiet
```

Operationally, start in Intake, enter concrete observations and optional strategic context, review the Canvas, refresh Live Intel, use the automatic Interview, and only then run the isolated AI review. Treat every result as a documented-tradecraft hypothesis requiring analyst validation and disconfirmation.
