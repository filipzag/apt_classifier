# Free text to ATT&CK group: research study

Question: can a model read the sentences of a report (before any technique mapping) and rank the
ATT&CK group behind it, when it is trained on ATT&CK text only? The study compares six families
of text models with the shipped pipeline and with the technique + software group network
(`apt_classifier/group_mlp.py`), alone and combined.

This folder is a port of a finished scratch study so that it can be rerun. Algorithms, defaults,
`BEST` configurations and seeds are unchanged; only paths (and lint) changed.

## Where things live

- Code: this folder. `harness.py` holds the protocol and metrics; `exp/<family>.py` holds one
  model family each (`make_fit(**config)`, its `BEST` config and its dev search under `__main__`).
- Data: `STUDY_DIR` = `$TEXT_GROUP_STUDY_DIR`, default `<repo>/data/text_group_study/`
  (gitignored). It holds `corpus.json`, `test_sets.json`, `emb.npz`, the search results
  `exp/<family>.json` (and `exp/<family>_trials.jsonl` resume logs when a search runs), and
  `final_eval.json`, `final_paired.json`.
- Pipeline artifacts: `$TEXT_GROUP_AUDITS_DIR`, default `<repo>/data/audits/`. In a worktree
  `data/audits` is empty, so point it at the main checkout's `data/audits`.
- Technique network: `<repo>/data/models/mlp/group_mlp.npz`; ATT&CK: `<repo>/data/kb`,
  `<repo>/data/raw`.

| File | What it does |
| --- | --- |
| `build_corpus.py` | Labelled ATT&CK text: group procedure examples, software procedure examples (labelled with every group using that software), group description sentences, campaign procedure examples (dev set). Citations stripped, group names and aliases replaced by "the adversary", campaign names by "the operation". 18,086 records. |
| `export_tests.py` | CTI-HAL (55 reports) and Elastic (8 attributed + 22 open-set posts) texts exactly as the attribution benchmarks read them, at four levels: `as_reported`, `names`, `strict`, `tradecraft`. |
| `embed.py` | ATTACK-BERT embeddings of every unique text (23,796), on CPU (about 10 minutes). |
| `harness.py` | Train/dev/test construction, metrics (top-1/3/5, MRR, macro-F1), `dev_eval`, `test_eval`. |
| `baselines.py` | Smoke baselines (group centroid, sentence kNN). |
| `exp/knn.py`, `exp/knn_baseline.py` | Sentence nearest-neighbour voting (tuned, and the fixed k=10 baseline). |
| `exp/sentence_clf.py` | Sentence-level linear/MLP classifier, aggregated per report. |
| `exp/lexical.py`, `exp/lexical_fixed.py` | Words only, no embeddings: one TF-IDF document per group. `lexical_fixed` adds the corpus placeholder "operation" to the stop words. |
| `exp/mlp_pooled.py` | MLP over mean-pooled report embeddings, trained on synthetic reports. |
| `exp/attention_mil.py` | Gated-attention multiple-instance network over the sentences. |
| `exp/prototypes.py` | Per-group k-means prototypes. |
| `final_eval.py` | Test run of each family's `BEST`, a near-copy ablation, and the combination with the technique network. |
| `paired.py` | Per-report paired comparison of `lexical_fixed` and its combination against the pipeline. |

## Protocol

- Tuning uses only `harness.dev_eval(fit)`: the 23 single-group ATT&CK campaigns with at least 3
  texts. In dev mode, training drops every campaign text and every training text with cosine
  >= 0.95 to a campaign text (77 texts). Selection score = MRR + macro-F1.
- Test sets are scored only by `final_eval.py` and `paired.py`, once per finalist. They refit in
  final mode (all ATT&CK text, campaigns included). Headlines: CTI-HAL `names` and Elastic
  `strict`. Nothing is chosen from a test number, including the combination weight, which is
  picked on the dev campaigns from 0, 0.25, 0.5, 1, 2, 4.

## Run order

From the repository root. `PY` is the project's `.venv` (in a worktree, the main checkout's
`.venv`). `embed.py` needs the ATTACK-BERT cache in `.huggingface` (in a worktree, a link to the
main checkout's).

```sh
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=   # every step runs on CPU

# 1. ATT&CK training text
PYTHONPATH=. $PY scripts/text_group_study/build_corpus.py

# 2. Test texts. The strict/tradecraft anonymiser (scripts/report_anonymizer.py) and the Elastic
#    loader (load_report_folds(..., anonymizer=...)) exist only on branch
#    claude/apt-elastic-security-data-5fa6a3 (the study used its tip, 74ca87b). Run against an
#    archive of that branch, with this repo's data linked into it.
ARCH=$(mktemp -d)
git archive claude/apt-elastic-security-data-5fa6a3 | tar -x -C "$ARCH"
for d in kb raw models benchmarks; do ln -s "$PWD/data/$d" "$ARCH/data/$d"; done
PYTHONPATH="$ARCH" $PY scripts/text_group_study/export_tests.py

# 3. Embeddings
PYTHONPATH=. $PY scripts/text_group_study/embed.py

# 4. Dev searches (dev_eval only); each writes STUDY_DIR/exp/<family>.json
for f in knn sentence_clf lexical mlp_pooled attention_mil prototypes; do
  $PY scripts/text_group_study/exp/$f.py
done

# 5. Test, once per finalist (reads the pipeline artifacts and the technique network)
export TEXT_GROUP_AUDITS_DIR=/path/to/main/checkout/data/audits   # only needed in a worktree
$PY scripts/text_group_study/final_eval.py lexical_fixed lexical sentence_clf knn mlp_pooled attention_mil prototypes knn_baseline

# 6. Paired statistics for the headline
$PY scripts/text_group_study/paired.py
```

`build_corpus.py`, `export_tests.py` and `embed.py` also accept an explicit output path (folder for
`embed.py`) as their first argument. The CTI-HAL and Elastic artifacts that `final_eval.py` reads
are `attribution_cti_hal_current.json`, `attribution_cti_hal_as_reported.json`,
`attribution_elastic_current.json`, `attribution_elastic_tradecraft.json` and
`attribution_elastic_names.json`.

Not ported: the scratch study's smoke, inspect and standalone-check scripts, its logs and its
`*_trials.jsonl` resume logs. `make_fit(**BEST)` works for every family without them. The
`mlp_pooled` search's stage-2 held-out-sentence validation is `exp/mlp_pooled_internal.py`
(training data only, no dev or test data); it appends to `STUDY_DIR/exp/mlp_pooled_internal.jsonl`,
which the search reuses.

## Results (2026-10-02)

From `final_eval.json` and `final_paired.json`. Pipeline = the shipped attribution benchmark
artifacts on the same reports.

| Model | Dev MRR / F1 | CTI-HAL names: top-1 / top-5 / MRR (55) | Elastic strict: top-1 / top-5 / MRR (8) |
| --- | --- | --- | --- |
| Pipeline | | 0.455 / 0.600 / 0.512 | 0.500 / 0.625 / 0.537 |
| Technique network (technique + software) | 0.649 / 0.508 | 0.364 / 0.545 / 0.460 | 0.375 / 0.500 / 0.442 |
| **lexical_fixed** | 0.723 / 0.592 | 0.491 / 0.691 / 0.584 | 0.250 / 0.500 / 0.368 |
| **lexical_fixed + technique network (weight 0.25)** | | **0.545 / 0.727 / 0.628** | 0.375 / 0.625 / 0.490 |
| lexical (no "operation" fix) | 0.767 / 0.667 | 0.491 / 0.691 / 0.587 | 0.250 / 0.500 / 0.368 |
| knn (tuned) | 0.517 / 0.330 | 0.345 / 0.600 / 0.469 | 0.375 / 0.500 / 0.433 |
| knn_baseline (k=10) | 0.398 / 0.242 | 0.364 / 0.582 / 0.465 | 0.250 / 0.500 / 0.394 |
| sentence_clf | 0.606 / 0.409 | 0.236 / 0.545 / 0.401 | 0.250 / 0.500 / 0.339 |
| mlp_pooled | 0.456 / 0.294 | 0.345 / 0.545 / 0.444 | 0.250 / 0.500 / 0.358 |
| attention_mil | 0.371 / 0.253 | 0.127 / 0.309 / 0.223 | 0.125 / 0.375 / 0.278 |
| prototypes | 0.358 / 0.271 | 0.127 / 0.345 / 0.245 | 0.125 / 0.375 / 0.220 |

- The headline text model is `lexical_fixed`: word TF-IDF, one document per group (its own
  sentences plus its software's), prior 0.5. Every embedding family scores lower on CTI-HAL.
- `lexical` scored higher on dev only because "the operation" (the campaign-name placeholder)
  marks groups that have ATT&CK campaigns; adding the token to the stop words took dev from 1.43
  to 1.32 before any test run.
- Paired against the pipeline on CTI-HAL (MRR difference, per-report bootstrap 95% interval):
  `lexical_fixed` +0.072 [-0.031, 0.181] (23 better, 12 worse); the combination +0.116
  [0.039, 0.201] (26 better, 5 worse; top-1 7 gained, 2 lost). On Elastic (8 reports) both are
  below the pipeline and the intervals are wide: text -0.170 [-0.496, 0.089], combination -0.047
  [-0.364, 0.253].
- Near-copy ablation (training texts within cosine 0.92 of a test statement removed: 533 for
  CTI-HAL, 279 for Elastic): `lexical_fixed` is unchanged (CTI-HAL MRR 0.584 -> 0.585).

## Caveats (from the audits)

- **Same-source ATT&CK text.** ATT&CK procedure examples are written from the same public reports
  that CTI-HAL annotates, and the cosine ablation does not catch reworded copies. Dropping the 159
  training records that cite the 13 source reports takes the combination to +0.063
  [-0.003, 0.138] against the pipeline, and the text model alone to -0.021 [-0.121, 0.080].
- **Malware name lookups.** Software names stay in the text, and much of the lexical signal is a
  name lookup (software name -> groups that use it). Masking malware names in the training and
  dev text drops the lexical model's dev selection from 1.43 to 0.95; on CTI-HAL `tradecraft` (names
  masked) `lexical_fixed` falls from 0.584 to 0.519 MRR.
- **The prominence prior** helps on CTI-HAL and misfires on Elastic. With prior 0, the CTI-HAL
  gains shrink (text +0.017, combination +0.067), while the Elastic combination goes from -0.047
  to +0.020.
- **No naming comparison.** These are ranking metrics. The pipeline names 10 of the 55 CTI-HAL
  reports with no wrong name. The combination has no calibrated reject: at the same coverage of
  10 (its 10 most confident reports) it gives 1 wrong name.
- Small samples: 23 dev campaigns, 55 CTI-HAL reports from 6 groups, 8 Elastic reports.

## Licence

CTI-HAL and Elastic texts are local evaluation data only. `test_sets.json` and `emb.npz` contain
them, which is why `data/text_group_study/` is gitignored. Never commit them, and never commit a
model trained on them.
