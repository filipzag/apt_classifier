# cribl-decision fine-tuning experiment

Can a local, fine-tuned decision model pick the right ATT&CK technique for a report sentence?
The model is [`cribl-ai/cribl-decision-1.0`](https://huggingface.co/cribl-ai/cribl-decision-1.0)
@ `75cb0dd9`: a 58 MB LoRA (r=16) on `Qwen/Qwen3.5-4B-Base` @ `1001bb4d`. Both are Apache-2.0. It
scores options it is given and returns probabilities, the same request shape as TypeSafe Jev.

The task is the Jev **C1** question (docs 16.4b, "Question framing"): one sentence, the pipeline's
top-10 candidate techniques plus `none_of_these`, and three option orders combined. Because of that
the result compares directly with the pipeline's rank-1 and with hosted Jev C1 on the same cases.
Experiment only: no product code, setting or `*_current.json` artifact changes. The pre-registration
is `docs/preregistration/cribl_finetune_2026-10-11.json`.

## Readout

Cribl's own prompt format and "decision slot" are not published, so this folder uses its own readout:
- **Prompt**: the C1 state and instructions, then the options lettered `A`, `B`, … and a final `Answer:`.
- **Readout**: the logits of the tokens ` A`, ` B`, … at the last position, through the model's own
  `lm_head`, softmaxed over the offered options.
- **Formats**: three exist (`plain`, `chat`, `json`). `probe.py` picks one on the first 200 dev cases.

## Environments

- **`.venv-cribl`** (`make cribl-venv`, about 5 GB): isolated, transformers 5 + peft + bitsandbytes.
  Qwen3.5 does not load in the project `.venv`, whose transformers 4.57 is kept for ATTACK-BERT.
  transformers picks up flash-linear-attention on its own; it cuts a training step from 3.0 to
  2.2 s (forward speed is the same, because the dense matrix multiplies dominate on this GPU).
- **The project `.venv`**: runs only `export_cases.py`, which needs the KB, ChromaDB and the
  benchmark code.
- **`common.py`**: standard library only. Both sides import it, and the tests cover it.
- **Weights**: about 9.4 GB go to `/home/fzagar/apt_classifier/.huggingface` (`CRIBL_HF_CACHE`).
- **GPU**: the base runs in 4-bit NF4 on the GPU (RTX 4060 8 GB). The scripts refuse to run without CUDA.

## Data

| split | cases | source | used for |
|---|---|---|---|
| train | ~6.3k | public tram/expert/derived_procedures **train** (CC-BY-4.0); candidates from `benchmark_mapping.py --split train` with exemplars **off** | training |
| dev | 659 | public tram/expert dev, the exact candidate lists Jev C1 saw (`s1_dev_c1.json`) | format, checkpoint and arm selection |
| cti_hal | 968 | CTI-HAL statements, Jev C1's candidate lists (`s3_cti_hal_statements.json`); no licence, local only | confirmation, once |
| test | 1391 | public test (tram 713, expert 157, derived 521), pipeline top-10 | confirmation, once |

More detail on the splits:
- **Train dedup**: exact and near (token Jaccard ≥ 0.9) duplicates of any dev, test or CTI-HAL text are dropped.
- **Why exemplars are off for train**: the exemplar index holds the train sentences themselves, so
  they would vote for their own answer.
- **Not used**: ATT&CK procedure sentences, software "uses" text and synthetic sentences. The first
  two are indexed or leak into the `procedures` split; the synthetic set has only 300 sentences.

## Steps

```bash
make cribl-venv                         # once
make cribl-export                       # project .venv + ChromaDB; writes data/audits/cribl_ft/cases_*.jsonl
make cribl-probe                        # smoke test (adapter 304/304 tensors) + format probe, ~30 min
make cribl-eval CRIBL_ARM=B0 CRIBL_FORMAT=<chosen>            # zero-shot on full dev
make cribl-train CRIBL_ARM=B1 CRIBL_FORMAT=<chosen>           # Cribl adapter, fine-tuned, hours
make cribl-train CRIBL_ARM=B2 CRIBL_INIT=fresh CRIBL_FORMAT=<chosen>  # control: new LoRA on the plain base
make cribl-eval CRIBL_ARM=B1 CRIBL_SPLIT=cti_hal CRIBL_ADAPTER=<selected checkpoint> CRIBL_FORMAT=<chosen>
```

From a worktree, add `CRIBL_VENV=/home/fzagar/apt_classifier/.venv-cribl`. Outputs go to two places:
- **`data/audits/cribl_ft/`**: per-case predictions (`<arm>_<split>.jsonl`, resumable), metrics
  (`<arm>_<split>.json`) and training logs (`<arm>_train_log.json`, every checkpoint's dev score and
  the selected one).
- **`data/models/cribl_ft/<arm>/step_<n>/`**: adapters.

The metrics are those of `scripts/jev_benchmark.py`:
- **exact**: the pick is a gold id.
- **parent-aware**: the pick or its parent matches a gold id or its parent.
- **no pick** counts as a miss.
- **ECE**: 10 bins over P(pick).
- **paired deltas**: 95% case-bootstrap intervals, 1000 samples, seed 7.

The ceiling is that gold is among the candidates in 0.838 of dev and 0.741 of CTI-HAL cases.

### Never-abstain rule (prereg amendment 4)

A no-pick is always a miss, so saying `none_of_these` can never raise exact or parent-aware. The
trained arms said it on a fifth of dev and a third of test. `rescore.py` re-decides stored
predictions without the model: the pick is the most probable of the ten techniques, and
P(`none_of_these`) is kept as a separate flag (`none_flag`: its AUROC for cases whose gold is not
among the candidates). It also scores Jev C1 under the same rule from Jev's stored distributions.

```bash
python3 scripts/cribl_finetune/rescore.py --split cti_hal B1_cti_hal.jsonl   # writes B1_cti_hal_forced.json
```

### B3: real-report sentences (AnnoCTR, prereg `cribl_annoctr_2026-10-13.json`)

AnnoCTR (github.com/boschresearch/anno-ctr-lrec-coling-2024, CC-BY-SA 4.0) links sentences of 112
vendor reports to ATT&CK techniques, almost only at parent level. B3 continues B1 step_9462 on its
train sentences (family targets) plus an equal public replay sample, is selected on AnnoCTR dev
(parent-aware, never-abstain) and is confirmed once on CTI-HAL, AnnoCTR test and the public test.

```bash
git clone --depth 1 https://github.com/boschresearch/anno-ctr-lrec-coling-2024 data/benchmarks/annoctr/source
.venv/bin/python scripts/cribl_finetune/prepare_annoctr.py            # data/benchmarks/annoctr/normalized/
for s in dev test train; do PYTHONPATH=. .venv/bin/python scripts/benchmark_mapping.py \
  --root data/benchmarks/annoctr/normalized --datasets annoctr --split $s --modes hybrid \
  --max-candidate-rows 12 --output data/audits/cribl_ft/annoctr_${s}_candidates.json; done
.venv/bin/python scripts/cribl_finetune/export_cases.py --splits annoctr_dev annoctr_test annoctr_train
bash scripts/cribl_finetune/runpod/remote_b3.sh                         # on a GPU pod
```

AnnoCTR was rebuilt on 2026-10-13 after B3: the sentence is now `_context_left + mention +
_context_right` (the stripped `context_*` fields glued the mention to its neighbours, e.g.
"installsransomware"), markdown links with a title and images are stripped, and a sentence with
several mention spans is one case (train 1,500, dev 336, test 738). The candidate files were rebuilt
with `--max-candidate-rows 40`; B3's own inputs are kept in `data/audits/cribl_ft/b3_inputs/`.

### F1: families without retrieval (prereg `cribl_family_head_2026-10-13.json`)

The model reads only the sentence (no candidate options) and a cosine head scores all 222 ATT&CK
families; each head row starts as the model's reading of that family's own ATT&CK texts, so
`step_0` is a nearest-description classifier (F0). Trained on public train + clean AnnoCTR train
with family targets, selected on public dev + AnnoCTR dev, confirmed once on CTI-HAL (local), public
test and AnnoCTR test against pipeline rank-1, B1/B3 (retrieve-then-pick) and Jev C1.

```bash
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/export_cases.py --catalog-only   # catalog.json
cd scripts/cribl_finetune
../../.venv-cribl/bin/python family_head.py train --arm F1 --init cribl                # ~3.5 h, RTX 4060
../../.venv-cribl/bin/python family_head.py evaluate --checkpoint <selected> --name F1 --split cti_hal \
  --compare B1=B1_cti_hal.jsonl B3=B3_cti_hal.jsonl
```

### F2: family head with ATT&CK procedure examples (prereg `cribl_family_attack_2026-10-13.json`)

```bash
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/export_attack_procedures.py  # cases_attack_train.jsonl
bash scripts/cribl_finetune/runpod/remote_f2.sh                                   # on a GPU pod
```

`export_attack_procedures.py` turns ATT&CK's `uses` relationships into sentences and drops any that
overlap a dev or test text (near duplicates, containment, a shared 10-token run); CTI-HAL is read
for the filter on the laptop and never written out.

### F3: ATT&CK definitions + rival margin (prereg `cribl_family_margin_2026-10-13.json`)

```bash
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/export_attack_procedures.py --descriptions  # 693 rows
bash scripts/cribl_finetune/runpod/remote_f3.sh                                                  # on a GPU pod
```

F2's recipe plus one training row per technique definition and `--margin 1 1`
(`losses.rival_margin`: the gold families must beat their strongest rival by 1 nat).

Result (docs 21.3.18): no gain over F2 on CTI-HAL, the public test or AnnoCTR test, and the look-alike
pair mix-ups are unchanged; F2 stays the best family classifier.

### E1: exact techniques under the family head (prereg `cribl_technique_head_2026-10-14.json`)

```bash
bash scripts/cribl_finetune/runpod/remote_e1.sh   # on a GPU pod (F2's data + --technique-head)
cd scripts/cribl_finetune && ../../../.venv-cribl/bin/python family_head.py evaluate \
  --checkpoint <E1 step dir> --name E1 --split cti_hal --compare F2=F2_cti_hal.jsonl B3=B3_cti_hal.jsonl
```

P(technique) = P(family) x P(technique | family) (`hierarchy.py`), answer from the three most
probable families. Result (docs 21.3.18): CTI-HAL exact top-1 0.504, above B3 (0.441) and Jev C1
(0.460); public test 0.560, level with B1.

### E2: advisories, Atomic Red Team and cleaned rows (prereg `cribl_family_data_2026-10-14.json`)

```bash
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/prepare_advisories.py download  # IC3 joint advisories, ~200 MB
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/prepare_advisories.py build     # 765 rows
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/prepare_atomic.py               # 1,874 rows from data/benchmarks/atomic/raw/index.yaml
python3 scripts/cribl_finetune/clean_cases.py cases_train.jsonl cases_annoctr_train.jsonl cases_attack_train.jsonl
bash scripts/cribl_finetune/runpod/remote_e2.sh                                      # on a GPU pod
```

E1's recipe on the cleaned rows plus the two new sources. Result (docs 21.3.18): about +0.01 on
CTI-HAL family and exact, within noise; the headline fails and E1 stays the reference.

### M1: a yes/no probability per family, up to three answers (prereg `cribl_yesno_2026-10-14.json`)

```bash
bash scripts/cribl_finetune/runpod/remote_m1.sh   # on a GPU pod: E2's data + --family-loss yesno
python3 scripts/cribl_finetune/multi_answer.py --budget 1.3 E2=E2_cti_hal.jsonl M1=M1_cti_hal.jsonl --baseline E2
python3 scripts/cribl_finetune/blind_check.py sheet --split cti_hal E2=E2_cti_hal.jsonl M1=M1_cti_hal.jsonl
python3 scripts/cribl_finetune/blind_check.py score --split cti_hal   # after filling "supported"
```

Every family gets its own sigmoid (`losses.yes_no`, a missing label weighs 0.1); answers are the top
family plus up to two more above a threshold set to 1.3 answers per sentence. Result (docs 21.3.18):
by the labels about 2 points below E2 everywhere (headline fails); in a blind check of extra answers
64% of M1's were supported against 46% of E2's.

### Paragraphs read whole or sentence by sentence (prereg `cribl_sentence_split_2026-10-14.json`)

```bash
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/sentence_split_eval.py split          # expert + derived test paragraphs
cd scripts/cribl_finetune && ../../../.venv-cribl/bin/python sentence_split_eval.py predict --checkpoint <E2 step dir> --name E2
python3 sentence_split_eval.py score E2 M1
```

No training. Result (docs 21.3.18): at the same number of answers per paragraph, splitting finds no
more labelled techniques (expert −0.04 at three answers, n.s.; derived procedures 2–5 points lower).

### M2: multi-label rows weighted 4x (prereg `cribl_multilabel_2026-10-14.json`)

```bash
bash scripts/cribl_finetune/runpod/remote_m2.sh   # on a GPU pod: M1 + --multi-label-weight 4
python3 scripts/cribl_finetune/multi_answer.py --dataset expert --budget 2.0 E2=E2_test.jsonl M1=M1_test.jsonl M2=M2_test.jsonl --baseline E2
```

Result (docs 21.3.18): on the expert paragraphs M2 finds as many labelled families as E2 (+0.000 at
two answers), slightly more than M1, and loses 3 points of exact top-1 on the public test.

### E3: CTI-Rationale paragraphs added (prereg `cribl_cti_rationale_2026-10-15.json`)

```bash
.venv/bin/hf download EbruResul/CTI-Rationale cti_rationale_gold_v2.jsonl report_split.csv LICENSE README.md \
  --repo-type dataset --local-dir data/benchmarks/cti_rationale/raw   # gated: accept the terms, hf auth login
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/prepare_cti_rationale.py build   # 1,011 train + 209 test paragraphs
bash scripts/cribl_finetune/runpod/remote_e3.sh   # on a GPU pod: E2's recipe + cases_cti_rationale_train.jsonl
python3 scripts/cribl_finetune/multi_answer.py --dataset expert --budget 2.0 E2=E2_test.jsonl E3=E3_test.jsonl --baseline E2
python3 scripts/cribl_finetune/multi_answer.py --budget 1.3 E2=E2_cti_rationale_test.jsonl E3=E3_cti_rationale_test.jsonl --baseline E2
```

Result (docs 21.3.18): on the expert paragraphs E3 finds as many labelled families as E2 (−0.013 at
two answers, n.s.); on the CTI-Rationale test reports (same publishers) family top-1 rises 4.3 points;
CTI-HAL, public and AnnoCTR test unchanged. E2 stays the reference.

### Parent-level scoring, recipe sweep, Gemma 4 E4B (prereg `cribl_recipe_base_2026-10-15.json`)

```bash
python3 scripts/cribl_finetune/relabel_parents.py                       # cases_<split>_parents.jsonl (sub-techniques -> parents)
python3 scripts/cribl_finetune/parent_rescore.py E1=E1_cti_hal.jsonl E2=E2_cti_hal.jsonl --baseline E2   # parent level / family / exact, paired CIs
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/prepare_cti_rationale.py build     # + cases_cti_rationale_validation.jsonl (selection set)
bash scripts/cribl_finetune/runpod/remote_sweep.sh R3 "--epochs 3 --init fresh --lora-r 64"              # on a GPU pod
bash scripts/cribl_finetune/runpod/remote_sweep.sh G1 "--epochs 2 --init fresh --base gemma-4-e4b"       # Gemma 4 E4B (16 GB download on the pod)
cd scripts/cribl_finetune && ../../../.venv-cribl/bin/python family_head.py evaluate --checkpoint <dir> --name G1 --split cti_hal --compare E2=E2_cti_hal.jsonl
```

Parent level = the answered technique's parent is a gold id's parent (E2 on CTI-HAL: family 0.642,
parent level 0.639, exact 0.514). `--selection parent` picks checkpoints on the mean parent level over
public dev, AnnoCTR dev and CTI-Rationale validation; `--base` names a base of `decision_model.BASES`
(Cribl's adapter needs the default Qwen base; other bases start a fresh LoRA). On a GPU under 16 GB
Gemma's 5.2 GB per-layer embedding table stays in system RAM (`BaseSpec.cpu_modules`).

Result (docs 21.3.18): no recipe or base beats E2 on real reports. The best-on-dev recipe (LoRA lr
2e-4) is 3.4 points worse on CTI-HAL at parent level; 3 epochs and rank 64 are level; Gemma 4 E4B
ties E2 (−0.3) from a fresh LoRA. E2 stays the reference.
Amendment 1, `--base foundation-sec-8b` (Cisco's security-pretrained Llama 3.1 8B, Apache 2.0): −1.2
on CTI-HAL (n.s.), +1 to +5 on the public, AnnoCTR and CTI-Rationale tests (n.s.). On a small GPU its
token table and output head stay in RAM; scoring takes 7.0 GB at batch 4.

### Group head: the threat group from extracted technique lists (prereg `cribl_group_head_2026-10-15.json`)

```bash
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/export_group_cases.py build   # group_catalog.json + cases_group_*.jsonl (vendor dev needs E2_vendor_statements.jsonl)
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/group_baseline.py            # MLP_* and JACCARD_* predictions on the same inputs
TRAIN_ARGS="--task group --catalog ../../data/audits/cribl_ft/group_catalog.json --train-files cases_group_train.jsonl --dev-files cases_group_campaigns.jsonl cases_group_vendor.jsonl" \
  SCORE_SPLITS="" COMPARE_WITH="" bash scripts/cribl_finetune/runpod/remote_sweep.sh A1 "--epochs 1 --init fresh --checkpoints-per-epoch 4"
cd scripts/cribl_finetune && ../../../.venv-cribl/bin/python family_head.py evaluate --checkpoint <dir> --name A1 --split group_cti_hal_extracted --compare MLP=MLP_group_cti_hal_extracted.jsonl
```

Result (docs 21.3.18): both heads (Qwen3.5-4B, Foundation-Sec-8B) name the group first on 2–4% of
CTI-HAL reports from the extracted techniques, against 12.7% for the product's group network on the
same input; technique lists are not fingerprints. The attribution stage is unchanged.

### Refitting the group network for extracted inputs (prereg `cribl_group_mlp_refit_2026-10-16.json`)

```bash
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/refit_group_mlp.py measure        # E2's extraction noise on CTI-Rationale train+validation -> group_mlp_noise.json
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/refit_group_mlp.py fit --save     # 8 fits, selection on the campaigns, test vs the current artifact -> group_mlp_refit.json
```

Result (docs 21.3.18): the vendor reports as real cases lift the campaigns dev MRR 0.41 → 0.49 but
not CTI-HAL (top-1 0.127 both; top-5 0.27 → 0.35, n.s.); the measured draws change nothing. The
chosen model is `data/models/mlp/group_mlp_extracted.npz`; the product's artifact is untouched.

### Actor head: the group from anonymised report sentences (prereg `cribl_actor_head_2026-10-16.json`)

```bash
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/export_actor_cases.py build   # ATT&CK per-group texts (anonymised) + statement sets
TRAIN_ARGS="--task actor --catalog ../../data/audits/cribl_ft/group_catalog.json --train-files cases_actor_train.jsonl --dev-files cases_actor_vendor_dev.jsonl" \
  SCORE_SPLITS="" COMPARE_WITH="" bash scripts/cribl_finetune/runpod/remote_sweep.sh T1 "--epochs 2 --init fresh --checkpoints-per-epoch 2"
cd scripts/cribl_finetune && ../../../.venv-cribl/bin/python family_head.py evaluate --checkpoint <dir> --name T1 --split actor_cti_hal --top-kept 160
python3 scripts/cribl_finetune/group_text_eval.py aggregate --name T1 --split actor_cti_hal      # per-report mean probability
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/group_text_eval.py tfidf --split actor_cti_hal   # the product's text-group model on the same statements
```

Result (docs 21.3.18): report-level top-1 on CTI-HAL 0.255 against 0.491 for the TF-IDF text-group
model (−0.24, significant) and 0.127 for the group network on extracted techniques; Elastic 3 of 8
vs 1 of 8. Text beats technique lists; the bag of words still beats the LLM head.

### CTIBench and AthenaBench (prereg `cribl_external_bench_2026-10-19.json`, evaluation only)

```bash
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/external_bench.py prepare    # raw files under data/benchmarks/{cti_bench,athenabench}/raw (non-commercial licences)
cd scripts/cribl_finetune && ../../../.venv-cribl/bin/python external_bench.py predict --technique <E2 dir> --actor <T1 dir>
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/external_bench.py score      # -> data/audits/cribl_ft/external_bench.json
```

Result (docs 21.3.18): AthenaBench ATE accuracy 0.736 (published: Gemini-2.5-Pro 76.2, GPT-5 76.0,
Qwen3-4B 5.6); CTI-ATE F1 0.58 on all 60 texts (ChatGPT-4 0.64), 0.75 on the 47 Enterprise ones;
TAA correct 0.38 (CTIBench, both group models) and 0.16-0.23 (AthenaBench).

### Product integration: the technique head as a service (prereg `cribl_product_integration_2026-10-19.json`)

```bash
make technique-head-serve                       # .venv-cribl, GPU: serves E2 on http://localhost:8102 (serve.py + head_service.py)
TECHNIQUE_HEAD_ENABLED=true PYTHONPATH=. .venv/bin/python scripts/fit_acceptance_model.py --priors-split none \
  --require-technique-head --output data/models/acceptance/acceptance_model_technique_head.json   # schema-4 model, public dev only
TECHNIQUE_HEAD_ENABLED=true ACCEPTANCE_MODEL_PATH=data/models/acceptance/acceptance_model_technique_head.json \
  PYTHONPATH=. .venv/bin/python scripts/benchmark_mapping.py --split test --require-technique-head --output <file>
PYTHONPATH=. .venv/bin/python scripts/cribl_finetune/integration_report.py    # arms, paired bootstraps, checks -> integration_report.json
```

The product side is `apt_classifier/technique_head.py` (client, fail closed), the hooks in
`apt_classifier/retrieve.py` (head evidence on every candidate, up to three head proposals appended
after the pool) and acceptance feature schema 4 (docs 8.2). Result: docs 21.3.18, "Product
integration".
