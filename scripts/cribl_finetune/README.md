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
