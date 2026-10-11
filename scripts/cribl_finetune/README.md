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

- **`.venv-cribl`** (`make cribl-venv`, about 5 GB): isolated, transformers 5 + peft + bitsandbytes +
  flash-linear-attention. Qwen3.5 does not load in the project `.venv`, whose transformers 4.57 is
  kept for ATTACK-BERT.
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
