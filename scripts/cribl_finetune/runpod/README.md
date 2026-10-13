# Running the rest of the cribl experiment on RunPod

The pod finishes **B2** (its second epoch, resumed from the 1-epoch checkpoint) and scores **B0, B1 and B2**
on the public test split. It uses the same code, the same 4-bit base, and the same seeds and data order
as the laptop. CTI-HAL is scored back on the laptop because it has no licence and stays local. Recorded
as pre-registration amendment 3.

## 1. Build the upload (on the laptop)

```bash
scripts/cribl_finetune/runpod/pack.sh
```

This writes `data/audits/cribl_ft/runpod/cribl_runpod.tar.gz`, about 110 MB. It holds the code, the public
case files, the B1/B2 logs and the B1 and B2 checkpoints.

## 2. Start a pod

- **GPU**: one RTX 4090 (24 GB) is plenty. The run needs about 5 GB of GPU memory. An A100 or H100
  is faster but costs more per hour.
- **Template**: a RunPod PyTorch template with Python 3.11 or newer.
- **Container disk**: at least 50 GB. The Python environment is about 5 GB and the model download 9.3 GB.
- **Connect**: open JupyterLab from the pod's Connect menu.

## 3. Run (in a JupyterLab terminal)

Upload `cribl_runpod.tar.gz` to `/workspace` (drag it into the file browser), then:

```bash
mkdir -p /workspace/cribl && tar xzf /workspace/cribl_runpod.tar.gz --no-same-owner -C /workspace/cribl
nohup bash /workspace/cribl/scripts/cribl_finetune/runpod/remote_run.sh > /workspace/run.log 2>&1 &
tail -f /workspace/run.log
```

`nohup` keeps the run going if the browser tab closes. If the pod restarts, run the same `nohup` line
again. It resumes B2 from its last scored checkpoint and resumes scoring where it stopped.

The log moves through these stages:
1. Set-up, about 10 min: Python packages and the 9.3 GB model download.
2. B2 training, with `step …` lines and two `checkpoint …` dev scores.
3. Six `score …` lines.
4. `results: /workspace/cribl_results.tar.gz`.

## 4. Bring the results back

- Download `/workspace/cribl_results.tar.gz` from JupyterLab: right-click it, then Download.
- **Stop or terminate the pod** so billing stops.
- On the laptop, run:

```bash
python3 scripts/cribl_finetune/runpod/merge_results.py ~/Downloads/cribl_results.tar.gz
```

The CTI-HAL confirmation then runs locally, once for each arm:
- `B0`: `CRIBL_ADAPTER=cribl`.
- `B1`: `CRIBL_ADAPTER=` the B1 checkpoint selected on dev.
- `B2`: `CRIBL_ADAPTER=` the B2 checkpoint selected on dev.

```bash
make cribl-eval CRIBL_VENV=/home/fzagar/apt_classifier/.venv-cribl CRIBL_FORMAT=chat CRIBL_SPLIT=cti_hal CRIBL_ARM=B0 CRIBL_ADAPTER=cribl
```

## Confirmation runs on a pod (amendment 5)

`remote_confirm.sh` scores the list in `CRIBL_RUNS` (`split:name:adapter`; by default the CTI-HAL
confirmations of amendments 4 and 5 and B2 step_12616 on test) and writes
`/workspace/cribl_confirm.tar.gz`. Its upload holds the code, `cases_cti_hal.jsonl`,
`cases_test.jsonl` and the B1 step_9462, B2 step_6308 and B2 step_12616 checkpoints. CTI-HAL has no
licence: the owner allowed this one upload; delete it and terminate the pod right after downloading.

## F1 family head (prereg `cribl_family_head_2026-10-13.json`, amendment 1)

`remote_f1.sh` trains F1, selects it on public dev + AnnoCTR dev, scores B1/B3 (retrieve-then-pick) and
F1 once on clean AnnoCTR test, F1 on the public test, and writes `/workspace/cribl_f1.tar.gz` with the
selected checkpoint. The upload holds the code, `catalog.json`, the public and AnnoCTR case files, the
B1/B3 public test predictions and the B1 step_9462 / B3 step_7088 adapters; no CTI-HAL. CTI-HAL is
scored on the laptop afterwards (`family_head.py evaluate --split cti_hal`).
