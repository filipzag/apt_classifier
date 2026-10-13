#!/usr/bin/env bash
# On a RunPod GPU pod: train B3 for 4 epochs (B1 step_9462 trained further on AnnoCTR report sentences plus an
# equal replay sample of the public train set; prereg docs/preregistration/cribl_annoctr_2026-10-13.json),
# select its checkpoint on AnnoCTR dev, then score B1 on AnnoCTR dev/test and the selected B3 on
# CTI-HAL, AnnoCTR test and the public test. Writes /workspace/cribl_b3.tar.gz.
# Usage, from the unpacked upload: bash scripts/cribl_finetune/runpod/remote_b3.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
A=$ROOT/data/audits/cribl_ft
M=$ROOT/data/models/cribl_ft
VENV=${CRIBL_VENV:-/workspace/venv-cribl}
export CRIBL_HF_CACHE=${CRIBL_HF_CACHE:-/workspace/hf-cache}
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
log() { echo "[$(date -Is)] $*" | tee -a "$A/remote_b3.log"; }

python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "Python >= 3.11 needed"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$A/remote_b3.log"
if [ ! -x "$VENV/bin/python" ]; then
  log "creating $VENV"
  python3 -m venv "$VENV"
  driver_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
  req=$(mktemp)
  if [ "${driver_cuda:-0}" -ge 13 ]; then
    cp "$ROOT/scripts/cribl_finetune/requirements.txt" "$req"
  else  # older driver: the same torch 2.11.0 built for CUDA 12.8
    sed 's/cu130/cu128/g' "$ROOT/scripts/cribl_finetune/requirements.txt" > "$req"
  fi
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$req"
fi
PY=$VENV/bin/python
# Hopper (H100/H200): flash-linear-attention refuses Triton 3.4-3.7.0 there (wrong gated backward,
# fla #640); torch 2.11 pins 3.6.0, so take 3.7.1 (torch only needs Triton for torch.compile)
if nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -q '^9\.'; then
  "$VENV/bin/pip" install -q "triton==3.7.1"
fi
"$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$A/remote_b3.log"
"$PY" -m pip freeze > "$A/remote_b3_pip_freeze.txt"

cd "$ROOT/scripts/cribl_finetune"
if [ ! -f "$A/B3_train_log.json" ] || ! "$PY" -c "import json,sys; sys.exit('finished_at' not in json.load(open(sys.argv[1])))" "$A/B3_train_log.json"; then
  [ -f "$A/B3_train_log.json" ] && { log "B3 log exists but is unfinished: restart from scratch"; rm -rf "$A"/B3_train_log.json "$A"/B3_step*_dev.jsonl "$M/B3"; }
  replay=$(grep -c . "$A/cases_annoctr_train.jsonl")
  log "train B3 (AnnoCTR train + $replay public replay rows)"
  "$PY" train.py --arm B3 --init "$M/B1/step_9462" --format chat --lr 5e-5 --epochs 4 \
    --train-file cases_annoctr_train.jsonl --replay-file cases_train.jsonl --replay-count "$replay" \
    --family-target-datasets annoctr --dev-file cases_annoctr_dev.jsonl --select forced_parent_aware \
    2>&1 | tee -a "$A/B3_train_remote.out" \
    | { grep --line-buffered -E '^step|^checkpoint|^selected|Error|out of memory' || true; }
fi
B3_SEL=$M/B3/$("$PY" -c "import json,sys,pathlib; print(pathlib.Path(json.load(open(sys.argv[1]))['selected']).name)" "$A/B3_train_log.json")
log "selected on AnnoCTR dev: $(basename "$B3_SEL")"

for run in "annoctr_dev:B1:$M/B1/step_9462" "annoctr_test:B1:$M/B1/step_9462" \
           "cti_hal:B3:$B3_SEL" "annoctr_test:B3:$B3_SEL" "test:B3:$B3_SEL"; do
  IFS=: read -r split name adapter <<< "$run"
  log "score $name on $split"
  "$PY" evaluate.py --split "$split" --adapter "$adapter" --name "$name" --format chat \
    > "$A/${name}_${split}_remote.out" 2>&1
  log "$name $split done"
done

cd "$ROOT"
results=/workspace/cribl_b3.tar.gz
find data/audits/cribl_ft -maxdepth 1 -type f \( -name 'B3_*' -o -name 'B1_annoctr_*' -o -name 'remote_b3*' \) -print0 \
  | tar czf "$results" --null -T - data/models/cribl_ft/B3
log "results: $results ($(du -h "$results" | cut -f1)); download it, then delete the upload and terminate the pod"
