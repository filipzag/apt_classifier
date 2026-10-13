#!/usr/bin/env bash
# On a RunPod GPU pod: train F1 (the family head without retrieval; prereg
# docs/preregistration/cribl_family_head_2026-10-13.json, amendment 1), select its checkpoint on
# public dev + AnnoCTR dev, score it once on the public test and clean AnnoCTR test, and score the
# retrieve-then-pick references B1/B3 once on clean AnnoCTR test. CTI-HAL is not uploaded: it is
# scored on the laptop with the downloaded checkpoint. Writes /workspace/cribl_f1.tar.gz.
# Usage, from the unpacked upload: bash scripts/cribl_finetune/runpod/remote_f1.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
A=$ROOT/data/audits/cribl_ft
M=$ROOT/data/models/cribl_ft
VENV=${CRIBL_VENV:-/workspace/venv-cribl}
export CRIBL_HF_CACHE=${CRIBL_HF_CACHE:-/workspace/hf-cache}
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
log() { echo "[$(date -Is)] $*" | tee -a "$A/remote_f1.log"; }

python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "Python >= 3.11 needed"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$A/remote_f1.log"
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
# Hopper: flash-linear-attention refuses Triton 3.4-3.7.0 (fla #640); see remote_b3.sh
if nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -q '^9\.'; then
  "$VENV/bin/pip" install -q "triton==3.7.1"
fi
"$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$A/remote_f1.log"
"$PY" -m pip freeze > "$A/remote_f1_pip_freeze.txt"

cd "$ROOT/scripts/cribl_finetune"
if [ ! -f "$A/F1_train_log.json" ] || ! "$PY" -c "import json,sys; sys.exit('finished_at' not in json.load(open(sys.argv[1])))" "$A/F1_train_log.json"; then
  [ -f "$A/F1_train_log.json" ] && { log "F1 log exists but is unfinished: restart from scratch"; rm -rf "$A"/F1_train_log.json "$A"/F1_step*_*.jsonl "$M/F1"; }
  log "train F1"
  "$PY" family_head.py train --arm F1 --init cribl 2>&1 | tee -a "$A/F1_train_remote.out" \
    | { grep --line-buffered -E '^step|^checkpoint|^selected|Error|out of memory' || true; }
fi
F1_SEL=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['selected'])" "$A/F1_train_log.json")
log "selected on dev: $(basename "$F1_SEL")"

for run in "annoctr_test:B1clean:$M/B1/step_9462" "annoctr_test:B3clean:$M/B3/step_7088"; do
  IFS=: read -r split name adapter <<< "$run"
  log "score $name on $split (C1, retrieve-then-pick)"
  "$PY" evaluate.py --split "$split" --adapter "$adapter" --name "$name" --format chat \
    > "$A/${name}_${split}_remote.out" 2>&1
done
log "score F1 on test"
"$PY" family_head.py evaluate --checkpoint "$F1_SEL" --name F1 --split test \
  --compare B1=B1_test.jsonl B3=B3_test.jsonl 2>&1 | tee "$A/F1_test_remote.out" | tail -1
log "score F1 on annoctr_test"
"$PY" family_head.py evaluate --checkpoint "$F1_SEL" --name F1 --split annoctr_test \
  --compare B1=B1clean_annoctr_test.jsonl B3=B3clean_annoctr_test.jsonl 2>&1 \
  | tee "$A/F1_annoctr_test_remote.out" | tail -1

cd "$ROOT"
results=/workspace/cribl_f1.tar.gz
sel_rel=${F1_SEL#"$ROOT"/}
find data/audits/cribl_ft -maxdepth 1 -type f \( -name 'F1_*' -o -name 'B?clean_annoctr_test*' -o -name 'remote_f1*' \) -print0 \
  | tar czf "$results" --null -T - "$sel_rel"
log "results: $results ($(du -h "$results" | cut -f1)); download it, then delete /workspace and terminate the pod"
