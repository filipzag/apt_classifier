#!/usr/bin/env bash
# On a RunPod GPU pod: finish B2 (resume at its last recorded checkpoint) and score B0/B1/B2 on the
# public test split; CTI-HAL only with CRIBL_INCLUDE_CTI_HAL=1 and its case file present (it is not
# in the default upload: no licence, local evaluation only). Safe to rerun after an interruption:
# training resumes from the last checkpoint with a dev score, scoring resumes by case.
# Usage, from the unpacked upload: bash scripts/cribl_finetune/runpod/remote_run.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
A=$ROOT/data/audits/cribl_ft
M=$ROOT/data/models/cribl_ft
VENV=${CRIBL_VENV:-/workspace/venv-cribl}
export CRIBL_HF_CACHE=${CRIBL_HF_CACHE:-/workspace/hf-cache}
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
FMT=chat
log() { echo "[$(date -Is)] $*" | tee -a "$A/remote.log"; }

python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "Python >= 3.11 needed"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$A/remote.log"

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
"$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$A/remote.log"
"$PY" -m pip freeze > "$A/remote_pip_freeze.txt"

cd "$ROOT/scripts/cribl_finetune"

# B2: done when the log has the final checkpoint; else resume from the last checkpoint that has a
# dev score, dropping anything written after it (an interrupted checkpoint or dev scoring).
plan=$("$PY" - "$A/B2_train_log.json" "$M/B2" "$A" <<'EOF'
import json, shutil, sys
from pathlib import Path
log, models, audits = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3])
seen = [c["examples_seen"] for c in log["checkpoints"]]
if seen and seen[-1] >= log["examples_total"]:
    print("done")
    sys.exit()
last = max(seen)
for path in [*models.glob("step_*"), *audits.glob("B2_step*_dev.jsonl")]:
    step = int(path.name.split("step", 1)[1].lstrip("_").split("_")[0])
    if step > last:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
print(f"step_{last}")
EOF
)
if [ "$plan" != done ]; then
  log "B2 resume from $plan"
  "$PY" train.py --arm B2 --init fresh --format $FMT --resume-from "$M/B2/$plan" 2>&1 \
    | tee -a "$A/B2_train_remote.out" \
    | { grep --line-buffered -E '^step|^checkpoint|^selected|Error|out of memory' || true; }
fi

selected() { "$PY" -c "import json,sys,pathlib; print(pathlib.Path(json.load(open(sys.argv[1]))['selected']).name)" "$1"; }
B1_SEL=$M/B1/$(selected "$A/B1_train_log.json")
B2_SEL=$M/B2/$(selected "$A/B2_train_log.json")
[ -f "$B1_SEL/adapter_model.safetensors" ] && [ -f "$B2_SEL/adapter_model.safetensors" ] \
  || { log "missing selected checkpoint: $B1_SEL $B2_SEL"; exit 1; }
log "selected on dev: B1=$(basename "$B1_SEL") B2=$(basename "$B2_SEL")"

splits=test
if [ "${CRIBL_INCLUDE_CTI_HAL:-0}" = 1 ] && [ -f "$A/cases_cti_hal.jsonl" ]; then splits="test cti_hal"; fi
for split in $splits; do
  for arm in B0 B1 B2; do
    adapter=cribl; [ "$arm" = B1 ] && adapter=$B1_SEL; [ "$arm" = B2 ] && adapter=$B2_SEL
    log "score $arm on $split"
    "$PY" evaluate.py --split "$split" --adapter "$adapter" --name $arm --format $FMT \
      > "$A/${arm}_${split}_remote.out" 2>&1
    log "$(tail -1 "$A/${arm}_${split}_remote.out")"
  done
done

cd "$ROOT"
results=/workspace/cribl_results.tar.gz
find data/audits/cribl_ft -maxdepth 1 -type f \( -name 'B2_train_log.json' -o -name 'B2_train_remote.out' \
  -o -name 'B2_step*_dev.jsonl' -o -name 'B?_test*' -o -name 'B?_cti_hal*' -o -name 'remote*' \) -print0 \
  | tar czf "$results" --null -T - data/models/cribl_ft/B2
log "results: $results ($(du -h "$results" | cut -f1)); download it, then stop the pod"
