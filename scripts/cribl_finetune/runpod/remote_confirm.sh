#!/usr/bin/env bash
# On a RunPod GPU pod: the confirmation runs of prereg amendment 5, each "split:name:adapter" with
# the adapter "cribl" or a checkpoint path relative to the unpacked upload. Scoring resumes by case
# when rerun. Writes /workspace/cribl_confirm.tar.gz with the predictions, metrics and logs.
# The upload holds the code, cases_cti_hal/test.jsonl and the three checkpoints (README.md here).
# Usage, from the unpacked upload: bash scripts/cribl_finetune/runpod/remote_confirm.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
A=$ROOT/data/audits/cribl_ft
VENV=${CRIBL_VENV:-/workspace/venv-cribl}
export CRIBL_HF_CACHE=${CRIBL_HF_CACHE:-/workspace/hf-cache}
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
RUNS=${CRIBL_RUNS:-"cti_hal:B1:data/models/cribl_ft/B1/step_9462 cti_hal:B2s12616:data/models/cribl_ft/B2/step_12616 cti_hal:B0:cribl cti_hal:B2:data/models/cribl_ft/B2/step_6308 test:B2s12616:data/models/cribl_ft/B2/step_12616"}
log() { echo "[$(date -Is)] $*" | tee -a "$A/remote_confirm.log"; }

python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "Python >= 3.11 needed"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$A/remote_confirm.log"
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
"$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$A/remote_confirm.log"
"$PY" -m pip freeze > "$A/remote_confirm_pip_freeze.txt"

cd "$ROOT/scripts/cribl_finetune"
for run in $RUNS; do
  IFS=: read -r split name adapter <<< "$run"
  [ "$adapter" = cribl ] || adapter=$ROOT/$adapter
  log "score $name on $split"
  "$PY" evaluate.py --split "$split" --adapter "$adapter" --name "$name" --format chat \
    > "$A/${name}_${split}_remote.out" 2>&1
  log "$name $split done"
done

cd "$ROOT"
results=/workspace/cribl_confirm.tar.gz
find data/audits/cribl_ft -maxdepth 1 -type f \( -name 'B*_cti_hal*' -o -name 'B2s12616_test*' \
  -o -name 'remote_confirm*' \) -print0 | tar czf "$results" --null -T -
log "results: $results ($(du -h "$results" | cut -f1)); download it, then delete the upload and terminate the pod"
