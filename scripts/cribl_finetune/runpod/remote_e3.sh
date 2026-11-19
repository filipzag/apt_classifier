#!/usr/bin/env bash
# On a RunPod GPU pod: train E3, E2's recipe with the CTI-Rationale train and validation report
# paragraphs added (prereg docs/preregistration/cribl_cti_rationale_2026-10-15.json), select it as E2
# and score it once on the public test, clean AnnoCTR test and the CTI-Rationale test reports
# (multi-answer scoring offline: multi_answer.py). CTI-HAL is not uploaded: it is scored on the
# laptop. Writes /workspace/cribl_e3.tar.gz.
# Usage, from the unpacked upload: bash scripts/cribl_finetune/runpod/remote_e3.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
A=$ROOT/data/audits/cribl_ft
M=$ROOT/data/models/cribl_ft
VENV=${CRIBL_VENV:-/workspace/venv-cribl}
export CRIBL_HF_CACHE=${CRIBL_HF_CACHE:-/workspace/hf-cache}
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
log() { echo "[$(date -Is)] $*" | tee -a "$A/remote_e3.log"; }

python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "Python >= 3.11 needed"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$A/remote_e3.log"
if [ ! -x "$VENV/bin/python" ]; then
  log "creating $VENV"
  python3 -m venv "$VENV"
  driver_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1)
  req=$(mktemp)
  if [ "${driver_cuda:-0}" -ge 13 ]; then
    cp "$ROOT/scripts/cribl_finetune/requirements.txt" "$req"
  else
    sed 's/cu130/cu128/g' "$ROOT/scripts/cribl_finetune/requirements.txt" > "$req"
  fi
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r "$req"
fi
PY=$VENV/bin/python
if nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -q '^9\.'; then
  "$VENV/bin/pip" install -q "triton==3.7.1"  # Hopper, see remote_b3.sh
fi
"$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$A/remote_e3.log"
"$PY" -m pip freeze > "$A/remote_e3_pip_freeze.txt"

cd "$ROOT/scripts/cribl_finetune"
if [ ! -f "$A/E3_train_log.json" ] || ! "$PY" -c "import json,sys; sys.exit('finished_at' not in json.load(open(sys.argv[1])))" "$A/E3_train_log.json"; then
  [ -f "$A/E3_train_log.json" ] && { log "E3 log exists but is unfinished: restart from scratch"; rm -rf "$A"/E3_train_log.json "$A"/E3_step*_*.jsonl "$M/E3"; }
  log "train E3"
  "$PY" family_head.py train --arm E3 --init cribl --epochs 2 --checkpoints-per-epoch 4 \
    --train-files cases_train_clean.jsonl cases_annoctr_train_clean.jsonl cases_attack_train_clean.jsonl \
      cases_advisories_train.jsonl cases_atomic_train.jsonl cases_cti_rationale_train.jsonl \
    --technique-head --families-considered 3 --family-only-datasets annoctr --exact-selection dev \
    2>&1 | tee -a "$A/E3_train_remote.out" \
    | { grep --line-buffered -E '^step|^checkpoint|^selected|Error|out of memory' || true; }
fi
E3_SEL=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['selected'])" "$A/E3_train_log.json")
log "selected on dev: $(basename "$E3_SEL")"
log "score E3 on test"
"$PY" family_head.py evaluate --checkpoint "$E3_SEL" --name E3 --split test \
  --compare E2=E2_test.jsonl 2>&1 | tee "$A/E3_test_remote.out" | tail -2
log "score E3 on annoctr_test"
"$PY" family_head.py evaluate --checkpoint "$E3_SEL" --name E3 --split annoctr_test \
  --compare E2=E2_annoctr_test.jsonl 2>&1 | tee "$A/E3_annoctr_test_remote.out" | tail -2
log "score E3 on cti_rationale_test"
"$PY" family_head.py evaluate --checkpoint "$E3_SEL" --name E3 --split cti_rationale_test 2>&1 \
  | tee "$A/E3_cti_rationale_test_remote.out" | tail -2

cd "$ROOT"
results=/workspace/cribl_e3.tar.gz
sel_rel=${E3_SEL#"$ROOT"/}
find data/audits/cribl_ft -maxdepth 1 -type f \( -name 'E3_*' -o -name 'remote_e3*' \) -print0 \
  | tar czf "$results" --null -T - "$sel_rel"
log "results: $results ($(du -h "$results" | cut -f1)); download it, then delete /workspace and terminate the pod"
