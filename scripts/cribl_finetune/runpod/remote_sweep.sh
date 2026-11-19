#!/usr/bin/env bash
# On a RunPod GPU pod: train one recipe or base-model run of the sweep (prereg
# docs/preregistration/cribl_recipe_base_2026-10-15.json) on E2's training rows, select the
# checkpoint on the widened dev set (public dev, AnnoCTR dev, CTI-Rationale validation; parent-level
# rule), and score it once on the public test, clean AnnoCTR test and CTI-Rationale test against E2.
# CTI-HAL is not uploaded: it is scored on the laptop. Writes /workspace/cribl_<arm>.tar.gz.
# Usage, from the unpacked upload:
#   bash scripts/cribl_finetune/runpod/remote_sweep.sh R1 "--epochs 3"
#   bash scripts/cribl_finetune/runpod/remote_sweep.sh R2 "--epochs 3 --lr 2e-4"
#   bash scripts/cribl_finetune/runpod/remote_sweep.sh R3 "--epochs 3 --init fresh --lora-r 64"
#   bash scripts/cribl_finetune/runpod/remote_sweep.sh G1 "--epochs 2 --init fresh --base gemma-4-e4b"
#   TRAIN_ARGS="--task group --catalog ../../data/audits/cribl_ft/group_catalog.json --train-files cases_group_train.jsonl --dev-files cases_group_campaigns.jsonl cases_group_vendor.jsonl" \
#     SCORE_SPLITS="" bash scripts/cribl_finetune/runpod/remote_sweep.sh A1 "--epochs 1 --init fresh --checkpoints-per-epoch 4"
set -euo pipefail
ARM=${1:?arm name}
EXTRA=${2:-}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
A=$ROOT/data/audits/cribl_ft
M=$ROOT/data/models/cribl_ft
VENV=${CRIBL_VENV:-/workspace/venv-cribl}
export CRIBL_HF_CACHE=${CRIBL_HF_CACHE:-/workspace/hf-cache}
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1
LOG=$A/remote_${ARM}.log
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || { echo "Python >= 3.11 needed"; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a "$LOG"
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
"$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, torch.cuda.get_device_name(0))" | tee -a "$LOG"
"$PY" -m pip freeze > "$A/remote_${ARM}_pip_freeze.txt"

cd "$ROOT/scripts/cribl_finetune"
if [ ! -f "$A/${ARM}_train_log.json" ] || ! "$PY" -c "import json,sys; sys.exit('finished_at' not in json.load(open(sys.argv[1])))" "$A/${ARM}_train_log.json"; then
  [ -f "$A/${ARM}_train_log.json" ] && { log "$ARM log exists but is unfinished: restart from scratch"; rm -rf "$A/${ARM}_train_log.json" "$A/${ARM}_step"*_*.jsonl "$M/$ARM"; }
  log "train $ARM: $EXTRA"
  # TRAIN_ARGS replaces the technique recipe's data and heads (the group head sets it to its own
  # catalogue and case files, with no technique head); EXTRA comes last and wins on repeats
  TRAIN_ARGS=${TRAIN_ARGS:-"--train-files cases_train_clean.jsonl cases_annoctr_train_clean.jsonl cases_attack_train_clean.jsonl cases_advisories_train.jsonl cases_atomic_train.jsonl --dev-files cases_dev.jsonl cases_annoctr_dev.jsonl cases_cti_rationale_validation.jsonl --technique-head --families-considered 3 --family-only-datasets annoctr"}
  # shellcheck disable=SC2086
  "$PY" family_head.py train --arm "$ARM" --init cribl --checkpoints-per-epoch 2 \
    $TRAIN_ARGS --selection parent \
    $EXTRA \
    2>&1 | tee -a "$A/${ARM}_train_remote.out" \
    | { grep --line-buffered -E '^step|^checkpoint|^selected|Error|out of memory' || true; }
fi
SEL=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['selected'])" "$A/${ARM}_train_log.json")
log "selected on dev: $(basename "$SEL")"
# SCORE_SPLITS: the confirmation sets scored on the pod (empty: none, e.g. the group head, whose
# test sets stay on the laptop); COMPARE_WITH: the run compared against (E2) or empty
SCORE_SPLITS=${SCORE_SPLITS-"test annoctr_test cti_rationale_test"}
COMPARE_WITH=${COMPARE_WITH-E2}
for split in $SCORE_SPLITS; do
  log "score $ARM on $split"
  # shellcheck disable=SC2086
  "$PY" family_head.py evaluate --checkpoint "$SEL" --name "$ARM" --split "$split" \
    ${COMPARE_WITH:+--compare "$COMPARE_WITH=${COMPARE_WITH}_${split}.jsonl"} 2>&1 | tee "$A/${ARM}_${split}_remote.out" | tail -3
done

cd "$ROOT"
results=/workspace/cribl_${ARM}.tar.gz
sel_rel=${SEL#"$ROOT"/}
find data/audits/cribl_ft -maxdepth 1 -type f \( -name "${ARM}_*" -o -name "remote_${ARM}*" \) -print0 \
  | tar czf "$results" --null -T - "$sel_rel"
log "results: $results ($(du -h "$results" | cut -f1)); download it, then delete /workspace and terminate the pod"
