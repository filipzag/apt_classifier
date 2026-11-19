#!/usr/bin/env bash
# Build the RunPod upload (README.md here): the experiment code, the public case files (train, dev,
# test; CC-BY-4.0), B1's and B2's logs and the checkpoints the pod needs. CTI-HAL is left out (no
# licence, local evaluation only) unless --with-cti-hal is given.
# Usage: scripts/cribl_finetune/runpod/pack.sh [--with-cti-hal] [output.tar.gz]
set -euo pipefail
WT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
A=data/audits/cribl_ft
M=data/models/cribl_ft
with_cti_hal=0
if [ "${1:-}" = --with-cti-hal ]; then with_cti_hal=1; shift; fi
OUT=${1:-$WT/$A/runpod/cribl_runpod.tar.gz}
cd "$WT"

if ps -eo args | grep -v grep | grep -q 'cribl_finetune.*train.py\|^[^ ]*python train.py'; then
  echo "a cribl training process is running here; stop it first" >&2
  exit 1
fi
b1=$(python3 -c "import json,pathlib; print(pathlib.Path(json.load(open('$A/B1_train_log.json'))['selected']).name)")
b2=$(python3 -c "import json; print(max(c['examples_seen'] for c in json.load(open('$A/B2_train_log.json'))['checkpoints']))")
files=(
  scripts/cribl_finetune/common.py scripts/cribl_finetune/decision_model.py
  scripts/cribl_finetune/evaluate.py scripts/cribl_finetune/train.py
  scripts/cribl_finetune/requirements.txt scripts/cribl_finetune/runpod/remote_run.sh
  $A/cases_train.jsonl $A/cases_dev.jsonl $A/cases_test.jsonl $A/cases_manifest.json
  $A/B1_train_log.json $A/B2_train_log.json $A/B2_step*_dev.jsonl
  $M/B1/$b1 $M/B2/step_$b2
)
if [ "$with_cti_hal" = 1 ]; then files+=($A/cases_cti_hal.jsonl); fi
mkdir -p "$(dirname "$OUT")"
tar czf "$OUT" "${files[@]}"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)): B1 $b1, B2 resumes at step_$b2, CTI-HAL included: $with_cti_hal"
