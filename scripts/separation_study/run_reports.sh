#!/bin/bash
# What separates the groups (development only): every member on, per-member truth ranks recorded
# (classifier_truth_ranks), vendor / campaigns / advisories at names, strict and tradecraft, all
# statements. The fusion is irrelevant here (combiner inactive with the optional members on).
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-$ROOT/.venv/bin/python}
OUT=${OUT:-$ROOT/data/audits/separation_study}
B=$ROOT/data/benchmarks
mkdir -p "$OUT"; cd "$ROOT" || exit 1
export PYTHONPATH=. OPENROUTER_API_KEY= LOCAL_LLM_ENABLED=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEXT_GROUP_WEIGHT=1.0 BEHAVIOR_TEXT_WEIGHT=1.0 GROUP_MLP_WEIGHT=1.0
nvidia-smi > /dev/null 2>&1 || { echo "== GPU unavailable"; exit 2; }
curl -sf -m 5 localhost:8102/health > /dev/null || { echo "== technique head service down"; exit 2; }
run() {
  local name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "== $name done earlier"; return; fi
  echo "== $name start $(date +%T)"
  "$@" > "$OUT/$name.log" 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    local avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    if [ "$avail" -lt 600 ]; then kill $pid; echo "== $name KILLED low memory ${avail} MB $(date +%T)"; fi
    sleep 3
  done
  wait $pid; echo "== $name exit $? $(date +%T)"
}
declare -A FILE=([vendor]=$B/vendor/reports_hide_cited_attack20.0.jsonl [campaigns]=$B/campaigns/reports_attack20.0.jsonl [advisories]=$B/advisories/reports.jsonl)
for level in names strict tradecraft; do
  for set in advisories campaigns vendor; do
    run ${set}_$level $PY scripts/attribution_benchmark.py --report-set $set --cti-hal "${FILE[$set]}" --anonymize $level --report-observations 0 --allow-inactive-combiner --output "$OUT/${set}_$level.json"
  done
done
echo "== STUDY DONE $(date +%T)"
