#!/usr/bin/env bash
# Detach full experiment from any interactive/Cursor terminal.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
mkdir -p Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/logs
LOG="Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/logs/run_full_detached.log"
# Drop incomplete Cursor-tied run if present and empty of artifacts
INC="Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/results/20260805T114443Z"
if [[ -d "$INC" && ! -f "$INC/latency.csv" ]]; then
  rm -rf "$INC"
fi
nohup bash Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/scripts/run_full.sh >>"$LOG" 2>&1 </dev/null &
echo $! > Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/logs/run_full_detached.pid
echo "started pid=$(cat Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/logs/run_full_detached.pid)"
echo "log=$LOG"
disown || true
