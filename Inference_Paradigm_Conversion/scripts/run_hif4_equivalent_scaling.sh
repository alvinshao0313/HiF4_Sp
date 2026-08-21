#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/shaoyuantian/program/HiF4_Sp"
PY="/home/shaoyuantian/anaconda3/envs/hif4/bin/python"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

OUT="${OUT_DIR:-Inference_Paradigm_Conversion/results}"
FORMAL_CONFIG="${FORMAL_CONFIG:-Inference_Paradigm_Conversion/configs/qwen3_8b_hif4_equivalent_scaling.yaml}"
SMOKE_CONFIG="${SMOKE_CONFIG:-Inference_Paradigm_Conversion/configs/qwen3_8b_hif4_equivalent_scaling_smoke.yaml}"
FORMAL_RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_hif4_equivalent_scaling}"
SMOKE_RUN_ID="${SMOKE_RUN_ID:-${FORMAL_RUN_ID}_smoke}"
MAX_GPUS="${MAX_GPUS:-4}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_FORMAL="${RUN_FORMAL:-1}"

mkdir -p "$OUT/$FORMAL_RUN_ID/logs" "$OUT/$SMOKE_RUN_ID/logs"

if [[ -n "${GPU_LIST:-}" ]]; then
  IFS=',' read -r -a ALL_GPUS <<< "$GPU_LIST"
else
  mapfile -t ALL_GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 < 2048) print $1}')
fi
if [[ "${#ALL_GPUS[@]}" -lt 1 ]]; then
  echo "No free GPUs (<2GB used). Set GPU_LIST explicitly or free a GPU." >&2
  exit 1
fi
NUM_GPUS=${#ALL_GPUS[@]}
if (( NUM_GPUS > MAX_GPUS )); then
  NUM_GPUS=$MAX_GPUS
fi
GPUS=("${ALL_GPUS[@]:0:$NUM_GPUS}")
PRIMARY_GPU="${GPUS[0]}"

echo "HiF4 equivalent scaling: FORMAL_RUN_ID=$FORMAL_RUN_ID SMOKE_RUN_ID=$SMOKE_RUN_ID GPUS=${GPUS[*]}"

run_stage() {
  local config="$1" run_id="$2" stage="$3" gpu="$4" shard_id="${5:-0}" num_shards="${6:-1}" split="${7:-discovery}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m Inference_Paradigm_Conversion.run_hif4_scaling \
    --scaling-config "$config" --out-dir "$OUT" \
    stage --stage "$stage" --run-id "$run_id" --device cuda:0 \
    --split "$split" --shard-id "$shard_id" --num-shards "$num_shards"
}

run_merge() {
  local config="$1" run_id="$2" stage="$3" num_shards="$4"
  "$PY" -m Inference_Paradigm_Conversion.run_hif4_scaling \
    --scaling-config "$config" --out-dir "$OUT" \
    merge --stage "$stage" --run-id "$run_id" --num-shards "$num_shards"
}

run_report() {
  local config="$1" run_id="$2"
  "$PY" -m Inference_Paradigm_Conversion.run_hif4_scaling \
    --scaling-config "$config" --out-dir "$OUT" report --run-id "$run_id"
}

run_parallel_stage() {
  local config="$1" run_id="$2" stage="$3" split="${4:-discovery}"
  local -a pids=()
  for ((i=0; i<NUM_GPUS; i++)); do
    local gpu="${GPUS[$i]}"
    (
      run_stage "$config" "$run_id" "$stage" "$gpu" "$i" "$NUM_GPUS" "$split"
    ) >"$OUT/$run_id/logs/${stage}_shard${i}.log" 2>&1 &
    pids+=("$!")
    echo "[$stage] shard=$i gpu=$gpu pid=${pids[-1]}"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" == "1" ]]; then
    echo "Parallel stage '$stage' failed; inspect $OUT/$run_id/logs/" >&2
    return 1
  fi
}

run_discovery_chain() {
  local config="$1" run_id="$2" parallel="$3"
  local n=1
  if [[ "$parallel" == "1" ]]; then n=$NUM_GPUS; fi

  if [[ "$parallel" == "1" ]]; then
    run_parallel_stage "$config" "$run_id" stats discovery
  else
    run_stage "$config" "$run_id" stats "$PRIMARY_GPU" 0 1 discovery \
      >"$OUT/$run_id/logs/stats_shard0.log" 2>&1
  fi
  run_merge "$config" "$run_id" stats "$n" | tee "$OUT/$run_id/logs/merge_stats.log"
  run_stage "$config" "$run_id" build-candidates "$PRIMARY_GPU" 0 1 discovery \
    | tee "$OUT/$run_id/logs/build_candidates.log"

  if [[ "$parallel" == "1" ]]; then
    run_parallel_stage "$config" "$run_id" eval discovery
  else
    run_stage "$config" "$run_id" eval "$PRIMARY_GPU" 0 1 discovery \
      >"$OUT/$run_id/logs/eval_shard0.log" 2>&1
  fi
  run_merge "$config" "$run_id" eval "$n" | tee "$OUT/$run_id/logs/merge_eval.log"
  run_stage "$config" "$run_id" select-full "$PRIMARY_GPU" 0 1 discovery \
    | tee "$OUT/$run_id/logs/select_full.log"

  if [[ "$parallel" == "1" ]]; then
    run_parallel_stage "$config" "$run_id" eval-full discovery
  else
    run_stage "$config" "$run_id" eval-full "$PRIMARY_GPU" 0 1 discovery \
      >"$OUT/$run_id/logs/eval_full_shard0.log" 2>&1
  fi
  run_merge "$config" "$run_id" eval-full "$n" | tee "$OUT/$run_id/logs/merge_full.log"

  run_stage "$config" "$run_id" build-refine "$PRIMARY_GPU" 0 1 discovery \
    | tee "$OUT/$run_id/logs/build_refine.log"
  if [[ "$parallel" == "1" ]]; then
    run_parallel_stage "$config" "$run_id" eval-refine discovery
  else
    run_stage "$config" "$run_id" eval-refine "$PRIMARY_GPU" 0 1 discovery \
      >"$OUT/$run_id/logs/eval_refine_shard0.log" 2>&1
  fi
  run_merge "$config" "$run_id" eval-refine "$n" | tee "$OUT/$run_id/logs/merge_refine.log"

  run_stage "$config" "$run_id" select-policy "$PRIMARY_GPU" 0 1 discovery \
    | tee "$OUT/$run_id/logs/select_policy.log"
  run_report "$config" "$run_id" >"$OUT/$run_id/logs/report_discovery.log" 2>&1
}

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[1/3] Running IPC test suite in hif4 ..."
  "$PY" -m pytest Inference_Paradigm_Conversion/tests -q \
    | tee "$OUT/$FORMAL_RUN_ID/logs/pytest.log"
fi

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "[2/3] Running single-GPU smoke through ES5-COMB ..."
  run_discovery_chain "$SMOKE_CONFIG" "$SMOKE_RUN_ID" 0
  echo "$SMOKE_RUN_ID" > "$OUT/latest_hif4_equivalent_scaling_smoke_run_id.txt"
  echo "Smoke passed: $SMOKE_RUN_ID"
fi

if [[ "$RUN_FORMAL" == "1" ]]; then
  echo "[3/3] Running formal multi-GPU discovery ..."
  run_discovery_chain "$FORMAL_CONFIG" "$FORMAL_RUN_ID" 1

  echo "Running independent representative validation ..."
  run_stage "$FORMAL_CONFIG" "$FORMAL_RUN_ID" validate "$PRIMARY_GPU" 0 1 validation \
    | tee "$OUT/$FORMAL_RUN_ID/logs/validation.log"
  run_report "$FORMAL_CONFIG" "$FORMAL_RUN_ID" >"$OUT/$FORMAL_RUN_ID/logs/report_validation.log" 2>&1

  if ! grep -q '"validation_pass": true' "$OUT/$FORMAL_RUN_ID/es6_validation.json"; then
    echo "Representative validation gate failed. Per plan, stop before ES6.5/E2E; final failure report is preserved."
    echo "$FORMAL_RUN_ID" > "$OUT/latest_hif4_equivalent_scaling_run_id.txt"
    exit 0
  fi

  echo "Instantiating frozen recipe on all decoder layers ..."
  run_stage "$FORMAL_CONFIG" "$FORMAL_RUN_ID" all-layer "$PRIMARY_GPU" 0 1 discovery \
    | tee "$OUT/$FORMAL_RUN_ID/logs/all_layer.log"

  echo "Running full target-trajectory sanity check ..."
  run_stage "$FORMAL_CONFIG" "$FORMAL_RUN_ID" trajectory "$PRIMARY_GPU" 0 1 validation \
    | tee "$OUT/$FORMAL_RUN_ID/logs/trajectory.log"

  echo "Running ARC-Easy / ARC-Challenge semantic E2E ..."
  run_stage "$FORMAL_CONFIG" "$FORMAL_RUN_ID" e2e "$PRIMARY_GPU" 0 1 validation \
    | tee "$OUT/$FORMAL_RUN_ID/logs/e2e.log"

  run_report "$FORMAL_CONFIG" "$FORMAL_RUN_ID" \
    | tee "$OUT/$FORMAL_RUN_ID/logs/report_final.log"
  echo "$FORMAL_RUN_ID" > "$OUT/latest_hif4_equivalent_scaling_run_id.txt"
  echo "HiF4 equivalent scaling formal run complete: $FORMAL_RUN_ID"
fi
