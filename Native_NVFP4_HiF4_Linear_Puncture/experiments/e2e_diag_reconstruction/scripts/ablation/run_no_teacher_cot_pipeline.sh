#!/usr/bin/env bash
# Detached no-Teacher-CoT ablation pipeline. GPU 0 is never used.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

export GPU_POOL="${GPU_POOL:-1,2,3}"
export STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
# s1k_original 自然变长（P50≈9k / max≈22k），batch=4 会 OOM；整条消融固定同一 batch。
export DIAG_BATCH_SIZE=1
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_PHASE_A_TRAIN="${SKIP_PHASE_A_TRAIN:-0}"
STAGE_ROOT="${RESULTS_ROOT}/no_teacher_cot_${STAMP}"
mkdir -p "${STAGE_ROOT}"
LOG="${STAGE_ROOT}/pipeline.log"

echo "GPU_POOL=${GPU_POOL} STAMP=${STAMP} DIAG_BATCH_SIZE=${DIAG_BATCH_SIZE} STAGE_ROOT=${STAGE_ROOT} SKIP_SMOKE=${SKIP_SMOKE} SKIP_PHASE_A_TRAIN=${SKIP_PHASE_A_TRAIN}"

echo "=== bash syntax ==="
bash -n "${SCRIPT_DIR}/../common.sh"
bash -n "${SCRIPT_DIR}/../utils/gpu_pool.sh"
for f in \
  run_phase1_matrix.sh \
  run_fusable_component_ablation.sh \
  run_train_scope_ablation.sh \
  run_input_mode_ablation.sh \
  run_loss_ablation.sh \
  run_rollback_ablation.sh \
  run_clamp_ablation.sh \
  run_calib_ablation.sh
do
  bash -n "${SCRIPT_DIR}/${f}"
done
bash -n "${SCRIPT_DIR}/../eval/run_final_eval.sh"
bash -n "${SCRIPT_DIR}/../eval/run_finalists.sh"
bash -n "${SCRIPT_DIR}/../smoke/run_train_smoke.sh"
bash -n "${SCRIPT_DIR}/../smoke/run_s1k_original_smoke.sh"

echo "=== GPU pool dry-run ==="
init_stage_gpu_pool
for g in "${AVAILABLE_GPUS[@]}"; do
  if [[ "${g}" == "0" ]]; then
    echo "GPU 0 must stay empty; GPU_POOL=${GPU_POOL}" >&2
    exit 1
  fi
done

echo "=== smokes on two GPUs ==="
if [[ "${SKIP_SMOKE}" == "1" ]]; then
  echo "SKIP_SMOKE=1"
else
SMOKE_A="${RESULTS_ROOT}/smoke_s1k_question_${STAMP}"
SMOKE_B="${RESULTS_ROOT}/smoke_s1k_original_${STAMP}"
CUDA_VISIBLE_DEVICES="${AVAILABLE_GPUS[0]}" RUN_ID="smoke_s1k_question_${STAMP}" \
  bash "${SCRIPT_DIR}/../smoke/run_train_smoke.sh" &
pid_a=$!
if [[ "${PARALLEL_SLOTS}" -ge 2 ]]; then
  CUDA_VISIBLE_DEVICES="${AVAILABLE_GPUS[1]}" RUN_ID="smoke_s1k_original_${STAMP}" \
    bash "${SCRIPT_DIR}/../smoke/run_s1k_original_smoke.sh" &
  pid_b=$!
  wait "${pid_a}"
  wait "${pid_b}"
else
  wait "${pid_a}"
  CUDA_VISIBLE_DEVICES="${AVAILABLE_GPUS[0]}" RUN_ID="smoke_s1k_original_${STAMP}" \
    bash "${SCRIPT_DIR}/../smoke/run_s1k_original_smoke.sh"
fi
fi

echo "=== Phase A ==="
if [[ "${SKIP_PHASE_A_TRAIN}" == "1" ]]; then
  SKIP_TRAIN=1 bash "${SCRIPT_DIR}/run_phase1_matrix.sh"
else
  bash "${SCRIPT_DIR}/run_phase1_matrix.sh"
fi
PHASE1="${RESULTS_ROOT}/phase1_${STAMP}"
[[ -f "${PHASE1}/summary.json" ]] || { echo "missing ${PHASE1}/summary.json" >&2; exit 1; }

read -r BEST_FUSABLE BEST_ONLINE < <(python - <<PY
import json
s = json.loads(open("${PHASE1}/summary.json", encoding="utf-8").read())
print(s["best_fusable_preset"], s["best_online_preset"])
PY
)
echo "best_fusable=${BEST_FUSABLE} best_online=${BEST_ONLINE}"

echo "=== Phase B ==="
bash "${SCRIPT_DIR}/run_fusable_component_ablation.sh" "${PHASE1}"
PHASE_B="${RESULTS_ROOT}/fusable_components_${STAMP}"

echo "=== Phase C1 train_scope ==="
bash "${SCRIPT_DIR}/run_train_scope_ablation.sh" "${PHASE1}" "${BEST_ONLINE}"
echo "=== Phase C2 input_mode ==="
bash "${SCRIPT_DIR}/run_input_mode_ablation.sh" "${PHASE1}" "${BEST_FUSABLE}" "${BEST_ONLINE}"
echo "=== Phase C3 loss ==="
bash "${SCRIPT_DIR}/run_loss_ablation.sh" "${PHASE1}" "${BEST_FUSABLE}" "${BEST_ONLINE}"
echo "=== Phase C4 rollback ==="
bash "${SCRIPT_DIR}/run_rollback_ablation.sh" "${PHASE1}" "${BEST_FUSABLE}"
echo "=== Phase C5 clamp ==="
bash "${SCRIPT_DIR}/run_clamp_ablation.sh" "${PHASE1}" "${BEST_FUSABLE}"

echo "=== Phase D ==="
bash "${SCRIPT_DIR}/run_calib_ablation.sh" "${PHASE1}" "${BEST_FUSABLE}" "${BEST_ONLINE}"
PHASE_D="${RESULTS_ROOT}/calib_ablation_${STAMP}"

echo "=== Finalists AIME25 ==="
bash "${SCRIPT_DIR}/../eval/run_finalists.sh" "${PHASE1}" "${PHASE_D}"

STAGE_MAP="${STAGE_ROOT}/stage_map.json"
python - <<PY > "${STAGE_MAP}"
import json
stamp = "${STAMP}"
root = "${RESULTS_ROOT}"
print(json.dumps({
  "phase_a": f"{root}/phase1_{stamp}",
  "phase_b": f"{root}/fusable_components_{stamp}",
  "train_scope": f"{root}/train_scope_{stamp}",
  "input_mode": f"{root}/input_mode_{stamp}",
  "loss": f"{root}/loss_{stamp}",
  "rollback": f"{root}/rollback_{stamp}",
  "clamp": f"{root}/clamp_{stamp}",
  "phase_d": f"{root}/calib_ablation_{stamp}",
  "calib_stats": {
    "s1k_original": f"{root}/shared_calibration/s1k_original_n128_v32_seed42",
    "s1k_question": f"{root}/shared_calibration/s1k_question_n128_v32_seed42",
    "wikitext2": f"{root}/shared_calibration/wikitext2_n128_v32_seed42_len1024",
    "c4": f"{root}/shared_calibration/c4_n128_v32_seed42_len1024",
  },
}, indent=2))
PY

REPORT="${RESULTS_ROOT}/ABLATION_NO_TEACHER_COT_REPORT_2026-08-18.md"
conda run --no-capture-output -n hif4 python -m \
  Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli.write_no_teacher_cot_report \
  --stage_map "${STAGE_MAP}" \
  --output_md "${REPORT}"
echo "report=${REPORT}"
echo "pipeline_done STAGE_ROOT=${STAGE_ROOT}"
