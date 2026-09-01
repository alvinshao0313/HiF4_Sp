#!/usr/bin/env bash
# Serial standard-MMLU (lm_eval + vLLM TP=2) for Phase-A E0-E4 on GPUs 2,3.
#
# Detached launch (recommended; does not need an interactive terminal):
#   cd /home/shaoyuantian/program/HiF4_Sp
#   setsid nohup env CUDA_VISIBLE_DEVICES=2,3 \
#     bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/eval/run_mmlu_e0_e4.sh \
#     </dev/null >"${RESULTS}/mmlu_lm_eval_gpu23_launch.out" 2>&1 &
# Check: cat .../phaseA_refactor_*/mmlu_lm_eval_gpu23.status.txt
set -euo pipefail
trap '' HUP
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

PHASEA_BASE="${RESULTS_ROOT}/phaseA_refactor_20260825T035730Z"
GPU_PAIR="${CUDA_VISIBLE_DEVICES:-2,3}"
EVAL_SEED="${EVAL_SEED:-42}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MASTER_LOG="${PHASEA_BASE}/mmlu_lm_eval_gpu23_${STAMP}.log"
echo $$ > "${PHASEA_BASE}/mmlu_lm_eval_gpu23.pid"

require_gpu_ids_in_pool "${GPU_PAIR}"
n_gpu=0
IFS=',' read -r -a _eval_gpus <<< "${GPU_PAIR}"
for tok in "${_eval_gpus[@]}"; do
  tok="${tok// /}"
  [[ -n "${tok}" ]] && n_gpu=$((n_gpu + 1))
done
if [[ "${n_gpu}" -lt 2 ]]; then
  echo "MMLU lm_eval needs CUDA_VISIBLE_DEVICES with >=2 GPUs, got ${GPU_PAIR}" >&2
  exit 1
fi

# Keep shared materializations across E1-E4; do not delete after each eval.
export KEEP_EVAL_CKPT=1

run_one() {
  local name="$1"
  local variant="$2"
  local artifact="${3:-}"
  local out_dir="${PHASEA_BASE}/${name}"
  local run_log="${out_dir}/eval/mmlu/run.log"
  mkdir -p "${out_dir}/eval/mmlu"

  {
    echo "===== $(date --iso-8601=seconds) START ${name} variant=${variant} gpus=${GPU_PAIR} ====="
    runtime_abi_prepare "${out_dir}" "${variant}" adopted
    local extra=()
    if [[ -n "${artifact}" ]]; then
      extra+=(--artifact_path "${artifact}")
    fi
    CUDA_VISIBLE_DEVICES="${GPU_PAIR}" e2e_eval \
      --variant "${variant}" \
      --output_dir "${out_dir}" \
      --groups mmlu \
      --eval_seed "${EVAL_SEED}" \
      "${extra[@]}"
    if [[ "${variant}" == "r64_only" || "${variant}" == "artifact" ]]; then
      runtime_abi_stamp "${out_dir}"
    fi
    echo "===== $(date --iso-8601=seconds) DONE ${name} ====="
  } 2>&1 | tee -a "${run_log}" "${MASTER_LOG}"
}

echo "master log: ${MASTER_LOG}" | tee "${MASTER_LOG}"

run_one E0_native_nvfp4 native_nvfp4
run_one E1_direct_hif4 direct_hif4
run_one E2_r64_only r64_only
run_one E3_fusable artifact \
  "${PHASEA_BASE}/E3_fusable/checkpoint/final_model/conversion_state.pt"
run_one E4_fusable_r64 artifact \
  "${PHASEA_BASE}/E4_fusable_r64/checkpoint/final_model/conversion_state.pt"

echo "===== MMLU summary =====" | tee -a "${MASTER_LOG}"
PHASEA_BASE="${PHASEA_BASE}" conda run --no-capture-output -n hif4 python - <<'PY' | tee -a "${MASTER_LOG}"
import json
import os
from pathlib import Path

base = Path(os.environ["PHASEA_BASE"])
names = [
    "E0_native_nvfp4",
    "E1_direct_hif4",
    "E2_r64_only",
    "E3_fusable",
    "E4_fusable_r64",
]
print(f"{'exp':<22} {'mmlu_acc':>10}")
for name in names:
    path = base / name / "eval" / "mmlu" / "metrics.json"
    if not path.is_file():
        print(f"{name:<22} {'MISSING':>10}")
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores") or {}
    acc = scores.get("mmlu")
    if acc is None:
        for key, value in scores.items():
            if key == "mmlu" or key.endswith(",none") and "mmlu" in key:
                acc = value
                break
    print(f"{name:<22} {acc if acc is not None else 'N/A':>10}")
PY

echo "done. master log: ${MASTER_LOG}"
