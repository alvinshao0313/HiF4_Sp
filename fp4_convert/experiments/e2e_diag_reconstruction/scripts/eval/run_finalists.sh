#!/usr/bin/env bash
# Finalist AIME25 avg@5. ARC/MMLU-Pro are reused from fast eval.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"

if [[ $# -lt 2 ]]; then
  echo "usage: $0 PHASE1_DIR CALIB_ABLATION_DIR" >&2
  exit 1
fi
PHASE1_DIR="$1"
CALIB_ABLATION_DIR="$2"
FINAL_EVAL="${SCRIPT_DIR}/run_final_eval.sh"

mapfile -t JOBS < <(python - <<PY
import json
from pathlib import Path

p1 = Path("${PHASE1_DIR}").resolve()
d = Path("${CALIB_ABLATION_DIR}").resolve()
phase_a = json.loads((p1 / "summary.json").read_text(encoding="utf-8"))
phase_d = json.loads((d / "summary.json").read_text(encoding="utf-8"))

jobs = [
    ("native_nvfp4", str((p1 / "E0_native_nvfp4").resolve()), ""),
    ("direct_hif4", str((p1 / "E1_direct_hif4").resolve()), ""),
]

def artifact_job(row):
    run_dir = Path(row["run_dir"]).resolve()
    art = run_dir / "checkpoint" / "final_model" / "conversion_state.pt"
    return ("artifact", str(run_dir), str(art))

for row in (phase_a.get("best_fusable"), phase_a.get("best_online")):
    if row:
        jobs.append(artifact_job(row))
for row in (phase_d.get("best_fusable"), phase_d.get("best_online")):
    if row and row.get("calib_source") != "s1k_original":
        jobs.append(artifact_job(row))

seen = set()
for variant, run_dir, art in jobs:
    if run_dir in seen:
        continue
    seen.add(run_dir)
    print(f"{variant}\t{run_dir}\t{art}")
PY
)

init_stage_gpu_pool
require_reasoning_eval_pool
eval_slots="$(reasoning_eval_slots)"
i=0
n=${#JOBS[@]}
while (( i < n )); do
  pids=()
  slot=0
  while (( slot < eval_slots && i < n )); do
    gpu="$(reasoning_eval_pair "${slot}")"
    IFS=$'\t' read -r variant run_dir art <<< "${JOBS[${i}]}"
    echo "launch gpu=${gpu} aime ${variant} ${run_dir}"
    if [[ "${variant}" == "artifact" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" bash "${FINAL_EVAL}" "${run_dir}" artifact "${art}" &
    else
      CUDA_VISIBLE_DEVICES="${gpu}" bash "${FINAL_EVAL}" "${run_dir}" "${variant}" &
    fi
    pids+=("$!")
    slot=$((slot + 1))
    i=$((i + 1))
  done
  wait_gpu_wave "${pids[@]}"
done
echo "finalists_done PHASE1_DIR=${PHASE1_DIR} CALIB_ABLATION_DIR=${CALIB_ABLATION_DIR}"
