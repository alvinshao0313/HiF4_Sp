#!/usr/bin/env bash
# Smoke: reorder a few layers of Qwen3.5-4B and print hierarchical vs identity metrics.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXP_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/HiFloat4:${PYTHONPATH:-}"

HIF4_PY="${HIF4_PY:-/home/shaoyuantian/anaconda3/envs/hif4/bin/python}"
export PATH="$(dirname "${HIF4_PY}"):${PATH}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hif4
  set -u
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
GPUS="${GPUS:-4}"
LAYERS="${LAYERS:-0,15,31}"
OUT_DIR="${OUT_DIR:-${EXP_DIR}/results/smoke_layers}"
LOG="${EXP_DIR}/logs/smoke_layers.log"
mkdir -p "${OUT_DIR}" "$(dirname "${LOG}")"

echo "[$(date --iso-8601=seconds)] smoke layers=${LAYERS} gpu=${GPUS}" | tee "${LOG}"

# refine_passes=0: verify construction alone beats identity before paying refine cost.
CUDA_VISIBLE_DEVICES="${GPUS}" "${HIF4_PY}" -m permutation_optimization.run_mlp_reorder \
  --model "${MODEL}" \
  --calibration-dataset wikitext2 \
  --calibration-nsamples 64 \
  --calibration-seqlen 2048 \
  --activation-rows 512 \
  --weight-rows 256 \
  --refine-passes 0 \
  --refine-bad-blocks 8 \
  --device cuda \
  --layers "${LAYERS}" \
  --output-dir "${OUT_DIR}" \
  --trust-remote-code \
  2>&1 | tee -a "${LOG}"

"${HIF4_PY}" - "${OUT_DIR}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
summary = json.loads((out / "summary.json").read_text())
print("layer | accepted | id_hif4 | hier_hif4 | id_nrmse | hier_nrmse")
for r in summary["results"]:
    print(
        f"{r['layer_index']:5d} | {str(r['accepted']):8s} | "
        f"{r['identity_hif4_loss']:.6f} | {r['optimized_hif4_loss']:.6f} | "
        f"{r['identity_output_nrmse']:.6f} | {r['optimized_output_nrmse']:.6f}"
    )
accepted = [r for r in summary["results"] if r["accepted"]]
# Require mid/late layers (indices > 0) to show improvement; early layers may keep identity.
mid_late = [r for r in summary["results"] if r["layer_index"] > 0]
if not mid_late or not all(
    r["optimized_hif4_loss"] < r["identity_hif4_loss"]
    and r["optimized_output_nrmse"] < r["identity_output_nrmse"]
    for r in mid_late
):
    raise SystemExit("Smoke failed: mid/late layers did not beat identity")
print(f"SMOKE_OK accepted={len(accepted)}/{len(summary['results'])}")
PY

echo "[$(date --iso-8601=seconds)] smoke done" | tee -a "${LOG}"
