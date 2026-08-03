#!/usr/bin/env bash
set -euo pipefail

# 统一从仓库根目录运行；所有新增代码与输出均限制在 ChuanCi/。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results/comprehensive_nvfp4_hif4/final"
PACKED_CHECKPOINT="${REPO_ROOT}/Qmodel/Qwen3.5-27B-NVFP4"

if [[ ! -d "${PACKED_CHECKPOINT}" ]]; then
  echo "Packed NVFP4 checkpoint not found: ${PACKED_CHECKPOINT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
conda run -n hif4 python "${SCRIPT_DIR}/nvfp4_hif4_study.py" \
  --device cuda \
  --samples-per-repeat 320000 \
  --repeats 10 \
  --packed-checkpoint "${PACKED_CHECKPOINT}" \
  --layers 3,31,63 \
  --chunk-groups 65536 \
  --output-dir "${OUTPUT_DIR}"

echo "Results: ${OUTPUT_DIR}/NVFP4_HiF4_comprehensive_results.json"
echo "Report:  ${OUTPUT_DIR}/NVFP4_HiF4_comprehensive_report.html"
