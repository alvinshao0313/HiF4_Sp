#!/usr/bin/env bash
#SBATCH -p Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:1
#SBATCH -J hif4-retry
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/hif4-retry-%j.out
#SBATCH -e logs/hif4-retry-%j.err

set -euo pipefail
: "${MISSING_TSV:?必须设置 MISSING_TSV}"
: "${INPUT_BASE:?必须设置 INPUT_BASE}"
: "${OUT_BASE:?必须设置 OUT_BASE}"
: "${IMAGE_FOLDER:?必须设置 IMAGE_FOLDER}"

python -m hif4_vbench_i2v.retry_missing \
  --missing-tsv "$MISSING_TSV" \
  --input-base "$INPUT_BASE" \
  --out-base "$OUT_BASE" \
  --image-folder "$IMAGE_FOLDER" \
  --devices "${DEVICES:-0}" \
  --max-rounds "${MAX_ROUNDS:-1}"
