#!/usr/bin/env bash
#SBATCH -p Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:1
#SBATCH -J hif4-vbench
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/hif4-vbench-%j.out
#SBATCH -e logs/hif4-vbench-%j.err

set -euo pipefail

# 这个文件只是 Slurm wrapper，核心逻辑在 python -m hif4_vbench_i2v.run_eval_10dim。
# 必填环境变量：CASE, MODE, INPUT_BASE, OUT_BASE, IMAGE_FOLDER。

: "${CASE:?必须设置 CASE}"
: "${MODE:=quant}"
: "${INPUT_BASE:?必须设置 INPUT_BASE}"
: "${OUT_BASE:?必须设置 OUT_BASE}"
: "${IMAGE_FOLDER:?必须设置 IMAGE_FOLDER}"

SCRATCH_PARENT="${SCRATCH_PARENT:-$PWD/.scratch_vbench_i2v}"
mkdir -p "$SCRATCH_PARENT" logs

CASE_INPUT="$INPUT_BASE/$CASE"
FINAL_OUT="$OUT_BASE/$CASE/$MODE"
SCR="$SCRATCH_PARENT/job_${SLURM_JOB_ID}_${CASE}_${MODE}"
rm -rf "$SCR"
mkdir -p "$SCR/input" "$SCR/output"

rsync -aL --delete "$CASE_INPUT/" "$SCR/input/"

python -m hif4_vbench_i2v.run_eval_10dim \
  --case-input "$SCR/input" \
  --output-dir "$SCR/output" \
  --image-folder "$IMAGE_FOLDER" \
  --mode "$MODE" \
  --dims "${DIMS:-all}" \
  --device cuda \
  --continue-on-error

mkdir -p "$FINAL_OUT"
cp -a "$SCR/output/." "$FINAL_OUT/"
rm -rf "$SCR"
