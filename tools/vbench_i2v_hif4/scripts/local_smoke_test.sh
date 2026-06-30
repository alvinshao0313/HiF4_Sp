#!/usr/bin/env bash
set -euo pipefail

# 本 smoke test 不需要真实 VBench，也不需要 GPU。
# 它验证：包可 import、preflight 可运行、输入构建/repair/validate/scan 的基础逻辑可工作。
# 关键检查：build_eval_inputs 必须要求 exact filename repeat，不能把 base-0.mp4 复制成 base-1..4.mp4。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT_DIR"

TMP="${TMPDIR:-/tmp}/hif4_vbench_i2v_smoke_$$"
rm -rf "$TMP"
mkdir -p "$TMP/template/i2v_subject_background/videos_quant_sb" "$TMP/template/i2v_camera_only/videos_quant_camera"
mkdir -p "$TMP/generated_broken" "$TMP/generated_full" "$TMP/results/hif4_seed42/quant/i2v_subject"

# 构造假模板：2 个 subject/background base × 5 repeats = 10；1 个 camera base × 5 repeats = 5。
for base in "cat on grass" "dog near lake"; do
  echo "fake only repeat 0 for $base" > "$TMP/generated_broken/${base}-0.mp4"
  for i in 0 1 2 3 4; do
    echo "generated full $base repeat $i" > "$TMP/generated_full/${base}-${i}.mp4"
    echo "template $base repeat $i" > "$TMP/template/i2v_subject_background/videos_quant_sb/${base}-${i}.mp4"
  done
done
for base in "camera pans left"; do
  echo "fake only repeat 0 for $base" > "$TMP/generated_broken/${base}-0.mp4"
  for i in 0 1 2 3 4; do
    echo "generated full $base repeat $i" > "$TMP/generated_full/${base}-${i}.mp4"
    echo "template $base repeat $i" > "$TMP/template/i2v_camera_only/videos_quant_camera/${base}-${i}.mp4"
  done
done

echo '[]' > "$TMP/template/i2v_subject_background/i2v_subject_full_info.json"
echo '[]' > "$TMP/template/i2v_camera_only/camera_motion_full_info.json"

python -m hif4_vbench_i2v.preflight --skip-import --scratch-dir "$TMP/scratch" --json-report "$TMP/preflight.json"

# 缺少 -1..-4 的源目录必须失败；这是为了防止把单个视频复制成 5 份的非官方输入。
if python -m hif4_vbench_i2v.build_eval_inputs --template-case "$TMP/template" --generated-dir "$TMP/generated_broken" --out-case "$TMP/evaluation_inputs/broken" --copy-mode physical >/tmp/hif4_vbench_i2v_unexpected_success.log 2>&1; then
  cat /tmp/hif4_vbench_i2v_unexpected_success.log
  echo "ERROR: build_eval_inputs unexpectedly accepted generated_broken with only repeat-0 videos" >&2
  exit 1
fi

python -m hif4_vbench_i2v.build_eval_inputs --template-case "$TMP/template" --generated-dir "$TMP/generated_full" --out-case "$TMP/evaluation_inputs/hif4_seed42" --copy-mode physical
python -m hif4_vbench_i2v.validate_case_input --case-input "$TMP/evaluation_inputs/hif4_seed42" --expected-sb 10 --expected-camera 5 --forbid-symlink

mkdir -p "$TMP/results/hif4_seed42/quant/i2v_subject"
echo '{"i2v_subject": 0.9}' > "$TMP/results/hif4_seed42/quant/i2v_subject/hif4_seed42_quant_i2v_subject_eval_results.json"
python -m hif4_vbench_i2v.scan_missing --out-base "$TMP/results" --cases hif4_seed42 --modes quant --dims i2v_subject

echo "LOCAL_SMOKE_TEST_OK tmp=$TMP"
