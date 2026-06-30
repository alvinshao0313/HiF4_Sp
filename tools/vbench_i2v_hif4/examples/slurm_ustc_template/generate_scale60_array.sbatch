#!/usr/bin/env bash
#SBATCH -p Students
#SBATCH --qos=qos_stu_medium_2gpu
#SBATCH --gres=gpu:A100:1
#SBATCH -J hif4-gen-s60
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 08:00:00
#SBATCH --array=0-2%2
#SBATCH -o logs/hif4-gen-s60-%A_%a.out
#SBATCH -e logs/hif4-gen-s60-%A_%a.err

set -euo pipefail

# 这是占位 wrapper。原仓库应把自己的 HiF4/Wan 生成命令填入 run_generate_one_video 函数。
# 重要：VBench-I2V 官方要求每个 image-prompt pair 真实采样 5 个视频，命名为 prompt-0..prompt-4.mp4。
# 不能只生成 prompt-0.mp4 再复制成 5 份。详见 docs/generation_contract_zh.md。
# array 映射：0=i2v_subject, 1=i2v_background, 2=camera_motion。

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) DIM=i2v_subject ;;
  1) DIM=i2v_background ;;
  2) DIM=camera_motion ;;
  *) echo "非法 SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

: "${SEED:=42}"
: "${FULL_INFO_JSON:?必须设置 FULL_INFO_JSON}"
: "${OUT_ROOT:?OUT_ROOT is required}"
: "${REPEATS:=5}"

mkdir -p "$OUT_ROOT/full_info" "$OUT_ROOT/shards/$DIM/generated" logs

python -m hif4_vbench_i2v.build_scale60_full_info \
  --full-info-json "$FULL_INFO_JSON" \
  --out-dir "$OUT_ROOT/full_info" \
  --dims "$DIM" \
  --start-offset "${START_OFFSET:-10}" \
  --n-per-dim "${N_PER_DIM:-20}"

run_generate_one_video() {
  local dim="$1"
  local repeat_index="$2"
  local seed="$3"
  local out_dir="$4"
  echo "TODO: 调用原仓库 HiF4/Wan 生成入口 dim=$dim repeat=$repeat_index seed=$seed out_dir=$out_dir" >&2
  echo "TODO: 生成入口必须按 full_info prompt 保存为 <prompt>-${repeat_index}.mp4" >&2
  return 1
}

for repeat_index in $(seq 0 $((REPEATS - 1))); do
  repeat_seed=$((SEED + repeat_index))
  run_generate_one_video "$DIM" "$repeat_index" "$repeat_seed" "$OUT_ROOT/shards/$DIM/generated"
done
