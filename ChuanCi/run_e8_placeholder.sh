#!/usr/bin/env bash
# E8 真实权重实验（请先填入 checkpoint 路径后再运行）
# 必须在 hif4 conda 环境中执行。
set -euo pipefail

# ========= 用户需要填写 =========
BF16_CHECKPOINT="${BF16_CHECKPOINT:-/path/to/bf16_model}"
NVFP4_FAKE_CHECKPOINT="${NVFP4_FAKE_CHECKPOINT:-/path/to/nvfp4_fake_decoded_float_model}"
PTS_SCALES="${PTS_SCALES:-}"   # 可选：JSON 或 .pt；留空则只评测 direct
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-results/e8}"
# ==============================

if [[ "${BF16_CHECKPOINT}" == /path/to/* ]] || [[ "${NVFP4_FAKE_CHECKPOINT}" == /path/to/* ]]; then
  echo "请先设置 BF16_CHECKPOINT 与 NVFP4_FAKE_CHECKPOINT（已解码的浮点 NVFP4 fake 权重，不是 packed uint8）。"
  exit 2
fi

LINEAR_RE='(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$'
EMBED_RE='(embed_tokens|lm_head)\.weight$'

mkdir -p "${OUT_ROOT}"

# 单层 q_proj 预检
NV_PTS_ARGS=()
if [[ -n "${PTS_SCALES}" ]]; then
  NV_PTS_ARGS=(--pts-scales "${PTS_SCALES}")
fi

python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint "${NVFP4_FAKE_CHECKPOINT}" \
  --input-kind nvfp4_fake \
  "${NV_PTS_ARGS[@]}" \
  --tensor-name model.layers.0.self_attn.q_proj.weight \
  --device "${DEVICE}" \
  --output-dir "${OUT_ROOT}/nv_single_q_proj"

python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint "${BF16_CHECKPOINT}" \
  --input-kind bf16 \
  --tensor-name model.layers.0.self_attn.q_proj.weight \
  --device "${DEVICE}" \
  --output-dir "${OUT_ROOT}/bf16_single_q_proj"

# 七类 Linear 主实验
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint "${NVFP4_FAKE_CHECKPOINT}" \
  --input-kind nvfp4_fake \
  "${NV_PTS_ARGS[@]}" \
  --include-regex "${LINEAR_RE}" \
  --device "${DEVICE}" \
  --group-size 64 \
  --group-dim -1 \
  --chunk-groups 16384 \
  --output-dir "${OUT_ROOT}/model_nvfp4_linear"

python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint "${BF16_CHECKPOINT}" \
  --input-kind bf16 \
  --include-regex "${LINEAR_RE}" \
  --device "${DEVICE}" \
  --group-size 64 \
  --group-dim -1 \
  --chunk-groups 16384 \
  --output-dir "${OUT_ROOT}/model_bf16_linear"

# embedding / lm_head 单独报告，不并入主全局
python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint "${NVFP4_FAKE_CHECKPOINT}" \
  --input-kind nvfp4_fake \
  "${NV_PTS_ARGS[@]}" \
  --include-regex "${EMBED_RE}" \
  --device "${DEVICE}" \
  --output-dir "${OUT_ROOT}/model_nvfp4_embedding_head"

python nvfp4_hif4_torch.py evaluate-checkpoint \
  --checkpoint "${BF16_CHECKPOINT}" \
  --input-kind bf16 \
  --include-regex "${EMBED_RE}" \
  --device "${DEVICE}" \
  --output-dir "${OUT_ROOT}/model_bf16_embedding_head"

echo "E8 完成，结果在 ${OUT_ROOT}"
