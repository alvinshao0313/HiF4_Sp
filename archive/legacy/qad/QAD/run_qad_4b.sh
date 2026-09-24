#!/usr/bin/env bash
# Qwen3.5-4B S0 微调恢复实验：
#   Stage 0: HiFloat4/main.py 产出 HiF4 RTN ckpt（作为 QAD 初始化 + baseline 参照）
#   Stage 1: 精确恢复 sanity check（exact_grid 分解 RTN ckpt，e6m2(s0)⊙B 须逐 bit 等于原权重）
#   Stage 2: QAD S0 训练（冻 B + 只训 S0，S1K 完整轨迹，导出 reconstructed_model）
#   Stage 3: 同协议评测（lm_eval ARC/MMLU + lighteval MMLU-Pro 300）并汇总 dense/RTN/S0 对照
# 必须在 hif4 conda 环境中运行。各阶段按产物存在与否幂等跳过，重跑不浪费已完成阶段。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HIF4_PY="${HIF4_PY:-/home/shaoyuantian/anaconda3/envs/hif4/bin/python}"
if [[ ! -x "${HIF4_PY}" ]]; then
  echo "错误：找不到 hif4 python: ${HIF4_PY}" >&2
  exit 1
fi
export PATH="$(dirname "${HIF4_PY}"):${PATH}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate hif4
  set -u
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "hif4" ]]; then
  echo "错误：需要 hif4 conda 环境" >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}/ScaleTuning:${REPO_ROOT}/ChuanCi:${REPO_ROOT}/Block_Sparse:${REPO_ROOT}/HiFloat4:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED="${PYTHONHASHSEED:-31}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
EXP_DIR="${EXP_DIR:-${REPO_ROOT}/QAD/.result/qad_qwen3_5_4b}"
RTN_CKPT="${EXP_DIR}/rtn_ckpt"
TRAIN_DIR="${EXP_DIR}/train"
RESULTS_DIR="${EXP_DIR}/results"
RECON="${TRAIN_DIR}/reconstructed_model"
LOG="${EXP_DIR}/run_qad_4b.log"

TRAIN_GPUS="${TRAIN_GPUS:-0}"
DTYPE="${DTYPE:-bfloat16}"
HIF4_WEIGHT_FORMAT="${HIF4_WEIGHT_FORMAT:-hif4}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,gate_proj,up_proj,down_proj}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-32768}"
LOGIT_CHUNK_SIZE="${LOGIT_CHUNK_SIZE:-32}"
MAX_STEPS="${MAX_STEPS:-2000}"

LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_easy,arc_challenge,mmlu}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-8}"
MMLU_PRO_DATASET="${MMLU_PRO_DATASET:-mmlu_pro|0}"
MMLU_PRO_MAX_SAMPLES="${MMLU_PRO_MAX_SAMPLES:-300}"

ABLATION_DIR="${REPO_ROOT}/HiF4_exp/qwen35_4b_w4a4_proj_ablation"
DENSE_BASELINE_JSON="${REPO_ROOT}/Block_Sparse/experiments/qwen35_4b_dense/dense_baseline.json"

export REPO_ROOT RTN_CKPT TARGET_MODULES EXP_DIR DENSE_BASELINE_JSON ABLATION_DIR

mkdir -p "${EXP_DIR}" "${RESULTS_DIR}"

echo "[$(date --iso-8601=seconds)] === Stage 0: HiF4 RTN ckpt -> ${RTN_CKPT} ===" | tee -a "${LOG}"
if [[ -f "${RTN_CKPT}/config.json" ]]; then
  echo "[$(date --iso-8601=seconds)] RTN ckpt 已存在，跳过量化" | tee -a "${LOG}"
else
  rm -rf "${RTN_CKPT}"
  mkdir -p "${RTN_CKPT}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" "${HIF4_PY}" HiFloat4/main.py \
    --model "${MODEL}" \
    --dtype "${DTYPE}" \
    --hif4w true \
    --hif4_weight_format "${HIF4_WEIGHT_FORMAT}" \
    --gptq false \
    --gptq_save_path "${RTN_CKPT}" \
    --exclude-layers lm_head \
    2>&1 | tee -a "${LOG}"
  "${HIF4_PY}" - <<PY 2>&1 | tee -a "${LOG}"
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("${MODEL}", trust_remote_code=True)
tok.save_pretrained("${RTN_CKPT}")
print("tokenizer saved to ${RTN_CKPT}")
PY
fi
if [[ ! -f "${RTN_CKPT}/config.json" ]]; then
  echo "RTN 量化失败，缺少 config.json: ${RTN_CKPT}" | tee -a "${LOG}" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] === Stage 1: 精确恢复 sanity check（逐 bit） ===" | tee -a "${LOG}"
"${HIF4_PY}" - <<'PY' 2>&1 | tee -a "${LOG}"
import os
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

repo = Path(os.environ["REPO_ROOT"])
for sub in ("QAD", "ScaleTuning", "ChuanCi"):
    sys.path.insert(0, str(repo / sub))

from nvfp4_hif4_torch import HiF4Config  # noqa: E402
from hif4_frozen_b import build_frozen_b_and_s0  # noqa: E402
from hif4_fixed_s0 import apply_e6m2_ste  # noqa: E402

ckpt = os.environ["RTN_CKPT"]
targets = {x.strip() for x in os.environ["TARGET_MODULES"].split(",") if x.strip()}
config = HiF4Config(group_size=64, group_dim=-1, scale_mode="hardware")

model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)

checked = 0
total_elems = 0
with torch.no_grad():
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        checked += 1
        w = module.weight.data
        # 与训练初始化同路径：伪量化 ckpt 走 exact_grid 精确分解
        s0, frozen_b = build_frozen_b_and_s0(w, config=config, exact_grid=True)
        s0q = apply_e6m2_ste(s0, scale_mode=config.scale_mode)
        g = int(config.group_size)
        out_f, in_f = w.shape
        s0_exp = s0q.unsqueeze(-1).expand(out_f, in_f // g, g).reshape(out_f, in_f)
        recon = (s0_exp.float() * frozen_b.float()).to(w.dtype)
        if not torch.equal(recon, w):
            bad = (recon != w).nonzero()[0].tolist()
            raise SystemExit(
                f"[sanity] FAIL: {name} 不是逐 bit 精确恢复（首个差异元素 {bad}）"
            )
        total_elems += w.numel()
    del model

if checked == 0:
    raise SystemExit("[sanity] FAIL: no target Linear modules matched")
print(f"[sanity] OK: {checked} Linear modules / {total_elems} elems 全部逐 bit 精确恢复；训练起点 ≡ RTN baseline")
PY

echo "[$(date --iso-8601=seconds)] === Stage 2: QAD S0 训练 -> ${TRAIN_DIR} ===" | tee -a "${LOG}"
if [[ -f "${RECON}/config.json" ]]; then
  echo "[$(date --iso-8601=seconds)] reconstructed_model 已存在，跳过训练" | tee -a "${LOG}"
else
  mkdir -p "${TRAIN_DIR}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" "${HIF4_PY}" "${SCRIPT_DIR}/train_qad.py" \
    --model_path "${MODEL}" \
    --pseudo_quant_model_path "${RTN_CKPT}" \
    --output_dir "${TRAIN_DIR}" \
    --trace_source "deepseek" \
    --seed "31" \
    --deterministic "true" \
    --target_modules "${TARGET_MODULES}" \
    --tune_modules "${TARGET_MODULES}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "1" \
    --gradient_accumulation_steps "8" \
    --learning_rate "2e-5" \
    --weight_decay "0.001" \
    --logging_steps "10" \
    --temperature "1.0" \
    --task_alpha "0.05" \
    --eakld_alpha "2.0" \
    --lafd_alpha "0.5" \
    --confidence_k "16" \
    --lafd_topk "3" \
    --logit_chunk_size "${LOGIT_CHUNK_SIZE}" \
    --gradient_checkpointing "true" \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --optim "adamw_torch" \
    --max_grad_norm "1.3" \
    --warmup_ratio "0.05" \
    --lr_scheduler_type "constant_with_warmup" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --allow_truncate "false" \
    --attn_implementation "sdpa" \
    --fp16 "false" \
    --bf16 "true" \
    --export_reconstructed_model "true" \
    --parallel_mode "layer" \
    2>&1 | tee -a "${LOG}"
fi
if [[ ! -f "${RECON}/config.json" ]]; then
  echo "训练未产出 reconstructed_model: ${RECON}" | tee -a "${LOG}" >&2
  exit 1
fi

echo "[$(date --iso-8601=seconds)] === Stage 3a: lm_eval (${LM_EVAL_TASKS}) ===" | tee -a "${LOG}"
LM_JSON="${RESULTS_DIR}/lm_eval_arc_mmlu.json"
if [[ -f "${LM_JSON}" ]]; then
  echo "[$(date --iso-8601=seconds)] lm_eval 结果已存在，跳过" | tee -a "${LOG}"
else
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" "${HIF4_PY}" \
    "${ABLATION_DIR}/eval_lm_eval_hif4a.py" \
    --model_path "${RECON}" \
    --tasks "${LM_EVAL_TASKS}" \
    --num_fewshot "0" \
    --batch_size "${LM_EVAL_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --fake_act_quant "hif4" \
    --fake_act_quant_exclude "lm_head" \
    --output_json "${LM_JSON}" \
    2>&1 | tee -a "${LOG}"
fi

echo "[$(date --iso-8601=seconds)] === Stage 3b: lighteval ${MMLU_PRO_DATASET} max_samples=${MMLU_PRO_MAX_SAMPLES} ===" | tee -a "${LOG}"
MMLU_PRO_DIR="${RESULTS_DIR}/mmlu_pro"
if compgen -G "${MMLU_PRO_DIR}/reconstructed_model/results/results_*.json" > /dev/null; then
  echo "[$(date --iso-8601=seconds)] mmlu_pro 结果已存在，跳过" | tee -a "${LOG}"
else
  mkdir -p "${MMLU_PRO_DIR}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" "${HIF4_PY}" main.py \
    --model_path "${RECON}" \
    --datasets "${MMLU_PRO_DATASET}" \
    --max_samples "${MMLU_PRO_MAX_SAMPLES}" \
    --tensor_parallel_size "1" \
    --max_model_length "32768" \
    --max_new_tokens "32768" \
    --temperature "0.7" \
    --top_p "0.8" \
    --top_k "20" \
    --gpu_memory_utilization "0.9" \
    --fake_act_quant "hif4" \
    --fake_act_quant_exclude "lm_head" \
    --disable_thinking \
    --output_dir "${MMLU_PRO_DIR}" \
    2>&1 | tee -a "${LOG}"
fi

echo "[$(date --iso-8601=seconds)] === Stage 3c: 汇总 dense / RTN / S0 ===" | tee -a "${LOG}"
"${HIF4_PY}" - <<'PY' 2>&1 | tee -a "${LOG}"
import glob
import json
import os
from pathlib import Path

exp = Path(os.environ["EXP_DIR"])
results_dir = exp / "results"
dense = json.load(open(os.environ["DENSE_BASELINE_JSON"]))
abl = json.load(open(Path(os.environ["ABLATION_DIR"]) / "results" / "summary.json"))
rtn = abl["variants"]["full"]

lm = json.load(open(results_dir / "lm_eval_arc_mmlu.json"))
pro_files = sorted(glob.glob(str(results_dir / "mmlu_pro" / "reconstructed_model" / "results" / "results_*.json")))
if not pro_files:
    raise SystemExit(f"mmlu_pro results json not found under {results_dir}/mmlu_pro")
pro = json.load(open(pro_files[-1]))["results"]["mmlu_pro|0"]

table = [
    # (指标, dense, rtn, s0)
    ("ARC-E", dense["arc_easy_acc"], rtn["lm_eval"]["arc_easy"], lm["arc_easy"]),
    ("ARC-C", dense["arc_challenge_acc"], rtn["lm_eval"]["arc_challenge"], lm["arc_challenge"]),
    ("MMLU", dense["mmlu_acc"], rtn["lm_eval"]["mmlu"], lm["mmlu"]),
    ("MMLU-Pro300", dense["mmlu_pro_300_extractive_match"], rtn["mmlu_pro"]["mmlu_pro"], pro["extractive_match"]),
]

rtn_pro_file = rtn["mmlu_pro"].get("file")
rtn_pro_stderr = None
if rtn_pro_file and Path(rtn_pro_file).is_file():
    rtn_pro_stderr = json.load(open(rtn_pro_file))["results"]["mmlu_pro|0"].get("extractive_match_stderr")

lines = [
    "# Qwen3.5-4B S0 微调恢复实验汇总",
    "",
    "- 训练: QAD frozen-B + 只训 S0, S1K-1.1 完整轨迹 (max_len=32768 不截断), init=HiF4 RTN ckpt",
    "- 评测: lm_eval 0-shot ARC/MMLU (fake_act_quant=hif4, exclude=lm_head); lighteval mmlu_pro|0 max_samples=300 disable_thinking",
    f"- MMLU-Pro300 stderr: dense ±{dense['mmlu_pro_300_stderr']:.4f} / s0 ±{pro['extractive_match_stderr']:.4f}"
    + (f" / rtn ±{rtn_pro_stderr:.4f}" if rtn_pro_stderr is not None else ""),
    "",
    "| 指标 | dense BF16 | HiF4 RTN | RTN+S0 微调 | Δ(S0−RTN) | 恢复率 |",
    "|------|-----------:|--------:|------------:|----------:|-------:|",
]
for name, d, r, s in table:
    delta = s - r
    gap = d - r
    if abs(gap) < 5e-3:
        rec = "— (基线差<0.5pt)"
    else:
        rec = f"{delta / gap * 100:.1f}%"
    lines.append(f"| {name} | {d*100:.2f} | {r*100:.2f} | {s*100:.2f} | {delta*100:+.2f} | {rec} |")

out = results_dir / "summary_qad_4b.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nwritten: {out}")
PY

echo "[$(date --iso-8601=seconds)] 全部完成。日志: ${LOG}" | tee -a "${LOG}"
