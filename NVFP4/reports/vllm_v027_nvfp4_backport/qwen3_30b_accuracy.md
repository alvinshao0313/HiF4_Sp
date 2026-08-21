# Qwen3-30B-A3B-NVFP4 MMLU-Pro 300 (Task 12)

- Status: **IN PROGRESS**
- Started (local): 2026-08-21 ~03:25 UTC+8
- RUN_ID: `20260821T032536Z_nvfp4_emulation_mmlu_pro`
- Checkpoint: `/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`
- GPU: CUDA_VISIBLE_DEVICES=0 (A800 80GB), TP=1
- Backends: `linear_backend=emulation`, `moe_backend=emulation`, `enforce_eager=true`
- Script: `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_mmlu_pro.sh`

## Live evidence (from kv_bfloat16 log)

- Detected ModelOpt NVFP4 checkpoint
- `Using Nvfp4QuantizationEmulationTritonExperts MOE backend`
- `a13_scale = a13_scale.max()` / `a2_scale = a2_scale.max()` message present (upstream semantics)
- Evaluation started: 300 prompts rendered; generation in progress

## Results

| KV mode | run_ok | score_key | accuracy | notes |
|---|---|---|---|---|
| bfloat16 | pending | — | — | first run |
| auto (checkpoint → FP8) | pending | — | — | second run |

Absolute difference (auto − bf16): **pending**

This file will be overwritten by the script when both runs complete.
