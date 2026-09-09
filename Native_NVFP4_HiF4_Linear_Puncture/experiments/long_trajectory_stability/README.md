# Qwen3-30B-A3B NVFP4→HiF4 长轨迹稳定性诊断

本目录是**完全隔离的诊断实验包**。正式机制结论只来自：

```text
真实 vLLM TP2
  + single-request incremental KV-cache decode
  + E0 exact token trajectory 强制解码
  + worker 内 hook
```

旧 semantic / fused / TP2 / shared-attention replay **已删除，不再是正式路径**。  
历史教训见 `results/long_trajectory_stability/LEGACY_SEMANTIC_REPLAY_POSTMORTEM.md`。

## 方案

| ID | 方案 | runtime |
|---|---|---|
| E0 | native NVFP4 | Phase-A E0 vLLM NVFP4 emulation |
| E1 | Direct HiF4 | Phase-A ABI-3 HiF4 |
| E2 | R64-only | Phase-A ABI-3 HiF4 + R64 |
| E3 | adopted DIAG | Phase-A adopted DIAG + ABI-3 HiF4 |
| E4 | DIAG + R64 | Phase-A adopted DIAG + R64 + ABI-3 |

固定：`TP=2`，`KV=BF16`，`enforce_eager=true`，`max_num_seqs=1`，禁 prefix cache / speculative decoding。  
E1–E4 必须 `runtime_abi_version == 3`。不修改 `3rdparty/vllm/**` 与 production HiF4 kernel。

## 正式入口（唯一）

```bash
conda activate hif4

# helper tests
pytest -q Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability/real_vllm_hooks

# 1) shape-matched isolated free-run（一次一个 request）
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/run_isolated_free_run_matrix.py \
  --run_root Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_real_vllm_isolated \
  --variants E0 E1 E2 E3 E4

# 2) 生成 divergence / probe_plan
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/prepare_isolated_analysis.py \
  --run_root Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_real_vllm_isolated

# 3) forced-core hook capture（每 variant 独立进程）
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/run_hook_matrix.py \
  --run_root Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_real_vllm_hook \
  --variants E0 E1 \
  --mode forced_core \
  --probe_plan Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_real_vllm_isolated/analysis/probe_plan.json

# 4) 校验
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/validate_capture.py \
  --run_root .../E0/forced_core --variant E0 --mode forced_core --require_e0_target_top1

python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/compare_hook_states.py \
  --reference_root .../E0/forced_core \
  --variant_root .../E1/forced_core \
  --variant E1 \
  --output .../analysis/e0_e1_compare.jsonl
```

GPU 门禁：`gpu_pool.py` 默认只用 `PROJECT_GPU_POOL` 中 free ratio≥0.90 且 util≤10% 的卡；正式 TP2 不允许降门槛或改成 TP1。

## Canonical trajectory 规则

- 正式 forced target **必须**来自同 execution shape 的 isolated E0 free-run（`max_num_seqs=1`，一次一个 request）。
- 旧 lighteval batched free-run 只作历史对照；与 isolated 的 token 差异标记为 `EXECUTION_SHAPE_NUMERIC_DIFFERENCE`，不是 hook bug。

## Smoke 已验证项

- `LLM.apply_model` + `VLLM_ALLOW_INSECURE_SERIALIZATION` 把 hook 装到两个真实 TP worker；
- 位置 pre-hook 挂在外层 `Qwen3MoeForCausalLM`（内层 `@support_torch_compile` 会绕过 Module hook）；
- 48 层 core boundary 覆盖；rank0/rank1 artifact；forced exact；hook neutrality。

## 辅助保留

- `nvfp4_operator_parity/`：frozen-input 算子 parity（非正式长轨迹证据）；
- `capture_main.py` / `trajectory_io.py` / `build_probe_plan.py` / `compare_free_runs.py` / `gpu_pool.py`。

## 禁止

- 恢复 semantic simulator；
- 用 full-prefix / full-max-decode 替代 incremental decode；
- 为过 smoke 降低 GPU / parity 门禁；
- 修改 production runtime。
