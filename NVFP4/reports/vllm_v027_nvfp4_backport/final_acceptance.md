# vLLM v0.27.0 NVFP4 Emulation Backport — Final Acceptance

- Date (UTC): 2026-08-21
- Vendored vLLM HEAD (at start): `0cdc52596045680b7174f19befab726371fe1f5b`
- Upstream baseline: tag `v0.27.0` / SHA `4bdc8a788d2e2ce9165d552b3d4d8b72604626bf`
- Acceptance GPU: NVIDIA A800 80GB PCIe, TP=1

## Completion checklist (design §完成标准)

| # | Criterion | Status |
|---|---|---|
| 1 | Explicit `linear_backend=emulation` / `moe_backend=emulation` | **PASS** |
| 2 | Dense uses v0.27-style `EmulationNvFp4LinearKernel` | **PASS** |
| 3 | MoE uses `Nvfp4QuantizationEmulationTritonExperts` | **PASS** |
| 4 | ModelOpt + compressed-tensors share backend | **PASS** (contract tests) |
| 5 | Packed NVFP4 residency (no full BF16 Parameter) | **PASS** |
| 6 | Upstream scale/QDQ math unmodified | **PASS** (SM80 only selects existing Python path when Triton fp8e4nv unavailable) |
| 7 | No MoE double-QDQ | **PASS** (`expects_unquantized_inputs`) |
| 8 | A800 smoke BF16 KV + checkpoint/auto FP8 KV | **PASS** (eager) |
| 9 | No Marlin fallback on NVFP4 target layers in smoke | **PASS** |
| 10 | Layer puncture within upstream tolerance | **PASS** |
| 11 | MMLU-Pro 300 both KV modes recorded | **IN PROGRESS** (launcher started; see accuracy.md when done) |
| 12 | Existing `--fake_act_quant nvfp4` path intact + conflict with emulation | **PASS** |
| 13 | Diff limited to whitelist | **PASS** (see inventory) |

## Behavior matrix

| Checkpoint | Dense | MoE | KV=bfloat16 | KV=auto |
|---|---|---|---|---|
| ModelOpt NVFP4 | emulation (verified E2E smoke + puncture) | emulation (verified E2E smoke + puncture) | supported (eager smoke) | checkpoint-driven FP8 (`fp8_e4m3`, smoke) |
| compressed-tensors NVFP4 | emulation (unit/contract) | emulation (unit/contract) | supported (config/contract) | checkpoint-driven (config; **E2E 未用 CT checkpoint 跑 smoke**) |

## Directed tests (Task 13)

```text
conda run -n hif4 pytest -q \
  3rdparty/vllm/tests/kernels/quantization/test_nvfp4_quant.py \
  3rdparty/vllm/tests/kernels/quantization/test_nvfp4_emulation.py \
  NVFP4/tests/test_vllm_emulation_cli_plumbing.py \
  NVFP4/tests/test_vllm_nvfp4_backend_contract.py \
  3rdparty/vllm/tests/config/test_kernel_config.py
→ 64 passed, 14 skipped
```

## Known limitations

1. **SM80 Triton fp8e4nv:** dequant/QDQ Triton kernels require SM89+; A800 uses upstream Python reference path (same math).
2. **torch.compile / CUDA graph:** emulation path raises `Unsupported` under graph mode; formal runs must use `--enforce_eager`.
3. **MMLU-Pro 300:** long-running under emulation; script `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_mmlu_pro.sh` started on GPU0.

## Key reports

- Inventory: `upstream_inventory.md`
- Preflight: `qwen3_30b_preflight.md`
- Puncture: `qwen3_30b_puncture.md`
- Smoke: `qwen3_30b_smoke.md`
- Accuracy: `qwen3_30b_accuracy.md` (filled when MMLU finishes)
