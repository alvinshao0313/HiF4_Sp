# Qwen3-30B-A3B-NVFP4 Puncture (Task 10)

- Checkpoint: `/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`
- Device: `cuda`
- Overall: **PASS**

## Dense

- Layer prefix: `model.layers.0.self_attn.q_proj`
- Weight: shape=`[4096, 1024]`, dtype=`uint8`
- Activation: BF16 seed=0, shape=`[4, 2048]`
- Kernel: `EmulationNvFp4LinearKernel` vs `run_nvfp4_emulations`
- SM80 Python fp8e4nv path expected: `True`
- Tolerance: atol=0.0, rtol=0.0
- Metrics: max_abs=0, mean_abs=0, rel_l2=0, nmse=0
- Finite: `True`
- Result: **PASS**

## MoE (W13 → silu-and-mul → W2)

- Layer `0`, experts `[0, 1]`
- W13 shape `[2, 1536, 1024]`, W2 shape `[2, 2048, 384]`
- Tokens=4, top_k=2
- Experts class: `Nvfp4QuantizationEmulationTritonExperts` vs torch reference
- SM80 Python fp8e4nv path expected: `True`
- Tolerance: atol=0.05, rtol=0.01
- Metrics: max_abs=0.015625, mean_abs=0.000690698, rel_l2=0.00336843, nmse=1.13463e-05
- Finite: `True`
- Result: **PASS**

## Gate

- Dense and MoE both passed upstream tolerance → E2E smoke allowed.
