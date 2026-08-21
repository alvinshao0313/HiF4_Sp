# Qwen3-30B-A3B-NVFP4 Smoke (Task 11)

- Checkpoint: `/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`
- Final TP: **1** (no OOM; TP=1 sufficient on A800-80G)
- CUDA_VISIBLE_DEVICES: `0`
- GPU: `NVIDIA A800 80GB PCIe`
- max_model_len=512, max_new_tokens=16
- Backends requested: linear=emulation, moe=emulation
- Entry: `NVFP4/scripts/run_qwen3_30b_nvfp4_emulation_smoke.py` (same kwargs as root `main.py`)
- Note: `kv_cache_dtype=bfloat16` requires Attention to pass kernel string `auto` while allocating BF16 tensors (local CUDA DISPATCH only accepts auto/fp8*).

## Runs

| tag | passed | linear | moe | kv resolved | fp8_kv | no_marlin | error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `eager_auto_tp1` | True | `emulation` | `emulation` | `fp8_e4m3` | True | True | `None` |
| `eager_bf16_tp1` | True | `emulation` | `emulation` | `bfloat16` | None | True | `None` |
| `graph_auto_tp1` | False | `None` | `None` | `None` | None | None | `torch._dynamo.exc.Unsupported: Unsupported method call` |
| `graph_bf16_tp1` | False | `None` | `None` | `None` | None | None | `torch._dynamo.exc.Unsupported: Unsupported method call` |

## Acceptance

- Eager BF16 KV: **PASS**
- Eager checkpoint/auto KV: **PASS**
- BF16 mode resolved cache_dtype: `bfloat16`
- auto KV resolved dtype: `fp8_e4m3` (fp8_kv=True)
- Graph-mode `graph_auto_tp1`: FAIL (documented upstream limitation)
  - Root cause: `torch._dynamo.exc.Unsupported` inside `EmulationNvFp4LinearKernel` → `run_nvfp4_emulations` / `_nvfp4_triton_fp8e4nv_supported` under compile/CUDA graph. Do **not** fall back to Marlin; use `enforce_eager=true` for formal runs.
- Graph-mode `graph_bf16_tp1`: FAIL (documented upstream limitation)
  - Root cause: `torch._dynamo.exc.Unsupported` inside `EmulationNvFp4LinearKernel` → `run_nvfp4_emulations` / `_nvfp4_triton_fp8e4nv_supported` under compile/CUDA graph. Do **not** fall back to Marlin; use `enforce_eager=true` for formal runs.

### Sample generations (`eager_auto_tp1`)

- prompt='Hello' -> ' of the 1000000000000'
- prompt='1+1=' -> '2, 2+2=4, 4+4=8,'
- prompt='The capital of France is' -> ' Paris. Which of the following is the capital of the United Kingdom? A.'
- prompt='Write one word: ok' -> ", I'm going to be honest, I'm not sure if I'm doing"

## Backend log evidence

- Log contains `Using EmulationNvFp4LinearKernel for NVFP4 GEMM`
- Log contains `Using Nvfp4QuantizationEmulationTritonExperts MOE backend`
- No Marlin fallback warning for NVFP4 target path in eager BF16 log

## Logs

- Directory: `/home/shaoyuantian/program/HiF4_Sp/NVFP4/reports/vllm_v027_nvfp4_backport/smoke_logs`

