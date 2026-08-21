# Qwen3-30B-A3B-NVFP4 Preflight (Task 9)

- Checkpoint: `/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`
- Architecture: `Qwen3MoeForCausalLM` / `qwen3_moe`
- Layers/experts: hidden_layers=48, num_experts=128, topk=8
- Index tensor keys: 74835

## Quantization metadata

| Field | Value |
| --- | --- |
| quant_method | `modelopt` |
| quant_algo | `NVFP4` |
| group_size | `16` |
| kv_cache_quant_algo | `FP8` |
| exclude_modules | count=49; sample=['model.layers.0.mlp.gate', 'model.layers.1.mlp.gate', 'model.layers.10.mlp.gate', 'model.layers.11.mlp.gate', 'model.layers.12.mlp.gate'] |

## Dense NVFP4 sample (first complete layer from index)

- Prefix: `model.layers.0.self_attn.k_proj`

| key | shape | dtype | shard |
| --- | --- | --- | --- |
| `model.layers.0.self_attn.k_proj.weight` | `[512, 1024]` | `U8` | `model-00001-of-00004.safetensors` |
| `model.layers.0.self_attn.k_proj.weight_scale` | `[512, 128]` | `F8_E4M3` | `model-00001-of-00004.safetensors` |
| `model.layers.0.self_attn.k_proj.weight_scale_2` | `[]` | `F32` | `model-00001-of-00004.safetensors` |
| `model.layers.0.self_attn.k_proj.input_scale` | `[]` | `F32` | `model-00001-of-00004.safetensors` |

## Expert W13/W2 packed sample (+ scales)

- Layer `0`, expert `0` (checkpoint stores gate_proj + up_proj (fused to W13 at load))

### gate_proj (W13/gate)

| key | shape | dtype | shard |
| --- | --- | --- | --- |
| `model.layers.0.mlp.experts.0.gate_proj.weight` | `[768, 1024]` | `U8` | `model-00001-of-00004.safetensors` |
| `model.layers.0.mlp.experts.0.gate_proj.weight_scale` | `[768, 128]` | `F8_E4M3` | `model-00001-of-00004.safetensors` |
| `model.layers.0.mlp.experts.0.gate_proj.weight_scale_2` | `[]` | `F32` | `model-00001-of-00004.safetensors` |
| `model.layers.0.mlp.experts.0.gate_proj.input_scale` | `[]` | `F32` | `model-00001-of-00004.safetensors` |

### up_proj (W13/up)

| key | shape | dtype | shard |
| --- | --- | --- | --- |
| `model.layers.0.mlp.experts.0.up_proj.weight` | `[768, 1024]` | `U8` | `model-00001-of-00004.safetensors` |
| `model.layers.0.mlp.experts.0.up_proj.weight_scale` | `[768, 128]` | `F8_E4M3` | `model-00001-of-00004.safetensors` |
| `model.layers.0.mlp.experts.0.up_proj.weight_scale_2` | `[]` | `F32` | `model-00001-of-00004.safetensors` |
| `model.layers.0.mlp.experts.0.up_proj.input_scale` | `[]` | `F32` | `model-00001-of-00004.safetensors` |

### down_proj (W2)

| key | shape | dtype | shard |
| --- | --- | --- | --- |
| `model.layers.0.mlp.experts.0.down_proj.weight` | `[2048, 384]` | `U8` | `model-00001-of-00004.safetensors` |
| `model.layers.0.mlp.experts.0.down_proj.weight_scale` | `[2048, 48]` | `F8_E4M3` | `model-00001-of-00004.safetensors` |
| `model.layers.0.mlp.experts.0.down_proj.weight_scale_2` | `[]` | `F32` | `model-00001-of-00004.safetensors` |
| `model.layers.0.mlp.experts.0.down_proj.input_scale` | `[]` | `F32` | `model-00001-of-00004.safetensors` |

## Consistency gate vs plan assumptions

- Assumed: quant_method=`modelopt`, quant_algo=`NVFP4`, group_size=`16`, kv=`FP8`, ModelOpt keys `weight/weight_scale/weight_scale_2/input_scale`
- **PASS**: metadata consistent with plan; smoke may proceed.

## Notes

- Inspection used safetensors **headers only** (no full tensor payload load to CPU/GPU).
- Expert weights are stored per `gate_proj`/`up_proj`/`down_proj`; vLLM fuses gate+up into W13 at load time.
