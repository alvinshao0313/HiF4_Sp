# Inference Paradigm Conversion

Qwen3-8B `NVFP4-QAT → HiF4` 推理范式转换误差拆解与根因分析。

## 关键语义（必读）

正式主 checkpoint：

```text
Qmodel/Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-NoHadamard
```

**BF16 只是容器 dtype。** Source 语义是 `nvfp4_qat_fake_dequant_bf16`：

- 权重来自 NVFP4 QAT，数值已经反量化并以 BF16 保存；
- 不是普通 BF16 原模型，也不是 packed NVFP4 payload；
- `W_N = FP32(stored BF16 NVFP4-QAT value)`；
- Hadamard 已折叠进权重，推理阶段禁止再执行。

## 环境

```bash
/home/shaoyuantian/anaconda3/envs/hif4/bin/python
```

所有新增代码只在本目录；只读复用仓库现有 NVFP4 / HiF4 / MXFP8 实现。

## 快速开始

```bash
# F0 预检
bash Inference_Paradigm_Conversion/scripts/run_f0_preflight.sh

# 单元测试
/home/shaoyuantian/anaconda3/envs/hif4/bin/python -m pytest Inference_Paradigm_Conversion/tests -q

# Attention / Injection / Synthetic / Report / E2E（需空闲 GPU）
GPU_LIST=0,1,6,7 bash Inference_Paradigm_Conversion/scripts/run_attn_propagation.sh
GPU_LIST=0,1,6,7 MODE=n1_n2 bash Inference_Paradigm_Conversion/scripts/run_injection.sh
bash Inference_Paradigm_Conversion/scripts/run_synthetic.sh
bash Inference_Paradigm_Conversion/scripts/build_report.sh
GPU_LIST=6,7 PATH_ID=P1_semantic bash Inference_Paradigm_Conversion/scripts/run_e2e.sh
```

结果指针在 `results/latest_*_run_id.txt`；汇总报告见 `results/<report_id>/report.html`。

## Path IDs

`P1_semantic` / `P1_runtime` / `P2_matched_*` / `P2_deployment_*` / `W_storage_probe`

不同 `path_id` 禁止直接平均。
