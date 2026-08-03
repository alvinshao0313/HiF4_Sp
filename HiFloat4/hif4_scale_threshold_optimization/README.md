# HiF4 层级 Scale 与阶段阈值优化实验

本目录独立研究 HiF4 顶层 `S0`、每 8 元素指数 `e8` 和每 4 元素指数 `e4` 的生成策略，不修改 `HiFloat4` 现有量化实现。

## 环境

```bash
conda activate hif4
cd HiFloat4/hif4_scale_threshold_optimization
```

## 快速自检

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/ -v
```

## 分阶段脚本

| 阶段 | 命令 |
| --- | --- |
| Phase 2/3 合成+真实权重切片 | `python scripts/run_synthetic.py --model Qwen/Qwen3.5-4B --device cuda` |
| Phase 4 权重搜索（抽样） | `python scripts/run_weight_search.py --layers sample --budget fast` |
| Phase 4 全模型 | `python scripts/run_weight_search.py --layers all --budget fast --save-state` |
| Phase 5 激活采集 | `python scripts/collect_activation_stats.py --model Qwen/Qwen3.5-4B` |
| Phase 5 激活标定 | `python scripts/calibrate_activation_params.py --store <activation_store.pt>` |
| Phase 6 端到端 | `python scripts/evaluate_model.py --schemes baseline_standard,weight_fixed_best,weight_search_fast,act_calib_only,joint --weight-updates <pt> --act-param-map <pt>` |
| 一键流水线 | `python scripts/run_pipeline.py --stages synthetic,weight_all,act,e2e` |

端到端评测：WikiText2 PPL + ARC-e/c + MMLU（lm_eval / HF 可配阈值）+ MMLU-Pro（lighteval + vLLM，`max_samples=300`，`fake_act_quant=hif4`），不含 GSM8K。

## 目录

```text
configs/   基线与搜索范围
plans/     实验设计文档
src/       参考量化器、搜索、标定、评测
scripts/   可执行入口
tests/     对齐与正确性门控
results/   原始指标与 summary
```
