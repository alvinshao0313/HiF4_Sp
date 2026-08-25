# Qwen3.5-4B Input-Only Dynamic Block Sparse E2E

实现与评测入口见计划：`Block_Sparse/plans/2026-08-07-qwen35-4b-input-only-dynamic-block-sparse-e2e-plan.md`。

## 包位置

- 运行时：`Block_Sparse/dynamic_input_sparse/`
- 单测：`Block_Sparse/tests/dynamic_input_sparse/`
- 本实验：本目录

## 快速开始

```bash
# 后台跑完整矩阵（关闭 Cursor 也可继续）
bash Block_Sparse/experiments/qwen35_4b_dynamic_input_sparse_e2e/scripts/launch_detached.sh

# 查看进度
tail -f Block_Sparse/experiments/qwen35_4b_dynamic_input_sparse_e2e/logs/run_full_detached_*.log
cat Block_Sparse/experiments/qwen35_4b_dynamic_input_sparse_e2e/results/LATEST_RUN_DIR.txt
```

空闲多卡：默认 `GPU_POOL=0,1,6,7`（可覆盖）。

## 协议摘要

| 任务 | 后端 | 备注 |
|---|---|---|
| ARC Easy/Challenge | lm_eval + HF `DynamicInputSparseMLPReference` | 0-shot, batch=8 |
| MMLU-Pro-300 | `main.py` vLLM+lighteval | `disable_thinking`, max_samples=300 |
| AIME25 avg@5 | `main.py` vLLM+lighteval | thinking on |

方法：DENSE / M8@{0.75,0.50,0.25} / M1@{0.75,0.50,0.25}；TP=1；`--enforce_eager`。
