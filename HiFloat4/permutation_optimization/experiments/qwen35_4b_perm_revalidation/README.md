# Qwen3.5-4B HiF4 MLP Permutation Revalidation

本目录只用于修复后复验，不覆盖 qwen35_4b_perm_rtn、qwen35_4b_perm_s1k、qwen35_4b_perm_s1k_v2。

## 固定模型与环境
- model: Qwen/Qwen3.5-4B
- conda env: hif4
- dtype: bfloat16
- weight quant: hifx4 RTN
- activation quant: hifx4 fake quant
- calibration: simplescaling/s1K-1.1_tokenized
- calibration samples: 128
- activation rows: 512
- search seed: 42
- validation split seeds: 42,43,44

## 阶段
1. layer audit: layers 0, 15, 31
2. proxy audit: layers 0, 15, 31
3. full 32-layer search
4. BF16-only control
5. W4A4 paired evaluation

## 目录约定
所有输出只落在本目录下：
- `results/`：各阶段 JSON/JSONL/报告（运行时创建，不提交伪结果）
- `logs/`：各阶段运行日志
- `tmp/`：临时文件
