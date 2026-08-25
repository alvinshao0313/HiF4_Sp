# Qwen3.5-4B HiF4 MLP Threshold-Gated Permutation Downstream Validation

该实验不判断 proxy 是否准确，只判断：
当仅保留 W4A4 层输出收益超过阈值的重排序层时，
Qwen3.5-4B 的下游任务准确率是否优于 W4A4 identity。

## 固定配置
- model: Qwen/Qwen3.5-4B
- conda env: hif4
- calibration: simplescaling/s1K-1.1_tokenized, 128 samples, full length
- activation rows: 512; weight rows: 512; search seed: 42
- 排序搜索只运行一次；所有阈值来自同一个 `candidate_permutations.pt`
- 量化: HiF4 W4A4 RTN（仅 lm_head 不量化），所有变体命令完全一致

## 阶段
1. `run_01_search_once.sh`：一次 32 层搜索，导出 selected 与 candidate 排列
2. `build_threshold_maps.py`：离线生成 tau ∈ {0, 0.25, 0.5, 1.0, 2.0}% 阈值映射
3. `run_02_quantize_variants.sh`：物化并对 identity/selected_default/5 个阈值模型做相同 RTN
4. `run_03_fast_eval.sh`：ARC-Easy/ARC-Challenge/PIQA 快速筛选（最多 2 个阈值晋级）
5. `run_04_full_eval.sh`：在未用于选阈值的任务（BoolQ/HellaSwag/WinoGrande/MMLU）上完整验证
6. `summarize_threshold_results.py`：汇总与最终报告

## 目录约定
所有输出只落在本目录下 `results/`、`logs/`、`tmp/`。
不覆盖 `qwen35_4b_perm_rtn`、`qwen35_4b_perm_s1k`、`qwen35_4b_perm_revalidation`。
