# HiF4 S0/e8/e4 阈值优化实验记录

- 日期：2026-07-30
- 模型：`Qwen/Qwen3.5-4B`
- 设备：NVIDIA A800 80GB（CUDA），conda 环境 `hif4`
- 代码：`HiFloat4/hif4_scale_threshold_optimization/`（未改动公共 HiFloat4 接口）
- 计划文档：[`plans/2026-07-30-hif4-scale-threshold-optimization-experiment-plan.md`](../plans/2026-07-30-hif4-scale-threshold-optimization-experiment-plan.md)

## 结果目录索引

| 内容 | 路径 |
| --- | --- |
| Phase2/3 合成分布基线与联合网格 | `results/20260730_044400_phase2_phase3_synthetic/` |
| Phase4 权重逐组搜索（抽样/全模型） | `results/20260730_phase4_weight_sample/`、`results/20260730_phase4_weight_all/` |
| Phase5 激活采集与标定（WikiText） | `results/20260730_phase5_act_stats/`、`results/20260730_phase5_act_calib/` |
| Phase6 端到端五方案 | `results/20260730_phase6_e2e/` |
| S1K 激活采集与标定 | `results/20260730_s1k_act_stats/`、`results/20260730_s1k_calib/` |
| 激活诊断（泛化/目标函数/逐 block/ oracle） | `results/20260730_act_diagnosis/` |
| 跨域标定检验（WikiText ↔ S1K） | `results/20260730_cross_domain/` |
| 权重量化耗时 benchmark | `results/20260730_weight_time_bench/` |
| reasoning 评测（AIME25 avg@5 + LCB v6） | `results/20260730_phase6_e2e/<scheme>/reasoning/` |
| 汇总报告（自动生成） | `results/20260730_final_report/summary.md` |

## 实验设置

- 权重搜索预算 `fast`：S0 offset ∈ {-1,0,+1}，e8/e4 精确 8 组合；`full`：S0 offset ∈ {-2..+2}
- 激活标定网格：d ∈ [5.5,7.5] step 0.25；t8 ∈ [3.4,4.1] step 0.1；t4 ∈ [1.70,2.05] step 0.05
- 端到端：WikiText2 PPL（seqlen 2048）+ ARC-e/c + MMLU（lm_eval，0-shot）+ MMLU-Pro（lighteval+vLLM，`max_samples=300`，`fake_act_quant=hif4`，`--disable_thinking`）
- 评测路径：PPL/ARC/MMLU 走本目录 HF fake-quant（支持可配激活阈值）；MMLU-Pro 走 vLLM，激活只能是标准 (7,4,2)

## 核心结论（重构误差）

### 固定阈值三基线（合成分布 NMSE）

| distribution | standard (7,4,2) | scalar_mse (3.75,1.875) | no_clip (3.5,1.75) |
| --- | ---: | ---: | ---: |
| gaussian | 6.905e-03 | 7.070e-03 | 7.792e-03 |
| laplace | 7.984e-03 | 8.176e-03 | 9.028e-03 |
| student_t3 | 8.852e-03 | 9.004e-03 | 9.707e-03 |
| outlier_0p1pct_20x | 8.348e-03 | 8.463e-03 | 8.981e-03 |
| phase_boundary | 3.095e-03 | 3.095e-03 | 3.095e-03 |

`scalar_mse` 与 `no_clip` **均未**稳定优于 standard；联合网格最优多落在 `(d,t8,t4)≈(7.0, 3.9~4.0, 1.95)`，相对 standard 增益很小。

### 权重逐组搜索（全模型 128 层，fast）

- mean NMSE：standard `6.966e-03` → 只搜 S0 `6.491e-03` → S0+e8/e4 `6.395e-03`
- **收益主要来自 S0 邻域搜索**（约 83%），e8/e4 精确枚举额外贡献更小（约 17%）
- 局部枚举 MSE 不高于同 S0 下标准阈值（验收通过）

### 激活离线标定（WikiText，128 层）

- 94/128 层验证集对角近似输出 MSE 优于 standard
- 参数集中在 `(7.0, 3.9~4.0, 1.95)`；平均验证集改善 `2.58e-02`（相对量级约 0.2%）

## 端到端结果（Phase6）

| scheme | PPL ↓ | ARC-e | ARC-c | MMLU | MMLU-Pro(300) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_standard | 10.2546 | 0.7984 | 0.5094 | 0.7276 | 0.6967 |
| weight_fixed_best (7.0,3.9,1.95) | 10.1796 | 0.8001 | 0.5026 | 0.7260 | 0.7033 |
| weight_search_fast | **9.9694** | **0.8018** | 0.5017 | 0.7247 | 0.6867 |
| act_calib_only（WikiText 标定） | 10.2984 | 0.7946 | 0.5077 | 0.7258 | 0.6967 |
| act_fixed_best（激活全局 7.0,3.9,1.95） | 10.2599 | 0.8009 | 0.5060 | 0.7242 | 0.6967 |
| act_calib_s1k（S1K 标定） | 10.2638 | 0.8026 | 0.5009 | 0.7272 | 0.6967 |
| joint（权重搜索+激活标定） | 10.0018 | 0.8005 | 0.5068 | 0.7215 | 0.7033 |

注：MMLU-Pro 采样温度 0.7，300 样本下 stderr≈0.027，个位数百分比差不宜过度解读。

## 追加分析一：激活标定为什么没有下游收益

诊断代码：`scripts/diagnose_activation.py`；结果：`results/20260730_act_diagnosis/`。

**A. 泛化**：94/128 层验证集改善，12 层校准改善但验证未改善 —— 泛化基本成立，不是主因。

**B. 目标函数**：抽 8 层对比对角近似 `Σ||W[:,j]||²(x−xq)²` 与真实 `||(X−Xq)W||²` 选参：5/8 层选参完全一致；不一致层在真实目标下差距也小于 1%。更关键：两种目标下**所有候选配置的输出 MSE 曲线都非常平**（如 `layers.27.q_proj`：diag pick 2.6573e3 vs real pick 2.6714e3 vs standard 2.7037e3，全程差距 <2%）。目标函数不是主因，**主因是激活量化对 (d,t8,t4) 本身不敏感**。

**C/D. NMSE 阶梯**（验证集，128 层平均）：

| 方案 | mean NMSE | mean 加权输出 MSE |
| --- | ---: | ---: |
| standard (7,4,2) | 7.991e-03 | 7.658e+02 |
| per-layer 标定 | 8.024e-03（更差） | 7.642e+02（-0.2%） |
| per-block 标定 | 8.081e-03（更差） | 7.627e+02（-0.4%） |
| oracle（每行每 block 在线搜索） | **7.303e-03（-8.6%）** | **6.978e+02（-8.9%）** |

结论：
1. 逐层/逐 block **离线**固定参数只能拿到 oracle 收益零头（<5%），因为收益来自「适配每一行 block 的实际内容」，离线参数做不到；
2. 离线参数还轻微**恶化**激活 NMSE（标定目标是加权输出 MSE，两者方向不一致），且逐 block 参数更多、过拟合校准行更明显（per-block 比 per-layer NMSE 更差）；
3. 权重量化误差（NMSE≈6.4~7.0e-3）与激活误差同量级，W4A4 下激活侧 0.2% 的改善被淹没，下游自然无感。

## 追加分析二：逐 block 离线记录参数值得做吗

不值得。上面阶梯表中的 per-block 行就是该方案：每层每 block 独立 (d,t8,t4)，存储量每层 40~152 组参数，推理零额外开销，但验证集加权输出 MSE 仅再降 0.4%、NMSE 反而更差。真正的收益上限在 oracle（在线逐 block 搜索，-8.6% NMSE），代价是推理时对每个 64-block 枚举候选——正是计划禁止的在线搜索，且 vLLM/HF 推理路径都不支持。

## 追加分析三：换 S1K 长序列校准集更好吗

S1K（`simplescaling/s1K-1.1_tokenized`，全长不截断，48 条）采集 128 层激活并逐层标定，与 WikiText 标定对比：

**参数分布几乎一致**（top 参数均为 `(7.0,4.0,1.95)`/`(7.0,3.9,1.95)`，占 ~80% 层）。

**跨域验证**（`results/20260730_cross_domain/`）：

| 验证域 | standard | 本域标定 | 跨域标定 |
| --- | ---: | ---: | ---: |
| WikiText（NMSE） | 7.991e-03 | 8.024e-03 (wt) | 8.041e-03 (s1k) |
| S1K（NMSE） | 8.106e-03 | 8.169e-03 (s1k) | 8.153e-03 (wt) |

本域标定在本域验证集上只有约 0.2% 的加权 MSE 优势，跨域后基本消失甚至为负。**校准集换 S1K 不改变结论**——因为两域激活分布对阈值的敏感度本来就低。S1K 标定版全套 e2e（上表 `act_calib_s1k` 行）：PPL 10.2638，介于 wikitext 标定（10.2984）与 baseline（10.2546）之间，ARC/MMLU/MMLU-Pro 与 baseline 打平——端到端同样无收益。

## 追加分析四：权重量化搜索的时间开销

`scripts/bench_weight_quant_time.py`，128 层目标 Linear，GPU 单次离线量化：

| 方案 | 总耗时 | 相对 standard |
| --- | ---: | ---: |
| standard 解析阈值 | 3.31 s | 1.0x |
| 搜索 fast（S0±1 × 8 组合） | 14.48 s | 4.4x |
| 搜索 full（S0±2 × 8 组合） | 18.15 s | 5.5x |

搜索是纯离线一次性成本：全模型多花约 **11 秒**（fast），换来 PPL 10.25→9.97。注：benchmark 原始输出覆盖 safetensors 中全部 135 个匹配 Linear（含 7 个 `mtp.*` 层）；上表已过滤为与实验对齐的 128 层口径，逐层数据见 `results/20260730_weight_time_bench/raw_metrics.json`。

## 追加分析五：权重 scale 搜索在 reasoning 任务上有效吗

三方案复用 Phase6 的权重量化 checkpoint，vLLM + lighteval 评测（thinking 开启，temperature 0.7，`fake_act_quant=hif4` 标准 (7,4,2) 激活）。AIME25 为 avg@5（30 题 × 5 采样）；LiveCodeBench 为 `lcb:codegeneration_v6`（175 题 × 16 采样估计 pass@1）；MMLU-Pro 为 300 样本。

| scheme | AIME25 avg@5 | LCB v6 pass@1 | MMLU-Pro(300) |
| --- | ---: | ---: | ---: |
| baseline_standard（RTN） | 0.3867 ± 0.062 | 0.1657 ± 0.028 | 0.6967 ± 0.027 |
| weight_fixed_best | 0.3800 ± 0.068 | **0.1829** ± 0.029 | **0.7033** ± 0.026 |
| weight_search_fast | **0.3933** ± 0.070 | **0.1829** ± 0.029 | 0.6867 ± 0.027 |

结论：
1. **三项指标的方案间差异全部落在各自标准误范围内**，权重 scale 搜索在 reasoning 任务上没有统计显著的收益；PPL 的明显改善（-0.29）不转化为 reasoning 精度。
2. LCB 上 fixed_best 与 search 同为 0.1829（32/175），比 baseline 高 1.7pt，方向上与 PPL 一致但幅度仅约 0.6 个标准误，只能算弱信号。
3. AIME 上 search 最高（0.393）、fixed 最低（0.380），进一步说明单次采样噪声（±0.06~0.07）远大于方案差异。
4. 已知瑕疵：每方案有 2 个超长样本触发 `context_size + max_new_tokens > 32768`，prompt 被截到 0 token，对 LCB/AIME 分数有轻微负面影响，三方案同条件，相对比较公平。

综合 Phase6 与本节：**权重搜索的收益确定体现在语言建模质量（PPL）上，对选择题/推理类下游在当前 4B 模型与采样预算下观察不到稳定提升；如要采信 LCB 的 +1.7pt，需要更大采样预算复测。**

## 回答计划中的四个问题

1. `(3.75,1.875)` **未**稳定优于 `(4,2)`；略降到 `(3.9,1.95)` 有极小收益。
2. `(3.5,1.75)`（no_clip）在合成分布上更差：过早放大 scale 增加范围内舍入误差。
3. 权重逐组联合搜索稳定降低重构 NMSE（约 8% 相对），**主要来自 S0**；e2e 上 PPL 明显改善（-0.29），下游选择题指标基本打平。
4. 激活离线标定在校准/验证输出误差上有约 0.2% 改善，但**不转化为下游收益**，原因见追加分析一。后补的 `act_fixed_best`（激活全局 `(7.0,3.9,1.95)`）e2e 同样与 baseline 打平（PPL 10.2599 vs 10.2546），至此激活侧所有离线方案（全局固定 / 逐层 WikiText 标定 / 逐层 S1K 标定）均已验证无下游收益。

## 复现指引

```bash
# 激活诊断
python scripts/diagnose_activation.py \
  --store results/20260730_phase5_act_stats/activation_store.pt \
  --calib-dir results/20260730_phase5_act_calib \
  --out-dir results/20260730_act_diagnosis
# S1K 采集 + 标定
python scripts/collect_activation_stats.py --dataset s1k --seqlen 0 \
  --n-samples 48 --max-rows 512 --max-rows-per-batch 32 \
  --out-dir results/20260730_s1k_act_stats
python scripts/calibrate_activation_params.py \
  --store results/20260730_s1k_act_stats/activation_store.pt \
  --granularity per_layer --out-dir results/20260730_s1k_calib
# 跨域检验
python scripts/cross_domain_calib_check.py \
  --wt-store results/20260730_phase5_act_stats/activation_store.pt \
  --s1k-store results/20260730_s1k_act_stats/activation_store.pt \
  --wt-map results/20260730_phase5_act_calib/param_map_per_layer.pt \
  --s1k-map results/20260730_s1k_calib/param_map_per_layer.pt \
  --out-dir results/20260730_cross_domain
# 权重耗时 benchmark
python scripts/bench_weight_quant_time.py --out-dir results/20260730_weight_time_bench
# reasoning 评测（AIME25 avg@5 + LCB v6，复用 Phase6 ckpt）
python scripts/run_reasoning_eval.py --e2e-root results/20260730_phase6_e2e \
  --schemes baseline_standard --gpu 5
```
