# AX1：HiF4 S0 除数 Oracle

## 1. 实验目标

判断固定 `amax/7` 是否适合真实激活；量化把 S0 除数调到 Oracle 后，相对 NVFP4 Source 能恢复多少 Linear 输出误差。

## 2. 实验设置

- 模型 checkpoint：`Qwen3-8B-FPQuant-QAT-NVFP4-Dequant-BF16-NoHadamard`
- run_id：`20260811T_ax_final_consolidated`
- 搜索：alpha∈[4,10]，步长 0.125，再对最优邻域精搜 33 点
- 样本行数：672
- 指标：逼近 NVFP4 的激活恢复率 / Linear 输出恢复率 R_Y

## 3. 实验结果

- 当前 alpha=7
- 最优 alpha：中位数=6.7656，p10=6.3438，p90=7.0938
- 平均 R_Y=0.0197，p50=0.0071，p90=0.0759
- 分投影：
- down_proj: alpha 中位数=6.5078, R_Y 均值=0.0515
- gate_proj: alpha 中位数=6.6719, R_Y 均值=0.0532
- k_proj: alpha 中位数=6.8984, R_Y 均值=0.0083
- o_proj: alpha 中位数=6.7656, R_Y 均值=0.0109
- q_proj: alpha 中位数=6.8984, R_Y 均值=-0.0002
- up_proj: alpha 中位数=6.6719, R_Y 均值=0.0090
- v_proj: alpha 中位数=6.8984, R_Y 均值=0.0050

## 4. 实验分析

若最优 alpha 稳定靠近 7 且 R_Y 很小，说明「除数位置」不是主因；若某些投影显著偏离 7 且 R_Y 较大，则存在 projection-specific 的 S0 定位问题。alpha_A*（逼近 NVFP4）与 alpha_X*（逼近 BF16）若不一致，说明转换目标与重建目标要求不同的 S0。

## 5. 实验结论

- S0 位置不是主要问题
- 全样本平均输出恢复率约为 1.97%
- 最优除数中位数相对 7 的偏移为 -0.234

## 6. 对算法设计的启示

只有在 R_Y 足够大时，才值得做低开销在线 S0 规则（AX5-R）；否则应优先其他机制（group / scale system / payload）。
