# H4 4维 Hadamard 块旋转实验

验证固定 `R4 = H4 / 2` 的 G4 块旋转，会不会在不改变 Linear 浮点结果的前提下，降低保存激活转到 HiF4 的量化误差，以及同步旋转权重后 Linear 输出误差会不会一起降。

本实验不做端到端任务，不重新 capture，不搜索旋转矩阵，不改现有 DIAG / HiF4 数学。

## 现有数据语义

审计对象：`Native_NVFP4_HiF4_Linear_Puncture` 已完成的 Linear puncture 正式 run。

- **capture run**：`20260812T103800Z_native_nvfp4_hif4_linear_puncture`
- **捕获点**：`X_rot`，checkpoint 指定的 16×16 `forward_hadamard_matrix` 作用在最后一维之后、NVFP4 activation quant 之前。张量名 `x_rot_bf16`，dtype=BF16。
- **不允许撤销**这段在线 rotation。H4 是叠在 `X_rot` 坐标系上的第二次、按 HiF4 G4 对齐的 4×4 旋转。
- **HiF4 转换源**与现有 E4/E5 相同：对 `X_rot` 做 `Q_HiF4`，不是对 `A_N` 再转一次。
- **权重源**：packed NVFP4 `qweight + scales + weight_global_scale` 反量化得到的 `W_N`。Identity 对 `W_N` 做 `Q_HiF4`；H4 对 `W_N R` 做同样的 `Q_HiF4`。
- **Linear 对照**与 `linear_cases.py` 相同：`Y_NN = Linear(A_N, W_N, bias)`，其中 `A_N = Q_NVFP4(X_rot)`。不是另造一套 BF16 `Linear(X_rot, W_N)` 当主 reference。
- **DIAG 对照**：直接读取该 capture run 的 `diagonal_scales/*.pt`（cal 上搜好的 `d`），validation 上评估 `Q_H(X_rot/d)` 与 `Q_H(W_N*d)`。不重新搜索，不扩大搜索空间。
- **评估 split**：`val`，与现有 E4/E5 相同。H4 本身不拟合数据；用 val 才能和 DIAG 比。
- **模块列表**：config 正式层 `[2,10,18,26,34]` × 7 projection = 35。找不到保存激活或 DIAG 文件就直接报错，禁止静默 recapture。

Identity HiF4 必须和 `formats.qdq_hif4_direct` 在同一 tensor 上逐元素一致。Full run 还会把 Identity / DIAG 的 Linear output NMSE 对上现有 `linear_results.csv` 的 `E4_WH_AH_RTN` / `E5_WH_AH_DIAG`。

## 四组对照

| Case | 激活 | 权重 |
|---|---|---|
| Identity | `Q_H(X_rot)` | `Q_H(W_N)` |
| DIAG | `Q_H(X_rot / d)` | `Q_H(W_N * d)` |
| H4_FP32 | `Q_H(X_rot R)` | `Q_H(W_N R)` |
| H4_BF16 | 先把激活/权重收到 BF16，再做同一个 G4 变换，再 HiF4 | 同左 |

`R` 是 `R4=H4/2` 的 block-diag，严格对齐连续 4 元素，不跨 G4、不跨 64-group。主结论只用 H4_FP32；H4_BF16 只回答低精度在线实现后收益还在不在。

## 运行

必须在仓库根目录、`hif4` conda 环境。

```bash
# smoke：第一个 Linear（layer02 q_proj）的 val 前 256 行
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/h4_block_rotation/run_smoke.sh

# full：35 个 Linear 的全部 val 样本
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/h4_block_rotation/run_full.sh
```

结果写到：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/h4_block_rotation/<run_id>/
```

## 输出

- `config.json` / `resolved_inputs.json`
- `group_metrics.csv` / `layer_metrics.csv` / `cf4_hist.csv`
- `summary.json` / `report.md`
- `figures/fig01` … `fig06`
