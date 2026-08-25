# 梯度 DIAG 实验（同目录）

## 1. 逐通道 DIAG

`Y(d)=Linear(Q_H(X*d), Q_H(W_N/d))`

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/run.sh
```

## 2. H4 + 四元组共享 DIAG

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/run_h4_group.sh
```

## 3. 先 H4 再逐通道 DIAG

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/run_h4_channel.sh
```

## 4. 先逐通道 DIAG 再 H4

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/run_diag_then_h4.sh
```

## 5. 先 R64 再逐通道 DIAG

`R64=kron(kron(H4,H4),H4)/8`，G64 block-diag：

`Y=Linear(Q_H((X R)*d), Q_H((W_N R)/d))`

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/run_r64_channel.sh
```

## 6. 先逐通道 DIAG 再 R64

`Y=Linear(Q_H((X*d) R), Q_H((W_N/d) R))`

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/run_diag_then_r64.sh
```

## 7. 去掉 z∈[-4,4] 钳位消融

默认仍钳位；加 `--no-log2-clamp` 只关掉 `z.clamp_`（仍用 `d=2^z` 保正）。批量跑 5 套：

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/run_no_log2_clamp_ablation.sh
```

有约束对照（已有 formal，可复用）：

- 纯 DIAG：`20260815T063200Z_diag_gradient`
- 先 H4 后 DIAG：`20260815T125500Z_h4_channel_diag_gradient`
- 先 DIAG 后 H4：`20260817T033700Z_diag_then_h4_gradient`
- 先 R64 后 DIAG：`20260817T060700Z_r64_channel_diag_gradient`
- 先 DIAG 后 R64：`20260817T060700Z_diag_then_r64_gradient`

无约束 run id 后缀为 `*_nolimit`；`summary.json` 含 `log2_clamp`、`max_abs_log2_d`、`min_d`、`max_d`。

## 共享

- 数据：`captures/`（X）、`nvfp4_qdq/`（A_N）、ckpt（W_N）
- 工具：[`common.py`](common.py)；R64：[`r64_transform.py`](r64_transform.py)
- 改超参：`LR` / `STEPS` 环境变量
