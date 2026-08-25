# Qwen3.5-4B：MLP 层级排序 + HiF4 W4A4 RTN 对比

对比：

| 变体 | 含义 |
|------|------|
| `rtn_baseline` | 不排序直接 W4A4 RTN（引用 `HiF4_exp/qwen35_4b_w4a4_proj_ablation` 的 `full`） |
| `perm_rtn` | 本目录搜索的 MLP 中间维层级排列 → 再 W4A4 RTN |

## 运行

冒烟（前/中/后各一层）：

```bash
GPUS=4 bash run_smoke_layers.sh
```

全量评测：

```bash
GPUS=4 bash run_perm_rtn_eval.sh
```

结果写在 `results/`。

## 搜索耗时（工程加速后）

`d_ff=9216, rows=512, refine_passes=0` 单层大约：G4 ~45s / G8 ~1s / G64 ~1s / 全层 `optimize` ~47s（相对加速前 ~119s，约 2.5×）。G8/G64 稠密惩罚矩阵走 CUDA；G4 beam 仍以 CPU 为主。
