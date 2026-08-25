# Qwen3.5-4B HiF4 代理输入掩码反推消融

在固定第 15 层上投影线性层、固定 s1K 激活缓存上，比较 8 种「从输出块掩码反推输入 K 维块掩码」的方法。

**数据语义：** 每条样本 1024 token = 32 行块；正式实验 8 样本 → 激活 `[8192, 2560]`。  
**详细报告：** [`reports/paper_report_input_mask_proxy_ablation.md`](reports/paper_report_input_mask_proxy_ablation.md)  
**正式结果（八方法，含 M8）：** [`results/20260807T070223Z/`](results/20260807T070223Z/)  
**历史七方法：** [`results/20260807T025747Z/`](results/20260807T025747Z/)  
**历史六方法：** [`results/20260805T114616Z/`](results/20260805T114616Z/)

## 一句话结论（新 run `20260807T070223Z`）

- **推荐 / 精度最优：方案二**（相对方案一输入重叠率 **0.906**）
- **反推速度最优：方案七**（`mean(S0)`，约 **0.34 毫秒**）
- **方案八（去掉 MY 条件化）：** 与方案三重叠 **0.958**，相对方案一重叠仅低 **0.009**，NRMSE 差约 **+0.0011**；在输出保留 25% 时差距最大，75% 时几乎等价 → MY 条件化价值与输出稀疏度相关（Case C）
- 方案八反推仅约 **1.04×** 快于方案三；`online_total` 几乎不变。去掉 MY 不等于可以删掉 proxy-output GEMM

## 八种方法

| 编号 | 中文含义 | 输出掩码 | 激活/贡献统计 | 权重统计 | 反推 |
|---|---|---|---|---|---|
| 方案一 | 全精度精确对照 | 真实输出 | 真实部分积 | 真实 W | 精确 |
| 方案二 | 只代理激活 + 自有输出 + 精确 | 单代理输出 | Xp 部分积 | 真实 W | 精确 |
| 方案三 | 只代理激活 + 自有输出 + 能量 | 单代理输出 | `mean(Xp²)` | 离线 `mean(W²)`，**乘 MY** | 能量 |
| 方案四 | 真实输出 + 真实能量 | 真实输出 | `mean(X²)` | 离线 `mean(W²)` | 能量 |
| 方案五 | 真实输出 + 双代理精确 | 真实输出 | 双代理部分积 | Wp | 精确 |
| 方案六 | 双代理全链路 | 双代理输出 | 双代理部分积 | Wp | 精确 |
| 方案七 | 只代理激活 + S0 均值能量 | 单代理输出 | `mean(S0)` | 离线 `mean(W²)`，乘 MY | S0 均值能量 |
| 方案八 | 只代理激活 + 无 MY 能量 | 单代理输出（最终仍用） | `mean(Xp²)` | 离线 `sum_j mean(W²)`，**不乘 MY** | 无条件能量 |

### 方案八公式（唯一相对方案三的差异）

方案三：`Score = E_Xp(i,k) * sum_j MY_xp(i,j) E_W(j,k)`  
方案八：`Score = E_Xp(i,k) * sum_j E_W(j,k)`  

- 方案二/三/七/八共享同一份 `MY_xp`（最终 joint sparse 仍用它）。
- 「去掉 MY」只作用于**输入评分**，不是删除输出稀疏。
- 固定输入保留率时，方案八的 `MX` 对三个输出保留率 bitwise 相同。

## 命令

```bash
# 冒烟（results/smoke_s0mean/<run_id>/）
bash Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/scripts/run_smoke.sh

# 完整实验
DEVICES=0,1,6,7 \
bash Block_Sparse/experiments/qwen35_4b_input_mask_proxy_ablation/scripts/run_full.sh
```

环境：`hif4` conda，需要 GPU。

## 产物

| 文件 | 用途 |
|---|---|
| `reports/paper_report_input_mask_proxy_ablation.md` | 详细实验报告 |
| `results/<run_id>/manifest.json` | 元数据与门禁 |
| `condition_summary.csv` | 条件汇总（完整实验 72 行） |
| `latency.csv` | 耗时（完整实验 329 行，冒烟 41 行） |
| `aggregate_summary.json` | 优胜者 / 帕累托 / `m7_vs_m3` / `m8_vs_m3` |
| `results/captured/layer15_up_proj_s8_t1024.pt` | 冻结缓存 |

## 重要限制

- 原型延迟 ≠ 最终融合稀疏 HiF4 算子延迟。
- 方案八不宣称删除 proxy-output GEMM。
- 不做完整模型下游评测。
- 不同 run 的延迟不得混表。
