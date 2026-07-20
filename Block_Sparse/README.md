# MLP 梯度块稀疏使用说明

可对 Qwen3.5-27B 的 MLP（`gate_proj` / `up_proj` / `down_proj`）做结构化块剪枝，导出标准 HF 权重，并用仓库根目录 `main.py`（vLLM + lighteval）做评测。

**环境：一律使用 conda 环境 `hif4`。**

理论细节见同目录设计文档  
`mlp_128x128_gradient_block_pruning_design_fixed.md`。

---

## 目录与产物

```text
Block_Sparse/
├── block_pruning/          # 核心库（打分、分配 mask、置零、导出）
├── scripts/
│   ├── prune_mlp.sh             # 独立剪枝（推荐日常改参数后执行）
│   ├── score_and_prune_mlp.py   # 剪枝 Python 入口
│   ├── eval_pruned.sh           # 评测单个剪枝模型
│   └── run_baselines.sh         # fisher/magnitude/random/fisher_budget_wanda 批跑 + 评测
├── tests/                  # 单元测试
├── experiments/
│   └── wikitext2_calib/    # WikiText-2 校准阶段归档（ckpt + 评测 + report.html）
├── outputs/                # 新实验剪枝模型（tag 含校准集后缀，如 _s1k）
└── results/                # 新实验评测结果
```

剪枝成功后，每个 `output_dir` 大致包含：

| 路径 | 含义 |
|------|------|
| `output_dir/` | 标准 HF 模型目录（`config.json`、权重、tokenizer），可直接给 vLLM / `main.py` |
| `output_dir/pruning_artifacts/block_scores.pt` | 各 MLP 矩阵的块分数（普通方法） |
| `output_dir/pruning_artifacts/block_masks.pt` | 块 mask（`True`=保留，`False`=已剪） |
| `output_dir/pruning_artifacts/pruning_summary.json` | 稀疏率、配置摘要 |
| `output_dir/pruning_artifacts/per_matrix_report.csv` | 逐矩阵稀疏率与 score 统计 |

`fisher_budget_wanda` 额外产物：

| 路径 | 含义 |
|------|------|
| `fisher_block_scores.pt` / `wanda_block_scores.pt` | 两阶段各自的块分数 |
| `fisher_reference_masks.pt` | Fisher 预算参考 mask（**不写权重**） |
| `module_prune_budget.csv` | 每矩阵 Fisher 预算与最终剪块数 |
| `hybrid_per_matrix_report.csv` | Fisher/Wanda 分数统计与 mask IoU |

---

## 1. 独立剪枝脚本：`prune_mlp.sh`（推荐）

风格与 `scripts/test.sh` 一致：先 `conda activate hif4`，在脚本顶部改参数后执行。

```bash
conda activate hif4
bash Block_Sparse/scripts/prune_mlp.sh
```

顶部常用项：

| 变量 | 含义 |
|------|------|
| `MODEL_PATH` | 被剪模型 |
| `SCORE_TYPE` | `fisher` / `magnitude` / `random` / `fisher_budget_wanda` |
| `SPARSITY` | 目标块稀疏率 |
| `BLOCK_SIZE` | 块尺寸：`128` 或 `64x128`（见下） |
| `CALIBRATION_DATASET` | `s1k` / `wikitext2` / `c4` / `ptb`（`fisher` / `fisher_budget_wanda`） |
| `CALIB_SAMPLES` / `SEQ_LEN` / `SEED` / `DTYPE` | 校准与精度；`SEQ_LEN=0` 表示 s1k 不截断 |
| `OUTPUT_DIR` | 剪枝输出目录 |

### `--block_size` / `BLOCK_SIZE` 写法（一个参数控制长宽）

| 写法 | 含义 |
|------|------|
| `128` | 正方形：高=128、宽=128 |
| `64x128` | 矩形：高=64（沿权重 `d_out`）、宽=128（沿 `d_in`） |

`d_out` 必须能被高整除，`d_in` 必须能被宽整除，否则直接报错。

---

## 2. Python 入口：`score_and_prune_mlp.py`

`prune_mlp.sh` 最终调用的就是它；也可以直接：

```bash
conda activate hif4
python Block_Sparse/scripts/score_and_prune_mlp.py \
  --model_path Qwen/Qwen3.5-27B \
  --output_dir Block_Sparse/outputs/qwen35_27b_fisher_s0.3 \
  --score_type fisher \
  --target_block_sparsity 0.30 \
  --block_size 128 \
  --calibration_dataset s1k \
  --sequence_length 0
```

### 推荐顺序

1. 先跑 **`magnitude` / `random`**（不用反向，验证导出与 vLLM 加载）
2. 再跑 **`fisher`**（全模型 LM loss 反向，显存与时间更大）
3. 对比 **`fisher_budget_wanda`**（Fisher 定预算 + Block-Wanda 选坐标）

### 参数说明

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `--model_path` | `Qwen/Qwen3.5-27B` | 模型路径或 HF id（可用本地 cache） |
| `--output_dir` | `Block_Sparse/outputs/qwen35_27b_fisher_s0.3` | 输出目录；相对路径相对**仓库根** |
| `--score_type` | `fisher` | `fisher` / `magnitude` / `random` / `fisher_budget_wanda` |
| `--target_block_sparsity` | `0.30` | 全局目标块稀疏率 |
| `--block_size` | `128` | 块尺寸：`128` 或 `HxW`（如 `64x128`）；不可整除则报错 |
| `--calibration_dataset` | `wikitext2` | 校准数据：`s1k` / `wikitext2` / `c4` / `ptb`（`fisher` 与 `fisher_budget_wanda`） |
| `--calibration_samples` | `128` | 校准样本条数（`fisher` / `fisher_budget_wanda`） |
| `--sequence_length` | `2048` | wiki/c4/ptb 为固定窗长；`s1k` 下 `0`=不截断（按样本实际长度），`>0` 时超长直接报错 |
| `--score_batch_size` | `1` | 打分 batch size；**必须为 1**（保证 Fisher 可比较） |
| `--max_prune_ratio_per_matrix` | `0.60` | 单个 Linear 最多可剪掉的块比例上限，防止某层被剪空 |
| `--min_keep_blocks_per_matrix` | `1` | 每个矩阵至少保留的块数 |
| `--share_up_gate_mask` | 默认关闭 | 打开后同层 `up`/`gate` 共享同一二维块坐标（联合分数）；默认各自独立 |
| `--pruning_rounds` | `1` | 剪枝轮数；`>1` 时按 `ρ·r/R` 增量剪并每轮重打分 |
| `--seed` | `42` | 随机种子（校准抽样、`random` 基线） |
| `--dtype` | `bfloat16` | 加载权重精度：`bfloat16` / `float16` / `float32` |
| `--device` | `cuda` | 设备，如 `cuda`、`cuda:0`、`cpu`（`magnitude`/`random` 可用 CPU） |
| `--no_gradient_checkpointing` | 默认开启 checkpoint | 加上该 flag 则关闭 gradient checkpointing（更费显存） |
| `--trust_remote_code` | 默认 True | 传给 transformers 加载 |

### `--score_type` 区别

| 类型 | 是否需要前向/反向 | 含义 |
|------|-------------------|------|
| `fisher` | 需要完整 causal LM loss + 反向 | 块分数 \(S_b=\mathrm{mean}_t(\langle\nabla W_b,W_b\rangle_t)^2\)，先平方再平均；全局排序选块 |
| `magnitude` | 否 | 块能量 \(\|W_b\|_F^2\)，剪最小的 |
| `random` | 否 | 固定 seed 随机选块（公平对照） |
| `fisher_budget_wanda` | 需要校准前向 + Fisher 反向 | Fisher 只定每矩阵剪块预算；Block-Wanda 在矩阵内选坐标 |

`fisher` / `magnitude` / `random` 共用同一套 **global ranking + 每矩阵上限** 的分配器。`fisher_budget_wanda` 第一阶段仍用该全局分配器得到预算，第二阶段改为按模块精确预算 + 局部 Wanda 排序。达不到目标则报错，不会静默降稀疏率。

### Fisher 预算 + Block-Wanda 选块

直接用 Fisher 选细粒度块坐标容易过拟合校准集。`fisher_budget_wanda` 把职责拆开：

1. 用现有 Fisher 分数做全局约束分配，得到临时参考 mask（**不应用到权重**）。
2. 从参考 mask 提取每个 MLP Linear 的累计剪块数 \(K_m\)。
3. 用实际输入通道 RMS 计算 Block-Wanda：\(S_b=\sum_{i\in R_b}\sum_{j\in C_b}|W_{ij}|a_j\)。
4. **每个矩阵内部**按 Wanda 分升序剪掉 \(K_m\) 个活跃块；不做跨矩阵 Wanda 全局排序，也不与 Fisher 分相加。
5. 打开 `--share_up_gate_mask` 时，up/gate 用联合 Wanda 分 \(S_{\mathrm{up}}+S_{\mathrm{gate}}\)，两边预算必须相等，坐标保持一致。

推荐首轮对比实验：

```text
model: Qwen/Qwen3.5-27B
block_size: 128
target sparsity: 0.20
calibration: s1k, 128 samples, sequence_length=0
pruning_rounds: 1
methods: magnitude / fisher / fisher_budget_wanda
```

开启方式：`--score_type fisher_budget_wanda`，或在 `prune_mlp.sh` 设 `SCORE_TYPE=fisher_budget_wanda`；批跑可把 `METHODS` 设为 `(magnitude fisher fisher_budget_wanda)`。

### Qwen3.5 注意点

- Hub 上的 `Qwen/Qwen3.5-27B` 配置是多模态包装；脚本会用 `text_config` 加载为 `Qwen3_5ForCausalLM` 再导出，方便本仓库 vLLM 加载。
- 只剪 `*.mlp.{gate,up,down}_proj`，不碰 Attention / embedding / lm_head。
- `5120` 与 `17408` 对常见块尺寸（如 64/128）可整除；换尺寸前请自行确认整除关系。

---

## 3. 评测脚本：`eval_pruned.sh`

对**已经剪好**的 HF 目录调用仓库根 `main.py`。

```bash
# 基本：评 AIME25 avg@5
bash Block_Sparse/scripts/eval_pruned.sh \
  Block_Sparse/outputs/qwen35_27b_fisher_s0.3

# 换任务 / GPU / 并行度
DATASETS=aime25_avg5,hellaswag \
GPUS=0,1,2,3 \
TP=4 \
bash Block_Sparse/scripts/eval_pruned.sh \
  Block_Sparse/outputs/qwen35_27b_fisher_s0.3
```

位置参数：

| 参数 | 含义 |
|------|------|
| `$1` | 剪枝模型目录（需含 `config.json`）；相对路径相对仓库根 |

环境变量：

| 变量 | 默认 | 作用 |
|------|------|------|
| `DATASETS` | `aime25_avg5` | 传给 `main.py --datasets`，逗号分隔多任务 |
| `GPUS` | `0,1,2,3` | `CUDA_VISIBLE_DEVICES` |
| `TP` | `4` | `--tensor_parallel_size` |
| `MAX_MODEL_LENGTH` | `32768` | `--max_model_length` |
| `MAX_NEW_TOKENS` | `32768` | `--max_new_tokens` |
| `OUTPUT_DIR` | `Block_Sparse/results` | lighteval 结果根目录 |

脚本末尾可再跟 `main.py` 的额外参数，例如：

```bash
bash Block_Sparse/scripts/eval_pruned.sh some/model --max_samples 10 --enforce_eager
```

评测结果写在：`Block_Sparse/results/<模型目录最后一级名>/`。

也可用 `lm-eval` / 其它工具直接读该 HF 目录；本仓库主路径是 `main.py` + lighteval。

---

## 4. 批跑脚本：`run_baselines.sh`

同一稀疏率下依次跑多种方法并评测。参数同样写在脚本顶部。

```bash
conda activate hif4
bash Block_Sparse/scripts/run_baselines.sh
```

主要项：`MODEL_PATH`、`SPARSITY`、`BLOCK_SIZE`、`CALIBRATION_DATASET`、`METHODS`、`SKIP_PRUNE` / `SKIP_EVAL`、`PRUNE_GPUS` / `EVAL_GPUS`、评测相关任务等。

剪枝多卡：把 `PRUNE_GPUS` 设成你要用的卡，例如 `6,7`。进程会按该 `CUDA_VISIBLE_DEVICES` 对可见卡做 `device_map=auto` 切分，**不会**自行扫描整机空卡。

输出示例：`Block_Sparse/outputs/qwen35_27b_fisher_s0.20_b128/`。

---

## 5. 单元测试

```bash
cd Block_Sparse
conda run -n hif4 --no-capture-output python -m pytest tests/ -v
```

覆盖：块 reduce、mask 梯度等价、Fisher 先平方后累计、allocator 剪块数、置零正确性、Qwen3.5 维度注册。

---

## 6. 建议工作流（精度对比）

1. Dense 基线（未剪原文）：
   ```bash
   bash Block_Sparse/scripts/eval_pruned.sh /path/to/Qwen3.5-27B
   # 或直接用 HF id：在仓库根
   # conda run -n hif4 python main.py --model_path Qwen/Qwen3.5-27B --datasets aime25_avg5 ...
   ```
2. 同一稀疏率下对比三种方法：
   ```bash
   SPARSITY=0.30 bash Block_Sparse/scripts/run_baselines.sh
   ```
3. 看 `Block_Sparse/results/` 下各模型分数，并对照  
   `outputs/*/pruning_artifacts/pruning_summary.json` 确认实际稀疏率一致。

---

## 7. 常见问题

| 现象 | 处理 |
|------|------|
| Fisher OOM | 剪枝用多卡：`PRUNE_GPUS=6,7`（可见卡 `device_map=auto`）；s1k 完整样本很长，需足够显存；保持 checkpointing 与 grad hook |
| `Cannot reach target sparsity` | 目标稀疏率与 `max_prune_ratio_per_matrix` 冲突；提高 cap 或降低目标稀疏率 |
| 维度不可整除 | 非 Qwen3.5-27B 类尺寸时直接报错；不要改代码做 padding |
| vLLM 加载失败 | 确认目录含 `config.json` 且 `architectures` 为 `Qwen3_5ForCausalLM`；用本仓库编译的 vLLM |

首版**不做**：SparseGPT 补偿、剪枝后微调、Attention 剪枝、真正的 block-sparse CUDA kernel（权重仍是 dense + 零块，靠现有 dense 算子推理）。
