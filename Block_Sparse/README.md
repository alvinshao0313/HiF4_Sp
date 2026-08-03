# MLP 梯度块稀疏使用说明

可对 Qwen3.5-27B 的 MLP（`gate_proj` / `up_proj` / `down_proj`）做结构化块剪枝，导出标准 HF 权重，并用仓库根目录 `main.py`（vLLM + lighteval）做评测。

**环境：一律使用 conda 环境 `hif4`。**

理论细节见同目录设计文档  
`mlp_128x128_gradient_block_pruning_design_fixed.md`。

---

## 目录与产物

```text
Block_Sparse/
├── block_pruning/          # 核心库（打分、分配 mask、置零、导出、LoRA）
├── scripts/                # 仅 shell 入口（改参数后执行）
│   ├── prune_mlp.sh
│   ├── run_baselines.sh
│   ├── run_mlp_lora_sft.sh
│   ├── eval_pruned.sh
│   └── eval_*.sh
├── tools/                  # Python CLI（由 scripts 调用）
│   ├── score_and_prune_mlp.py
│   ├── train_mlp_lora_sft.py
│   ├── eval_ppl.py
│   ├── eval_lm_eval.py
│   └── download_lighteval_mmlu.py
├── tests/                  # 单元测试
├── experiments/
│   ├── wikitext2_calib/    # WikiText-2 校准阶段归档（ckpt + 评测 + report.html）
│   └── s1k_calib/          # s1K 校准评测归档（ppl/lm_eval 日志；ckpt 仍在 outputs/）
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
| `output_dir/pruning_artifacts/mlp_permutations.pt` | （仅 `wanda_shared`）中间维置换与分数 |
| `output_dir/pruning_artifacts/mlp_permutation_summary.json` | （仅 `wanda_shared`）置换元信息 |
| `output_dir/pruning_artifacts/residual_permutation.pt` | （仅 `block_loss`）全局 residual 置换与 \(L\) |
| `output_dir/pruning_artifacts/residual_permutation_summary.json` | （仅 `block_loss`）residual 置换元信息 |

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
| `MLP_PERMUTATION` | `none` / `wanda_shared`：剪枝前是否做 FFN 中间维共享 Wanda 重排 |
| `RESIDUAL_PERMUTATION` | `none` / `block_loss`：剪枝前是否做全局 residual 隐空间置换（最小化被剪块分数和） |
| `RESIDUAL_PERM_SEARCH_STEPS` | `block_loss` 通道交换搜索步数（默认 2000） |
| `RESIDUAL_CHANNEL_AGG` | `equal` / `layer_fisher` / `matrix_fisher` / `raw_wanda` / `sparsity_raw_wanda` / `density_raw_wanda`：π0 残差通道跨层聚合（仅 `block_loss`） |

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
python Block_Sparse/tools/score_and_prune_mlp.py \
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
| `--projection_prune_shares` | 空（关闭） | 可选：在全局 `SPARSITY` 预算内按投影类型分配剪块份额，如 `gate_proj=1,up_proj=1,down_proj=2`；空=跨 u/g/d 全局排序 |
| `--min_keep_blocks_per_matrix` | `1` | 每个矩阵至少保留的块数 |
| `--share_up_gate_mask` | 默认关闭 | 打开后同层 `up`/`gate` 共享同一二维块坐标（联合分数）；默认各自独立 |
| `--mlp_permutation` | `none` | `none` / `wanda_shared`：剪枝前一次性共享 Wanda 重排 FFN 中间维 |
| `--residual_permutation` | `none` | `none` / `block_loss`：全局 residual 隐空间置换，最小化被剪块分数和（在 `mlp_permutation` **之前**） |
| `--residual_perm_search_steps` | `2000` | `block_loss` 的通道交换搜索步数 |
| `--residual_channel_agg` | `equal` | π0 残差通道聚合：`equal`（每矩阵 L1 后等权）/ `layer_fisher` / `matrix_fisher` / `raw_wanda` / `sparsity_raw_wanda`（×ρ_m）/ `density_raw_wanda`（×(1−ρ_m)；后三者 fisher 类需 fisher 类 score_type） |
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

### MLP 中间维共享重排

可选开关 `--mlp_permutation wanda_shared`：在 mask 初始化与一切块打分之前，对每层 FFN 中间维做一次共享置换。

对 SwiGLU：

\[
W_u'=PW_u,\qquad W_g'=PW_g,\qquad W_d'=W_dP^\top
\]

实现上对 `up`/`gate` 行与匹配的 `down` 列用同一 `perm` 做 `index_select`。因 SiLU 与 Hadamard 按元素运算，前向输出不变。

神经元重要性用 Wanda：up/gate 为 \(\sum_j|W_{k,j}|a_j\)，down 为 \(a_k\sum_i|W_{i,k}|\)；三路各自 L1 归一化后等权相加，再稳定降序排序。

与 `--share_up_gate_mask` 正交：前者重排神经元顺序，后者共享块坐标；可同时打开。

无运行时 gather/scatter；剪枝开始后不再重算/重 apply；导出 checkpoint **保持置换坐标系**（不做 unpermute）。产物：

| 路径 | 含义 |
|------|------|
| `pruning_artifacts/mlp_permutations.pt` | 每层 `permutation` / `inverse` / `combined_score` |
| `pruning_artifacts/mlp_permutation_summary.json` | 置换元信息 |

`wanda_shared` 会强制需要校准数据（即便 `score_type=magnitude`）。默认 `none`，现有剪枝行为不变。

### Residual 全局置换（`--residual_permutation block_loss`）

对 **residual 隐空间**（Qwen3.5-27B 为 5120）做全模型同一个置换 π，吸进 `embed` / RMSNorm / Attention 投影 / MLP / `lm_head`。**不改** GDN `conv1d`、`A_log`、`dt_bias`、head 内 RoPE 维。

目标：在给定稀疏率与打分器下，最小化将被剪掉的块分数之和 \(L(\pi)\)。流程：

1. Wanda 通道重要性得到初值 \(\pi_0\)（`--residual_channel_agg`：`equal` / `layer_fisher` / `matrix_fisher` / `raw_wanda` / `sparsity_raw_wanda` / `density_raw_wanda`）
2. 通道交换局部搜索直接最小化 \(L\)（`magnitude` 用块能量；`fisher` / `fisher_budget_wanda` 搜索用 Wanda；后者额外冻结一次 Fisher 每矩阵预算）
3. 最优 π 写进权重后，再跑现有 `--mlp_permutation`（若开启）与正式剪枝

`layer_fisher` / `matrix_fisher` / `sparsity_raw_wanda` / `density_raw_wanda` 需 `score_type` 为 `fisher` 或 `fisher_budget_wanda`。后两者用 Fisher 分配稀疏率 \(\rho_m=K_m/N_m\)：`sparsity_raw_wanda` 乘 \(\rho_m\)，`density_raw_wanda` 乘 \(1-\rho_m\)（均不做 L1）。搜索评估在各 MLP 参数所在 **GPU** 上做 float32 权重缓存与块归约（`device_map=auto` 时按层分散），块分再回 CPU 走现有分配器；搜完释放缓存。短暂会多占一份 MLP float32 显存。

与 `--mlp_permutation` **正交**：先 residual，再 FFN 中间维；中间维 `wanda_shared` 算法不变。与 `score_type=random` 同时开启会直接报错。

产物：

| 路径 | 含义 |
|------|------|
| `pruning_artifacts/residual_permutation.pt` | π / inverse / channel_score / \(L\) 轨迹 |
| `pruning_artifacts/residual_permutation_summary.json` | 置换元信息 |

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

---

## 8. 剪枝后 MLP Masked LoRA SFT（peft）

剪枝后精度掉点时，可对 **仅 MLP**（`gate_proj` / `up_proj` / `down_proj`）挂 peft LoRA，并用与剪枝相同的块 mask 约束 delta，使 `merge_and_unload` 后已剪块仍为零。

```bash
conda activate hif4
# 顶部改 PRUNED_MODEL_DIR / CUDA_VISIBLE_DEVICES / MAX_STEPS 等
bash Block_Sparse/scripts/run_mlp_lora_sft.sh
```

要点：

| 项 | 说明 |
|----|------|
| 输入 | 剪枝 HF 目录 + `pruning_artifacts/block_masks.pt` + `pruning_summary.json` |
| 数据 | S1K chat SFT（`simplescaling/s1K-1.1_tokenized`），逻辑在 `block_pruning/sft_data.py` |
| 适配 | `MaskedLoraLinear` 重写 peft 的 `get_delta_weight` / `forward`，ΔW 乘元素 mask |
| 导出 | merge 后标准 HF 目录，可直接 `eval_ppl` / `eval_lm_eval` / `eval_pruned.sh` |

Python 入口：`Block_Sparse/tools/train_mlp_lora_sft.py`。

### 蒸馏模式（EdgeRazor 式 QAD 损失）

`run_mlp_lora_sft.sh` 顶部 `TEACHER_MODEL_DIR` 非空时启用：加载**未剪基座**做冻结 teacher，损失换成 `task_alpha·CE + eakld_alpha·KL + lafd_alpha·LAFD`（与 QAD 同口径：last_hidden 取最后 decoder layer 输出即 pre-final-norm，分块 lm_head 全局 sum/count）。损失与 hook 实现是 `QAD/distill_losses.py` / `QAD/hidden_hooks.py` 在 `block_pruning/` 下的逐字副本。

| 项 | 说明 |
|----|------|
| `TEACHER_MODEL_DIR` | 未剪 HF 目录 / hub id；留空退回纯 CE SFT |
| `KL_MODE` | `eakld`（全词表熵自适应 KL）/ `eakld_topk`（top-k 版 EAKLD）/ `kl_topk`（仅 forward top-k KL，配 `KL_TOPK`、`KL_POST_ATTN`） |
| `TASK_ALPHA` / `EAKLD_ALPHA` / `LAFD_ALPHA` | 三分量权重（默认 0.05 / 2.0 / 0.5）；消融把对应 alpha 设 0（注意 `LAFD_ALPHA=0` 时 LAFD 仍计算，选层 pass 不省） |
| `TEMPERATURE` / `LAFD_TOPK` | KL 温度；LAFD 自适应选层数（按相邻层 cosine 变化选 top-k 层做 hidden MSE） |
| 代价 | 每 step 2 次 teacher 前向（选层 + 抓层）+ 1 次 student 前向；teacher/student 权重同时驻留（27B 级需两份 bf16 显存），teacher hidden 落 CPU |
| 日志 | `ce` / `eakld` / `lafd` / `qad_total` 分量随 `logging_steps` 输出 |

mask 约束与损失无关：`MaskedLoraLinear` 结构化保证 merge 后已剪块为零，蒸馏下同样成立。

**不做**：Attention LoRA、SparseGPT 补偿、真正的 block-sparse CUDA kernel（权重仍是 dense + 零块）；`parallel_mode=fsdp` 未适配蒸馏。
