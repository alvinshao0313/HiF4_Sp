# MLP Block 稀疏剪枝算法说明书（DiT 迁移版）

本文档是 `Block_Sparse/` MLP 结构化块剪枝方案的**完整实现规格**，
目标读者：要在另一个仓库（DiT 类扩散模型）里独立复现这套流程的人。
文档自包含——不需要读本仓库代码即可实现；每节固定结构为
**算法 → 实现规格 → DiT 适配**。只想估算工作量可直接看 §9 与附录 B。

---

## 1. 方案总览

### 1.1 核心思想

把每个 \(H\times W\) 权重块视为一个标量门控变量 \(\widetilde W_b = m_b W_b\)。
删除该块即 \(m_b: 1\to 0\)。在当前 dense 权重处，loss 对 \(m_b\) 的梯度为

\[
g_b = \Big\langle \nabla_{W_b}\mathcal{L},\ W_b \Big\rangle,
\]

即"缩放/删除整块"的一阶敏感度。整条流水线：**估计每块重要性 → 全局选最不重要的块
→ 整块置零 → 约束微调恢复 → 评测**。

### 1.2 流水线

```text
dense 模型
  │
  ├─ (可选) Residual 全局置换 π      ← 校准前向 + 交换搜索，见 §5.2
  ├─ (可选) FFN 中间维共享置换 P     ← 校准前向，见 §5.1
  ▼
[1] 校准打分    校准数据前向(+反向) → 每块一个重要性分数 S_b        §3
[2] Mask 分配   全局排序 + 每矩阵约束 → 块级布尔 mask（True=保留）   §4
[3] 置零导出    weight.mul_(element_mask) → 标准 HF 目录 + 产物      §6
[4] Masked LoRA SFT  已剪块冻结为零的约束下微调 ΔW，merge 导出       §7
[5] 评测        剪枝/恢复模型 vs dense 基线                          §8
```

### 1.3 主流程编排（伪代码，与实现一一对应）

```text
config.validate()                       # 参数合法性，非法直接报错
set_seed(config.seed)
model, tokenizer = load(model_path)     # bf16；多卡用 device_map=auto
targets = discover_mlp_linears(model)   # §2.3 注册规则 + 整除校验（失败即停）
batches = build_calibration(...)        # 仅 fisher/wanda/置换类需要，§3.4

if residual_permutation == "block_loss":
    π = search_residual_permutation(...) # §5.2；写进权重 + 保存产物
if mlp_permutation == "wanda_shared":
    P = compute_intermediate_perms(...)  # §5.1；写进权重 + 保存产物

masks = all_ones(targets)                # 全部 True 起步
for r in range(pruning_rounds):
    cumulative = target_sparsity * (r+1) / pruning_rounds
    scores = score_blocks(...)           # §3；在“当前已剪权重”上重新打分
    masks  = allocate(scores, masks, cumulative)   # §4；失败即报错
    model.apply(masks)                   # §6 置零
    save_round_artifacts(r)
verify_masks_and_weights(masks)          # 逐块断言 nnz==0
save_final_artifacts(); save_hf_model()  # §6.3 schema
```

### 1.4 设计原则（迁移时必须保留）

- **结构化**：只剪整个 \(H\times W\) 块，不剪零散元素；mask 粒度与未来 block-sparse kernel 的物理 tile 一致。
- **Let it crash**：维度不可整除、稀疏率达不到、mask 与权重不一致、hook 没收到梯度，全部直接报错；不做 padding、不静默降稀疏率、不做兜底。
- **权重仍是 dense + 零块**：本方案不含运行时稀疏 kernel，导出标准 dense 权重。

---

## 2. 剪枝目标与块划分

### 2.1 剪枝范围

只剪 MLP 的 Linear 权重，不碰 Attention / embedding / norm / 输出头 / MoE expert。

| 模型 | 目标模块 | 形状 |
|------|----------|------|
| LLM（SwiGLU） | `mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj` | \(W_g,W_u\in\mathbb{R}^{d_{ff}\times d}\)，\(W_d\in\mathbb{R}^{d\times d_{ff}}\) |
| DiT（标准 GELU MLP） | `mlp.fc1` / `mlp.fc2` | \(W_{fc1}\in\mathbb{R}^{d_{ff}\times d}\)，\(W_{fc2}\in\mathbb{R}^{d\times d_{ff}}\) |
| Flux / SD3 类 | `ff.net.0.proj` / `ff.net.2` | 同上 |

对应关系：**fc1 ≈ up_proj，fc2 ≈ down_proj，DiT 无 gate**。

### 2.2 块划分

权重 \(W\in\mathbb{R}^{d_{out}\times d_{in}}\) 划分为不重叠块：

\[
W_{a,b} = W[\,aH:(a+1)H,\ bW:(b+1)W\,],\qquad
a\in[0,\tfrac{d_{out}}{H}),\ b\in[0,\tfrac{d_{in}}{W})
\]

- 块尺寸一个参数控制：`"128"` → \(H{=}W{=}128\)；`"64x128"` → \(H{=}64\)（沿 \(d_{out}\)），\(W{=}128\)（沿 \(d_{in}\)）。
- 硬约束：\(d_{out}\bmod H = 0\) 且 \(d_{in}\bmod W = 0\)，否则直接报错。
- 块 mask：形状 \((d_{out}/H,\ d_{in}/W)\) 的 bool 矩阵，`True`=保留。
- 元素级展开：`element_mask = mask.repeat_interleave(H, dim=0).repeat_interleave(W, dim=1)`。

### 2.3 目标注册规则（实现规格）

对每个 `nn.Linear`，按**模块全名**判定是否为目标：

```text
1. 名字含 ".experts."            → 跳过（MoE expert 不剪）
2. 名字以 ".mlp.{proj}" 结尾      → 命中，proj ∈ {gate_proj, up_proj, down_proj}
   （DiT：proj ∈ {fc1, fc2}，或按仓库实际命名）
3. 层号解析：正则 (?:^|\.)(?:layers|h|blocks)\.(\d+)\. 的第一个捕获组
4. 命中后立即做整除校验（§2.2），失败直接抛错
5. 目标列表排序键：(layer_index, projection_type 固定顺序, module_name)
```

产物中的 mask key 就是这里返回的模块全名（如 `model.layers.12.mlp.down_proj`），
后续打分、分配、置零、LoRA 全部用同一套 key。

### 2.4 DiT 适配

只需重写注册规则（§2.3 的第 2 条）：模块名匹配换成 `fc1/fc2`
（或 `ff.net.0.proj/ff.net.2`，Flux 注意 expert/共享分支命名）。层号正则对
`blocks.N.` 天然兼容。DiT 常见 \(d\)=1152/1536/3072、\(d_{ff}=4d\)，对 64/128 均可整除，先验证再跑。

---

## 3. 校准与块重要性打分

四种打分器都产出每块一个非负分数（float64 CPU 张量，形状 = 块网格），**分越小越先剪**。
§4 的分配器不关心分数来源。

### 3.1 Block Empirical Fisher（主指标）

对每个校准样本独立做一次完整前向 + loss 反向，先平方再跨样本平均：

\[
g_b^{(t)} = \Big\langle \nabla_{W_b}\mathcal{L}_t,\ W_b \Big\rangle,
\qquad
\boxed{\ S_b^{\mathrm{Fisher}} = \frac{1}{T}\sum_{t=1}^{T}\Big(g_b^{(t)}\Big)^2\ }
\]

禁止"先平均梯度再取绝对值"——不同样本梯度符号相消。

**实现规格**（LLM 已验证，逐条保留）：

1. `score_batch_size` 必须为 1：不同 batch size 下 mean loss 的梯度尺度不同，分数不可比。
2. 冻结全部参数（`requires_grad_(False)`），只把目标 Linear 的 weight 打开。
3. 模型置 train 模式以启用 gradient checkpointing，但把所有 `nn.Dropout` 单独置回 eval；
   `config.use_cache=False`。
4. 前向在 `torch.autocast(bfloat16)` 下做，loss 反向；块归约在 float32 下计算
   （`weight.float() * grad.float()`），累加器为 **CPU float64**。
5. 每个目标 weight 注册 `register_post_accumulate_grad_hook`：反向中梯度一就绪就
   归约成块、搬 CPU、`param.grad = None` 释放——全层梯度从不同时驻留显存。
   需要 PyTorch ≥ 2.1。hook 逐 batch 注册、batch 结束移除。
6. 同时维护三种统计（只用第一种排序，其余诊断）：
   `score_sq += block_signal²`、`score_abs += |block_signal|`、`score_signed += block_signal`，
   最终分别除以 batch 数。
7. 已剪块（多轮剪枝）的 `block_signal` 乘当前 mask 置零；候选也只从 `mask==True` 的块产生（§4）。
8. 每个 batch 后校验所有目标都收到过梯度，任一缺失立即报错。

**块归约公式**（三种打分器共用同一个 4D reshape 技巧）：

\[
\mathrm{signal} = W \odot G \in \mathbb{R}^{d_{out}\times d_{in}}
\;\xrightarrow{\mathrm{reshape}}\;
\Big(\tfrac{d_{out}}{H}, H, \tfrac{d_{in}}{W}, W\Big)
\;\xrightarrow{\mathrm{sum\ over\ dim\ 1,3}}\;
S \in \mathbb{R}^{(d_{out}/H)\times(d_{in}/W)}
\]

### 3.2 Block-Wanda

\[
S_b^{\mathrm{Wanda}} = \sum_{i\in R_b}\sum_{j\in C_b}|W_{ij}|\,a_j,
\qquad
a_j = \sqrt{\frac{1}{N_{tok}}\sum_{tok} x_{tok,j}^2}
\]

**实现规格**：

1. 在每个目标 Linear 上挂 `register_forward_pre_hook`，取 `inputs[0]`（形状 `[..., d_in]`），
   reshape 成 `[-1, d_in]`，累加 `x.float().square().sum(dim=0)`（CPU float64）与 token 数；
2. `a = sqrt(square_sum / num_tokens)`，长度必须等于权重的 \(d_{in}\)，否则报错；
3. 元素分 `|W|.float() * a[None, :]`，用 §3.1 的同一 reshape 归约到块；
4. **只需前向**（`torch.no_grad()`），可与 Fisher 共用同一次校准前向（hook 同时挂着，不用跑两遍）。

### 3.3 免校准基线

| 打分器 | 公式 | 用途 |
|--------|------|------|
| `magnitude` | \(S_b=\|W_b\|_F^2\)（同一 reshape 对 \(W^2\) 求和） | 不需校准数据，验证链路 |
| `random` | 块网格上 i.i.d. `rand`（CPU generator，固定 seed） | 公平对照 |

### 3.4 校准数据规格（LLM 侧）

batch 是 `dict`，每条样本一个 batch（batch size = 1）：

```python
{"input_ids": LongTensor[1, T],           # add_special_tokens=False
 "attention_mask": LongTensor[1, T],      # 全 1
 "labels": LongTensor[1, T]}              # = input_ids.clone()，HF 内部做 shift
```

- 数据集：`s1k`（`simplescaling/s1K-1.1_tokenized`，`sequence_length=0` 表示不截断，
  `>0` 截断到该长度）；`wikitext2/c4/ptb`（固定窗长，`sequence_length` 必须 >0）。
- 抽样：`random.Random(seed).sample(range(len(ds)), k=num_samples)`，默认 128 条。
- T 必须 ≥ 2（LM loss 需要）。

### 3.5 两阶段 `fisher_budget_wanda`（推荐配置）

直接用 Fisher 选细粒度块坐标容易过拟合校准集，拆分职责：

1. **联合校准前向**：按 §3.1 跑 Fisher，同时挂 §3.2 的 RMS hook（一次前向拿两样）；
2. **Fisher 定预算**：Fisher 分跑 §4 全局分配器得参考 mask（**不写权重**），
   数出每矩阵剪块预算 \(K_m\)，并校验 \(\sum_m K_m\) = 参考 mask 总剪块数；
3. **Wanda 定坐标**：用第 1 步的 RMS 算 Wanda 块分，每矩阵内部按 Wanda 升序剪恰好 \(K_m\) 个活跃块；
4. **收尾校验**：最终每矩阵剪块数 == \(K_m\)，总剪块数 == 参考 mask 总数，不等即报错。

不做跨矩阵 Wanda 排序，不与 Fisher 分相加。

### 3.6 DiT 适配：校准数据与 loss（迁移中改动最大的部分）

Wanda（3.2）、magnitude/random（3.3）、Fisher 的 hook/归约/offload 机制（3.1）**全部复用**，
要换的只有 **loss 来源**：causal-LM CE → diffusion 训练 loss。

单个校准样本的构造必须与训练时完全一致：

```python
# 一个校准 batch（batch size = 1）
x0   = vae.encode(image)                      # latent，按仓库训练 pipeline
c    = text_encoder(prompt)                   # 条件，冻结
t    = rand_t(generator=seed_gen)             # 与训练同分布（如 U{1..T} 或 U(0,1)）
eps  = randn_like(x0, generator=seed_gen)
x_t  = schedule(x0, eps, t)                   # DDPM: √ᾱ·x0 + √(1-ᾱ)·eps
                                              # flow-matching: (1-t)·x0 + t·eps
pred = model(x_t, t, c)
loss = mse(pred, target).mean()               # target = eps（或 v = eps - x0），与训练同款
loss.backward()                               # 只开了 fc1/fc2 的 requires_grad
```

三个注意点：

- \(t\) 与 \(\epsilon\) 的随机性使单样本分数是期望估计，**所有随机数走固定 seed 的 generator**；
  样本数与 LLM 侧同量级（128 起）。
- loss 的归约方式（对 batch/空间维 mean 还是 sum）直接决定梯度尺度，所有样本必须统一
  且与训练配置一致，否则分数不可比。
- 冻结全模型后只开 fc1/fc2 的 `requires_grad`；t_embedder / text encoder 照常前向但不收梯度。

---

## 4. Mask 分配器（与模型无关，完全复用）

输入：块分数 dict + 当前 mask dict；输出：新 mask dict。对每种 `score_type` 打分器通用。

### 4.1 全局排序 + 每矩阵约束（默认路径）

```text
K = floor(ρ * N_total)                        # 全局剪块数；N_total = 所有矩阵块数之和
candidates = []
for name in targets:
    for (ob, ib) in active_blocks(mask[name]):          # 只从仍保留的块产生候选
        candidates.append((score[name][ob,ib], name, ob, ib))
candidates.sort(stable, key=(score, name, ob, ib))      # 升序，分越小越先剪
for each candidate until K 个被选:
    skip if pruned_count[name] >= cap[name]
    cap[name] = min(floor(N_m * max_prune_ratio_per_matrix),   # 默认 0.6
                    N_m - min_keep_blocks_per_matrix)          # 默认 ≥1 块保留
选不满 K → RuntimeError（不静默降稀疏率）
```

### 4.2 变体

- **`share_up_gate_mask`**：同层 up/gate 共享二维块坐标。联合分数 \(S_u+S_g\)；
  up/gate 对与 down 单块进入**同一候选流**，对的 cost=2、单块 cost=1；
  要求剪块增量为偶数；两个矩阵各自仍受 §4.1 的 cap 约束。**DiT 无 gate，关闭**。
- **`projection_prune_shares`**：把全局预算 \(K\) 按投影类型份额拆分（如
  `gate=1,up=1,down=2`），份额归一化后用**最大余数法**整数化
  （floor 后按"小数部分降序 → 份额降序 → 名字升序"补齐到总数），
  各类型内部独立跑 §4.1。DiT 键名换成 `fc1/fc2`。
- **多轮剪枝**（`pruning_rounds=R>1`）：第 \(r\) 轮累计目标 \(\rho\cdot(r{+}1)/R\)；
  每轮**在已置零的模型上重新打分**（Fisher/Wanda 都重算），增量剪块、立即置零、
  保存带 `_round{r}` 后缀的产物。
- **每矩阵预算模式**（`fisher_budget_wanda` 第二阶段用）：输入 `{name: K_m}`，
  每矩阵内部按分升序剪恰好 \(K_m\) 个；预算超出 cap 或候选不足即报错；
  剪完逐矩阵校验实际数 == 预算。

---

## 5. 可选增强：剪枝前置换

两个置换都满足**前向输出严格不变**，目的是把重要性"聚拢"，让被剪块的分数之和更小。
执行顺序固定：先 residual 置换（§5.2），再中间维置换（§5.1），再正式剪枝。
导出 checkpoint **保持置换坐标系**（不 unpermute），无运行时 gather/scatter。

**置换方向约定（全套代码统一）**：`perm[k]` = 新坐标系第 \(k\) 个位置上的**旧索引**；
应用即 `param.index_select_(dim, perm)`；`inverse[perm] = arange`。

### 5.1 FFN 中间维共享置换

**算法**：SwiGLU 的中间维 \(d_{ff}\) 被三个矩阵共享，求每层一个置换 \(P\)：

\[
W_u' = P W_u,\qquad W_g' = P W_g,\qquad W_d' = W_d P^\top
\]

SiLU 与 Hadamard 逐元素作用 → 前向不变。神经元 \(k\) 的 Wanda 重要性：

- up/gate：\(s_k = \sum_j |W_{kj}|\,a_j\)（\(a_j\) 是该矩阵输入通道 RMS）
- down：\(s_k = a_k \sum_i |W_{ik}|\)
- 三路各自 **L1 归一化**后等权相加，`argsort(descending, stable)` 得 \(P\)

在任何打分与 mask 初始化之前应用一次，之后不再重算。强制需要校准数据（即便 `score_type=magnitude`）。

**DiT 适配**：三元组退化为二元组（GELU 同样逐元素）：

\[
W_{fc1}' = P W_{fc1},\qquad W_{fc2}' = W_{fc2} P^\top
\]

神经元分 = fc1 行分与 fc2 列分各自 L1 归一化后等权相加。

### 5.2 Residual 隐空间全局置换

**算法**：对 residual 通道维 \(d_{model}\) 做**全模型同一个**置换 \(\pi\)，吸进所有挂在
residual 轴上的参数。目标：在给定稀疏率与打分器下，最小化被剪块分数之和

\[
L(\pi) = \sum_{m}\sum_{b\in \mathrm{Pruned}_m(\pi)} S_b^{(m)}(\pi),
\]

其中 \(\mathrm{Pruned}_m(\pi)\) 由 §4 分配器在虚拟置换后的权重上决定
（搜索只做 `index_select` 得到虚拟权重，不写回模型）。

流程：

1. **初值** \(\pi_0\)：residual 通道重要性 = 各 MLP 矩阵 residual 轴 Wanda 质量的跨层聚合
   （`equal / layer_fisher / matrix_fisher / raw_wanda / sparsity_raw_wanda /
   density_raw_wanda`），`argsort(descending, stable)`；
   - 通道原始质量：输入侧矩阵 \((\sum_i|W_{ik}|)\,a_k\)；输出侧矩阵 \(\sum_j |W_{kj}|\,a_j\)
     （每矩阵先 L1 归一化，再按 agg 方式加权；raw 系不做 L1）；
2. **交换搜索**：每步随机取通道 \(i\ne j\) 交换，重算 \(L\)，**严格更小才接受**
   （默认 2000 步，CPU generator 固定 seed）。
   `magnitude` 用块能量评估 \(L\)；`fisher` / `fisher_budget_wanda` 用 Wanda 代理评估，
   后者先把每矩阵 Fisher 预算冻结一次（搜索中只重排坐标，预算不动）；
3. 最优 \(\pi\) 一次性 `index_select` 写进全部 mount 参数，再进入 §5.1 与正式剪枝。

**搜索缓存**：全部 MLP 权重 float32 + 输入 RMS 缓存在参数所在设备，搜完释放并
`torch.cuda.empty_cache()`；短暂多占一份 MLP float32 显存。

**LLM 侧 mount 分类规则（迁移时对照写 DiT 版）**：

| 参数 | 判定 | permute dim |
|------|------|-------------|
| `embed_tokens.weight` / `lm_head.weight` \([V, H]\) | 名字匹配 | 最后一维 |
| RMSNorm gain \([H]\)（`input_layernorm` / `post_attention_layernorm` / `*.norm.weight`） | 名字匹配 | 0 |
| 2D Linear：`*_{out,o,down}_proj.weight` | 输出侧（\(d_{out}{=}H\)） | 0 |
| 2D Linear：`*_{q,k,v,gate,up}_proj.weight`、`*in_proj*.weight` | 输入侧（\(d_{in}{=}H\)） | 1 |
| 形状 \([H,H]\) 且名字不在上两类的方阵 | **拒绝**（无法判定 in/out 侧，let it crash） | — |
| bias \([H]\) | 输出侧 | 0 |
| 忽略清单（不匹配 residual 轴） | `conv1d`、`A_log`、`dt_bias`、`q_norm`、`k_norm`、`linear_attn.norm` | — |
| tied weights（`lm_head` 与 `embed` 共享存储） | 按 `data_ptr` 去重，只置换一次 | — |
| 任何含 \(H\) 维但不在上表的参数 | **报错**，不允许静默跳过 | — |

**DiT 适配**：搜索算法不变，mount 分类器按 DiT 结构重写：

| DiT 参数 | residual 侧 | permute dim |
|----------|------------|-------------|
| `patch_embed.proj`（Conv2d \([d,C_{in},p,p]\)） | 输出 | 0 |
| `pos_embed`（\([1,N,d]\) 参数/buffer） | 全部 | -1 |
| `blocks.N.norm1/norm2`（有 gain 的 LayerNorm） | 通道 | 0 |
| `blocks.N.attn.qkv` / `attn.proj` | qkv 输入列 / proj 输出行 | 1 / 0 |
| `blocks.N.mlp.fc1` / `fc2` | fc1 输入列 / fc2 输出行 | 1 / 0 |
| `adaLN_modulation` Linear | 输入列 dim1；**输出是 \(k\cdot d\) 扇出**（如 6 路 shift/scale/gate），reshape \([k,d]\) 后对 dim1 permute | 1 / 分段 0 |
| `t_embedder` 第一层 | 输入是 sinusoidal 频率维，**不是** residual 轴，不碰 | — |
| 条件投影（text/pooled proj，输出 \(d\)） | 输出侧 | 0 |

两个坑：

- **RoPE 模型（Flux/SD3）**：置换只动 qkv 的**输入列**（residual 通道），不触碰输出侧
  head 维结构，与 RoPE 兼容；mount 分类时勿误匹配 head 内维度。
- **adaLN 扇出**：\(k\cdot d\) 输出必须 \(k\) 段共用同一个 \(\pi\)，否则 shift/scale/gate
  通道语义错位。这是 DiT 独有、LLM 没有对应物的 mount 类型。

首轮迁移建议 `residual_permutation=none`，先打通主链路。另外它与 `score_type=random`
在语义上矛盾（随机分与坐标无关），实现中直接拒绝该组合。

---

## 6. 置零、校验与导出产物

### 6.1 置零与校验

- 置零：`weight.mul_(element_mask)`（原地，`element_mask` 转到 weight 的 device/dtype），
  保留 `nn.Linear` 结构不变；
- 校验：逐块遍历，断言每个 `mask==False` 的块 `count_nonzero == 0`，任一非零即抛错。

### 6.2 导出

标准 HF 目录（`config.json` + safetensors + tokenizer），推理框架直接加载。
注意 config 的 `architectures` 必须是推理框架认识的因果 LM / 扩散模型类名
（LLM 侧踩过的坑：多模态包装的 config 要改写为 `*ForCausalLM`，否则 vLLM 拒绝加载）。

### 6.3 产物 schema（§7 的 LoRA 依赖这些文件，迁移时格式保持不变）

目录：`<output_dir>/pruning_artifacts/`

| 文件 | 内容 |
|------|------|
| `block_masks.pt` | `dict[module_name -> BoolTensor[n_ob, n_ib]]`（CPU） |
| `block_scores.pt` | `dict[name -> {layer_index, projection_type, weight_shape, block_height, block_width, fisher, abs_taylor, signed_mean, wanda?}]` |
| `pruning_summary.json` | 见下方字段表 |
| `per_matrix_report.csv` | 列：`module_name, layer_index, projection_type, weight_shape, num_blocks, num_pruned_blocks, block_sparsity, score_min/median/mean/max`（统计只算保留块） |
| `mlp_permutations.pt` + `_summary.json` | （仅 §5.1）每层 `permutation / inverse_permutation / combined_score` |
| `residual_permutation.pt` + `_summary.json` | （仅 §5.2）`permutation / inverse / channel_score / loss_init / loss_final / accepted_swaps / mount_names` |

`fisher_budget_wanda` 额外产物：

| 文件 | 内容 |
|------|------|
| `fisher_block_scores.pt` / `wanda_block_scores.pt` | 两阶段各自的块分数 |
| `fisher_reference_masks.pt` | Fisher 预算参考 mask（**未写权重**） |
| `module_prune_budget.csv` | 每矩阵 Fisher 预算 vs 最终剪块数（不等即报错） |
| `hybrid_per_matrix_report.csv` | Fisher/Wanda 分数统计 + 两套 mask 的 IoU |

`pruning_summary.json` 字段：

```json
{
  "model_path": "...", "block_size": "128", "block_height": 128, "block_width": 128,
  "target_block_sparsity": 0.3, "actual_block_sparsity": 0.2998,
  "num_total_blocks": 0, "num_pruned_blocks": 0, "num_pruning_rounds": 1,
  "score_type": "fisher_budget_wanda", "selection_mode": "global_constrained",
  "share_up_gate_mask": false, "projection_prune_shares": null,
  "mlp_permutation": "none", "residual_permutation": "none",
  "residual_perm_search_steps": 2000, "residual_channel_agg": "equal",
  "max_prune_ratio_per_matrix": 0.6, "min_keep_blocks_per_matrix": 1,
  "calibration_dataset": "s1k", "calibration_samples": 128,
  "sequence_length": 0, "seed": 42
}
```

多轮剪枝时每轮产物带 `_round{r}` 后缀，最终产物无后缀（LoRA 只读无后缀版）。

---

## 7. Masked LoRA SFT 恢复

剪枝掉点后，对**仅 MLP** 挂 peft LoRA，用同一份块 mask 约束增量，保证 merge 后已剪块仍为零。

### 7.1 机制（与模型无关，完全复用）

`MaskedLoraLinear` 继承 `peft.tuners.lora.Linear`，**同时**重写两条路径——
stock peft 的 forward **不走** `get_delta_weight`，只改一处会导致训练与 merge 不一致：

\[
\Delta W = \frac{\alpha}{r}(BA)\odot M,\qquad
\mathrm{forward}:\ y = Wx + \mathrm{dropout}(x)\,\Delta W^\top
\]

实现规格：

1. `get_delta_weight(adapter)`：`return super().get_delta_weight(adapter) * element_mask`；
2. `forward`：base 前向后，对每个 active adapter 算 `delta = get_delta_weight(adapter)`
   （已含 scaling 与 mask），`result += F.linear(dropout(x.to(A.dtype)), delta)`；
   `merged` / `disable_adapters` 分支直接走 base；
3. 挂接流程：先 `get_peft_model(model, LoraConfig(target_modules=[...]))`，
   再对每个 LoRA Linear：按**模块名后缀匹配**找到 mask key
   （`model.layers.N.mlp.fc1` 这类，逐层截掉前缀直到命中）→
   校验 mask 形状 == `(d_out/H, d_in/W)` → `module.__class__ = MaskedLoraLinear` →
   `register_buffer("element_mask", expanded, persistent=True)`；
   所有 mask key 必须全部被用掉，否则报错；
4. 训练前 `assert_only_lora_trainable`：遍历参数，`requires_grad=True` 的名字必须含
   `lora_A` 或 `lora_B`，否则报错；
5. merge：`merge_and_unload()` 后重新收集 MLP Linear，逐块校验已剪块 `nnz==0`；
6. 不支持 DoRA、不支持混合 adapter batch（实现中直接拒绝）。

### 7.2 LLM 侧参考配置

| 项 | 值 |
|----|----|
| 数据 | S1K chat SFT（与剪枝校准同分布） |
| LoRA | r=16, alpha=32, dropout=0, `target_modules={gate,up,down}_proj`, bias=none |
| 优化 | AdamW, lr=1e-4, cosine, warmup 3%, max_steps=500, max_grad_norm=1.0 |
| batch | per_device=1, grad_accum=8, bf16, grad ckpt（`use_reentrant=False`） |
| Loss | 分块 causal-LM CE：lm_head 按 512 token 分 chunk 累加 `CE(reduction=sum)`，最后除有效 token 数——与全序列 mean CE 严格等价，但 logits 不同时驻留显存 |
| 输入 | 剪枝 HF 目录 + `pruning_artifacts/block_masks.pt` + `pruning_summary.json`（从 summary 读 `block_height/block_width`） |
| 导出 | merge 后标准 HF 目录 + 拷贝 `block_masks.pt` + `lora_train_summary.json` |

### 7.3 DiT 适配：机制复用，训练循环重写

- §7.1 全部复用；`target_modules` 换成 `["fc1","fc2"]`（或 `ff.net.0.proj/ff.net.2`，
  按 peft 后缀匹配规则）。
- 训练循环不能复用 HF Trainer + 分块 CE，写成 **diffusion 微调循环**：

```text
model = load(pruned_dir)                       # bf16；VAE / text encoder 冻结不进图
peft_model = wrap_with_masked_lora(model, masks, H, W, r=16, alpha=32)
assert_only_lora_trainable(peft_model)
for step in range(max_steps):
    x0, c = next(train_loader)                 # 与剪枝校准同数据源，可放大样本量
    t = rand_t(gen); eps = randn(gen)          # 固定 seed generator
    x_t = schedule(x0, eps, t)
    pred = peft_model(x_t, t, c)               # MaskedLoraLinear forward，ΔW⊙M
    loss = mse(pred, target)                   # 与 §3.6 同一 loss 与归约方式
    loss.backward(); clip_grad_norm_(1.0); optimizer.step(); scheduler.step()
merged = merge_and_verify(peft_model, masks)   # 逐块验证已剪块仍为零
merged.save_pretrained(output_dir); copy block_masks.pt
```

- 训练数据与剪枝校准**同分布**（LLM 侧用同一 S1K 源就是这个原因），数据量与步数按恢复缺口调；
- 初始超参可照搬 §7.2：lr=1e-4、cosine、warmup 3%、grad ckpt、bf16。

---

## 8. 评测

**LLM 侧**（仅作参考）：ppl + 仓库根 `main.py`（vLLM + lighteval）跑下游任务；
非 reasoning 下游（ARC/MMLU）用 lm_eval，reasoning（MMLU-Pro/GSM8K）用 lighteval。

**DiT 侧**三层验收：

1. **固定 seed 出图对比**：同批 prompt + 同组初始噪声，dense vs 剪枝 vs 剪枝+LoRA 逐对出图，
   查模式崩塌与细节丢失；
2. **数值指标**：标准协议采样（固定采样器与步数）算 FID / IS / CLIP-score，与 dense 基线同表对比；
3. **稀疏率审计**：读 `pruning_summary.json` 确认各方法实际块稀疏率一致，
   `per_matrix_report.csv` 检查无矩阵被剪穿（单矩阵稀疏率应 ≤ `max_prune_ratio_per_matrix`）。

---

## 9. 迁移路线图

### 9.1 组件复用 / 适配对照表

| 组件 | 本仓库文件 | 动作 |
|------|-----------|------|
| 块归约 / mask 展开 / 整除校验 | `block_utils.py` | 复用（约 120 行，可直接抄） |
| Mask 分配器（全部变体） | `mask_allocator.py` | 复用（约 760 行，可直接抄） |
| magnitude / random 打分 | `magnitude_scorer.py` / `gradient_scorer.py` | 复用 |
| Fisher 打分框架（hook、offload、先平方后平均） | `gradient_scorer.py` | **适配**：loss 换 diffusion loss（§3.6） |
| Wanda 输入 RMS + 块分 | `wanda_scorer.py` | 复用 |
| 目标注册表 | `mlp_registry.py` | **适配**：识别 fc1/fc2（§2.4） |
| 置零与逐块校验 | `mask_apply.py` | 复用（75 行，可直接抄） |
| 产物读写（§6.3 schema） | `serialization.py` | 复用（格式不变） |
| 中间维置换 | `mlp_permutation.py` | **适配**：二元组版（§5.1） |
| Residual 置换 | `residual_permutation.py` | **适配**：mount 分类器（§5.2），首轮可关 |
| Masked LoRA 机制 | `peft_masked_lora.py` | 复用，改 `target_modules`（§7.3） |
| SFT 训练入口 | `train_mlp_lora_sft.py` | **重写**：diffusion 训练循环（§7.3） |
| 评测 | `eval_*.py` / `main.py` | **重写**：生成指标（§8） |

### 9.2 推荐顺序

1. `magnitude` 打分 + 分配 + 置零 + 导出 + 固定 seed 出图：验证 registry 与导出链路；
2. Wanda / Fisher 打分接入（§3.6），跑 `fisher_budget_wanda` 对比 magnitude 基线；
3. Masked LoRA diffusion SFT（§7.3），确认 merge 后已剪块逐块为零；
4. （可选）中间维二元置换（§5.1）；
5. （可选）residual 全局置换（§5.2）：先只接 MLP/attention mount，adaLN 扇出最后接。

---

## 附录 A. 完整配置参数表（剪枝入口）

| 参数 | 默认 | 说明 |
|------|------|------|
| `model_path` / `output_dir` | — | 输入模型与输出目录 |
| `block_size` | `128` | `128` 或 `HxW`（如 `64x128`）；不可整除即报错 |
| `target_block_sparsity` | `0.30` | 全局目标块稀疏率，∈(0,1) |
| `score_type` | `fisher` | `fisher` / `magnitude` / `random` / `fisher_budget_wanda` |
| `calibration_dataset` | `wikitext2` | LLM 侧：`s1k` / `wikitext2` / `c4` / `ptb` |
| `calibration_samples` | `128` | 校准样本条数 |
| `sequence_length` | `2048` | 固定窗长；s1k 下 `0`=不截断 |
| `score_batch_size` | `1` | **必须为 1**（Fisher 可比性） |
| `max_prune_ratio_per_matrix` | `0.60` | 单矩阵剪块比例上限 |
| `min_keep_blocks_per_matrix` | `1` | 单矩阵最少保留块数 |
| `share_up_gate_mask` | `false` | up/gate 共享块坐标（DiT 不适用） |
| `projection_prune_shares` | 空 | 如 `gate_proj=1,up_proj=1,down_proj=2` |
| `pruning_rounds` | `1` | >1 时每轮按 \(\rho\cdot r/R\) 增量剪并重打分 |
| `mlp_permutation` | `none` | `none` / `wanda_shared` |
| `residual_permutation` | `none` | `none` / `block_loss` |
| `residual_perm_search_steps` | `2000` | 交换搜索步数 |
| `residual_channel_agg` | `equal` | 6 种聚合，见 §5.2 |
| `seed` | `42` | 抽样、random 基线、搜索 |
| `dtype` | `bfloat16` | 权重加载精度 |
| `gradient_checkpointing` | `true` | Fisher 打分省显存 |

参数组合的合法性校验（非法即报错，实现时照做）：
`residual_channel_agg` 的 fisher 类取值要求 `score_type ∈ {fisher, fisher_budget_wanda}`；
`residual_permutation=block_loss` 拒绝 `score_type=random`；
`share_up_gate_mask` 要求 gate/up 份额相等。

## 附录 B. DiT 侧需新实现组件的接口规格

以下四个组件是 DiT 仓库里必须新写的，接口对齐后即可复用其余全部逻辑：

```python
# B1. 校准数据（§3.6）
def build_dit_calibration_batches(
    dataset, vae, text_encoder, scheduler,
    num_samples: int, seed: int,
) -> list[dict]: ...
    # 每条返回 {"x0": latent[1,C,H,W], "cond": c, "t_sampler": fn(gen), }
    # 或直接返回预采样好的 (x_t, t, eps, target, cond) 五元组

# B2. Fisher loss（§3.6）
def diffusion_loss(model, batch, generator) -> torch.Tensor: ...
    # 前向 + 训练同款 MSE；返回标量 loss，调用方 .backward()

# B3. MLP 注册表（§2.4）
def collect_dit_mlp_linears(model, H, W) -> list[Target]: ...
    # Target = (module_name, module, layer_index, proj_type∈{fc1,fc2})

# B4. Residual mount 分类器（§5.2，首轮可跳过）
def classify_dit_residual_dim(name, shape, hidden) -> int | None | str: ...
    # 返回 permute dim；"fanout:k" 表示 adaLN 扇出；None 表示忽略；无法判定即抛错
```

## 附录 C. LLM 侧已验证参考（Qwen3.5-27B）

- 维度：\(d_{model}=5120\)，\(d_{ff}=17408\)；128×128 块下
  gate/up 网格 136×40、down 网格 40×136，整除关系成立。
- 推荐首轮对比实验：`block_size=128`，`sparsity=0.20`，s1k 128 条不截断，
  `pruning_rounds=1`，方法对比 `magnitude / fisher / fisher_budget_wanda`。
- Fisher 打分显存：27B 模型 + s1k 长样本需多卡 `device_map=auto` + grad ckpt；
  块分数全部 CPU float64 累加，GPU 不留全层梯度。
