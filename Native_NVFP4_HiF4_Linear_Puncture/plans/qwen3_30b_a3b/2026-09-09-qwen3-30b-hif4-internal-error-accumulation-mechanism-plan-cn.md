# Qwen3-30B-A3B NVFP4→HiF4 模型内部误差累计机制完整实验计划

> 日期：2026-09-09  
> 模型：`nvidia/Qwen3-30B-A3B-NVFP4`  
> 状态：**当前模型内部误差累计机制唯一执行计划**  
> 正式数值路径：**真实 vLLM TP2 actual path**  
> 核心研究对象：**结构性误差累计、层保护、Router 对损失的贡献、累计感知优化目标**  
> 原则：不恢复 semantic replay；不研究 token trajectory；不重新验证 decode-length 单调累计；所有正式机制证据必须来自相同输入/相同 forced history 下的真实 vLLM 内部状态。

---

# 0. 本计划研究的到底是什么

本计划不再研究：

```text
生成越长
→ 数值误差越大
→ token trajectory 分叉
```

已有真实 vLLM TP2 forced-history 结果已经否定“固定 history 下误差随 decode token 数持续单调恶化”作为主机制。

本计划研究的是另一个问题：

> **在完全相同 token history 下，NVFP4→HiF4 引入的数值误差，如何沿 Transformer 层深和 Attention/MoE 结构被持续注入、传播、相干放大或抵消；哪些层真正贡献最终损失；Router 在其中贡献多少；以及什么优化目标最能抑制这种累计。**

因此主轴由：

```text
decode position
```

改为：

```text
layer depth / operator depth
```

Token 只作为模型真实状态样本，不作为 trajectory 研究对象。

---

# 1. 与已有实验的关系

## 1.1 继续承认并复用的事实

最新统一评测已经完成：

- clean controlled LCB：E0 95/175，E1 89/175，delta=-3.429pp，统计上未确认显著 clean format collapse；
- MMLU-Pro300：E0=70.67%，E1=70.67%；
- 因此不能再把旧 `95/175→34/175` 写成纯格式转换掉点；
- fixed-history decode-length monotonic accumulation 不支持；
- router top-k change 不是必要条件；
- 旧 real-vLLM frontier 显示主要大跳偏向 MoE，但旧 frontier 的 `layer_out rel-L2 jump` 不是严格的“累计 residual error”定义。

## 1.2 继续复用的基础设施

直接复用：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/real_vllm_hooks/
```

包括：

- 真实 vLLM TP2；
- single-request incremental KV decode；
- E0 exact forced token history；
- worker 内 hook；
- `input_norm` / `post_attn_norm` residual capture；
- `o_proj` / `moe_out` / `router_logits`；
- `raw_logits`；
- production same-input puncture；
- frozen-router primitive；
- repeatability / identity gate。

## 1.3 禁止恢复

```text
causal_replay.py
run_semantic_matrix.py
semantic decoder
手写 attention replay
Python expert loop
TP2 simulator
```

任何离线分析只能消费真实 vLLM 已捕获张量，不能重新实现一套模型语义。

---

# 2. 一个必须先修正的旧分析语义

当前 Qwen3-MoE vLLM decoder layer 的真实结构是：

```text
input residual
  ↓
input RMSNorm
  ↓
Attention
  ↓
post-attention RMSNorm + residual add
  ↓
MoE
  ↓
下一层 input RMSNorm + residual add
```

`layer_out.branch` 是当前层 MoE/MLP branch 输出，不是完整累计 residual stream。

因此旧分析：

```text
layer_out[l].rel_l2 - layer_out[l-1].rel_l2
```

只能描述相邻 branch output 差异，不能严格解释“模型内部误差怎么累计”。

本计划正式累计状态固定改成：

```text
input_norm.updated_residual
post_attn_norm.updated_residual
final_norm.updated_residual
```

其中：

- layer 0 起始 residual `R_0`：使用 layer0 `layer_in.branch`，因为 layer0 进入时 residual=None；
- layer l Attention 后 residual `R_l^A`：`post_attn_norm.updated_residual`；
- layer l MoE 后 residual `R_{l+1}`：下一层 `input_norm.updated_residual`；
- layer 47 MoE 后最终 residual `R_48`：`final_norm.updated_residual`。

这是本计划最重要的语义修正。

---

# 3. 核心数学定义：建立严格的“误差账本”

对同一个 calibration sample、同一个 forced E0 token history、同一个 decode state：

```text
E0 = Native NVFP4
E1 = Direct HiF4
```

定义第 l 层进入 Attention 前的完整 residual：

\[
R_l^0,\quad R_l^1
\]

累计误差：

\[
\delta R_l = R_l^1 - R_l^0
\]

Attention branch 输出：

\[
A_l^0,\quad A_l^1
\]

MoE branch 输出：

\[
M_l^0,\quad M_l^1
\]

branch 差异：

\[
\delta A_l = A_l^1-A_l^0
\]

\[
\delta M_l = M_l^1-M_l^0
\]

真实 residual 结构给出严格恒等式：

\[
R_l^A = R_l + A_l
\]

\[
R_{l+1}=R_l^A+M_l
\]

因此跨 E0/E1：

\[
\boxed{\delta R_l^A=\delta R_l+\delta A_l}
\]

\[
\boxed{\delta R_{l+1}=\delta R_l^A+\delta M_l}
\]

最终：

\[
\boxed{
\delta R_{48}
=
\delta R_0
+
\sum_{l=0}^{47}\delta A_l
+
\sum_{l=0}^{47}\delta M_l
}
\]

相同 token IDs 下 embedding 不应改变，因此预期：

\[
\delta R_0=0
\]

必须实测，不允许默认。

这条恒等式是整个实验的唯一“累计”主定义。

---

# 4. 误差为什么会累计：不能只看 local NMSE

对于一个已有累计误差 `e`，当前 branch 新增误差 `c`：

\[
e'=e+c
\]

误差能量变化：

\[
\boxed{
\|e'\|^2-\|e\|^2
=
\|c\|^2+2e^Tc
}
\]

因此 local error 很大，不代表最终损失一定大。

必须同时看：

1. `||c||`：当前 branch 本身产生多少误差；
2. `cos(e,c)`：当前误差与已有累计误差方向是否一致；
3. `ΔE`：这一 branch 最终让累计误差能量增加还是减少。

定义 Attention 累计能量增量：

\[
G_l^A
=
\|\delta R_l^A\|^2-\|\delta R_l\|^2
\]

定义 MoE 累计能量增量：

\[
G_l^M
=
\|\delta R_{l+1}\|^2-\|\delta R_l^A\|^2
\]

其中：

- `G > 0`：该结构净放大累计误差；
- `G < 0`：该结构净抵消已有误差；
- `G ≈ 0`：虽有 local error，但对累计误差净贡献小。

严格 telescoping：

\[
\boxed{
\|\delta R_{48}\|^2-\|\delta R_0\|^2
=
\sum_l G_l^A+
\sum_l G_l^M
}
\]

禁止使用各层不同分母的 rel-L2 直接做这个求和。

正式报告同时给：

- raw L2 / squared-L2：用于严格闭合；
- local rel-L2：用于跨状态尺度比较；
- 统一按 `||R_48^0||^2` 或 `||delta R_48||^2` 归一化的 signed contribution：用于跨样本聚合。

---

# 5. 结构性误差累计需要回答的四个研究问题

## RQ1：误差是随机叠加，还是具有结构性相干累计？

核心指标：

### 5.1 逐层累计曲线

\[
E_l=\|\delta R_l\|^2
\]

以及 Attention/MoE 后的中间状态。

### 5.2 方向对齐

\[
\rho_l^A=\cos(\delta R_l,\delta A_l)
\]

\[
\rho_l^M=\cos(\delta R_l^A,\delta M_l)
\]

若大量危险层 `rho>0`，说明误差不是随机噪声，而是顺着已有错误方向相干累积。

### 5.3 全模型 coherence ratio

设 96 个 branch error 向量：

```text
c_0^A, c_0^M, ..., c_47^A, c_47^M
```

定义：

\[
\boxed{
\kappa
=
\frac{\|\sum_i c_i\|^2}{\sum_i\|c_i\|^2}
}
\]

解释：

- `kappa≈1`：近似正交/随机叠加；
- `kappa>1`：存在相干放大；
- `kappa<1`：整体存在显著抵消。

分别计算：

- 全模型；
- Attention-only；
- MoE-only；
- early layers 0-15；
- middle layers 16-31；
- late layers 32-47。

### 5.4 final-error direction attribution

定义最终 residual error：

\[
u=\delta R_{48}
\]

每个 branch 的 signed attribution：

\[
\boxed{
s_i=
\frac{c_i^T u}{\|u\|^2}
}
\]

若 `delta R_0=0`，应满足：

\[
\sum_i s_i=1
\]

`s_i<0` 表示该 branch 在最终误差方向上起抵消作用。

### 5.5 branch-error cosine matrix

输出 96×96 pairwise cosine heatmap，并给 16-layer segment 聚合图。

目的不是展示漂亮图，而是回答：

> 不同层产生的误差方向是否在 residual space 中持续一致，从而解释为什么少量 local error 可以形成显著 final residual error。

这里必须明确：**residual-space cosine 只能解释内部误差能量为什么相干放大/抵消，不能直接等价为最终输出 KL 的贡献。** 最终 KL 还取决于后续层、final RMSNorm、LM Head 和 softmax 对误差方向的敏感性。

### 5.5A 从内部累计误差到最终输出 KL：建立正式桥梁

本计划最终不是只解释 `||delta R_48||`，而要回答：

> **哪些内部误差方向最终真的改变了输出概率分布，以及它们如何共同形成 E0/E1 的 KL 散度。**

首先验证 E0/E1 的 LM Head 权重/计算语义一致，并做 **raw-logit reconstruction identity gate**：分别用捕获的 `final_norm.normalized` 经过各自实际 LM Head 计算路径重建 logits，与同一次 actual-vLLM forward 捕获的 raw logits 比较。只有 E0/E1 两侧都落入各自 measured repeatability envelope，且两侧 LM Head 参数/TP 语义确认一致后，才允许使用共享线性映射 `W` 推导 `delta ell = W delta z`。若任一 gate 不通过，final exact KL 仍直接使用 actual raw logits，但禁止使用该线性桥梁/Fisher 分解下结论。若 identity gate 通过，记 final RMSNorm 后 hidden：

\[
z_0,\quad z_1
\]

\[
\delta z=z_1-z_0
\]

相同 LM Head `W` 下，logit 扰动严格为：

\[
\boxed{\delta \ell=W\delta z}
\]

其中：

\[
\ell_0=Wz_0,\qquad \ell_1=Wz_1
\]

正式输出损失始终使用 exact：

\[
\boxed{
KL_{exact}=KL(p_0\|p_1),\quad
p_0=softmax(\ell_0),\ p_1=softmax(\ell_1)
}
\]

为了理解 KL 为什么变大，再计算 softmax 的局部二阶近似。定义 E0 softmax Fisher：

\[
F_0=diag(p_0)-p_0p_0^T
\]

当 logit perturbation 足够局部时：

\[
\boxed{
KL(p_0\|p_1)
\approx
\frac{1}{2}\delta\ell^T F_0\delta\ell
}
\]

并且：

\[
\delta\ell^T F_0\delta\ell
=
Var_{j\sim p_0}[\delta\ell_j]
\]

因此最终 KL 不是由 hidden/residual L2 单独决定，而是由：

```text
内部误差方向
→ 后续网络传播
→ final RMSNorm
→ LM Head 投影
→ softmax Fisher 敏感方向
```

共同决定。

必须同时报告：

```text
exact KL
Fisher quadratic KL
quadratic / exact ratio
absolute approximation error
```

若二阶近似在某 state 上明显失效，该 state 的因果结论继续使用 exact KL；Fisher 分解只作为局部机制解释，禁止用近似值替代正式 KL。

### 5.5B 两种 cosine 必须严格区分

#### A. residual-space cosine：解释“内部误差为什么累计”

对于已有累计误差 `e` 和当前新 branch error `c`：

\[
\rho_{res}=cos(e,c)=\frac{e^Tc}{\|e\|\|c\|}
\]

因为：

\[
\|e+c\|^2-\|e\|^2
=
\|c\|^2+2\|e\|\|c\|\rho_{res}
\]

所以：

- `rho_res > 0`：当前误差与已有误差同向，产生**相干放大**；
- `rho_res ≈ 0`：近似正交，主要按能量独立叠加；
- `rho_res < 0`：当前误差在 residual space 中**抵消**已有误差。

这就是本计划中 `cos(delta R, delta A/M)` 的物理意义。

#### B. KL/Fisher-space cosine：解释“这些误差为什么共同放大最终 KL”

对经过真实后续网络以后，在 final logits 上形成的两个扰动：

\[
v_i,\quad v_j
\]

定义 Fisher 内积：

\[
\langle v_i,v_j\rangle_F=v_i^T F_0v_j
\]

以及：

\[
\boxed{
\rho_{KL}(i,j)=
\frac{v_i^T F_0v_j}
{\sqrt{(v_i^T F_0v_i)(v_j^T F_0v_j)}}
}
\]

则局部二阶 KL 中：

\[
\frac12\left(\sum_i v_i\right)^T
F_0
\left(\sum_i v_i\right)
=
\frac12\sum_i v_i^TF_0v_i
+
\sum_{i<j}v_i^TF_0v_j
\]

因此：

- `rho_KL > 0`：不同层/branch 的 logit effect 在 **模型决策敏感方向**上相互增强；
- `rho_KL ≈ 0`：它们对 KL 近似独立；
- `rho_KL < 0`：它们在最终概率空间中相互抵消。

这比普通 residual cosine 更直接对应最终 KL。

### 5.5C 如何在真实 vLLM actual path 上获得 branch 的 logit effect

不手写 Jacobian，不恢复 semantic replay。

对 selected branch `i` 使用后面的 one-step oracle/format-only repair：

```text
baseline logits = l_E1
repair-i logits = l_E1_without_i_error
```

定义真实下游传播后的 branch logit effect：

\[
\boxed{
v_i=\ell_{E1}-\ell_{repair-i}}
\]

它已经包含：

```text
当前 branch error
+ 后续 47-l 层真实 E1 propagation
+ final RMSNorm
+ LM Head
```

因此 selected layers 必须额外计算：

```text
residual-space cos
Fisher-space cos
exact single-layer KL gain
```

注意：由 single-branch repair 得到的 `v_i` 是真实 downstream causal logit effect，但后续 Transformer 是非线性的，因此多个 `v_i` **不能默认严格线性相加**。正式因果结论始终以 exact single-/multi-repair KL 为准；Fisher-space cosine 只解释局部二阶相互增强/抵消。若需要验证两个高贡献 branch 的交互，只对少量 Top pairs 增加 double-repair，并比较 exact pair gain 与单层 gain 之和，不做全 96×96 pair intervention。

三者联合回答：

1. 这层是不是在内部 residual 上放大已有误差？
2. 这层产生的误差方向是否最终落到 softmax 敏感方向？
3. 真正修掉这层误差后 exact KL 能下降多少？

### 5.5D 逐层累计误差对最终 KL 的 one-step state-reset 曲线

为了直接回答“累计到第 l 层的内部误差怎样影响最终 KL”，增加一个只看当前 predictor 的 state-reset 实验，不生成新 token。

该 state-reset 曲线只定义在 **layer 1–47**。layer0 进入模型时 `residual=None`，`input_norm` 不返回 `(normalized, updated_residual)` 二元组，因此禁止把 layer0 强行套入同一 tuple intervention。layer0 仅作为初始误差锚点 `delta R_0` 参与 residual ledger，不参与本 state-reset 曲线。

在 E1 full forward 的 layer `l` (`l=1..47`) `input_norm` 边界，把：

```text
(normalized_E1, updated_residual_E1)
```

替换为同一 state 捕获的：

```text
(normalized_E0, updated_residual_E0)
```

后续 layer `l..47` 全部继续真实 E1 forward。

定义：

\[
KL_{base}=KL(p_0\|p_1)
\]

\[
KL_{reset,l}=KL(p_0\|p_{E1,reset\ at\ l})
\]

\[
\boxed{
C_l^{upstream-KL}=KL_{base}-KL_{reset,l}
}
\]

解释：

- `C_l^{upstream-KL} > 0`：截至 layer l 已累计的 state error 对最终 KL 有净正贡献；
- `≈0`：此前累计误差被后续网络基本吸收/不敏感；
- `<0`：此前误差反而在补偿后续 E1 error，直接 reset 会让 KL 更差。

必须做 no-op：把 E1 tuple 替换成捕获的 E1 tuple，最终 logits 必须落入 baseline repeatability envelope。

该曲线与 `||delta R_l||` 曲线并排画，从而直接检验：

> **residual error 越大，是否真的意味着 final KL 越大；如果不一致，差异是否由方向/Fisher sensitivity 解释。**

---

## RQ2：哪些 Transformer layer / 子结构 / Linear 真正决定最终损失？

这里不预设 MoE 比 Attention 更重要，也不把“层保护”先定义成 MoE-only。正式采用**三级因果归因**：

```text
Level 1：Transformer layer importance
    ↓
Level 2：Attention vs MoE
    ↓
Level 3：若 Attention 重要，再拆 Q / K / V / O
```

所有 importance 都以同一固定 state 下对 **final exact KL** 的因果改善为主，不以 local NMSE 直接决定。

### 5.6 Level 1：whole-Transformer-layer oracle repair

对每个 layer `l`，在 E1 full forward 中联合修复这一层的两个 residual branch：

```text
Attention branch -> captured E0 Attention branch
MoE branch       -> captured E0 MoE branch
```

E1 进入该层之前的 upstream residual 保留，后续 layer `l+1..47` 继续真实 E1 forward。它回答：

> **如果第 l 个 Transformer layer 当前产生的总 branch deviation 被整体消掉，最终 KL 能恢复多少？是否存在少数 layer 对最终精度特别敏感？**

定义绝对 KL 因果改善：

\[
\boxed{
P_{l,abs}^{layer}
=
KL(E0\|E1)-KL(E0\|E1+repair\ whole\ layer\ l)
}
\]

并同时记录该 state 的相对恢复比例：

\[
\boxed{
P_{l,frac}^{layer}
=
\frac{KL(E0\|E1)-KL(E0\|E1+repair\ whole\ layer\ l)}
{KL(E0\|E1)}
}
\]

其中正式 **Transformer-layer causal importance** 的主排序只使用 sample 聚合后的 `P_layer_abs`；`P_layer_frac` 仅用于解释“该层能恢复当前 state 总 KL 的多少比例”。当 baseline KL 为 0 或落入 measured numerical floor 时，`P_layer_frac` 记为未定义，不参与排序。

禁止人为设置“恢复超过 X% 才算敏感层”之类阈值。正式结论直接报告 48 层 `P_layer_abs` 分布、Top-1/2/4/8、贡献集中度以及是否呈现明显少数层/长尾结构。

必须有 whole-layer no-op：联合替换 captured E1 Attention + captured E1 MoE branch，最终 raw logits 必须落入 E1 repeatability envelope。

### 5.7 Level 2：同一层拆 Attention vs MoE

对同一个 layer 独立做：

\[
P_l^{A}
=
KL(E0\|E1)-KL(E0\|E1+repair\ Attention_l)
\]

\[
P_l^{M}
=
KL(E0\|E1)-KL(E0\|E1+repair\ MoE_l)
\]

这样直接回答：

> **在同一个敏感 Transformer layer 内，最终损失主要来自 Attention 还是 MoE？**

同时记录 whole-layer 交互项：

\[
I_l^{A,M}
=
P_l^{layer}-P_l^A-P_l^M
\]

`I_l^{A,M}` 不是要求为 0；后续网络非线性意味着 Attention/MoE 的 causal gains 不必可加。它用于识别“两个子结构单独看都一般，但联合修复收益很大”或相互补偿的层。

### 5.8 Level 3：Attention 重要时继续拆 Q / K / V / O

如果某些 layer 的 `P_l^A` 稳定为正且在该层与 `P_l^M` 可比或更大，则进一步做 Attention Linear 级因果消融。

Qwen3-MoE vLLM 当前实际实现中：

```text
qkv_proj(hidden)
-> fused [Q | K | V] rank-local output
-> split(q_size, kv_size, kv_size)
-> Q/K norm + RoPE
-> attention core
-> o_proj
```

因此禁止为 Q/K/V 重新实现 Attention。直接在真实 forward 的 `qkv_proj` 输出上做 slice-level intervention：

- Q-only repair：只把 `[Q]` slice 替换为对应 E0 / source-format result；K、V 保留 E1；
- K-only repair：只替换 `[K]` slice；
- V-only repair：只替换 `[V]` slice；
- O-only repair：在真实 attention-core 输出不变时，只修复 `o_proj`。

定义最终 exact-KL causal gain：

\[
P_l^Q,\quad P_l^K,\quad P_l^V,\quad P_l^O
\]

每项仍定义为：

\[
P=KL(E0\|E1)-KL(E0\|E1+repair)
\]

这一级回答：

> **如果 Attention 确实是主要损失来源，究竟是 Q、K、V 还是 O projection 更值得保护/优化？**

Q/K/V intervention 必须使用运行时真实 `q_size/kv_size` 切片和 TP rank-local 输出；禁止按 checkpoint shape 猜切片边界。所有 slice repair 都必须先做 captured-E1 slice no-op identity。

### 5.9 Format-only causal protection：把“总 deviation”进一步收缩到当前格式误差

Oracle repair 衡量的是该 layer/branch 当前总 deviation 的因果上限。若要回答真正的 NVFP4→HiF4 当前格式误差，必须保持 E1 upstream input 不变，用 source E0 production operator 在同一个 E1 input 上计算。

MoE：

```text
x1 = E1 actual post_attn_norm input
M0(x1) vs M1(x1)
```

Attention Linear：

```text
x1 -> E0/E1 production qkv_proj(x1)，只替换 Q/K/V 指定 slice
E1 attention_core -> E0/E1 production o_proj，同输入比较 O
```

由此分别得到：

```text
P_format^M
P_format^Q
P_format^K
P_format^V
P_format^O
```

从而区分“该结构重要”与“该结构的 HiF4 格式误差重要”这两个不同问题。

### 5.10 Practical source-format protection：真正可部署的层/子结构保护

只有前述 causal study 确认稳定敏感的 Transformer layers / modules / linears 后才启动，不预设保护 scope。

当前源模型本身就是 NVFP4 checkpoint，没有 pre-quant BF16 teacher。W-source 只能表示**源 NVFP4 checkpoint 所表示权重值的 BF16 dense 反量化执行**。

对 Top sensitive Transformer layers，实际 protection scope 至少比较：

```text
Attention-only
MoE-only
whole Transformer layer = Attention + MoE
```

若 Attention-only 明显重要，再增加：

```text
Q-only
K-only
V-only
O-only
最佳 Attention linear 组合
```

每个 scope 内再做必要的 source ablation：

- A-only：保留 HiF4-converted weight value，关闭该 scope 的 HiF4 activation QDQ；
- W-source：source-NVFP4-dequant BF16 dense weight，activation 仍按 HiF4；
- W+A-source：source-NVFP4-dequant BF16 dense weight + 关闭 activation QDQ。

在把 W-source 解释为 source-format protection 前，必须通过对应 operator 的 native E0 NVFP4 vs source-NVFP4-dequant BF16 dense frozen-input parity gate；失败则只能标记为 dense-dequant control。

Router 始终保持原 BF16，不纳入保护位宽变化。

必须保证：

- 未保护层/未保护模块与 E1 execution path 完全不变；
- 不允许全局 backend 静默改变；
- 不允许用 state patch 冒充 deployable protection；
- 若无法局部实现，则标记 `PRACTICAL_PROTECTION_BLOCKED`。

最终保护预算仍做：

```text
K = 1, 2, 4, 8 Transformer layers
```

层选择至少比较：

1. local reconstruction/NMSE ranking；
2. structural positive gain ranking；
3. final-direction attribution ranking；
4. whole-layer causal `P_l^layer` ranking；
5. random control：每个 K 固定 3 组随机层集合。

每个 K 的正式 variant 需要报告：

- exact final KL；
- final residual / final hidden NMSE；
- Attention/MoE causal source；
- protected parameter ratio / memory overhead；
- 最终只对最佳 1-2 组跑 MMLU-Pro300。

不无条件重跑 LCB。

---

## RQ3：Router 对累计误差和最终 loss 到底贡献多少？

### 5.9 先纠正概念：Router 不是本计划里的“量化误差源”

当前实验约定 Router/Gate 为 BF16。

因此真正的问题不是：

```text
Router 自己被 HiF4 量化了多少？
```

而是：

> **上游 residual error 改变了 BF16 Router 输入，Router 因此改变 routing weight / expert ID，进而把已有误差继续放大了多少？**

所以 Router contribution 应从 upstream propagation `p` 里拆，不从 pure local format injection `q` 里拆。

### 5.10 MoE 总误差先做 q/p 分解

对于同一个 layer/state，先固定 Router 语义，避免后续 q/p/router 三项分解出现定义漂移。

记真实捕获的：

```text
r0 = E0 Gate(x0) router logits
r1 = E1 Gate(x1) router logits
```

定义：

\[
y_0=M_0(x_0;r_0)
\]

\[
y_{00}=M_1(x_0;r_0)
\]

\[
y_{11}=M_1(x_1;r_1)
\]

其中 `M1(x0;r0)` 必须显式 frozen E0 router，而不是默认“E1 natural Gate(x0) 应该等于 r0”。后者可以做 identity 审计，但不能作为代数定义的隐含前提。

定义：

\[
\boxed{q=y_{00}-y_0}
\]

`q` = 在**相同 E0 input + 相同 E0 routing**下，E0→E1 MoE operator/format 本身新注入的误差。

再定义：

\[
\boxed{p=y_{11}-y_{00}}
\]

`p` = 从 E0 state/routing 变成 E1 actual state/routing 后的 upstream propagation 总项。

必须验证：

\[
\boxed{\delta M_l=q+p}
\]

并记录：

- `||q||`；
- `||p||`；
- `cos(q,p)`；
- closure error；
- `q` / `p` 对 `G_l^M` 的所有能量展开项。

MoE energy gain 完整展开：

\[
G_l^M
=
\|q\|^2+
\|p\|^2+
2(\delta R_l^A)^Tq+
2(\delta R_l^A)^Tp+
2q^Tp
\]

禁止只用 `||q||` 大小解释累计损失。

### 5.11 把 p 进一步拆成 expert propagation + router propagation

记：

```text
r0 = Gate(x0)
r1 = Gate(x1)
```

构造：

\[
y_{00}=M_1(x_0;r_0)
\]

\[
y_{10}=M_1(x_1;r_0)
\]

\[
y_{11}=M_1(x_1;r_1)
\]

定义：

\[
\boxed{
p_{expert}=y_{10}-y_{00}}
\]

表示 routing 固定为 E0 时，上游输入误差经过 expert compute 的传播。

定义：

\[
\boxed{
p_{router}=y_{11}-y_{10}}
\]

表示在同一个 E1 输入 `x1` 下，从 E0 routing 恢复到 E1 routing 带来的额外误差。

必须验证：

\[
\boxed{p=p_{expert}+p_{router}}
\]

最终：

\[
\boxed{
\delta M_l
=q+p_{expert}+p_{router}
}
\]

这是本计划 Router contribution 的正式定义。

### 5.12 Router 不只看 top-k change

每个 layer/state 同时记录：

- router-logit KL；
- top-k exact；
- top-k overlap；
- routing weight L1/L2 difference；
- `||p_router||`；
- Router 对 MoE energy gain 的 signed contribution；
- one-step final KL causal gain。

即使 top-k exact=True，也继续计算 `p_router`，因为 routing weight 改变本身可能有贡献。

### 5.13 只有 Router 确实重要时，才继续拆 expert-ID 与 weight

若 selected layers 中 `p_router` 对正累计能量贡献明显，则增加：

```text
r1_on_S0 = 使用 E1 router logits，但 expert set 强制为 E0 top-k set，并在 S0 内重新归一化
```

定义：

\[
p_{weight}
=
M_1(x_1;r1\_on\_S0)-M_1(x_1;r_0)
\]

\[
p_{id}
=
M_1(x_1;r_1)-M_1(x_1;r1\_on\_S0)
\]

验证：

\[
\boxed{p_{router}=p_{weight}+p_{id}}
\]

当 E0/E1 top-k set 相同时，预期：

```text
p_id = 0
p_weight = p_router
```

必须通过 identity 控制。

### 5.14 Router 对最终 loss 的 one-step 因果干预

在 E1 full forward 当前 layer：

```text
baseline: Gate(x1) -> r1
intervention: Gate(x1) 强制输出 E0 captured r0
```

后续层继续真实 E1 forward，只看当前 predictor 的最终 logits，不生成新 trajectory。

定义：

\[
C_l^{router}
=
KL(E0\|E1)
-
KL(E0\|E1\text{ with router l frozen to }r0)
\]

正值表示恢复 E0 routing 能降低最终 output loss。

no-op：强制输出捕获的 E1 `r1`，必须复现 baseline logits。

这一步才是“Router 对 loss 的贡献”的最终因果量。

---

## RQ4：什么优化目标真正适合 HiF4？

当前默认训练目标主要是 local reconstruction，例如：

```text
block_delta_nmse
block_output_nmse
mse
```

但如果误差具有明显相干累计，单层 local reconstruction 可能不是最优目标。

本计划先用 layer protection 的真实 causal gain 作为“金标准”，再比较候选优化目标。

### 5.15 第一阶段：先做目标的预测能力研究，不立刻重训全模型

对每层统计：

1. 现有 local reconstruction score；
2. `Q_l = E ||q_l||^2`；
3. `G_l^M` / `G_l^A`；
4. final-direction attribution `s_l`；
5. Router contribution `C_l^router`；
6. whole-Transformer-layer causal protection gain `P_l^layer`。

以 `P_l^layer` 作为真实 Transformer-layer 重要度金标准；Attention/MoE 与 Q/K/V/O 的子结构预测能力分别以 `P_l^A/P_l^M` 和 `P_l^Q/P_l^K/P_l^V/P_l^O` 为对应金标准。比较：

- Spearman rank correlation；
- Top-K overlap，K={1,2,4,8}；
- sample-bootstrap 95% CI。

目标：回答：

> local NMSE、累计能量增量、final-direction attribution 中，哪个最能预测“保护这一层真的有用”。

### 5.16 第二阶段：正式比较 3+1 个优化目标

在相同 DIAG 自由度、相同初始化、相同数据、相同优化步数和 seed 下，比较：

#### O0：现有基线

```text
block_delta_nmse
```

#### O1：pure local format objective

O1 不再预设只研究 MoE，而是根据前面的 causal attribution 对 Attention / MoE 分别定义。

MoE：

\[
L_q^M
=
\frac{\|M_\theta(x_0)-M_0(x_0)\|^2}
{\|M_0(x_0)\|^2}
\]

Attention：

\[
L_q^A
=
\frac{\|A_\theta(x_0)-A_0(x_0)\|^2}
{\|A_0(x_0)\|^2}
\]

目标：只最小化当前 branch 在相同 E0 input 下的 pure format injection。若因果结果显示 MoE 主导，则主要优化 `D_GU/D_UD`；若 Attention 主导，则主要优化 `D_QKV/D_VO`；若两者都重要，则分别比较后再做联合。

#### O2：cumulative residual objective —— 本计划核心新目标

O2 同样必须与 causal attribution 对齐，不得固定成 MoE-only。

对于 Attention，当前 E1 进入 Attention 前 residual 为 `R_l^1`，candidate Attention 输出为 `A_theta(x1)`：

\[
\hat R_l^{A,\theta}
=R_l^1+A_\theta(x_1)
\]

以 E0 Attention 后 residual `R_l^{A,0}` 为目标：

\[
\boxed{
L_{cum}^A
=
\frac{\|R_l^1+A_\theta(x_1)-R_l^{A,0}\|^2}
{\|R_l^{A,0}\|^2}
}
\]

对于 MoE，当前 E1 Attention 后 residual 为 `R_l^{A,1}`，candidate MoE 输出为 `M_theta(x1)`：

\[
\hat R_{l+1}^{\theta}
=R_l^{A,1}+M_\theta(x_1)
\]

以 E0 下一层 residual `R_{l+1}^0` 为目标：

\[
\boxed{
L_{cum}^M
=
\frac{\|R_l^{A,1}+M_\theta(x_1)-R_{l+1}^0\|^2}
{\|R_{l+1}^0\|^2}
}
\]

它与 local reconstruction 的本质区别：

- local objective 想让当前 branch 自己像 E0；
- cumulative objective 直接让“加入已有 upstream error 后的完整 residual”接近 E0；
- 因而允许 candidate branch 对已有误差做有益抵消，而不是只追求局部最小 NMSE。

正式 objective study 的优化对象由 Task 7 的因果归因决定：

```text
MoE 主导       -> D_GU / D_UD + L_cum^M
Attention 主导 -> D_QKV / D_VO + L_cum^A
两者均重要     -> 先分别消融，再做联合 objective
```

若 Q/K/V/O 进一步显示某个 Linear 特别敏感，但当前 fusable 参数化无法给它独立自由度，则把这一点明确报告为“当前 DIAG 参数化自由度与 causal-sensitive operator 不完全匹配”，不得为了完成实验临时改变主参数化。

这是本计划最重要的优化目标候选。

#### O3：cumulative + Router-aware，正式拆成 full-router 与 production Top-k 两个目标

只有 Router causal study 显示该 layer 的 Router 对 final KL 有稳定正贡献，并且该 layer 已进入前述 `o3_layers` 时才启用。正式比较两个 Router 辅助目标，二者必须使用**同一个匹配 calibration token/state 的 E0 Router teacher**，不能一个用 E0 teacher、另一个用 candidate 自身 Top-k：

```text
O3_full
O3_topk
```

当前模型 `nvidia/Qwen3-30B-A3B-NVFP4` 的：

```text
num_experts = 128
num_experts_per_tok = 8
norm_topk_prob = true
```

但实现中禁止硬编码 `k=8`；`k` 必须从实际加载模型的 `config.num_experts_per_tok` 读取并写入 RUN manifest / objective artifact。若后续换模型，自动随 production 配置变化。

对同一 training/validation token state，记 E0 native Router logits 为 `r0`，candidate 当前 Router logits 为 `r_theta`。E0 teacher 必须来自 native E0 actual path、提前缓存并 `detach`，Router weight 本身保持冻结。

**O3_full：**

\[
p_0=\mathrm{softmax}(r_0),\qquad
p_\theta=\mathrm{softmax}(r_\theta)
\]

\[
\boxed{
L_{router\text{-}full}
=KL(p_0\|p_\theta)
}
\]

\[
\boxed{
L_{O3\_full}=L_{cum}+\lambda L_{router\text{-}full}
}
\]

**O3_topk：** 首先使用 E0 Router 的真实 production Top-k expert set：

\[
S_0=\operatorname{TopK}(r_0,k)
\]

由于当前 production `norm_topk_prob=true`，teacher routing weights 按 E0 Top-k 内归一化：

\[
w_i^0=
\frac{\exp(r_i^0)}{\sum_{j\in S_0}\exp(r_j^0)},
\qquad i\in S_0
\]

candidate 在 **teacher 的同一组 `S0` experts 内**计算归一化权重：

\[
\tilde w_i^\theta
=
\frac{\exp(r_i^\theta)}{\sum_{j\in S_0}\exp(r_j^\theta)},
\qquad i\in S_0
\]

Top-k 内权重 KL：

\[
\boxed{
L_{topk\text{-}weight}
=KL(w^0\|\tilde w^\theta)
}
\]

仅做这个 KL 还不能保证 candidate 的实际 Top-k expert IDs 不发生变化，因为 `S0` 外 logits 不进入上述归一化。因此再加入一个**零 margin 的 support hinge**，只在错误 expert 越过 E0 Top-k 边界时产生梯度：

\[
r_{in,min}^\theta=\min_{i\in S_0}r_i^\theta,
\qquad
r_{out,max}^\theta=\max_{j\notin S_0}r_j^\theta
\]

\[
\boxed{
L_{topk\text{-}support}
=\mathrm{ReLU}\left(r_{out,max}^\theta-r_{in,min}^\theta\right)
}
\]

由于 `S0` 本身来自 E0 的真实 Top-k，candidate=E0 时必有 `r_out,max <= r_in,min`，因此 support loss 为 0；同时 `\tilde w^\theta=w^0`，Top-k weight KL 也为 0。这样不会像“sparse teacher + candidate full-softmax KL”那样额外强迫 E0 Router 自身变尖锐。

正式 Top-k Router objective 固定为：

\[
\boxed{
L_{router\text{-}topk}
=L_{topk\text{-}weight}+L_{topk\text{-}support}
}
\]

其中 support 项内部系数固定为 `1.0`，不单独调参；唯一对外权重仍是与 `O3_full` 共享的 `lambda`：

\[
\boxed{
L_{O3\_topk}=L_{cum}+\lambda L_{router\text{-}topk}
}
\]

因此 `O3_topk` 精确约束的是 production 真正使用的两件事：

```text
E0 Top-k expert identity / support
E0 Top-k 内 renormalized routing weight
```

Top-k 外 full-softmax probability mass 仅作为诊断指标记录，不作为优化项；production 在 `norm_topk_prob=true` 时本来就会丢弃这些 experts 并对 Top-k 重新归一化。

每个 state 额外记录：

```text
router_full_kl
router_topk_weight_kl
router_topk_support_hinge
router_topk_total
candidate_topk_id_match / overlap
candidate_topk_margin = min_teacher_topk_logit - max_outside_logit
outside_teacher_topk_mass（诊断 only）
```

正式 O3 训练只允许 Router auxiliary gradient 进入该 MoE layer 的 Router-coupled DIAG 自由度（当前为 `D_GU`）；E0 teacher、Router weight、current Router input 对 Attention 路径均 detach/freeze。`D_UD` 仍由 `L_cum^M` 获得梯度；不得让 Router auxiliary 反向约束 `D_QKV/D_VO`。

现有 `router_alignment_kl()` 可以作为实现参考，但其旧的“same-current-hidden ref vs folded”语义**不能直接冒充正式 O3_full/O3_topk**。正式目标必须显式消费匹配 token/state 的 E0 teacher logits/cache，保证 full 与 top-k 的 teacher 语义一致。

不得在 Router contribution 接近 0 时为了“看起来完整”强行加 Router loss。`O3_full/O3_topk` 都只在 per-layer `o3_layers` 内运行。

`lambda` 不直接拍脑袋大扫：`O3_full` 与 `O3_topk` 使用完全相同的 `lambda`。先比较 `lambda=0`（即 O2）与当前已有实验常用候选权重；只有 Router objective 出现方向性收益后，再统一增加一个中间点。禁止给 top-k 单独更大的调参预算。

#### O3.1 代码修改细节锁定：Cursor 不得自行改接口或换实现语义

为了避免把本实验专用 Router teacher / Top-k objective 污染到旧 `e2e_diag_reconstruction` 正式路径，代码结构固定为：**最小共享 primitive + `internal_error_accumulation` 专用 selected-layer trainer**。禁止直接重写旧 `router_alignment_kl()` 的语义。

##### A. `core/moe_fold.py` 只做最小共享扩展

新增函数，名字与签名固定：

```python
def folded_router_logits_from_pre_dgu_input(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    diag_state: MoEFusableDiagState,
) -> torch.Tensor:
    ...
```

其数值语义必须逐项复用当前 `router_compensation_logits()` 的 folded branch：`D_GU -> BF16 input scaling -> transformed BF16 router weight -> matmul`。随后让旧 `router_compensation_logits()` 的 folded 分支调用这个 helper；旧 `original` 分支、`router_alignment_kl()`、`router_compensation_topk_gate()`、旧 CLI 和 rollback 语义不得改变。

新 helper 的唯一目的，是允许新 trainer 对 `pre-D_GU router input` 显式 `detach()` 后重算 candidate Router logits，使 Router auxiliary 的梯度只能进入 `D_GU`。

##### B. 新建 `internal_error_accumulation/router_objective.py`

该文件只放纯数学函数，不加载模型、不访问 RUN 状态。接口固定：

```python
def build_production_topk_teacher(e0_logits, *, top_k, norm_topk_prob): ...
def router_full_kl(e0_logits, candidate_logits): ...
def router_topk_weight_kl(topk_ids, topk_weights, candidate_logits): ...
def router_topk_support_hinge(topk_ids, candidate_logits): ...
def router_topk_total(topk_ids, topk_weights, candidate_logits): ...
```

所有正式 loss 输入统一为去掉 padding 后拼接的二维张量：

```text
e0_logits        [N_valid, num_experts]
candidate_logits [N_valid, num_experts]
topk_ids         [N_valid, k]
topk_weights     [N_valid, k]
```

函数内部禁止猜 sequence length 或处理 padding。

`build_production_topk_teacher()` 必须严格复刻 production：先 `softmax(float32)`，再 `topk`，当前正式模型要求 `norm_topk_prob=true`，因此最后在 Top-k 内重新归一化；若 runtime 配置为 false，直接 Gate fail。`k` 只能来自 runtime/model spec，禁止在 loss 函数里写死 8。

`router_full_kl()` 固定 `T=1.0`，实现为 `KL(softmax(E0 logits) || softmax(candidate logits))`，按有效行求 mean，不增加 temperature sweep。

`router_topk_weight_kl()` 必须直接对 `candidate_logits.gather(-1, topk_ids)` 做 `log_softmax`，再计算 `KL(E0 Top-k renormalized weights || candidate weights on the same E0 Top-k set)`；禁止构造 `[N,128]` 稀疏 teacher。

`router_topk_support_hinge()` 必须使用 **E0 teacher Top-k set** 作为 inside mask：

```text
min_in  = min(candidate logits on E0 Top-k experts)
max_out = max(candidate logits outside E0 Top-k experts)
loss    = mean(ReLU(max_out - min_in))
```

零 margin，不增加额外 margin 超参数。`router_topk_total = weight_kl + support_hinge`，support 系数固定 1.0；函数必须同时返回三个分量用于日志。

##### C. 新建 `internal_error_accumulation/router_teacher_cache.py`

对外入口固定：

```python
def build_router_teacher_cache(
    *,
    run_root: Path,
    model_path: str,
    objective_split_manifest: dict,
    layers: list[int],
    batch_size: int,
) -> dict:
    ...
```

只为 `o3_layers` 建 cache；`o3_layers=[]` 时直接返回 EMPTY，不允许无条件扫 48 层。

Teacher reference 必须来自现有 `NativeQwen3MoELayerRuntime` 的 E0 progressive propagation，不允许复用旧 `router_alignment_kl()` 的 same-current-hidden original 分支，也不能用 S1 只有四个位置的 capture 代替训练 teacher。

实现流程锁定：

1. 从 `objective_split_manifest.json` 读取固定 train/val IDs；
2. 加载和 objective trainer 完全相同的 calibration samples 与 `DynamicCalibrationCollator`；
3. 用 `build_initial_moe_hidden_cache(...)` 建立 E0 layer-0 input；
4. 从 layer0 顺序 native forward 到 `max(o3_layers)`；
5. 每层都用 native output 更新 E0 progressive hidden cache，不能把 E1/student hidden 混入 teacher 链；
6. 当前 layer 属于 `o3_layers` 时保存 native `router_logits`；
7. native Router logits 是 flatten 的 `[B*T,E]` 时，必须先按当前 `B,T,E` reshape，再按每个 sample 的真实 length 切 padding；禁止按 flat row 猜 sample 边界；
8. 用同一份 E0 logits 调 `build_production_topk_teacher()` 生成 Top-k IDs / weights；
9. 每层结束释放该 layer master state，不能常驻全部层权重。

目录固定：

```text
runs/<RUN_ID>/60_objective/router_teacher_cache/
  manifest.json
  Lxx_train.pt
  Lxx_val.pt
```

每个 `Lxx_<split>.pt` 是 `dict[sample_id] -> payload`，payload 固定包含：

```text
sample_id
layer
length
router_logits   # [length, num_experts], CPU BF16
topk_ids        # [length, k], CPU int64
topk_weights    # [length, k], CPU float32
```

`manifest.json` 固定保存：`model_path / layers / num_experts / top_k / norm_topk_prob / train_ids / val_ids / source_ratio / status`。`top_k` 必须从实际 loaded spec/HF config 读取并互相校验；当前模型应为 8，但代码不得从计划文字硬编码。`norm_topk_prob` 必须为 true，否则直接失败。

##### D. 新建 `internal_error_accumulation/selected_layer_objective_trainer.py`

旧 `e2e_diag_reconstruction/training/moe_trainer.py` 不承载 O1/O2/O3/O4 的新实验语义，只复用其底层 runtime / model-state / calibration / optimizer primitives。

正式入口固定：

```python
def train_selected_layer_recipe(
    *,
    recipe: dict,
    run_root: Path,
    model_path: str,
    phasea_root: Path,
) -> dict:
    ...
```

O3 recipe 必须显式包含：`layer / loss(O3_full|O3_topk) / params / train_ids / val_ids / router_lambda / router_teacher_cache_dir / seed / steps-or-epochs / optimizer / lr / weight_decay`。禁止继续使用 `O3_conditional`。

一个 MoE O3 batch 的计算顺序固定：

1. 用冻结的 baseline/current-state cache 组装当前层输入，执行 `StudentQwen3MoELayerRuntime(..., use_ste=True)`；
2. 先计算正式 `O2_M` cumulative residual base loss；
3. 从 `out.router_input` 按 batch 每个 sample 的真实 length 拼接有效 `pre-D_GU` rows；
4. 对这份有效 Router input **先 detach**，再调用 `folded_router_logits_from_pre_dgu_input()` 重算 candidate Router logits；
5. 从 cache 按同样 sample 顺序与 length 拼接 E0 `router_logits/topk_ids/topk_weights`，四个张量的 `N_valid` 必须完全相等；
6. `O3_full` 计算 `base_loss + router_lambda * router_full_kl(...)`；
7. `O3_topk` 计算 `base_loss + router_lambda * router_topk_total(...)`。

禁止直接拿 `out.router_logits` 参与 Router auxiliary backward，因为 joint/both scope 下它可能把梯度沿 Attention 路径传回 `D_QKV/D_VO`。`out.router_logits` 只用于数值 identity 检查：在 no-grad 下，真实有效 Router logits 与 helper 重算 logits 必须一致；失败直接停止 recipe，不能训练 surrogate Router。

梯度规则锁定：

```text
D_GU       Router auxiliary 可有梯度
D_UD       Router auxiliary 自身必须 0/None；可由 O2_M 获得梯度
D_QKV/D_VO Router auxiliary 必须 0/None
router_weight requires_grad=False
E0 teacher tensors detached
```

##### E. `objective_study.py / objective_train.py / protection_objective.py / objective_train_exec.py` 的修改锁定

```text
objective_study.py
  OBJECTIVES 改为 O0, O1, O2, O3_full, O3_topk, O4
  decide_objective_scopes(...) 新增显式 o3_layers 参数
  只有 layer in o3_layers 时，MoE/both scope 才增加 O3_full 与 O3_topk
  删除 O3_conditional

objective_train.py
  run_formal_objective_ablation(...):
    enable_o3: bool -> o3_layers: list[int]
  training_recipes.json 对每个入选 layer 分别生成 O3_full 与 O3_topk 两条 recipe

protection_objective.py
  禁止继续读取 objective["enable_o3"]
  改为读取 objective["o3_layers"]
  新 FROZEN scope 缺少 o3_layers 时直接 fail，不做旧字段兼容

objective_train_exec.py
  不再只写 READY_FOR_LAYERWISE_TRAIN
  每条正式 recipe 必须调用 train_selected_layer_recipe(...)
  O3 recipe 启动前必须确认对应 Router teacher cache status=COMPLETE
  只有 checkpoint + train metrics + validation metrics + cost artifact 齐全才算 recipe 完成
```

用户已要求统一新命名，因此这里**不保留 `enable_o3` / `O3_conditional` 兼容逻辑**。

##### F. O3 结果目录与 checkpoint 选择规则锁定

每个 layer 的两个 Router candidate 独立隔离：

```text
60_objective/objective_candidates/Lxx/O3_full/
  recipe.json
  checkpoint.pt
  train_metrics.jsonl
  val_metrics.json
  cost.json

60_objective/objective_candidates/Lxx/O3_topk/
  recipe.json
  checkpoint.pt
  train_metrics.jsonl
  val_metrics.json
  cost.json
```

`O3_topk/val_metrics.json` 至少包含：

```text
cumulative_residual_nmse
final_logit_kl
router_causal_contribution
router_topk_weight_kl
router_topk_support_hinge
router_topk_total
topk_id_match_ratio
topk_overlap_mean
topk_margin_mean
outside_teacher_topk_mass
```

`O3_full/val_metrics.json` 必须包含相同主指标，并额外记录 `router_full_kl`。

O0 / O1 / O2 / O3_full / O3_topk 的正式公平对比**不做 objective-specific best-epoch selection**：统一使用相同初始化、相同 optimizer step 数后的 `final checkpoint`。训练过程中可以记录各自 loss 曲线，但不得让 O3_full 用 Router KL 挑 best、O3_topk 用 Top-k loss 挑 best、O2 用 cumulative NMSE 挑 best。训练结束后再对这些固定 final checkpoints 统一跑同一 holdout actual-path evaluation。这样既避免 selection bias，也避免为了选 epoch 在训练过程中反复启动完整 downstream final-KL 评估。

##### G. 单元测试锁定

新增：

```text
Native_NVFP4_HiF4_Linear_Puncture/tests/internal_error_accumulation/test_router_objective.py
Native_NVFP4_HiF4_Linear_Puncture/tests/internal_error_accumulation/test_router_teacher_cache.py
Native_NVFP4_HiF4_Linear_Puncture/tests/internal_error_accumulation/test_selected_layer_o3.py
```

至少覆盖：

1. candidate logits 与 E0 logits 完全相同时，`router_full_kl` 为 0（浮点误差范围内）；
2. candidate=E0 时，`router_topk_weight_kl=0` 且 `support_hinge=0`；
3. 只改变 teacher Top-k 内相对 logits，weight KL 必须 >0；
4. Top-k 内相对权重保持不变，但把一个 outside expert 提升到越过 `min(S0)`，support hinge 必须 >0；
5. outside expert 没越界时 support hinge 必须为 0；
6. teacher Top-k IDs/weights 与当前 production `softmax -> topk -> norm_topk_prob` 逐元素一致；
7. 用 synthetic `k != 8` 验证函数没有硬编码 8；
8. teacher cache 的 sample / length / layer 对齐正确，padding rows 不落盘；
9. train 与 val cache 的 sample IDs 不串；
10. helper 重算 candidate logits 与真实 student Router logits identity；
11. 单独 backward `router_full_kl` 或 `router_topk_total` 时，只有 `z_gu` 有非零梯度，`z_ud/z_qkv/z_vo` 为 0 或 None；
12. 总 O3 loss backward 时 `z_ud` 可由 `O2_M` 获得梯度，但 Router auxiliary 本身不能给 `z_ud` 梯度；
13. 新 formal scope 中出现 `enable_o3` 或 `O3_conditional` 必须测试失败，防止旧语义回流。

以上文件位置、接口、梯度流、teacher cache schema 与数学公式全部视为冻结内容。若既有 runtime API 名字略有差异，只允许写最小 adapter；**不得自行换成 sparse-teacher/full-softmax KL、candidate-own-topk teacher、额外 temperature/margin/support-weight，或修改旧 e2e trainer 的默认行为。**

#### O4：final-KL oracle —— 只做 Top-4 selected layers

O4 不作为全 48 层正式优化方案，而是作为机制上界/金标准，专门检验 `L_cum` 是否真的抓住最终输出分布损失。

对选中的单个 layer `l`，只训练该层 DIAG/缩放参数，其余参数全部冻结；candidate 从当前层继续通过真实后续 Transformer、final RMSNorm 和 LM Head，直接优化：

\[
\boxed{
L_{final-KL}
=
KL\left(p_{E0}\|p_{\theta,l}\right)
}
\]

其中 `p_{theta,l}` 必须来自真实后续网络输出，禁止用手写 Jacobian、semantic replay 或 Fisher 二阶近似代替训练目标。

O4 固定只在 **Top-4 selected sensitive Transformer layers** 上运行，且每层独立训练。具体开放的 DIAG 参数必须跟随前述 causal attribution：Attention 主导时只开放对应 `D_QKV/D_VO`，MoE 主导时只开放 `D_GU/D_UD`，两者都重要时允许做一个联合 oracle 对照。O4 不扩展为 48 层 sweep，也不作为最终 deployable 主方法。

O4 的研究问题只有一个：

> **如果直接最小化最终 KL 能显著优于 O2，而 O2 无法同步降低 final KL，说明 cumulative residual 仍缺少决策方向敏感性；如果 O2 已接近 O4，则说明 `L_cum` 是 final-KL 的有效低成本代理。**

O4 因计算图跨越后续全部层，训练成本天然高于 O0/O1/O2，因此不要求 wall-clock/step-cost 等价；公平性要求改为：同一 Top-4 层、同一初始化、同一训练样本、同一优化 step 数，并单独报告每 step 成本和总训练成本。

### 5.17 Objective study 的公平性

正式 objective training 的 calibration 分布固定为：

```text
50% wikitext2 + 50% s1k_original
```

O0/O1/O2/O3_full/O3_topk/O4 必须使用同一组 source mix、同一组 train sample IDs / validation sample IDs 和同一抽样 seed。旧的纯 `s1k_original` O0 结果只作为历史参考，**禁止直接进入正式 O0/O1/O2/O4 公平对比**；正式 O0 必须在新的 50:50 calibration 上重新训练。O3_full 与 O3_topk 必须额外共享同一份 E0 Router teacher cache、同一个 `lambda` 和同一训练预算，唯一变量只能是 Router loss 的 full-vs-topk 定义。objective train/validation IDs 必须在 causal cohort 固定后再抽取，并显式排除全部 discovery+holdout 的 16 个 causal sample IDs；causal holdout 因参与 S4 scope selection，禁止再次充当 objective validation。

所有 objective candidate：

- 同一 WikiText2:S1K = 50:50 source ratio；
- 同一 train sample split；
- 同一 validation sample split；
- 同一 DIAG init；
- 同一 optimizer / LR / steps；
- 同一 quantizer；
- 同一 R64 开关；
- 不因某 objective 单独增加训练预算；
- research candidate 阶段不使用 objective-specific rollback 掩盖失败。

统一在与 causal cohort 完全隔离的 objective validation actual-vLLM states 上评价：

1. local q NMSE；
2. cumulative residual NMSE；
3. positive energy gain；
4. final norm hidden NMSE；
5. one-step logit KL；
6. Router causal contribution；
7. protection ranking 是否改变。

只有统一 holdout 指标改善后，才讨论 adoption/rollback。

---

# 6. 正式 state cohort：不按 trajectory divergence 采样

旧 first-divergence cohort 有机制选择偏差，不作为本实验主 cohort。

新建：

```text
internal_error_state_cohort.jsonl
```

内部机制数据与下游 benchmark 测试集彻底分离。正式来源固定为仓库已有 calibration source：

- `wikitext2`；
- `s1k_original`；
- **禁止从 MMLU-Pro / LiveCodeBench 测试集抽内部机制样本**；
- 不按 E0/E1 输出差异、pass/fail、first divergence、Router change 等结果筛选；
- sample IDs / source indices 必须在任何 E0/E1 机制结果产生前由固定 seed 冻结；
- **不让 E0 为机制实验额外自由生成 continuation**；WikiText2 / `s1k_original` 的原始 calibration token 序列本身就是唯一 fixed teacher-forced history；
- 只选择 tokenizer 后原始 calibration 序列长度 `>=129` 的样本，从而能无歧义覆盖最大 `j=128` predictor state。

每个 sample 固定采样 4 个 teacher-forced predictor state：

```text
j = 8, 32, 64, 128
```

`j` 的语义固定为 **prefix length**：对 token 序列 `ids[0:L]`，state `j` 使用完全相同的真实 calibration prefix `ids[:j]` 构造 KV/history，并读取“基于该 prefix 预测 `ids[j]`”这一当前 predictor 的 48 层内部状态与 logits。E0/E1 的 `ids[:j]` 和 target `ids[j]` 必须逐 token 完全一致。

因此这些 `j` 只是同一 calibration sample 中 4 个不同真实上下文状态，不是 E0 生成长度，不做 decode-length / trajectory 趋势结论。

正式划分为：

```text
discovery: 8 samples = 4 wikitext2 + 4 s1k_original = 32 states
holdout:   8 samples = 4 wikitext2 + 4 s1k_original = 32 states
```

其中最昂贵的 48-layer whole-layer causal screening **只在 discovery 上全扫**：

```text
8 samples × 4 states × 48 layers = 1536 whole-layer interventions
```

holdout 不再重复 48 层全扫，只对 discovery 预先选出的 Top sensitive / neutral / cancellation layers 做确认，从而避免层选择完全在同一批状态上自证。

内部结构 capture 总 cohort 为：

```text
16 samples × 4 states = 64 real decode states
```

所有统计以 sample 为独立单位；同一 sample 的 4 个 state 属于 repeated measures。Objective training 可以使用同源 calibration train split，但 validation/causal confirmation 必须与训练 sample IDs 隔离；MMLU-Pro/LCB 只用于最终候选的外部任务验证。

---

# 7. 正式运行协议

内部机制捕获固定：

```text
model = nvidia/Qwen3-30B-A3B-NVFP4 / 对应 E1 HiF4 artifact
TP = 2
KV = BF16
single request
max_num_seqs = 1
enforce_eager = true
prefix cache = false
speculative decoding = false
same exact calibration token sequence
same exact teacher-forced prefix ids[:j] and target ids[j]
NO E0-generated continuation for mechanism cohort
real incremental KV decode under fixed calibration history
```

禁止：

- E0/E1 各自自由生成后比较内部状态；
- E0/E1 分别重新 tokenize；
- 为方便分析改成 TP1；
- 用 batch E0 对 isolated E1；
- 用 semantic replay 补缺失 tensor。

---

# 8. 新代码与结果目录

新实验代码必须独立，不继续把“内部累计”塞进 `long_trajectory_stability` 命名空间：

```text
Native_NVFP4_HiF4_Linear_Puncture/
  experiments/
    internal_error_accumulation/
      __init__.py
      config.py
      build_state_cohort.py
      capture_states.py
      residual_ledger.py
      structural_accumulation.py
      moe_decomposition.py
      router_contribution.py
      one_step_intervention.py
      layer_protection.py
      objective_study.py
      variant_validation.py
      report.py
      run_pipeline.py
      scripts/
        run_formal.sh
        launch_detached.sh
        show_status.sh
```

只复用 `real_vllm_hooks` 的通用 primitives，不复制 worker hook 实现。

`internal_error_accumulation/` 必须形成独立实验命名空间。除非现有 `real_vllm_hooks` 确实缺少无法在新目录组合得到的通用 primitive，否则不得把本实验专用逻辑写回 `long_trajectory_stability/`、`e2e_diag_reconstruction/` 或 `3rdparty/vllm`。若确需补共享 primitive，只允许做最小通用扩展，并由 `tests/internal_error_accumulation/` 覆盖本实验依赖的行为。

测试：

```text
Native_NVFP4_HiF4_Linear_Puncture/tests/internal_error_accumulation/
```

只写必要测试：

- BF16 runtime residual-add reference + rounding-residue strict closure；
- q/p/router closure；
- signed energy decomposition；
- layer0/final-layer boundary mapping；
- layer0 state-reset 明确拒绝、layer1–47 tuple reset 映射；
- LM Head raw-logit reconstruction identity gate；
- Q/K/V fused slice 边界 + captured-E1 slice no-op identity；
- intervention no-op identity；
- S3 full-48 计数/唯一键完整性：4608 discovery + 6144 holdout；
- Attention-sensitive / MoE-sensitive / unstable 的 holdout sample-level 固定判定；
- Q/K/V/O 只能由冻结 layer set 触发，禁止 per-state outcome-dependent filtering；
- per-layer `o3_layers` 判定：mean/median>0 且 positive_count>=5/8；
- `O3_topk` 数学正确性：candidate=E0 时 Top-k weight KL 与 support hinge 都为 0；
- `O3_topk` support sensitivity：保持 E0 Top-k 内相对权重不变、只把一个 outside expert logit 提升到越过 `min(S0)` 时 support hinge 必须严格变正；
- `k` 必须来自 runtime `config.num_experts_per_tok`，当前模型验证为 8，测试禁止依赖硬编码；
- O3_full/O3_topk 必须消费同一 E0 Router teacher cache，cache key 覆盖 layer/sample/token-position 且 train/val 不串；
- Router auxiliary gradient 只能进入 D_GU；E0 teacher / Router weight / Attention 路径无梯度；
- `S3_FULL48_CAUSAL` 在同一 RUN_ID 的 resume/partial-key skip，不覆盖旧 partial-S3；
- detached launcher 的 RUN_ID/PID duplicate-start guard 与 run_state 原子恢复。

结果：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/internal_error_accumulation/
  qwen3_30b_a3b_e0_e1_formal/
    runs/
      <RUN_ID>/
        run_manifest.json
        run_state.json
        launcher.pid
        logs/
        00_protocol/
        10_capture/
        20_structural/
        30_moe_qp/
        40_router/
        50_protection/
        60_objective/
        70_variant_validation/
        analysis/
```

结果隔离要求：

- 不覆盖旧 `results/long_trajectory_stability`；
- 每次正式运行必须使用唯一 `RUN_ID`，所有日志、manifest、中间张量、报告和候选 artifact 只写入该 `runs/<RUN_ID>/`；
- 禁止多个 `RUN_ID` 共写同一结果子目录；
- 恢复运行必须显式传入原 `RUN_ID`，不得通过扫描“最新目录”猜测；
- 新 run 不得静默复用旧 run 的中间结果；需要复用时必须在 `run_manifest.json` 中显式记录来源路径和配置身份，并仅复用计划明确允许的 immutable 输入 artifact；
- `run_state.json` 使用原子替换写入，作为 Cursor 后续唤醒时的唯一运行状态真值来源。

---

# 9. Task 0：基础设施审计，只补真正缺的能力

执行前确认现有 hook 已能捕获：

```text
layer_in.branch/residual
input_norm.normalized/updated_residual
attention_core
o_proj.tp_reduced
post_attn_norm.normalized/updated_residual
router_logits
moe_out.tp_reduced
layer_out.branch/residual
final_norm.normalized/updated_residual
raw_logits
```

当前代码已经包含这些边界，不要重新实现。

需要新增的能力仅限：

1. residual ledger 分析；
2. Router propagation decomposition；
3. one-step branch/router intervention；
4. practical protection artifact；
5. objective comparison。

除非现有通用 hook 本身无法表达必要状态，否则禁止改 `3rdparty/vllm`。

---

# 10. Task 1：构建 unbiased state cohort

输出：

```text
00_protocol/internal_error_state_cohort.jsonl
00_protocol/internal_error_state_cohort.meta.json
```

每条 state 至少记录：

```text
sample_key
source = wikitext2 | s1k_original
source_index / calibration_sample_id
calibration_input_ids hash / length
fixed_history_source = calibration_tokens
prefix_length_j
target_token_id = ids[j]
abs_position
split = discovery | holdout
```

禁止记录/使用：

```text
first divergence
trajectory survival
regression/robust label
```

---

# 11. Task 2：E0/E1 actual-path 全 48 层 capture + 严格 closure gate

对完整 64-state calibration cohort 捕获 E0/E1 core tensors。

## 11.1 TP replica gate

对以下 global/replicated states：

```text
input_norm.updated_residual
post_attn_norm.updated_residual
o_proj.tp_reduced
router_logits
moe_out.tp_reduced
final_norm.updated_residual
final_norm.normalized
```

先验证 rank0/rank1 一致性。

正式分析可只存/读 rank0 的 global vector，但只有 replica gate 通过后才允许。

rank-local `attention_core` 不允许直接当全局 residual contribution。

## 11.2 residual identity gate：必须按真实 BF16 fused-add 存储语义闭合

这里禁止把 FP64 离线 `R + branch` 与 BF16 `updated_residual` 直接做“exact identity”。vLLM 的 RMSNorm residual 路径会执行 fused residual-add，并把 residual-out 按运行 dtype 写回；当前正式协议下 residual dtype 为 BF16。PyTorch-native 等价路径明确是：

```text
sum_fp32 = x.float() + residual
updated_residual = sum_fp32.to(orig_dtype)   # 当前为 BF16
```

CUDA `fused_add_rms_norm` 必须与该运行语义等价；不能用“重复运行恰好 deterministic，所以 tolerance=0”去要求 FP64 `R+A` 等于 BF16 存储值。

对 E0 和 E1 的每个 layer，先构造 runtime-add reference：

\[
\widetilde R_l^A=\operatorname{BF16}(\operatorname{FP32}(R_l)+\operatorname{FP32}(A_l))
\]

\[
\widetilde R_{l+1}=\operatorname{BF16}(\operatorname{FP32}(R_l^A)+\operatorname{FP32}(M_l))
\]

并验证：

```text
captured R_l^A == runtime-add reference R_l^A
captured R_{l+1} == runtime-add reference R_{l+1}
```

只允许落在 **actual runtime repeatability envelope** 内；不允许另拍经验 tolerance。若该 reference 与真实 fused kernel 不一致，直接 `STOP`，先修正 runtime-add reference，而不是放宽阈值。

同时为了做高精度误差账本，显式记录 BF16 storage rounding residue：

\[
\epsilon_l^A
=
\operatorname{FP64}(R_l^A)-\operatorname{FP64}(R_l)-\operatorname{FP64}(A_l)
\]

\[
\epsilon_l^M
=
\operatorname{FP64}(R_{l+1})-\operatorname{FP64}(R_l^A)-\operatorname{FP64}(M_l)
\]

于是高精度账本的真实 identity 是：

\[
R_l^A=R_l+A_l+\epsilon_l^A
\]

\[
R_{l+1}=R_l^A+M_l+\epsilon_l^M
\]

对 l=47 仍要求：

```text
R_48 = final_norm.updated_residual
```

## 11.3 cross-variant error closure

跨 E0/E1 时必须把 storage-rounding difference 一并计入，而不是假设其为 0：

\[
\delta R_l^A
=
\delta R_l+\delta A_l+\delta\epsilon_l^A
\]

\[
\delta R_{l+1}
=
\delta R_l^A+\delta M_l+\delta\epsilon_l^M
\]

全 48 层严格账本为：

\[
\delta R_{48}
=
\delta R_0
+\sum_l\delta A_l
+\sum_l\delta M_l
+\sum_l\delta\epsilon_l^A
+\sum_l\delta\epsilon_l^M
\]

正文主机制图仍可以单独展示 Attention/MoE branch contribution，但报告必须同时给 rounding-residue 总量及其相对占比；只有确认其相对主要 branch error 可忽略时，才允许把简化式 `delta R_48 ≈ delta R_0 + sum delta A + sum delta M` 作为解释性近似，禁止把近似写成 strict closure。

若 closure 不过：

```text
STOP
```

禁止继续做任何累计结论。

输出：

```text
20_structural/RESIDUAL_LEDGER_GATE.md
20_structural/residual_ledger_rows.jsonl
20_structural/residual_ledger_summary.json
```

---

# 12. Task 3：结构性误差累计主实验

对每个 state × 48 layer 计算：

```text
||delta R_l||
||delta A_l||
||delta M_l||
G_A
G_M
cos(delta R_l, delta A_l)
cos(delta R_l^A, delta M_l)
final-direction attribution s_A / s_M
final exact KL(E0||E1)
final Fisher-quadratic KL
Fisher/exact approximation error
```

再计算：

```text
kappa_all
kappa_attn
kappa_moe
kappa_early
kappa_middle
kappa_late
```

聚合方式：

- 先 state 内计算；
- 再 sample 内对 4 个 state 聚合；
- 最后跨 sample 给 median / mean / sample-bootstrap 95% CI。

必须输出以下图：

### Figure 1：48-layer cumulative residual error

横轴 layer depth；纵轴 residual rel-L2 / energy。

同时标 Attention 后、MoE 后。

### Figure 2：signed energy gain

每层两根柱：

```text
G_A
G_M
```

正值放大，负值抵消。

### Figure 3：alignment

```text
cos(delta R, delta A)
cos(delta R^A, delta M)
```

### Figure 4：96×96 branch cosine heatmap

### Figure 5：final-direction signed contribution

### Figure 6：Attention vs MoE total decomposition

给出：

\[
C_A=\sum_l\delta A_l
\]

\[
C_M=\sum_l\delta M_l
\]

以及：

```text
||C_A||
||C_M||
cos(C_A,C_M)
```

### Figure 7：Residual error vs final KL

对每个 state 画：

```text
x = final residual / final-norm hidden error
 y = exact KL(E0||E1)
```

并同时给 Fisher-quadratic KL，用于判断“大 residual error”是否真的等价于“大输出分布损失”。

### Figure 8：Exact KL vs Fisher quadratic KL

检查二阶近似适用范围；只把 Fisher 近似用于机制解释，不替代 exact KL。

### Figure 9：Residual cosine vs KL/Fisher cosine

对后续已经做 one-step branch repair 的 selected layers，比较：

```text
rho_res
rho_KL
exact KL gain after repair
```

用于区分“内部同向累计”与“最终决策空间同向累计”。

输出：

```text
20_structural/STRUCTURAL_ACCUMULATION_REPORT.md
20_structural/structural_metrics.jsonl
20_structural/figures/*
```

---

# 13. Task 4：建立无先验的结构候选，不按旧 divergence frontier

先从 discovery samples 的结构账本中建立候选池，但**不据此直接宣布层重要度**。至少保留：

```text
Top positive whole-layer gain: G_layer = G_A + G_M
Top positive G_A layers
Top positive G_M layers
neutral / cancellation controls
```

所有排名使用 sample-mean signed gain（同一 sample 的 4 个 state 先聚合），不使用单个 token/state 极值。

结构 score 只用于安排后续 causal screening 的计算预算。正式“哪个 Transformer layer 最重要”必须由 `P_l^layer` 决定；正式“Attention vs MoE”必须由 `P_l^A / P_l^M` 决定。

禁止因为旧 frontier 偏向 MoE 就预先只深钻 MoE。若 Attention causal gain 更大，后续实验自然转向 Attention，并继续做 Q/K/V/O 三级归因。

---

# 14. Task 5：MoE q / p / Router 三项分解

对选中 layer × discovery states，运行真实 production puncture。

正式保存：

```text
q
p_expert
p_router
delta M
```

验证：

\[
\delta M=q+p_{expert}+p_{router}
\]

输出所有能量项：

```text
||q||^2
||p_expert||^2
||p_router||^2
2 q·p_expert
2 q·p_router
2 p_expert·p_router
2 deltaR_A·q
2 deltaR_A·p_expert
2 deltaR_A·p_router
```

这些项之和必须闭合到：

```text
G_M
```

从而回答：

- final positive G_M 主要来自 pure local q？
- 还是已有误差经过 expert compute 被放大？
- 还是 Router 变化贡献明显？
- q 与 upstream error 是否持续同向？

输出：

```text
30_moe_qp/MOE_ERROR_DECOMPOSITION_REPORT.md
30_moe_qp/moe_decomposition_rows.jsonl
```

---

# 15. Task 6：Router 对 loss 的完整研究

## 15.1 Basic Router decomposition：所有 selected MoE layers 必做

计算：

```text
router KL
top-k exact/overlap
routing weight diff
p_router norm / energy terms
```

不设置“top-k change 才分析”的前置条件。

## 15.2 ID/weight split：条件执行

仅当 Router basic decomposition 表明 `p_router` 对正 G_M 有稳定贡献时，执行：

```text
p_weight
p_id
```

并做 exact closure。

## 15.3 one-step causal Router intervention

对 Top Router-contribution layer：

```text
E1 actual router
vs
E1 x1 + frozen E0 router logits
```

后续层继续 E1，输出 raw logits。

主要统计：

```text
Delta final logit KL
Delta final residual NMSE
Delta E0-target NLL
```

按 calibration sample paired bootstrap；同一 sample 的 4 个 state 作为 repeated measures 一起重采样。

输出：

```text
40_router/ROUTER_CONTRIBUTION_REPORT.md
40_router/router_rows.jsonl
40_router/router_intervention_rows.jsonl
```

最终 Router 结论必须区分：

1. Router top-k ID change；
2. same top-k 下 routing weight change；
3. Router local MoE output contribution；
4. Router 对最终 one-step output loss 的 causal contribution。

禁止再用单独的 top-k change rate 代表 Router importance。

---

# 16. Task 7：三级层级化因果保护实验

这一 Task 不再只对 MoE 做“单层保护”，而是严格按三级归因执行。

## 16.1 Level 1：48-layer Transformer-layer causal importance

必须获得 48 个 layer 都有真实 `P_l^layer` 的 causal result，不能用结构 score 冒充未测层的因果结果。

48 层 whole-layer oracle screening 的预算现在固定，不再留给执行阶段决定：

```text
discovery = 8 calibration samples
          = 4 wikitext2 + 4 s1k_original
states/sample = {8, 32, 64, 128}
full scan = 8 × 4 × 48 = 1536 interventions
```

这 32 个 discovery states 对全部 48 层一视同仁全扫，形成完整 `P_layer_abs` 排名。S2 只负责 discovery 侧的 whole-layer 排名；**holdout 侧的 48-layer whole-layer confirmation 统一放到 S3 full-48 causal stage 完成**，不在 S2 重复执行。sample IDs 必须在任何 causal result 产生前固定。

输出至少包含：

```text
P_layer = whole Transformer layer oracle KL gain
Delta final residual NMSE
Delta final norm NMSE
Delta target NLL/rank
```

主要回答：是否存在少量 Transformer layer 对最终输出 KL 具有特别大的因果贡献。

## 16.2 Level 2：48-layer Attention / MoE / Router 全覆盖因果图谱

Attention、MoE、Router 三类针对性实验必须对 **全部 48 个 Transformer layers** 完整执行，禁止只测 Top/controls。正式输出对每个 `l=0..47` 都必须包含：

```text
Attention-only oracle repair -> P_A
MoE-only oracle repair       -> P_M
whole-layer oracle repair    -> P_layer
interaction                  -> I_A,M = P_layer - P_A - P_M
Router freeze to captured E0 -> C_router
```

执行口径固定：

- discovery 8 samples × 4 states：48 层全部跑 `Attention-only / MoE-only / Router-freeze`；`P_layer` 直接复用 S2 已有 48-layer whole-layer result，禁止重复计算；
- holdout 8 samples × 4 states：48 层全部跑 `Attention-only / MoE-only / whole-layer / Router-freeze`，形成独立完整确认图谱；
- 因此最终每层都有 discovery 与 holdout 两套 `P_A/P_M/P_layer/I_A,M/C_router`；
- 正式 Attention-vs-MoE 与 Router scope 冻结只允许使用 **holdout sample-level 聚合**；discovery 仅用于探索、排序稳定性和 discovery↔holdout 一致性分析；
- 同一 sample 的 `j={8,32,64,128}` 先聚合，再跨 sample 统计；禁止把 4 个 state 当独立样本。

禁止只看 `G_A/G_M` 就下 Attention/MoE 重要性结论；`G` 是 residual-space 结构量，`P_A/P_M` 才是 final-KL causal quantity。Router 重要性必须用 `C_router` 的 final-KL causal contribution，而不是 top-k change rate。

### S3 full-48 review 后的 S4 scope 冻结规则（2026-09-11）

当前正式 RUN 中 discovery↔holdout 的 whole-layer ranking 稳定性较弱：Spearman `rho≈0.164`，Top-4 overlap=`0/4`，Top-8 overlap=`2/8`。因此旧 S2 discovery Top-4 `[44,31,11,12]` 保留为**预注册 discovery 结果**，但禁止继续把它当成“独立确认后的最终 sensitive Top-4”。也禁止直接改用 holdout Top-4，因为那会把 holdout 同时用于选层和后续效果验证。

S4 主机制 scope 改为固定的 cross-split stable + causal-source balanced 规则：

1. 只保留 `P_layer_discovery>0` 且 `P_layer_holdout>0` 的层；
2. 分别对 48 层按 discovery / holdout `P_layer` 降序排名；
3. 定义 `cross_split_rank=(rank_discovery+rank_holdout)/2`；
4. 在 holdout causal source=`attention` 中取 cross-split rank 最好的 2 层；
5. 在 holdout causal source=`moe` 中取 cross-split rank 最好的 2 层；
6. 当前 RUN 按此规则得到主机制层 `[41,43,39,31]`：Attention=`[41,43]`，MoE=`[39,31]`；O4 也只在这四层运行；
7. Router objective 仅在 `o3_layers` 且 discovery/holdout `P_layer` 都为正的层中，按 holdout `C_router` 取 Top-2；当前 RUN 得到 `[19,31]`；
8. 因为 causal holdout 已参与 S4 scope 选择，它从此只承担**selection/confirmation evidence**，不得再作为 O0/O1/O2/O3/O4 的最终 validation split。

S4 objective 的独立 validation 必须使用重新冻结的 50:50 WikiText2:S1K split，并显式排除全部 16 个 causal discovery+holdout sample IDs。objective train/val 之间也必须严格不重叠。`objective_scope.json` 只能为 `selected_objective_layers=[41,43,39,31,19]` 生成 recipe；禁止把 full-48 `causal_source_by_layer` 直接传给 trainer 后对 48 层铺开训练。

O3 也不再使用全局 `enable_o3=true/false`。必须从 48-layer holdout Router 图谱冻结。判定不能留给执行阶段自由解释，固定如下：

1. 对每个 holdout sample，先把该 layer 的 4 个 `j` 的 `C_router` 取均值，得到 8 个 sample-level `C_router`；
2. 该 layer 的 causal source 必须是 `moe` 或 `both`；
3. 8 个 sample-level `C_router` 的 **mean > 0、median > 0，且至少 5/8 为正**；
4. 同时满足时才写入：

```text
o3_layers = [...]
```

否则该 MoE layer 只比较 `O0/O1_M/O2_M`，不加入 `O3_full/O3_topk`。报告同时保留每层 Router mean / median / positive_count / bootstrap CI，不能用全局 Router 均值替代 per-layer 判定。S3 full-48 还必须在同一 E0/E1 Router logits 上计算 `router_full_kl`、`router_topk_weight_kl`、`router_topk_support_hinge` 与 `router_topk_total`，并分别报告 `router_full_kl` / `router_topk_total` 与真实 `C_router` 的 48-layer Spearman/Top-K overlap；这一步只用于判断哪种 Router proxy 更接近 causal loss，不替代 `C_router` 的 O3 eligibility Gate。

## 16.3 Level 3：Attention-sensitive layer 内 Q/K/V/O

先基于 **layer-level holdout 聚合后的 `P_A/P_M`** 冻结 Attention-sensitive layers。判定固定为：

1. 对每个 holdout sample，分别把 4 个 state 的 `P_A/P_M` 取均值；
2. 跨 8 个 holdout samples 计算 `mean(P_A)` 与 `mean(P_M)`；
3. 若 `mean(P_A)>0` 且至少 5/8 个 sample-level `P_A>0`，并满足以下之一，则该层进入 Attention-sensitive set：
   - `mean(P_A) > 1.05 * max(mean(P_M), 0)`：`attention`；
   - `mean(P_A)>0` 且 `mean(P_M)>0`，两者相对差不超过 5%：`both`；
4. `moe` 对称定义：`mean(P_M)>0`、至少 5/8 个 sample-level `P_M>0`，且显著高于 `P_A`；
5. 若两者都不满足稳定正贡献条件，标记为 `unstable/compensatory`，不得强行分到 Attention 或 MoE。

随后对冻结后的 Attention-sensitive layers 的 **全部 discovery+holdout states** 一视同仁执行 Q/K/V/O，禁止按单个 state 的 `P_A>0`、`P_A>=P_M` 等 outcome-dependent 条件临时触发。

```text
Q-slice oracle repair -> P_Q
K-slice oracle repair -> P_K
V-slice oracle repair -> P_V
O-proj oracle repair  -> P_O
```

然后在同一批 Attention-sensitive layers 上做对应 format-only intervention，得到：

```text
P_format_Q
P_format_K
P_format_V
P_format_O
```

这一步必须区分：

```text
oracle total-deviation importance
vs
HiF4 current-format importance
```

每次 intervention 只修改当前 layer / module / linear，后续网络继续真实 E1 forward；所有 intervention 必须有 captured-E1 no-op identity。

最终形成的不是单一 MoE protection table，而是层级化表：

```text
layer
P_layer
P_A
P_M
I_A,M
P_Q / P_K / P_V / P_O（条件）
P_format_M
P_format_Q/K/V/O（条件）
```

并比较：

```text
local NMSE vs P_layer / P_A / P_M
G_layer / G_A / G_M vs causal gain
final-direction s vs causal gain
q norm vs P_M
router contribution vs P_M / P_layer
```

S3 full-48 的正式产物必须与当前已存在的 partial-S3 结果隔离，禁止覆盖旧文件。固定输出：

```text
50_protection/s3_full48/
  full48_causal_rows.jsonl
  full48_holdout_layer_summary.json
  full48_discovery_layer_summary.json
  full48_layer_importance_table.csv
  attention_sensitive_layers.DRAFT.json
  qkvo_rows.jsonl
  attention_linear_importance.csv
  S3_FULL48_CAUSAL_REPORT.md
40_router/s3_full48/
  router_intervention_rows.jsonl
  router_layer_summary.json
  o3_layers.DRAFT.json
```

计数 Gate（不含 no-op 行）：

```text
discovery: 32 states × 48 layers × 3 = 4608
            # Attention-only + MoE-only + Router-freeze; whole-layer复用S2
holdout:   32 states × 48 layers × 4 = 6144
            # Attention-only + MoE-only + whole-layer + Router-freeze
base full-48 total = 10752 interventions
```

`full48_causal_rows.jsonl` 必须对上述基础 intervention 做唯一键去重/完整性检查，键至少包含 `(split, sample_key, j/decode_index, layer, kind)`；任何缺行、重复行、非 48-layer 覆盖都不能进入 scope 冻结。Q/K/V/O 行数由冻结后的 Attention-sensitive layer 数决定，但对每个入选层必须完整覆盖 `16 samples × 4 states × 4 operators`，不能按 state 结果筛选。

当前旧文件 `50_protection/s3_substructure_rows.jsonl`、旧 Router 行和两个旧 DRAFT scope 全部保留为 pre-full48 历史证据，只允许复制/归档，不允许拿它们继续驱动 S4。

---

# 17. Task 8：实际 source-format protection

先对 Level-1 选出的 Top-4 sensitive Transformer layers 做 protection-scope 消融，而不是默认 MoE-only：

```text
Attention-only
MoE-only
whole Transformer layer
```

若 Task 7 已确认 Attention 重要，则对这些 Attention-sensitive layers 继续加入：

```text
Q-only
K-only
V-only
O-only
最佳 Attention-linear 组合
```

对每个真正值得继续的 scope，再做 source ablation：

```text
A-only
W-source
W+A-source
```

W-source / W+A-source 在进入正式结果前，必须通过对应 operator 的 native-E0-vs-source-dequant-dense parity gate；未通过时只能作为 dense-dequant control。

因此这一步要回答两个正交问题：

1. **保护哪里最重要？** Transformer layer / Attention / MoE / Q/K/V/O；
2. **保护什么最重要？** activation format / weight format / both。

随后按 whole-Transformer-layer causal ranking 做保护预算：

```text
K=1,2,4,8 Transformer layers
```

层选择至少比较：

```text
local NMSE ranking
structural G ranking
whole-layer causal P_layer ranking
random ranking
```

对于每个被选中的 layer，内部 protection scope 使用前述 Top-4 source-scope 消融得到的最佳可部署方案，而不是统一硬编码 MoE-only。

实际 variant 必须重新跑同一个 64-state calibration actual-path capture，重新计算完整 residual ledger，而不是只看最终 KL。

这样可以回答：

> 为什么保护某些 Transformer layer 有效？收益究竟来自 Attention、MoE，还是某个 Q/K/V/O linear；以及保护后是直接降低 local format injection，还是改变了后续误差累计方向？

最佳 1-2 个保护 variant 最后跑 MMLU-Pro300。

输出：

```text
50_protection/PRACTICAL_LAYER_PROTECTION_REPORT.md
50_protection/protection_budget_results.json
```

---

# 18. Task 9：优化目标预测研究

把 discovery 全 48 层测得的 whole-Transformer-layer absolute causal gain `P_layer_abs` 当层级金标准；子结构预测只与对应的 `P_A/P_M/P_Q/P_K/P_V/P_O` 比较，禁止混用不同层级的 causal target。

对 48 layers 比较：

```text
current block_delta_nmse
local q NMSE
positive G
final-direction s
router causal contribution C_router
router_full_kl = KL(E0 full-softmax || E1 full-softmax)
router_topk_total = Top-k weight KL on E0 support + zero-margin support hinge
router outside-teacher-topk probability mass
```

指标：

```text
Spearman rho with P_layer_abs
Top-K overlap with causal P_layer_abs ranking
K = 1,2,4,8
```

Router proxy 另做同层级比较，禁止拿 Router KL 去和 whole-layer `P_layer_abs` 混排：

```text
Spearman(router_full_kl, C_router)
Spearman(router_topk_total, C_router)
Top-K overlap with positive C_router ranking
K = 1,2,4,8
```

该比较的研究问题是：完整 Router 分布偏移还是 production Top-k 路由偏移更能预测真实 final-loss Router causal contribution。无论相关性谁更高，正式 `o3_layers` eligibility 仍由 `C_router` Gate 决定。

bootstrap 以 calibration sample 为单位，同一 sample 的 4 个 state 成组重采样。

必须输出一个明确结论：

> 当前 local reconstruction 是否真的能找对危险层；如果不能，哪一种累计感知 metric 更接近真实保护收益。

输出：

```text
60_objective/OBJECTIVE_PREDICTIVENESS_REPORT.md
```

---

# 19. Task 10：优化目标训练消融

先只在 selected sensitive layers 上跑，不直接全 48 层铺开。

候选层不预设 MoE。先由 Task 7 的三级 causal attribution 决定研究对象：

```text
Top sensitive Transformer layers
+ neutral/cancellation controls
```

随后按每层 causal source 分流：

```text
Attention 主导 -> 训练 D_QKV / D_VO，比较 O0/O1^A/O2^A/O4
MoE 主导       -> 训练 D_GU / D_UD，比较 O0/O1^M/O2^M/(O3_full/O3_topk)/O4
两者均重要     -> 先分别训练，再增加一个联合 objective 对照；Router auxiliary 仍只作用于 MoE 的 D_GU
```

训练：

```text
O0 block_delta_nmse
O1^A / O1^M local q NMSE（按 causal source）
O2^A / O2^M cumulative residual NMSE（按 causal source）
O3_full = O2^M + lambda * KL(E0 full Router || candidate full Router)
O3_topk = O2^M + lambda * (Top-k weight KL on E0 support + zero-margin support hinge)
O4 final-KL oracle（仅 Top-4 selected Transformer layers）
```

`O3_full/O3_topk` 只对 `o3_layers` 运行。二者在正式训练前必须为 objective train/val split 的相同 token positions 生成同一份 E0 Router teacher cache；cache 至少保存 layer、sample/token position、full E0 Router logits、production Top-k IDs、Top-k renormalized weights、`k` 与 `norm_topk_prob`。teacher cache 必须位于当前 RUN_ID 的 `60_objective/router_teacher_cache/`，不得使用其他 RUN 的隐式缓存。

O0/O1/O2/O3_full/O3_topk 的训练预算完全一致；O3_full 与 O3_topk 的 `lambda` 也完全一致。O4 由于必须穿过真实后续网络反传，只保持相同 layer / init / sample / optimizer step 数，并单独报告计算成本，不要求 wall-clock 等价。

Validation 不允许只看训练目标本身，统一看：

```text
holdout cumulative residual NMSE
holdout positive G
holdout one-step logit KL
holdout final hidden NMSE
holdout Router causal contribution C_router
holdout router_full_kl
holdout router_topk_total
holdout Top-k ID match / overlap
holdout outside_teacher_topk_mass
```

若 O2 在 selected layer 上没有稳定优于 O0/O1：

```text
STOP objective expansion
```

报告 negative result，不全模型重训。

若 O2 稳定胜出，再做：

```text
accumulation-aware DIAG on selected Top-K layers
K = 4, 8
```

其他层 identity/direct HiF4。

这里不能继续使用最初 baseline E1 捕获的 stale `x1 / R_l^{A,1}` 逐层训练到底。Top-K candidate 必须采用**当前模型状态**：每采用/更新一个更早层后，后续待优化层的输入 residual / normalized hidden 必须由当前 candidate 重新传播或重新 capture，再计算 `L_cum`。否则所谓 cumulative objective 实际是在优化一个已经不存在的 upstream state。

selected-layer objective 对比阶段可以冻结 baseline state 以保证 O0/O1/O2 公平；但一旦进入多层 Top-K candidate，必须切换为 sequential current-state 语义，并在 manifest 中显式记录每层的 upstream-state source。

如果该层进入 `o3_layers`，正式比较 `O2 vs O3_full vs O3_topk`。若 `O3_topk` 优于 `O3_full`，必须同时检查是否伴随更低的 `outside_teacher_topk_mass`、更高的 Top-k ID match/overlap 和更低 final KL；若只降低 Router proxy 而不改善 final KL，则不能宣称 Top-k Router objective 更优。

输出：

```text
60_objective/OBJECTIVE_TRAINING_ABLATION_REPORT.md
60_objective/objective_candidates/
```

---

# 20. Task 11：用现有 E2/E3/E4 做结构指标外部验证

这一步不做新的 trajectory mechanism。

在同一个 64-state calibration cohort 上，对：

```text
E1 Direct HiF4
E2 R64
E3 adopted Fusable DIAG
E4 adopted Fusable DIAG + R64
```

只跑 core structural capture，比较：

```text
final residual NMSE
sum positive G
kappa
MoE positive gain
Attention positive gain
final-direction concentration
logit KL
```

目的：

> 检查我们新提出的“累计结构指标”是否能解释为什么某些方案 local reconstruction 看似改善，但端到端未必改善；以及 E4 MMLU-Pro +2pp 是否伴随更合理的 internal accumulation profile。

这里 adopted E3/E4 可以用于“真实 variant 的结构结果比较”；

但禁止用 adopted E3 反推“DIAG 训练目标本身为什么有效”。优化目标机制仍由 Task 10 的 controlled candidate ablation 回答。

输出：

```text
70_variant_validation/E1_E4_STRUCTURAL_VALIDATION.md
```

---

# 21. Task 12：最终端到端验证

只对以下最多 2 个新 candidate 做 MMLU-Pro300：

1. 最佳 practical layer protection；
2. 最佳 accumulation-aware optimization。

统一复刻已经审计通过的 MMLU-Pro300 正式协议。

不要默认再跑 LCB；除非内部机制和 MMLU-Pro 都显示稳定收益，再另立 LCB 验证任务。

任务级结果只用于证明新机制指标具有外部效度，不反过来改变内部累计定义。

---

# 22. 统计规则

## 22.1 独立单位

独立单位是 calibration sample，不是 token/decode state。

同一 sample 的 4 个 decode state 属于 repeated measures，bootstrap 时必须成组重采样，不能拆开。

## 22.2 bootstrap

所有 CI 使用 calibration-sample-level paired bootstrap；WikiText2 / S1K source 标签随 sample 一起重采样，并额外报告两种 source 的分层描述统计，避免 50:50 混合掩盖 source-specific 现象。

## 22.3 layer correlation

先在每个 layer 上对每个 discovery sample 的 4 个 state 聚合，再跨 8 个 discovery samples 聚合形成该层 score，最后做 48-layer Spearman；

通过 calibration-sample bootstrap 得到 rho 的置信区间。

## 22.4 不做伪大样本

禁止把：

```text
64 states × 48 layers
```

当成 3072 个独立样本做显著性检验。

---

# 23. Stop Gates

## Gate 1：Residual closure

以下任一失败即停止：

```text
TP replica identity
BF16 runtime fused-add reference identity
R_A = R + A + epsilon_A
R_next = R_A + M + epsilon_M
cross-variant delta closure including delta-epsilon
48-layer final closure including storage-rounding residue
LM Head raw-logit reconstruction identity（线性桥梁使用前）
```

不允许把 FP64 离线相加当作 BF16 runtime exact identity，也不允许放宽阈值“让实验继续”。

## Gate 2：Production puncture identity

E0/E1 production call 必须落在真实 repeatability envelope 内。

失败的 operator/layer 不做 q/p 结论。

## Gate 3：Router full-forward intervention identity

freeze captured E1 router logits 必须复现 baseline E1 logits。

失败则 Router causal result 全部无效。

## Gate 4：Practical protection path

若无法保证未保护层路径与 E1 一致，停止 deployable protection，不用 oracle patch 冒充。

## Gate 5：Objective expansion

新目标只在 selected-layer holdout actual-path 指标稳定优于 baseline 后，才允许扩展到 Top-K/full candidate。

---

# 24. 本计划明确不做

1. decode-length monotonic accumulation scan；
2. first-divergence cohort 作为主样本；
3. token trajectory survival；
4. rescue/poison；
5. low-margin enrichment；
6. router top-k enrichment；
7. semantic replay；
8. 全模型 Q/K/V 大抓取；
9. 全 48 层 Attention 手写 puncture；
10. 为了得到正结论而无条件重训全部 DIAG；
11. 无条件重跑 LCB；
12. 把 local NMSE 当作 layer importance 真值。

---

# 25. 最终必须形成的核心图表

最终报告至少包含：

1. **48-layer residual cumulative error curve**；
2. **Attention/MoE signed energy gain bar chart**；
3. **error-direction alignment curve**；
4. **96×96 branch-error cosine heatmap**；
5. **final-direction signed attribution**；
6. **MoE q / p_expert / p_router stacked decomposition**；
7. **Router causal loss contribution by layer**；
8. **layer protection gain vs local NMSE scatter**；
9. **layer protection gain vs G / s scatter**；
10. **K={1,2,4,8} protection budget curve**；
11. **optimization objective holdout comparison**；
12. **E1/E2/E3/E4 structural validation table**。

---

# 26. 最终报告必须回答的结论

最终报告：

```text
results/internal_error_accumulation/qwen3_30b_a3b_e0_e1_formal/analysis/
INTERNAL_ERROR_ACCUMULATION_REPORT.md
```

必须明确回答：

### Q1

NVFP4→HiF4 的模型内部误差是：

```text
随机近正交累积
相干放大
还是大量互相抵消？
```

给 `kappa` 和 signed energy evidence。

### Q2

Attention 与 MoE 谁贡献更多 positive cumulative gain？

不能继续只引用旧 `44/48 frontier`。

### Q3

危险 MoE 层的误差主要来自：

```text
q local format injection
p_expert upstream propagation
p_router routing propagation
```

中的哪一部分？

### Q4

Router 对 final one-step logit loss 的真实因果贡献有多大？

必须区分：

```text
top-k ID
routing weight
local output
final loss
```

### Q5

哪些 **Transformer layer** 对最终 exact KL 特别敏感？在这些敏感层内部：

```text
Attention 还是 MoE 更重要？
如果 Attention 更重要，Q / K / V / O 哪个 Linear 更关键？
local error 大的层是否真的等价于 causal importance 大？
positive cumulative gain / final-direction / Router sensitivity 能否预测这种重要性？
```

必须同时给 whole-layer `P_layer`、Attention/MoE `P_A/P_M`，并在 Attention-sensitive 层给 `P_Q/P_K/P_V/P_O`。

### Q6

层保护的实际收益来自：

```text
Activation protection
Weight protection
两者共同
```

中的哪一部分？

### Q7

当前 `block_delta_nmse` 是否是合理优化目标？

如果不是：

> `cumulative residual objective` 是否更能降低真实 internal accumulation 和 final output KL？

---

# 27. Cursor 后台执行、恢复与进程生命周期协议

本实验禁止让 Cursor/Go 模式在前台持续 `watch`、`tail -f`、轮询 GPU 或等待训练结束。Cursor 的职责是：**第一次正式运行前，先把整套实验状态机、所有当前已知条件分支、配置、脚本、Gate、恢复逻辑和结果目录一次性编排完整，做完静态检查/必要 smoke，然后才允许启动 detached 后台任务并退出前台。** 已知条件分支至少包括 Attention/MoE 分流、条件 Q/K/V/O、per-layer `o3_layers`、共享 E0 Router teacher cache、`O3_full/O3_topk`、Top-4 O4、Top-K accumulation-aware candidate；不能等跑到中途再临时拼正式脚本。用户后续重新唤醒 Cursor 时，主要只做“读取持久化状态/结果 -> 按计划冻结 selection/objective scope -> 恢复同一 RUN_ID 的下一阶段”；除非发现真实 bug，否则不重新设计运行框架。

## 27.1 唯一正式入口

正式运行只允许经过：

```text
experiments/internal_error_accumulation/scripts/launch_detached.sh
  -> scripts/run_formal.sh
     -> run_pipeline.py
```

职责固定：

- `run_pipeline.py`：唯一实验状态机，负责 task 依赖、Gate、断点恢复、结果落盘；
- `run_formal.sh`：固定工作目录、`hif4` conda 环境、GPU 参数和 Python 入口；
- `launch_detached.sh`：只负责创建/恢复 `RUN_ID`、检查同一 RUN_ID 是否已有活进程、以 detached 方式启动并写 PID；
- `show_status.sh`：只做一次性状态读取，不实现持续监控。

禁止再为每个小实验各写一套独立 Go launcher；所有正式子实验必须由同一个 `run_pipeline.py` 调度。

## 27.2 必须真正脱离终端

`launch_detached.sh` 必须使用等价于以下语义的启动方式：

```bash
nohup setsid bash run_formal.sh --run-id "${RUN_ID}" --through-stage "${THROUGH_STAGE}" \
  </dev/null >>"${RUN_ROOT}/logs/launcher.log" 2>&1 &
echo $! > "${RUN_ROOT}/launcher.pid"
```

要求：

- stdin 固定 `/dev/null`；
- stdout/stderr 全部落当前 RUN_ID 的日志；
- `setsid + nohup` 使任务不依赖 VS Code、SSH、Cursor 会话或启动终端存活；
- 不依赖 `disown`、tmux/screen 会话或保持某个 shell 前台存在；
- `run_formal.sh` 不依赖调用者已经 `conda activate`，必须显式通过 `conda run --no-capture-output -n hif4 ...` 或等价稳定方式进入 `hif4` 环境；
- 正式启动后 Cursor 只回报 `RUN_ID / PID / run_state.json / launcher.log` 路径，然后结束当前前台执行，不继续盯跑。

## 27.3 阶段化后台运行，不自动越过科学 Gate

`run_pipeline.py` 必须支持从同一 RUN_ID 恢复，并只使用本计划第 28 节定义的**唯一具名 Stage ID**（例如 `S0_INFRA`、`S1_CAPTURE`）指定 `--from-stage / --through-stage`。后台 runner 禁止复用正文 `Task 0–12` 的数字作为 CLI stage 编号，避免两套编号歧义。Cursor 每次唤醒后：

1. 读取 `run_state.json`；
2. 读取刚完成阶段的报告/Gate；
3. 根据本计划已经写死的规则生成下一阶段 `selection_manifest.json` / objective scope；
4. 再次调用 `launch_detached.sh --run-id <same RUN_ID> ...`；
5. 启动成功后退出，不前台等待。

可以在一个后台阶段内连续执行所有**不依赖中间科学结论改变实验对象**的 task；一旦下一步需要根据 Attention-vs-MoE、Router contribution、O2-vs-O0/O1 等结果决定实验范围，必须结束当前后台进程，并将状态写成：

```text
WAITING_REVIEW
```

禁止脚本在未经复核时自行扩大到新的 Q/K/V/O sweep、O3、Top-K objective 或 MMLU-Pro300。

## 27.4 持久化状态与失败语义

每个 RUN_ID 的 `run_state.json` 至少记录：

```text
run_id
status = RUNNING | WAITING_REVIEW | FAILED | COMPLETED
current_stage
completed_stages
pid
started_at / updated_at
exit_code
failure_or_gate_reason
next_allowed_stage
```

要求：

- stage 只有在其全部 artifact 和 Gate 都成功落盘后才能记入 `completed_stages`；
- Gate 失败直接 `FAILED`，不得 fallback 到旧脚手架、缩小精度或跳过检查继续；
- 科学条件需要复核时写 `WAITING_REVIEW` 并正常退出 0；
- 进程异常退出必须留下非 0 exit code 和最后失败 stage；
- 恢复时只从 `completed_stages` 之后继续，不重复已完成正式 stage，也不凭日志猜测进度；
- 同一 RUN_ID 若 `launcher.pid` 对应进程仍存活，launcher 必须拒绝重复启动，避免两套进程同时写结果。

## 27.5 运行时隔离

除代码隔离外，运行时也必须隔离：

- 当前 run 的临时文件、selection manifest、候选 DIAG/ckpt、缓存索引、阶段日志全部放在 `runs/<RUN_ID>/` 内；
- 不使用固定公共 `tmp/`、固定 `candidate.pt`、固定 `latest/` 目录作为正式中间状态；
- 不覆盖其他 RUN_ID 的 artifact；
- GPU 选择、模型/variant 路径、calibration sample IDs、关键运行参数在 `run_manifest.json` 启动时冻结；
- 后续恢复同一 RUN_ID 时禁止静默改变这些固定协议；若研究决策确实要求改变实验 scope，则新增显式 phase/selection artifact，而不是偷偷改原配置。

## 27.6 Cursor 每次交接给用户时必须给出的信息

后台启动完成后，只需要给用户：

```text
RUN_ID
当前后台 Stage ID / through-stage
PID
run_state.json 路径
主日志路径
预计下一次唤醒后需要检查的 Gate 名称
```

不要让 Cursor 为“确认还在跑”持续消耗 Go 模式额度。用户之后唤醒 Cursor 时，再一次性检查进程是否结束、读取状态、分析产物并继续。

## 27.7 2026-09-10 当前 RUN 的 S3 full-48 修正恢复协议

当前正式 RUN：

```text
RUN_ID = iea_20260909T081436Z_1076681
```

已经完成并必须保留：`S0_INFRA / S1_CAPTURE_STRUCTURAL / S2_LAYER_CAUSAL / 旧 S3_SUBSTRUCTURE_CAUSAL(partial)`。旧 S3 的科学 scope 不再允许直接进入 S4，但其原始结果不得删除或覆盖。

为避免 `completed_stages` 导致旧 S3 被跳过或强行重跑覆盖，状态机新增一个明确的 mandatory corrective Stage：

```text
S3_FULL48_CAUSAL
```

固定插在旧 `S3_SUBSTRUCTURE_CAUSAL` 与 `S4_PROTECTION_OBJECTIVE` 之间。当前 RUN 的迁移规则：

1. 不删除 `completed_stages` 中已有的旧 S3；
2. 将 `next_allowed_stage` 从 `S4_PROTECTION_OBJECTIVE` 改为 `S3_FULL48_CAUSAL`；
3. `S3_FULL48_CAUSAL` 只消费已有 S1 capture + S2 discovery whole-layer artifact，写入新的 `s3_full48/` 隔离目录；
4. 完成 10752 基础 intervention、完整性 Gate、holdout layer-level attribution 后，再固定 Attention-sensitive set 并补全 Q/K/V/O；
5. 新 DRAFT scope 必须从 full-48 holdout summary 生成，并使用 `o3_layers`，禁止继续读旧 DRAFT 中的全局 `enable_o3`；
6. Stage 正常结束为 `WAITING_REVIEW`，`next_allowed_stage=S4_PROTECTION_OBJECTIVE`；
7. 人工冻结新的两个 scope 后才允许 S4。

状态机必须允许 resume 已完成部分的 `S3_FULL48_CAUSAL`：每条正式 intervention 以唯一键识别，已完整落盘且通过 identity 的键可跳过，缺失键补跑；禁止通过“文件存在”粗粒度跳过整个 Stage。

## 27.8 S4–S6 actual-execution 完整性 Gate：禁止 plan-only / stub 冒充完成

当前审计已发现以下现有实现仍属于 scaffold，Cursor 在启动新的 full-48 后台正式任务前必须一并补齐已知执行链，避免下一个 Gate 再临时拼脚本：

```text
objective_train_exec.py  # 不能只写 recipe/exec_plan；必须真正调用 selected-layer DIAG training
practical_protection.py  # 不能固定 raise PRACTICAL_PROTECTION_BLOCKED 作为正式实现
variant_validation.py / topk_validate.py  # 不能只写 stub/manifest；S5 必须实际 recapture + E1/E2/E3/E4 structural validation
e2e_validate.py          # 不能只写 MMLU-Pro plan；S6 必须实际调用已审计的 MMLU-Pro300 runner
```

“实现完整”不等于现在就运行 S4–S6。要求是：**代码路径、参数冻结、artifact 输出、resume、Gate 都先实现并通过必要 smoke；正式计算仍严格等 S3/S4 科学 Gate。**

S4 标记 completed 的最低条件：practical protection 的实际可部署路径要么真实执行并产出结果，要么在经过计划定义的 operator parity / path identity 后由 Gate 4 明确判定该研究分支不可行；O0/O1/O2/(per-layer O3_full/O3_topk)/O4 必须产生真实训练 checkpoint、训练/验证指标与成本记录。凡进入 `o3_layers` 的 layer，必须同时有共享 E0 Router teacher cache、O3_full 与 O3_topk 两条公平 recipe/result；单纯 `READY_FOR_LAYERWISE_TRAIN`、recipe JSON 或只实现其中一个 Router objective 都不算完成。

S5 标记 completed 的最低条件：若 expansion Gate 允许，必须真正执行 sequential current-state recapture 的 Top-K candidate，并对 E1/E2/E3/E4 + best candidate 完成同一 64-state actual-path structural capture；`Pending` 报告或 validation plan 不算完成。

S6 标记 completed 的最低条件：必须有正式 MMLU-Pro300 runner 的实际结果 artifact；只写 `e2e_mmlu_pro_plan.json` 不算完成。

---

# 28. 执行顺序

后台 runner 只认下面这套唯一 Stage ID；正文 `Task 0–12` 只是科学章节编号，不得作为 CLI 状态机编号：

```text
S0_INFRA
  - audit existing real_vllm_hooks / production puncture
  - 一次性实现 isolated experiment package、run_pipeline、所有已知条件分支、detached launcher、resume/state machine
  - 必要 unit/smoke/identity checks
  - 本 Stage 不启动正式大规模实验

S1_CAPTURE_STRUCTURAL
  - build calibration-only cohort: discovery 8 + holdout 8, each sample 4 states
  - E0/E1 actual-path 64-state full-layer capture
  - TP replica / BF16 runtime residual closure / LM-head raw-logit reconstruction gates
  - structural accumulation analysis

S2_LAYER_CAUSAL
  - discovery 8×4×48 whole-Transformer-layer oracle scan
  - 输出完整 P_layer_abs / P_layer_frac ranking
  - 正常结束为 WAITING_REVIEW
  - 用户/Cursor 复核后冻结 layer_selection_manifest.json

S3_SUBSTRUCTURE_CAUSAL
  - 当前 RUN 中已完成的旧 partial-S3，仅保留为历史 artifact；不再作为 S4 scope 真值
  - 对新的 fresh run，此旧 Stage 只作为轻量迁移/审计节点，不再执行 partial 9-layer 科学扫描

S3_FULL48_CAUSAL
  - 48 层全覆盖 Attention-only / MoE-only / Router-freeze causal scan
  - discovery 侧复用 S2 whole-layer P_layer；holdout 侧补全 48 层 whole-layer confirmation
  - 对每层输出 discovery+holdout 的 P_A / P_M / P_layer / I_A,M / C_router
  - 正式 Attention/MoE/Router scope 只用 holdout sample-level 聚合冻结；discovery 用于一致性分析
  - 先按固定判定冻结 layer-level Attention-sensitive set，再对这些固定层的全部 discovery+holdout states 执行 Q/K/V/O；禁止 per-state outcome filtering
  - MoE q + p_expert + p_router decomposition 可继续优先放在 MoE-sensitive / control 层，不替代 48-layer P_M/C_router 全扫描
  - Router objective 使用 per-layer o3_layers 固定判定，不使用全局 enable_o3；入选层在 S4 同时比较 O3_full 与 O3_topk
  - format-only causal protection + metric -> causal importance predictiveness
  - 10752 base interventions + Q/K/V/O 完整性 Gate 全通过后才生成新的 DRAFT scopes
  - 正常结束为 WAITING_REVIEW
  - 复核后冻结 practical_protection_scope.json 与 objective_scope.json

S4_PROTECTION_OBJECTIVE
  - practical source-format protection ablation
  - K=1/2/4/8 Transformer-layer protection budget
  - 50:50 WikiText2:S1K 上重新训练正式 O0
  - selected-layer O0/O1/O2/(conditional O3_full/O3_topk)
  - Top-4 O4 final-KL oracle
  - 正常结束为 WAITING_REVIEW
  - 复核后决定是否允许新 objective 进入 Top-K expansion

S5_TOPK_STRUCTURAL_VALIDATE
  - 仅当前一 Gate 允许时：accumulation-aware Top-K DIAG，逐层 current-state recapture
  - E1/E2/E3/E4 + best new variants 在 64-state calibration cohort 上做同口径结构验证

S6_E2E_VALIDATE
  - 只对 Gate 选出的最多 1–2 个新 candidate 跑 MMLU-Pro300 外部验证
  - 不默认跑 LCB

S7_REPORT
  - 汇总全部已完成 artifact
  - 生成 INTERNAL_ERROR_ACCUMULATION_REPORT.md
  - run_state -> COMPLETED
```

每个 Stage 先消费前一 Stage 的 immutable artifact；不允许旧 Go 脚本无条件把所有实验同时铺开。`WAITING_REVIEW` 是后台进程的正常阶段终点：Cursor 不前台等待，用户后续唤醒后读取结果再继续。

---

# 29. 计划成功标准

本计划成功不要求证明“HiF4 一定存在严重端到端掉点”。

成功标准是把模型内部数值机制回答清楚：

1. 建立真实 TP2 residual error ledger；
2. residual closure 全部通过；
3. 定量区分 local error magnitude 与 signed cumulative gain；
4. 给出结构相干/抵消证据；
5. q / p_expert / p_router 严格闭合；
6. Router final-loss causal contribution 被直接测量；
7. 层保护收益被真实 one-step intervention 验证；
8. local NMSE 与 causal layer importance 的关系被量化；
9. 至少完成 O0/O1/O2 的公平目标对比；
10. 如果新目标没有更好，明确报告 negative result；如果更好，再形成一个最小 accumulation-aware candidate。

最终研究主线应收敛为：

> **NVFP4→HiF4 的损失不是简单由各层局部 NMSE 相加决定，而应从 residual stream 上区分“本层新误差注入、已有误差经专家传播、已有误差经 Router 放大，以及不同层误差方向的相干/抵消”。真正值得保护和优化的，是对最终累计误差具有正因果贡献的层和方向，而不是 local reconstruction error 最大的层。**

## 2026-09-15 S4 门控审计修正（同 RUN，用户授权 GPU 2、3）

S4 的 23 个训练 checkpoint 已完成，保持不变。审计发现验证仅包含各自训练目标，
非 O3 的汇总 loss 重复累加，Router causal 字段来自旧因果实验；S5 原实现只写协议。
本次只补齐门控证据与执行实现，不授权运行 S5。

在 S4 与 S5 之间增加纠正阶段 `S4_HOLDOUT_ACTUAL_VALIDATE`，沿用正式 launcher 和同一 RUN_ID。
该阶段：

- 保留 `val_metrics.schema1.json` 原始记录，精确修正重复 loss，清空未实际测量的候选 Router 因果字段。
- 使用原冻结 objective val 的 16 个样本（8 WikiText2 + 8 S1K），沿用 j=8/32/64/128，形成独立 64-state 验证 cohort；不使用 causal cohort 做目标选优。
- 将 23 个现有 checkpoint 的选定层导出为真实 vLLM 模型，其余层直接引用 E1 原始分片。
- 对 E0/E1/各候选执行 TP2 forced-history capture、TP identity、BF16 residual closure、重复推理及 Router no-op identity。
- 同口径记录最终 exact KL、累计 residual NMSE、positive G、final hidden NMSE、候选自身 Router intervention、两种 Router proxy、Top-k 匹配/交集及 outside mass。
- 先将同一样本的四个 state 聚合，再计算配对 sample bootstrap；另报两个来源的统计。
- 评审规则在读取新验证结果前固定：O2 对 E1 与同层 O0/O1 的 KL、累计 NMSE 改善的 95% 配对 CI 下界均 >0，且两个来源均值改善均 >0，才记为稳定支持。K=4 至少需要 4 个受支持层；不得把不足 8 层的候选标作 K=8。
- O3 与 O2、O3_topk 与 O3_full 单独比较，不能仅凭 Router proxy 判优。
- 只生成 `WAITING_REVIEW` 的 `objective_expansion_gate.json`，不自动写 `ALLOW_TOPK`。

S5 实现改为先按批准的 importance ranking 取 K 层，再按层号前后顺序训练；每个后续层
用已经采用的前层 DIAG 重新传播完整训练/验证样本输入，保存 upstream checkpoint 哈希及
逐样本输入哈希。全部训练、导出与 E1/E2/E3/E4/实际新候选结构验证完成后，才标 COMPLETE。
实际启动 S5 仍须单独批准并明确 `topk_layers`、`objective_by_layer`、`k_values`。

## 2026-09-16 路径审计与逐样本机制复盘（用户授权步骤一、二，GPU 2、3）

同 RUN 增加 `S4_PATH_MECHANISM_AUDIT`，正式 launcher 单阶段运行，结束于
`S4_PATH_AUDIT_REVIEW`。禁止借此自动启动 S5、重训或更改已有 expansion gate。

已确认的实现问题：IEA 训练及 Router teacher 路径传 `attention_mask=None`，
共享 SDPA 同时为 `is_causal=False`，因此训练可读取未来 token；O4 后续层使用
Native NVFP4，但候选部署剩余层是 HiF4。修正 IEA 的调用处，不改共享 runtime 默认行为。
旧源码保存 SHA256；旧 checkpoint、teacher cache、split、实际推理结果原样保留。
新训练路径标记 version 2，禁止把旧非因果 Router teacher 缓存用作新版 O3 teacher。

审计冻结口径：

- GPU 2、3 各做 Native/Student 真实层未来 token 扰动 smoke；新 O4 后续层做 checkpoint backward。
- 从已有 objective val 每个来源按 sample ID 排序取前 2 个，共 4 个样本；每个截取前 128 token，观察 j=8/32/64/128。
- GPU 2 运行旧无掩码诊断，GPU 3 运行因果掩码诊断，独立单卡进程。
- 复用现有训练 runtime，逐层比较 E0/E1/L31 O2_M/O3_full/L41 O2_A/O4 的 residual、Router、最终 logits 与已有真实 TP2 captures。
- L41 O4 从完全相同的 L41 输出分别接 Native/HiF4 后续层，隔离后续格式差异。
- 旧 L19/L31 Router teacher cache 对全部 64 个验证 state 的真实 E0 Router 做对照。
- 诊断不是正式 inference semantic replay，不能替代真实 TP2 科学证据。128-token 旧路径也不声称复现原始全序列训练。
- 不预设修正后数值等价；完整保留 SDPA/TP1 与 vLLM/TP2 的剩余偏差。

机制复盘仅消费已有 actual-vLLM 结果，覆盖全部 16 个验证样本：L31 O3_full vs O2_M、
L41 O4 vs O2_A，按样本聚合 4 个 state，给出 bootstrap、来源差异、逐样本去除敏感性、
49 个 residual 边界、误差与更新方向、Fisher 二阶描述及其 exact-KL 近似误差。
Router 双侧冻结对照是干预交互，不是可加的中介贡献百分比；已有验证样本用于选择方向，
本次属于探索性分析，不得标为新的独立确认。

## 2026-09-16 用户批准修正阶段 v3 的实施

用户已明确批准：路径修正 → L31/L41 8 个候选 → 其余 15 个 → 独立确认；
以实际 KL 为主要输出指标，累计 NMSE 单独研究。沿用 RUN，GPU 2、3，不自动开 S5。

第一阶段入口为 `S4_CORRECTED_PATH_ALIGNMENT`。路径存在未解决差异时按用户指定条件
停止重训，记录 BLOCKED 门控与 WAITING_REVIEW；不代表整个后续计划已经完成。

v3 根目录为 `60_objective/corrected_objectives_v3/`。已冻结：
- 原 train64/val16 与 token 哈希、全部 23 个新 recipe（pilot 8 + remaining 15）；
- 64 个独立确认样本（32/来源），seed=20260916，排除原训练/验证/因果 ID、完整 token 内容及前 128 token 重复；
- 两个固定主要比较，正式 KL 比较 family=8；分来源样本配对 bootstrap 50000 次，Bonferroni family alpha=0.05；
- 后续复验 seed 为 20260910、20260911，不能把 seed 当成新增独立样本。

已实施并实测：O1_M 改成 E0 MoE 同输入；O1_A 取原始分支；IEA 显式启用原生 causal SDPA；
最长 17155 token 单层 Native/Student 前向；因果和右侧 padding 检查；O4 六层 checkpoint
前向/输入梯度一致。单样本与 batch 存在数值差异，不能视为完整等价。

v3 trainer 禁止旧 recipe 直接续跑；必须通过 alignment gate、实际 teacher manifest/token 校验；
实际 teacher 的消费接口已接入，完整 teacher 采集尚待路径门控通过。
完成标记校验五类产物的 SHA256，旧“仅文件存在就跳过”不再适用于新训练。

TP2 实际生产复核在新阶段内执行 E0/E1 全前缀捕获、QKV/O/MoE 三次 replay、Router no-op、
MoE y0/y00/y10/y11 分解及 BF16 能量闭合、实际 model.compute_logits identity 与 LM Head 参数哈希。
本阶段仅使用 4 个固定审计样本，不能冒充后续全部 16 个样本的机制实验。

后续仍需完成：训练前向与 TP2 差异修正、非零 DIAG 和完整 Attention 历史对照、完整实际 teacher
采集与训练调度、23 个候选重训、固定专家集合权重分解、历史 KV/联合修复、独立确认及多 seed。
当前实现绝不以本次局部测试替代上述未完成实验。
