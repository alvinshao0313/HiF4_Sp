# Qwen3-30B-A3B 原生 NVFP4 → HiF4 MoE E2E DIAG/R64 消融实验计划（不使用 Teacher-CoT）

> 本计划用于指导 Cursor 在当前仓库上做**增量适配**。目标模型从旧 `ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4` dense + online-H16 checkpoint，切换为 `nvidia/Qwen3-30B-A3B-NVFP4` ModelOpt NVFP4 MoE checkpoint。旧 8B 计划与旧 8B 实现保留作为历史路径，不覆盖、不删除。

## 1. Goal

在 `nvidia/Qwen3-30B-A3B-NVFP4` 上建立一套可复现的 NVFP4→HiF4 逐层 reconstruction 与端到端消融流程，回答：

1. Native NVFP4 → Direct HiF4 的 E2E 损失有多大；
2. 固定 R64、fusable DIAG、online expert-specific DIAG 各自带来多少恢复；
3. fusable 的 `D_QKV / D_VO / D_GU / D_UD` 谁贡献最大；
4. progressive student input、loss、rollback、clamp、calibration source 是否影响最终 E2E；
5. 在 MoE 结构下，fusable 低自由度结构是否仍能接近 online 高自由度上界。

第一阶段仍使用 **HiF4 fake-QDQ / numerical emulation** 验证算法精度，不实现 packed HiF4 真 kernel。Native NVFP4 baseline 必须使用当前仓库已经验证通过的 **真实 packed ModelOpt NVFP4 vLLM emulation**。

历史参考计划：

- `Native_NVFP4_HiF4_Linear_Puncture/plans/2026-08-18-qwen3-8b-native-nvfp4-to-hif4-e2e-ablation-no-teacher-cot-plan-cn.md`
- `Native_NVFP4_HiF4_Linear_Puncture/plans/2026-08-17-qwen3-8b-native-nvfp4-to-hif4-e2e-diag-r64-layerwise-reconstruction-plan-cn.md`

当前 NVFP4 vLLM 能力参考：

- `NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_preflight.md`
- `NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_puncture.md`
- `NVFP4/reports/vllm_v027_nvfp4_backport/qwen3_30b_smoke.md`
- `NVFP4/reports/vllm_v027_nvfp4_backport/final_acceptance.md`

**路径约定：** 下文 `Files` 中出现的 `.../` 不是让 Cursor 自行寻找目录，而是固定缩写为：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/
```

例如 `.../core/moe_model_spec.py` 的唯一含义就是 `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/moe_model_spec.py`。测试中的 `.../tests/e2e_diag_reconstruction/` 固定指 `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/`。不得把 `...` 解释成待选择目录。

## 1.1 Agent 执行协议（强制）

本节约束 **Cursor / Codex 如何执行本计划**，优先级高于“分小步提交/小步验证”之类的一般工作习惯。

### 单次实现阶段的停止边界

```text
实现阶段 = Task 0 ～ Task 13
正式实验阶段 = Task 14 ～ Task 19
```

执行 Task 0～13 时，agent 必须**连续向后执行**。完成一个 Task 后只能做三件事：

1. 运行该 Task 的验收；
2. 更新实现进度；
3. 验收通过后立即进入下一个 Task。

**不得因为“已经完成一组可 review 的基础模块”“本轮改动较多”“先交还用户确认”而在 Task 13 之前主动停止。** “small/reviewable steps”只表示代码和测试要分块实现，不表示每完成几个 Task 就返回用户。

### 唯一允许提前停止的条件

只有以下情况允许在 Task 13 前停止：

```text
A. 真实 checkpoint / Transformers / vLLM 接口与本计划的已锁死事实发生冲突，继续实现会改变算法语义；
B. 当前 Task 的数值门禁失败，已经定位到需要用户选择新的算法定义，而不是普通代码 bug；
C. 真实 GPU / 磁盘资源不足以完成该 Task，且在不改变 batch/TP/KV/算法语义的前提下没有可执行路径；
D. Task 13 全部 smoke/correctness gate 已 PASS。
```

普通单测失败、shape bug、import bug、实现遗漏、接口适配问题都不是提前停止理由；应在当前 Task 内修到通过再继续。

若触发 A/B/C，必须创建：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/QWEN3_30B_IMPLEMENTATION_STOP_REASON.md
```

只写：当前 Task、失败命令/测试、实际错误、已经确认的事实、为什么继续会改变已锁死语义、需要用户决定的唯一问题。禁止只写“Task N 完成，等待下一步”。

### Definition of Done 规则

**文件存在不等于 Task 完成。** 每个 Task 只有同时满足以下条件才允许标记完成：

```text
1. Files 列表中的 Create/Modify 项全部落地；
2. 本 Task 明确列出的行为全部实现；
3. 本 Task 明确列出的测试/数值门禁全部实际执行并 PASS；
4. 没有用 smoke 代替数值对齐测试；
5. 没有把后续 Task 才需要实现的关键文件留空/跳过；
6. 验收结果写入统一进度文件。
```

统一进度文件：

```text
Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/QWEN3_30B_IMPLEMENTATION_PROGRESS.md
```

每个 Task 只追加一行：`Task N | PASS/FAIL | tests | files | note`。这只是执行进度，不保存 tensor/hash/checksum。

### 当前仓库的恢复点（2026-08-21）

当前仓库已经出现 Task 0～4 的部分文件，但**禁止据此直接认为 Task 0～4 已完成**。恢复执行时固定先做一次 Task 0～4 DoD audit：

```text
Task 0: 检查 model spec / real checkpoint coverage gate 是否完整；
Task 1: 检查 30B model-spec dispatch 是否真的进入 train/prepare CLI；
Task 2: 检查 layer0/23/47、QKV collapse、W13/W2 collapse、release gate；
Task 3: 当前已知未完成，必须补齐；
Task 4: 重新跑完整 transform equivalence tests。
```

Task 3 当前已知缺口：

```text
training/moe_layer_runtime.py 尚不存在；
没有真实 attention projection vs vLLM emulation puncture；
没有完整 routed MoE output vs vLLM emulation puncture；
没有 router logits / top-k 与 HF/vLLM reference 的数值门禁。
```

因此新的 Codex/Cursor session **必须从 Task 0～4 audit 开始，首先补完 Task 3；不能直接从 Task 5 开始，也不能重新从零重写已正确的 Task 0～4 文件。**

---

# 2. 已锁死的模型事实

目标：

```text
model_id = nvidia/Qwen3-30B-A3B-NVFP4
architecture = Qwen3MoeForCausalLM
model_type = qwen3_moe
hidden_size = 2048
num_hidden_layers = 48
num_attention_heads = 32
num_key_value_heads = 4
head_dim = 128
num_experts = 128
num_experts_per_tok = 8
moe_intermediate_size = 768
decoder_sparse_step = 1
mlp_only_layers = []
norm_topk_prob = true
shared_expert_intermediate_size = 0 / absent
```

因此 48 层全部是 routed MoE，没有 shared expert，也没有普通 dense FFN 层。

Checkpoint：

```text
quant_method = modelopt
quant_algo = NVFP4
group_size = 16
kv_cache_quant_algo = FP8
```

但**本计划正式主实验强制 KV cache = BF16**，不使用 checkpoint FP8 KV，避免把 KV 误差混入 NVFP4→HiF4 的结构消融。

量化覆盖：

```text
Attention:
48 × {q_proj,k_proj,v_proj,o_proj} = 192 个 NVFP4 Linear

Experts:
48 × 128 × {gate_proj,up_proj,down_proj} = 18,432 个 NVFP4 Linear

总目标物理 Linear = 18,624
```

不量化：

```text
48 × model.layers.L.mlp.gate     # BF16 router
lm_head                           # high precision
embedding / norm                  # high precision
```

checkpoint **不存在旧 8B 的 `forward_hadamard_matrix`**，所以新实现不得继续把 native H16 塞成 identity 来兼容旧公式；MoE 路径必须直接使用无 H16 的数学定义。

---

# 3. 已确认的设计决策

以下内容不得再让 Cursor 自行选择：

1. **Fusable DIAG：**
   - `D_QKV`：每层共享；
   - `D_VO`：每层共享；
   - `D_GU`：每层共享；
   - `D_UD[e]`：**128 experts 各自独立**。
2. `D_GU` 最终融合进 `post_attention_layernorm` 时，同时对 **BF16 router** 做严格逆补偿；router 不做 HiF4 量化，也不做 R64。
3. **Online DIAG 保留，并且 expert-specific**：每个 expert 的 gate/up/down 各有独立 DIAG。
4. Online 只保留 **`layer_joint`**；本模型取消 `linear_independent` 消融。
5. Fusable E2E 使用 fused-MoE HiF4 numerical runtime；online 为自由度上界，允许 correctness-first runtime 拆开 q/k/v 和 gate/up，不要求保持 fused QKV/W13。
6. 主实验 KV cache 固定 `bfloat16`。
7. 训练不把完整 30B expert BF16 master 常驻 GPU；采用 **layer-scoped lazy materialization**。
8. Native teacher 数值定义必须与当前仓库 vLLM NVFP4 emulation 一致，包括 QKV/MoE scale collapse；禁止自定义“更合理”的 per-expert scale 语义。
9. 当前阶段不运行 `s1k_teacher_cot`、Teacher autoregressive generation、self-judge、regenerate/replace。
10. 正式默认 calibration 仍为 `s1k_original`，train=128，val=32，seed=42。

---

# 4. 数学定义

## 4.1 Native NVFP4 teacher

旧 8B：

```text
X -> H16 -> NVFP4 QDQ -> GEMM
```

新 30B：

```text
X -> NVFP4 QDQ -> GEMM
```

不再出现 native H16。

Native teacher 的 packed weight / scale 解析必须严格复现当前 vLLM ModelOpt NVFP4 emulation：

### QKV

vLLM 的 `QKVParallelLinear` 会将 q/k/v checkpoint 中的多个 global scale 合并。训练 semantic teacher 必须复现：

```text
qkv_input_global_scale = max(q.input_scale, k.input_scale, v.input_scale)
qkv_weight_global_scale = max(q.weight_scale_2, k.weight_scale_2, v.weight_scale_2)
```

q/k/v 各自保留自己的 block `weight_scale`，但 global scale 按 vLLM load 后语义统一。禁止训练侧使用三个独立 global scale、评测侧又使用 vLLM collapse。

### MoE W13/W2

严格跟随当前 vLLM：

```text
W13 weight global scale:
  gate/up 若不同，记录 warning；按 vLLM 使用 gate 分支 scale 作为 W13 每 expert global scale。

W13 activation scale:
  对全部 expert / gate-up input_scale 做 vLLM emulation 的 max-collapse，形成单一 activation global scale。

W2 activation scale:
  对全部 expert down input_scale 做 max-collapse，形成单一 activation global scale。
```

不得将其“修正”为 per-expert activation scale。

## 4.2 Online DIAG

对任意 Linear：

### diag_then_rot

\[
X' = X D R, \qquad W' = W D^{-1} R.
\]

### rot_then_diag

\[
X' = X R D, \qquad W' = W R D^{-1}.
\]

无 R64 时 `R=I`。

每层 online 参数：

```text
attention:
  z_q    [2048]
  z_k    [2048]
  z_v    [2048]
  z_o    [4096]

experts:
  z_gate [128, 2048]
  z_up   [128, 2048]
  z_down [128, 768]
```

约 632,832 个 DIAG 参数/层，只做 `layer_joint`。

Router 永远看原 post-attention norm 输出；online gate/up 的 D 只在 routing 完成后对 routed expert input 施加，因此 router 不需要补偿。

## 4.3 Fusable DIAG

每层：

```text
D_QKV       [2048]       shared
D_VO        [512]        shared, KV-head space
D_GU        [2048]       shared
D_UD        [128,768]    per-expert
```

约 102,912 参数/层。

通用无-H16公式：

\[
X' = (X D_{in})R,
\]

\[
W' = D_{out} W D_{in}^{-1} R,
\]

于是无量化时：

\[
X'(W')^T=(XW^T)D_{out}.
\]

具体角色：

```text
q/k:
  D_in  = D_QKV
  D_out = I

v:
  D_in  = D_QKV
  D_out = D_VO

o:
  attention output 已由 V 携带 D_VO_expanded
  D_in  = D_VO_expanded
  D_out = I

router:
  input = X * D_GU
  W_router' = W_router * D_GU^{-1}  # column compensation
  不做 HiF4，不做 R64

expert e gate:
  D_in  = D_GU
  D_out = I

expert e up:
  D_in  = D_GU
  D_out = D_UD[e]

expert e down:
  SwiGLU 输出已携带 D_UD[e]
  D_in  = D_UD[e]
  D_out = I
```

SwiGLU 等价性：

\[
\mathrm{SiLU}(g)\odot(uD_{UD}^{(e)})
= [\mathrm{SiLU}(g)\odot u]D_{UD}^{(e)}.
\]

禁止给 gate 输出再额外增加独立 fusable output DIAG。

## 4.4 GQA 的 D_VO

`D_VO` 只在 4 个 KV heads × 128 head_dim = 512 维学习。

映射到 O 输入时按 GQA 复制到 32 个 Q heads。每个 KV head 对应 8 个 Q heads；同一 KV-head 的 128 维 scale block复制 8 次。禁止给每个 Q head 学新的自由参数。

## 4.5 R64 位置

R64 仍固定为当前 `diag_gradient/r64_transform.py` 的 `R4⊗R4⊗R4` 64×64 正交矩阵，不学习。

```text
q/k/v K=2048       -> 32 个连续 G64
gate/up K=2048     -> 32 个连续 G64
down K=768         -> 12 个连续 G64
o K=4096           -> 32 heads × 2 个 G64/head
router              -> 不做 R64
```

O 的 G64 禁止跨 attention head。

---

# 5. 全局实现约束

- 所有 Python/pytest/训练命令使用 `hif4` 环境。
- 保留旧 8B 文件和旧测试，不将 `src/semantic_model.py` / `src/checkpoint.py` 改成混杂 8B+MoE 的巨型条件分支。
- 新 MoE 适配继续放在现有 `experiments/e2e_diag_reconstruction/` 分层内，新增 MoE 专用模块，不建第二套完整 pipeline。
- HiF4 数学只允许一个 source of truth；训练、fold、materialize、vLLM sidecar 必须共用相同定义/测试向量。
- 不实现 packed HiF4 kernel。
- 不允许 broad exception fallback、silent OOM fallback、自动降 batch、自动改 calibration。
- 不新增 SHA/hash/tensor fingerprint 作为实验门禁。
- Native NVFP4 formal baseline 不允许 fallback 到 Marlin W4A16。
- A800 Native E0 固定：

```text
linear_backend=emulation
moe_backend=emulation
kv_cache_dtype=bfloat16
enforce_eager=true
```

- 训练一张 GPU 一个完整 run；不同结构 run 可按 GPU pool 并行。
- E2E eval 统一按 2 GPU/任务规划，避免 30B materialized BF16 checkpoint 的单卡显存风险。
- calibration shared cache 继续复用现有 CPU tensor cache。

---

# Part I：模型与训练 runtime 适配

## Task 0：建立 30B MoE 模型契约与 checkpoint preflight

**Files**

- Create: `.../core/moe_model_spec.py`
- Create: `.../core/modelopt_moe_checkpoint.py`
- Create: `.../tests/e2e_diag_reconstruction/test_moe_model_spec.py`
- Reuse/read-only reference: `NVFP4/scripts/inspect_native_nvfp4_checkpoint.py`

- [ ] 从真实 snapshot 的 `config.json/hf_quant_config.json/model.safetensors.index.json` 建立 `Qwen3MoeModelSpec`。
- [ ] 明确断言 48 layers / 128 experts / top8 / hidden2048 / head32 / kv4 / head_dim128 / moe768。
- [ ] 明确断言无 shared expert。
- [ ] 明确断言 48 router + lm_head 在 exclude_modules 中。
- [ ] 统计 NVFP4 target：192 attention + 18432 expert = 18624。
- [ ] 明确断言 checkpoint 没有 `forward_hadamard_matrix`；新 MoE runtime 不接受 native H16。
- [ ] 校验每个 target 存在 `weight / weight_scale / weight_scale_2 / input_scale`。
- [ ] 不读取全部 tensor body做 coverage；header/index 足够。

测试：

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_moe_model_spec.py -q
```

验收：真实 snapshot preflight PASS，打印上述真实计数。

---

## Task 1：配置从 8B hard-code 改成模型 spec 驱动

**Files**

- Modify: `.../core/config.py`
- Modify: `.../cli/train.py`
- Modify: `.../cli/prepare_calibration.py`
- Modify: `.../tests/e2e_diag_reconstruction/test_config.py`

要求：

- 默认 model 改为 `nvidia/Qwen3-30B-A3B-NVFP4`。
- 不再用全局常量 `36 / TARGET_LINEAR_COUNT=252` 决定合法 layer range。
- 对目标 30B，从 model spec 得到 `last_layer=47`。
- `diag_train_scope` 在 30B MoE 正式路径**只允许 `layer_joint`**；显式传 `linear_independent` 直接报错。
- `fusable+rot_then_diag` 继续报错。
- calibration default 仍 `s1k_original`。
- 不删除旧 8B `linear_independent` 实现；只是新 MoE model spec 不允许走。
- config / manifest 中新增：

```text
model_type=qwen3_moe
num_layers=48
num_experts=128
top_k=8
kv_cache_dtype=bfloat16
native_has_online_rotation=false
```

`prepare_calibration.py` 只加载 tokenizer，不加载 30B 模型。

---

## Task 2：实现 layer-scoped ModelOpt NVFP4 lazy materialization

**Files**

- Implement: `.../core/modelopt_moe_checkpoint.py`
- Create tests: `.../test_modelopt_moe_checkpoint.py`

必须提供明确接口，名称可按下列定义直接实现：

```python
load_qwen3_moe_layer_state(snapshot, layer_idx, device) -> MoELayerMasterState
release_qwen3_moe_layer_state(state) -> None
```

`MoELayerMasterState` 只持有当前层：

```text
input_layernorm
post_attention_layernorm
q_norm / k_norm
router BF16 weight
q/k/v/o FP32 master
128 × gate/up/down FP32 master
Native NVFP4 activation/global scale metadata
```

加载规则：

1. source 始终是原 packed ModelOpt checkpoint；
2. 当前层进入时一次性从 safetensors 读取 packed tensors；
3. 按 vLLM ModelOpt 规则 normalize global scales；
4. 使用仓库现有 vLLM `nvfp4_emulation_utils.dequantize_to_dtype` 或单一 wrapper 解码；禁止再写第二份 E2M1/F8 解码表；
5. master transform 用 FP32；teacher GEMM/模型输出按 BF16 Qwen3 runtime需要 cast；
6. 当前层 20 epochs 内不重复读磁盘/反量化 master；
7. 当前层结束并完成下一层 hidden cache传播后立即删除 state + temporary transformed weight，并 `torch.cuda.empty_cache()` 一次；
8. 禁止 full 30B BF16 master 常驻 GPU。

需要测试：

- 随机抽 layer0/layer23/layer47；
- attention q/k/v/o shape；
- expert0/expert127 gate/up/down shape；
- QKV scale collapse 与 `ModelOptNvFp4LinearMethod.process_weights_after_loading()` 一致；
- W13/W2 scale collapse 与 `convert_to_nvfp4_moe_kernel_format(...EMULATION)` 一致；
- load→release 后不保留该层 master 引用。

---

## Task 3：实现与 vLLM E0 完全对齐的 Native semantic teacher

**Files**

- Create: `.../core/moe_semantic_hif4.py`
- Create: `.../training/moe_layer_runtime.py`
- Create: `.../tests/e2e_diag_reconstruction/test_moe_native_teacher.py`

实现原则：

- 使用 Transformers Qwen3-MoE decoder layer 的 attention、RoPE、RMSNorm、router/top-k 语义；禁止自己重新发明 routing。
- 用 meta/full skeleton 只负责结构和 layer-call 参数，不 materialize 全模型权重。
- layer0 hidden cache 只 materialize `embed_tokens` 与进入 layer0 前真正需要的非 layer 参数。
- 每次 replay 只 materialize 当前层。

Native target：

```text
attention q/k/v:
  shared vLLM-collapsed NVFP4 activation QDQ
  q/k/v native dequant master GEMM

attention o:
  own NVFP4 activation scale QDQ

MoE:
  BF16 router -> top8
  W13 input按 vLLM collapsed a13 scale QDQ
  gate/up using vLLM W13 dequant semantics
  SiLU(gate)*up
  W2 input按 vLLM collapsed a2 scale QDQ
  down using vLLM W2 dequant semantics
  原 routing weights combine
```

关键门禁：

- [ ] 用与 `NVFP4/scripts/run_nvfp4_emulation_puncture.py` 相同的固定输入，比较新 semantic attention projection 与 vLLM emulation。
- [ ] 比较完整 MoE routed output；tolerance 不得比已有 puncture 更松。
- [ ] 比较 router logits/top-k 与 vLLM/HF reference。
- [ ] **只有该 puncture PASS，才允许开始 DIAG training。**

Task 3 的 DoD 固定为以下四项全部满足：

```text
A. core/moe_semantic_hif4.py 存在并只实现 native semantic math；
B. training/moe_layer_runtime.py 存在，负责 position ids / RoPE / mask / lazy state 到 decoder-layer runtime 的调用边界；
C. test_moe_native_teacher.py 不得只有 smoke，必须包含真实 checkpoint attention / routed-MoE / router 三类 reference test；
D. 三类 reference test 实际 PASS，并把测试命令与 tolerance 记录进 IMPLEMENTATION_PROGRESS.md。
```

单纯的 `ref_nvfp4_quant_dequant()` 单元测试或“完整 layer forward 输出 finite”只能算辅助 smoke，**不能作为 Task 3 完成证据**。

禁止“teacher 大致接近”后继续实验。

---

## Task 4：实现 MoE 无-H16 DIAG/R64 数学唯一入口

**Files**

- Create: `.../core/moe_transforms.py`
- Create: `.../tests/e2e_diag_reconstruction/test_moe_transforms.py`

接口至少覆盖：

```text
apply_r64_no_cross_head
online_weight_transform_no_h
fusable_weight_transform_no_h
expand_vo_scale_qwen3_moe
transform_router_weight
```

测试必须使用无量化 FP32 等价性：

- q/k/v；
- o，并断言 R64 不跨 head；
- expert gate/up/down；
- `D_UD[e]` SwiGLU 等价；
- router：`(x*D_GU) @ (W_router*D_GU^-1)^T == x@W_router^T`；
- R64 orthogonality。

容差使用 FP32合理容差；不要因为 BF16 评测而放松数学单测。

---

## Task 5：实现 DIAG state 与当前层 student runtime

**Files**

- Extend: `.../core/moe_semantic_hif4.py`
- Create/extend: `.../training/moe_layer_runtime.py`
- Modify/reuse: `.../training/trainer.py`
- Create: `.../tests/e2e_diag_reconstruction/test_moe_semantic_hif4.py`

Task 5 不允许一次性“凭感觉”改 trainer。固定拆成 5A→5E，**每个子步骤测试通过后立即继续下一个，5E 完成前 Task 5 仍为未完成。**

### Task 5A：DIAG state 只定义参数，不接训练循环

Fusable state：

```text
z_qkv [2048]
z_vo  [512]
z_gu  [2048]
z_ud  [128,768]
```

Online state：

```text
z_q    [2048]
z_k    [2048]
z_v    [2048]
z_o    [4096]
z_gate [128,2048]
z_up   [128,2048]
z_down [128,768]
```

统一：

```text
D = 2^z
z init = 0
default clamp = [-4,4]
```

测试：shape、requires_grad、identity snapshot/load、clamp；fusable/online 参数集合不得互相泄漏。

### Task 5B：实现单个 Dense attention student projection

固定接口应能表达：

```text
forward_student_attention_proj(proj, x, master_weight, diag_state, use_r64, rot_order, step_cache)
```

要求：

- weight 始终从固定 FP32 Native master重新 transform；
- transformed weight 做 HiF4 STE-QDQ；
- 同一 optimizer step 同一 proj weight QDQ 一次；
- q/k/v/o 分别验证 fusable 与 online；
- 和 `moe_transforms.py` 的无量化公式共用 transform，不复制矩阵顺序。

### Task 5C：实现 routed expert student compute

固定流程：

```text
BF16 router -> selected_experts/topk_weights
for active expert e:
    gather tokens
    gate student
    up student
    SwiGLU
    down student
    routing-weighted scatter/add
```

要求：

- fusable 使用 shared `D_GU` + per-expert `D_UD[e]`；
- online 使用 `D_gate[e]/D_up[e]/D_down[e]`；
- 仅 active expert 做 transformed-weight QDQ/GEMM；
- 同一 step 同一 expert/proj transformed weight QDQ 一次；
- router 与 Task 3 Native teacher 使用同一 routing semantics。

测试必须人工构造多个 token 命中相同 expert，验证 cache 不会按 token重复 QDQ；同时覆盖多个 expert。

### Task 5D：拼成完整 current-layer student forward

固定顺序：

```text
input RMSNorm
 -> attention q/k/v/o student
 -> residual
 -> post-attention RMSNorm
 -> router
 -> routed MoE student
 -> residual
```

同一个输入下同时支持：

```text
native teacher forward
identity HiF4 student (all z=0)
trainable fusable student
trainable online student
```

此处只实现**当前层 runtime**，还不在 Task 5 内改 0..47 orchestration。

### Task 5E：expert coverage 与 step-cache 审计

每层必须能输出：

```text
active_experts_train
active_experts_val
per_expert_routed_token_count
min / median / max routed tokens
never_routed_experts
weight_qdq_calls_by_proj
```

规则：

- 未在 calibration 中被路由到的 expert 参数自然保持 `z=0`；
- 禁止伪造额外 routing 样本；
- 每个 optimizer step结束后明确清空 step-local transformed-weight cache；
- cache 只跨当前 batch 内 token复用，不跨 optimizer step复用 trainable transformed weight。

### Task 5 完成门禁

Task 5 只有以下全部 PASS 才算完成：

```text
1. state shape/identity/clamp tests；
2. attention fusable + online numerical tests；
3. expert fusable + online numerical tests；
4. repeated-routed-token QDQ-once test；
5. full current-layer forward smoke；
6. expert coverage统计 test。
```

不得仅创建 `test_moe_semantic_hif4.py` 后就标记 Task 5 完成。

---

## Task 6：实现 fusable fold，包括 BF16 router compensation

**Files**

- Create: `.../core/moe_fold.py`
- Create: `.../tests/e2e_diag_reconstruction/test_moe_fold.py`

最终 fold：

```text
input_layernorm.weight *= D_QKV
post_attention_layernorm.weight *= D_GU
router.weight columns *= D_GU^-1     # BF16 router，不量化

q/k:   W D_QKV^-1 [R]
v:     D_VO W D_QKV^-1 [R]
o:     W D_VO_expanded^-1 [R]

gate_e: W D_GU^-1 [R]
up_e:   D_UD[e] W D_GU^-1 [R]
down_e: W D_UD[e]^-1 [R]
```

`[R]` 表示 `use_r64=true` 时右乘 R64。

必须有两级 router gate：

1. FP32 algebra test：等价；
2. 用最终 BF16 folded router + calibration hidden states，要求 **top-k expert IDs 100%一致**，并记录 router-logit max_abs/rel_l2。

若 BF16 fold 导致 top-k 改变：直接 FAIL 当前实现/实验，不静默把 router 恢复、不局部关闭 D_GU。

同时验证 fusable fold 前后在**无 HiF4 QDQ**条件下完整 layer output 对齐。

---

## Task 7：改造逐层训练 orchestration 为 lazy layer runtime

**Files**

- Modify: `.../cli/train.py`
- Create/modify: `.../training/moe_layer_runtime.py`
- Modify: `.../training/trainer.py`
- Modify: `.../tests/e2e_diag_reconstruction/test_layer_runtime.py`
- Modify: `.../tests/e2e_diag_reconstruction/test_trainer.py`

流程固定：

```text
构造/加载 calibration CPU cache
  ↓
用 embedding 构造 X_0 CPU BF16 cache
  ↓
for layer l = 0..47:
    lazy load + normalize + dequant 当前 layer master
    构造当前 layer runtime
    预计算 Native teacher train/val target 一次
    训练当前层 DIAG 20 epochs
    held-out val 选 best epoch
    identity baseline 比较 / rollback
    fusable fold gate 或 online freeze
    用 accepted student 或 native teacher 传播 X_{l+1}
    保存 layer artifact + expert routing stats
    release 当前 layer masters/runtime
```

`teacher` input mode：下一层输入来自 Native layer；`progressive_student`：下一层输入来自 accepted/rollback student。

`start_layer>0`：必须逐层 lazy replay 0..start-1，不能一次性 materialize prefix。

主 training 保留：

```text
recon_loss=block_delta_nmse
lr=5e-3
epochs=20
scheduler=cosine
batch_size=4
rollback=on
```

不要新增 adaptive batch/OOM fallback。

---

## Task 8：升级 compact artifact / resume schema

**Files**

- Modify: `.../core/artifact.py`
- Modify: `.../tests/e2e_diag_reconstruction/test_fold_artifact.py`

新 artifact 必须能表达：

```text
schema_version = 2
model_type = qwen3_moe
source_model
num_layers = 48
num_experts = 128
diag_mode
use_r64
rot_order
calib_source
calib_seed
calib_input_mode
layer_rollback
kv_cache_dtype = bfloat16
per-layer z
per-layer accepted/rollback/best_epoch
```

Online 的 expert-specific z 允许使 artifact 达到百 MB 量级，但仍远小于完整模型；禁止每个 training run 保存完整 30B BF16 checkpoint。

旧 schema v1 读取逻辑保留给旧 8B；不要把 v1 artifact 当作 v2 MoE artifact。

`require_complete_train_run` 改为检查 `layer_00 ... layer_47`。

---

# Part II：E2E materialization 与 vLLM HiF4 runtime

## Task 9：实现 streaming evaluation checkpoint materializer

**Files**

- Create: `.../evaluation/moe_materialize.py`
- Modify: `.../evaluation/vllm_runner.py`
- Create: `.../tests/e2e_diag_reconstruction/test_moe_materialize.py`

目的：从 packed ModelOpt source + compact conversion artifact 生成**临时标准 BF16 HF checkpoint**，其中目标 weight 已完成 transform + HiF4 weight QDQ。

固定策略：

- 一层一个 safetensors output shard，外加 global tensors shard；
- 每次只在 CPU/GPU内存保存一个 layer 的输出 tensor，禁止一次性构建全 30B state_dict；
- packed NVFP4 aux tensors `weight_scale/weight_scale_2/input_scale` 不写入 materialized checkpoint；
- `config.json` 删除 `quantization_config`，dtype 标记 BF16；
- router/norm按 fusable artifact进行 fold；
- online 不 fold activation-side D，仅预变换 + QDQ weight；
- 写标准 `model.safetensors.index.json`；
- tokenizer/config/generation 等辅助文件 symlink/copy；
- materialized checkpoint 放 scratch/shared eval 目录，不放每个训练 run 内永久占 60GB。

默认 eval 完成后删除临时 checkpoint；仅 `KEEP_EVAL_CKPT=1` 时保留。

同时生成 compact runtime sidecar：

```text
hif4_runtime_spec.pt
```

内容：model type、variant、use_r64、rot_order、online activation DIAG、R64 placement、layer count；fusable 不保存已经 fold 的 D 作为 runtime乘法。

---

## Task 10：为普通 Dense attention 增加 sidecar-aware HiF4 runtime

**Files**

- Modify: `main.py`
- Modify minimal vLLM plumbing: `3rdparty/vllm/vllm/model_executor/layers/linear.py`
- Create: `3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_runtime.py`
- Reuse: `3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_fake.py`
- Tests: `.../test_moe_vllm_runtime.py`

`hif4_runtime.py` 是 sidecar 的唯一 vLLM 读取/查询入口：worker 内第一次访问时加载一次 `hif4_runtime_spec.pt`，之后按 `layer prefix + TP rank` 返回本地 transform view；禁止在每个 token/forward 重读 sidecar，也禁止在 `linear.py`、MoE experts 中各自复制 sidecar 解析逻辑。

新增 CLI 只使用一个明确入口，例如：

```text
--hif4_runtime_spec PATH
```

再通过 vLLM `additional_config["hif4_runtime_spec_path"]` 传入。没有 sidecar 时现有 vLLM 行为必须完全不变。

Dense semantics：

```text
Direct:
  x -> HiF4 QDQ -> pre-QDQ weight GEMM

R64-only/fusable+R64:
  x -> R64 -> HiF4 QDQ -> transformed weight GEMM

Online:
  q/k/v/o 分别按自己的 D 与 rot_order处理，再 QDQ
```

由于 online q/k/v DIAG 独立，不能继续把一次 transformed activation直接送进 fused QKV。Online correctness runtime必须在当前 `QKVParallelLinear` 的本地 weight shard 中按 vLLM 已有 q/k/v shard offset 拆出三段，分别计算后 concatenate；禁止另存一份完整 q/k/v weight。

TP=2 时必须显式处理 sidecar 的本地视图：q/k/v 的 input DIAG 长度 2048，在各 TP rank 上完整复用；`o_proj` 是 row-parallel，`z_o[4096]` 必须按该 rank 的 attention-head/K shard切成 2048，本地 R64 仍按 128-d head 内两组 G64执行。禁止直接拿全局 `z_o` 乘本地 activation。

这是确认的高自由度 correctness reference，不以速度为目标。必须比较 project PyTorch runtime 与 vLLM runtime 的 q/k/v/o 输出，并增加 TP1 reference 与 TP2 分片重组的一致性测试。

---

## Task 11：实现 fused fusable/direct/R64 MoE HiF4 runtime 与 online reference MoE runtime

**Files**

- Minimal hook: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py`
- Create: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/hif4_emulation_moe.py`
- Create: `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/hif4_online_reference_moe.py`
- Tests: `.../test_moe_vllm_runtime.py`

仅当 `hif4_runtime_spec_path` 存在时选择新路径；不得改现有 NVFP4 emulation Oracle/数学。

### Direct / R64 / fusable

保持 W13 fused：

```text
router BF16
 -> top8
 -> routed hidden
 -> [optional R64]
 -> HiF4 QDQ once
 -> fused W13
 -> SiLU-and-mul
 -> [optional R64]
 -> HiF4 QDQ once
 -> W2
 -> routing weight combine
```

Fusable D 已全部 fold 进 norm/router/weight，runtime 不再重复乘 D。

### Online

correctness-first：

```text
router BF16/top8 保持 vLLM
for active expert e:
    gather routed tokens
    gate activation: expert-specific D_gate[e]/R order -> HiF4 QDQ
    up activation:   expert-specific D_up[e]/R order   -> HiF4 QDQ
    gate/up 分开 GEMM
    silu(gate)*up
    down activation: expert-specific D_down[e]/R order -> HiF4 QDQ
    down GEMM
scatter + routed weight reduce
```

Online 不要求 fused W13，也不要求高吞吐。

TP=2 时 expert intermediate 768 会按 TP 分成每 rank 384。`D_gate[e] / D_up[e]` 是 hidden-input 2048 维，两个 rank 均使用完整向量；`D_down[e,768]` 必须按 vLLM row-parallel K shard切成本 rank的384维。384=6×64，因此 down 的 R64 可以在本 rank 完整按 G64执行，不跨 rank。禁止把768维 D直接作用到384维本地 activation，也禁止为了方便 all-gather intermediate 再做 online D/R64。

Materializer 必须在**全局逻辑 weight**上先执行 transform + HiF4 weight QDQ，再交给 vLLM 正常 TP shard。增加门禁：对 q/k/v/o/gate/up/down，`full weight HiF4-QDQ -> TP2 slice` 与 `对应本地完整G64 shard的 reference` 数值一致；若某维度未来不能整除64则直接 fail，不引入跨 rank量化通信。

测试至少覆盖 2/8/128 expert synthetic case、重复 expert token、未命中 expert、top-k routed weights，并与 project semantic runtime逐元素比较；额外做 TP1 reference 与 TP2 输出 all-reduce/reduce 后的一致性测试。

---

## Task 12：统一 ARC / MMLU-Pro / AIME E2E 评测语义

**Files**

- Modify: `.../evaluation/runner.py`
- Modify: `.../evaluation/vllm_runner.py`
- Create: `.../evaluation/lm_eval_vllm.py`
- Modify: `.../evaluation/common.py`
- Modify: `.../cli/evaluate.py`
- Modify: `.../scripts/common.sh`
- Modify: `.../scripts/eval/run_fast_eval.sh`
- Modify: `.../scripts/eval/run_final_eval.sh`
- Modify: `main.py`
- Tests: `.../test_evaluation.py`

### Native E0

直接加载原 packed checkpoint：

```text
linear_backend=emulation
moe_backend=emulation
kv_cache_dtype=bfloat16
enforce_eager=true
```

禁止 `fake_act_quant=nvfp4`，禁止 Marlin。

### Converted E1-E7

加载 materialized BF16 weight checkpoint + HiF4 runtime sidecar。

全部：

```text
kv_cache_dtype=bfloat16
enforce_eager=true
TP=2
```

### ARC

ARC-Easy / ARC-Challenge **仍必须使用 lm_eval**，但 30B 不再使用旧 `HFLM(pretrained=model)` 单卡内存模型。`evaluation/lm_eval_vllm.py` 固定做一层薄适配：复用 lm_eval 已有 vLLM model class/请求协议，只覆盖 engine 构造 kwargs，使其明确传入 `tensor_parallel_size=2`、`kv_cache_dtype=bfloat16`、`enforce_eager=True` 和 `additional_config={"hif4_runtime_spec_path": ...}`。禁止自己实现 ARC scoring，也禁止为了 sidecar 改成 lighteval。Native E0 同一 adapter 不传 sidecar，改传 `linear_backend=emulation/moe_backend=emulation`。

### MMLU-Pro

repo root `main.py` + lighteval + vLLM：

```text
datasets=mmlu_pro|0
max_samples=300
TP=2
KV=BF16
```

### AIME25

仅 finalists，`aime25_avg5|0`。

Winner 排序恢复为本计划正式口径：

```text
1. MMLU-Pro 300
2. ARC-Challenge
3. ARC-Easy
```

修改当前 `evaluation/summarize.py::choose_best()`；不做综合分。

---

## Task 13：实现前完整 smoke / correctness gate

在任何 48-layer 正式 training 前必须全部通过：

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction -q
```

以及保留旧 8B 基线测试通过，确认没有破坏历史路径。

新增 smoke：

```text
A. ModelOpt layer0 lazy load/release
B. Native semantic vs vLLM attention puncture
C. Native semantic vs vLLM MoE puncture
D. fusable no-R64, layer0, 2 train / 1 val / 1 epoch
E. fusable+R64 同规模 smoke
F. online expert-specific, layer0, 2/1/1 smoke
G. router compensation top-k identity gate
H. materialize E1/E2/E4/E6 各一个最小 checkpoint smoke
I. converted vLLM TP2 generate 8-16 token，KV=BF16，确认 sidecar 生效
J. Native E0 packed NVFP4 TP2 + BF16-KV smoke，确认 linear/moe 都是 emulation 且无 Marlin
K. 固定短 prompts 做 Native E0 TP1/TP2 numerical parity diagnostic：prefill next-token top-k/logit 数值指标、首次分叉 margin 分析、layer-level TP puncture；greedy generation exact-match 只作为 report，不作为第一 PASS/FAIL gate
```

每次打印 CUDA peak memory。任何一项失败都不得启动 Phase A。当前已有的 TP1 Native smoke 不能替代 J/K，因为正式 E2E eval 统一使用 TP2。若 K 中 greedy 文本不一致，但 identical input 下数值 parity、top-k overlap、首次分叉 margin 分析与 layer-level puncture 支持 Case A（小数值差异 + 小 margin 导致 greedy bifurcation），不得据此修改 NVFP4 数学；TP2 仍作为正式 E2E eval baseline。

---

# Part III：正式实验矩阵

## Task 14：Phase A E0–E7 核心结构

**Script**

- Modify: `.../scripts/ablation/run_phase1_matrix.sh`

固定 calibration：

```text
s1k_original
train=128
val=32
seed=42
batch=4
```

矩阵不变：

| ID | Method | Train |
|---|---|---:|
| E0 | Native NVFP4 | No |
| E1 | Direct HiF4 | No |
| E2 | Direct HiF4 + R64 | No |
| E3 | Fusable DIAG | Yes |
| E4 | Fusable DIAG → R64 | Yes |
| E5 | Online DIAG | Yes |
| E6 | Online DIAG → R64 | Yes |
| E7 | Online R64 → DIAG | Yes |

E3-E7 五个训练 run 可一 GPU 一 run按现有 GPU pool wave 并行。

所有 E0-E7 fast eval：

```text
ARC-Easy
ARC-Challenge
MMLU-Pro 300
KV=BF16
eval_seed=42
```

分析固定：

```text
E1 -> E2 : R64 only
E1 -> E3 : fusable DIAG
E1 -> E5 : online DIAG upper bound
E3 -> E4 : fusable 后 R64
E5 -> E6 : online DIAG 后 R64
E6 vs E7 : online order
E3 vs E5 : fusable freedom cost
E4 vs E6 : fusable freedom cost with R64
```

输出：best_fusable / best_online / best_overall。

---

## Task 15：Phase B fusable component ablation

**Script**

- Modify: `.../scripts/ablation/run_fusable_component_ablation.sh`

固定 `use_r64=false`，矩阵：

```text
B1 qkv   = D_QKV
B2 vo    = D_VO
B3 gu    = shared D_GU + router compensation
B4 ud    = 全部 128 个 per-expert D_UD
B5 attn  = QKV + VO
B6 mlp   = GU + all per-expert UD
B7 all   = reuse Phase A E3
```

报告除了 E2E/layer recovery，还增加：

```text
router top-k equivalence stats
overall expert coverage
D_UD per-expert z/d distribution
```

---

## Task 16：Phase C 训练机制与稳定性

**明确删除旧 8B 的 `Joint vs Linear-independent` 正式消融。**

`run_train_scope_ablation.sh` 可以保留历史文件，但 README/正式 orchestrator 标记为“不适用于 Qwen3-30B MoE”；不得执行。

### C1 input mode

复用 `run_input_mode_ablation.sh`：

```text
best fusable: progressive vs teacher
best online:  progressive vs teacher
```

新增 2 runs。

### C2 loss

复用 `run_loss_ablation.sh`：

```text
block_delta_nmse baseline
block_output_nmse
mse
```

best fusable + best online，新增 4 runs。

### C3 rollback

best fusable：on vs off，新增 1 run。

### C4 clamp

best fusable：`-4,4` vs none，新增 1 run。

Phase C 总新增 8 runs。

---

## Task 17：Phase D calibration source

复用 `run_calib_ablation.sh`。

比较：

```text
s1k_original
s1k_question
wikitext2
c4
```

对 best fusable / best online。

`s1k_original` 直接复用 Phase A，新增 6 runs。

仍需报告每个 source：

```text
train/val sample count
total tokens
max/P50/P95 length
```

当前是“实用 calibration recipe”比较，不宣称等 token-budget 数据域消融。

另外增加 MoE 特有统计：各 source 对 128 experts 的 routed coverage；用于解释某个 calibration source 是否只覆盖少数 expert。

---

## Task 18：Finalists AIME25 avg@5

Finalists：

1. E0 Native NVFP4；
2. E1 Direct HiF4；
3. Phase A best fusable；
4. Phase A best online；
5. Phase D best fusable calibration（若不是 s1k_original）；
6. Phase D best online calibration（若不是 s1k_original）。

去重后跑：

```text
AIME25 avg@5
TP=2
KV=BF16
enforce_eager=true
```

ARC/MMLU 复用 fast eval，不重复。

---

# Part IV：结果与资源管理

## Task 19：最终正式报告

输出：

`Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/QWEN3_30B_A3B_MOE_NVFP4_TO_HIF4_ABLATION_REPORT_2026-08-21.md`

主表：

1. 模型/量化结构与 Native semantic consistency；
2. E0–E7 核心结构；
3. Fusable component；
4. input/loss/rollback/clamp；
5. calibration source + expert coverage；
6. finalist AIME。

每个结构 run 至少报告：

```text
method/config
calib_source
accepted / rollback / would_rollback
mean/median layer recovery
active expert coverage
router compensation top-k mismatch count（fusable应为0）
ARC-E
ARC-C
MMLU-Pro300
AIME25（若执行）
vs Native NVFP4 pp drop
```

必须明确：online 是 correctness-first 高自由度上界，不是当前可部署高效 kernel 设计；fusable 才是主部署候选。

---

# 6. GPU / scratch 规则

训练：

```text
1 GPU / training run
当前 layer lazy materialization
```

评测：

```text
2 GPU / eval run
```

修改 `scripts/common.sh`：

- `require_complete_train_run` layer range 0..47；
- 30B fast eval 不再把 ARC 当成单卡 special case；ARC/MMLU 都按 2-GPU eval slot；
- `FAST_EVAL_GROUPS` 正式默认 `arc,mmlu_pro_300`；
- GPU pool 仍只是 shell wave，不进入 trainer/runtime。

临时 materialized checkpoint：

```text
results/e2e_diag_reconstruction/shared_vllm_qwen3_30b/<run_id>/<variant>/
```

`run_id` 直接使用已有结果目录 basename，不新增 hash/checksum 作为 cache key。默认任务结束删除；`KEEP_EVAL_CKPT=1` 才保留。不要同时 materialize 多份无关 60GB checkpoint把磁盘打满。

---

# 7. 正式新增训练 run 数

```text
Phase A: 5
Phase B: 6
Phase C: 8
Phase D: 6
----------------
总计: 25 个新增 48-layer training run
```

E0/E1/E2 不训练。

取消旧计划的 `linear_independent` 后比 8B 版本少 1 个正式 training run。

---

# 8. 执行顺序

```text
Task 0  model/checkpoint contract
  ↓
Task 1  config/model-spec dispatch
  ↓
Task 2  layer-scoped ModelOpt loader
  ↓
Task 3  Native semantic == vLLM E0 puncture
  ↓
Task 4  no-H16 DIAG/R64 math
  ↓
Task 5  fusable/online MoE student runtime
  ↓
Task 6  fusable fold + BF16 router compensation
  ↓
Task 7  lazy layerwise training orchestration
  ↓
Task 8  artifact/resume v2
  ↓
Task 9  streaming eval materialization
  ↓
Task 10 dense HiF4 vLLM sidecar runtime
  ↓
Task 11 fused fusable/direct/R64 MoE + online reference MoE
  ↓
Task 12 ARC/MMLU/AIME unified eval
  ↓
Task 13 全部 smoke/correctness gates
  ↓
Phase A E0-E7
  ↓
Phase B components
  ↓
Phase C input/loss/rollback/clamp
  ↓
Phase D calibration
  ↓
Finalists AIME25 avg@5
  ↓
正式报告
```

Cursor 禁止跳过 Task 3、Task 6、Task 11 的数值一致性 gate 后直接跑正式实验。

---

# 9. Final Acceptance Criteria

1. 旧 Qwen3-8B plan/实现仍可读、旧测试不被破坏；新 MoE 代码按专用模块增加。
2. 新 model spec 正确识别 48 layers / 128 experts / top8 / 18624 NVFP4 physical Linears / 48 BF16 routers。
3. 新 MoE runtime 中不存在 native H16/`forward_hadamard_matrix` 依赖。
4. layer-scoped loader 不把完整 30B BF16 expert master 常驻 GPU。
5. Native semantic teacher 的 QKV scale collapse、W13/W2 scale collapse与当前 vLLM emulation一致，并通过真实 layer puncture。
6. Fusable 参数严格为 shared `D_QKV/D_VO/D_GU` + per-expert `D_UD`。
7. `D_GU` fold 后 BF16 router top-k 在 fold gate calibration 上 100%一致；router 不做 HiF4/R64。
8. Online attention及128 experts均为独立 DIAG，并只支持 `layer_joint`。
9. Online E2E correctness runtime允许拆 q/k/v 与 gate/up，结果与 project semantic runtime一致。
10. Fusable/direct/R64 E2E MoE runtime保持 fused W13，并且 activation HiF4 QDQ次数/位置正确，无 double-QDQ。
11. R64 在 O projection 中不跨 128-d attention head；down 768 正好分12个G64。
12. Train/val 每层 Native teacher target只预计算一次；同 step同 expert/proj transformed weight只QDQ一次。
13. 每层输出 expert routing coverage；未路由 expert保持identity，不补伪样本。
14. Artifact为 compact schema v2，不为25个training run各保存完整30B BF16模型。
15. E0 使用真实 packed NVFP4 vLLM emulation + BF16 KV + eager，无 Marlin fallback。
16. E1-E7 使用同一 HiF4 numerical runtime；ARC 与 MMLU/AIME 不得使用不同量化语义。
17. ARC-E/C 走 lm_eval；MMLU-Pro300/AIME走仓库规定框架；MMLU-Pro固定300条。
18. 所有正式 E2E 主实验 KV cache固定 BF16。
19. Phase A完成 E0-E7，并按 MMLU-Pro300 → ARC-C → ARC-E 确定 winner。
20. Phase B完成 qkv/vo/gu/ud/attn/mlp/all；`ud` 明确表示128个 per-expert D_UD。
21. Phase C不再运行 `linear_independent`；只做 input/loss/rollback/clamp。
22. Phase D只比较 s1k_original/question/wikitext2/c4，并报告 token统计和expert coverage。
23. Finalist只新增 AIME25 avg@5，不重复已有 fast metrics。
24. 最终形成 `QWEN3_30B_A3B_MOE_NVFP4_TO_HIF4_ABLATION_REPORT_2026-08-21.md`。

执行时若真实 checkpoint、Transformers Qwen3-MoE forward signature 或现有 vLLM接口与本计划记录发生事实性不一致，Cursor必须停止在对应 Task，记录实际结构并反馈；不得凭猜测改模型语义后继续正式实验。
