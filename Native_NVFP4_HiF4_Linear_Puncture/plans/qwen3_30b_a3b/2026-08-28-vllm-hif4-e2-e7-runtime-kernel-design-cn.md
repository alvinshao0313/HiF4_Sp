# Qwen3-30B-A3B HiF4 E2–E7 vLLM Runtime Kernel 设计

日期：2026-08-28

适用模型：`nvidia/Qwen3-30B-A3B-NVFP4`

适用实验：`Native_NVFP4_HiF4_Linear_Puncture/plans/2026-08-25-qwen3-30b-a3b-native-nvfp4-to-hif4-moe-task14-plus-refactor-replan-cn.md`

本文只设计 E2–E7 正式评测所需的 vLLM HiF4 runtime，不改变任何训练算法定义。

---

# 1. 已锁死的算法语义

## 1.1 HiF4

- 激活使用当前 `hifx4` 64-value group QDQ 语义。
- 当前阶段权重仍以“HiF4 格点上的 BF16 值”保存和执行 grouped BF16 GEMM；真正 packed HiF4 weight kernel 不作为 E2–E7 精度实验的前置依赖。
- CUDA runtime 必须走 Triton，不允许重新退回 Python expert loop / per-expert `F.linear`。

## 1.2 R64

固定：

```text
R64 = kron(kron(H4, H4), H4) / 8
```

其中：

```text
H4 =
[[ 1,  1,  1, -1],
 [ 1,  1, -1,  1],
 [ 1, -1,  1,  1],
 [-1,  1,  1,  1]]
```

R64 只在连续 G64 内变换；attention `o_proj` 的输入必须按每个 head 内部独立 G64 变换，禁止跨 head。

## 1.3 Online DIAG

必须严格保持当前训练定义：

```text
Attention:
  D_q
  D_k
  D_v
  D_o

MoE 每个 expert:
  D_gate[e]
  D_up[e]
  D_down[e]
```

特别锁死：

```text
D_gate[e] != D_up[e] 允许且必须支持
```

禁止为了复用一次 W13 GEMM 而共享 `D_gate` / `D_up`。

Online 的两个顺序保持：

```text
E6: diag_then_rot
    x -> x*D -> R64 -> HiF4 QDQ

E7: rot_then_diag
    x -> R64 -> x*D -> HiF4 QDQ
```

E5：

```text
x -> x*D -> HiF4 QDQ
```

所有 DIAG / R64 计算必须以 FP32 中间值执行，再进入 HiF4 QDQ；输出 dtype 回到 BF16/FP16 输入 dtype，与训练 reference 的 `_student_activation(...)->qdq_hif4_direct(..., BF16)` 对齐。

---

# 2. 当前 runtime 缺口

正式 E2–E7 启动前必须解决以下问题。

## 2.1 E2/E4 R64 激活路径未实现

当前 `hif4_runtime.apply_dense_hif4_runtime()` 对 `direct / fusable / r64 / fusable_r64` 都只执行 HiF4 QDQ，没有真正执行 activation R64。因此 E2/E4 目前权重侧 R64 与激活侧不闭合。

## 2.2 Online 权重未离线 transform

当前 Online materialize 会直接将原权重做 `qdq_hif4_direct`，没有执行：

```text
online_weight_transform_no_h(W, D, use_r64, rot_order)
```

正式 E5–E7 必须先离线生成对应 transformed HiF4-grid BF16 weight。

## 2.3 Online sidecar 尚未被 vLLM runtime 完整消费

当前 sidecar 保存 `online_activation_diag`，但 dense runtime 没有真正取对应 layer/projection 的 D；MoE runtime 没有进入 Online grouped Triton 路径；expert-specific `D_gate/D_up/D_down` 没有 GPU runtime。

## 2.4 Online 不能复用单输入 W13

因为 `D_gate[e]` 和 `D_up[e]` 独立，同一个 routed token 必须生成两份不同的 HiF4 activation：

```text
A_gate(token,e)
A_up(token,e)
```

因此 E5–E7 exact runtime 先采用 Gate grouped GEMM + Up grouped GEMM + Down grouped GEMM，而不是错误地共享一份 W13 input。

---

# 3. 总体 runtime 架构

## 3.1 E1/E3

保持当前快路径：

```text
activation
  -> HiF4 QDQ Triton
  -> grouped W13 GEMM
  -> SiLU*up
  -> HiF4 QDQ Triton
  -> grouped W2 GEMM
  -> moe_sum
```

Fusable DIAG 已折入权重 / norm / router，不增加在线 DIAG kernel。

## 3.2 E2/E4

变为：

```text
activation
  -> R64 + HiF4 QDQ Triton
  -> grouped W13 GEMM
  -> SiLU*up
  -> R64 + HiF4 QDQ Triton
  -> grouped W2 GEMM
```

attention Q/K/V 采用 contiguous G64 R64；`o_proj` 采用 per-head R64。

## 3.3 E5–E7 Online MoE

Router 本身保持 BF16 native 语义，不加 expert DIAG。Router 得到 `topk_ids/topk_weights` 后，将 token-expert assignment 展平：

```text
M_route = num_tokens * top_k
route_expert_ids = topk_ids.reshape(-1)
route_weights    = topk_weights.reshape(-1, 1)
```

然后：

```text
                            routed assignment
                                  |
                   +--------------+--------------+
                   |                             |
           OnlineTransformQDQ             OnlineTransformQDQ
              D_gate[e]                      D_up[e]
                   |                             |
               A_gate                         A_up
                   |                             |
           grouped Gate GEMM             grouped Up GEMM
                   |                             |
                   +--------------+--------------+
                                  |
                            SiLU(gate)*up
                                  |
                     OnlineTransformQDQ
                         D_down[e]
                                  |
                           grouped Down GEMM
                                  |
                         route weight multiply
                                  |
                         view [M, top_k, H]
                                  |
                              moe_sum
```

所有 route tensor 都保持 GPU tensor；禁止 `.item()`、`.tolist()`、Python expert loop。

---

# 4. Kernel K0：现有 HiF4 QDQ

文件：`3rdparty/vllm/vllm/model_executor/layers/quantization/hif4_triton.py`

保留 `hif4_quantize_hifx4_triton(x, out=None)` 作为 Direct/Fusable 和其它 kernel 的数值参考。新增内部公共 helper 时必须复用同一个 `_hif4_fake_quant_block_tl()`，禁止复制出第二套稍有差异的 HiF4 rounding 实现。

---

# 5. Kernel K1：R64 + HiF4 QDQ

目标 API：

```python
hif4_r64_quantize_hifx4_triton(x, *, out=None, head_dim=None)
```

语义：

```text
head_dim=None:
  每个 contiguous 64-value group 独立 R64

head_dim=128 (o_proj):
  先按 [..., n_heads, head_dim] 划分，
  再在每个 head 内按 G64 独立 R64，禁止跨 head。
```

## 5.1 实现策略

R64 采用三层 H4 butterfly：

```text
stage0: base-4 digit 2
stage1: base-4 digit 1
stage2: base-4 digit 0
```

每层做 normalized H4：

```text
y0 = ( a + b + c - d) / 2
y1 = ( a + b - c + d) / 2
y2 = ( a - b + c + d) / 2
y3 = (-a + b + c + d) / 2
```

三层后总归一化即 `/8`，与 `R64` reference 一致。

优先实现单-program register 版本；如果当前 Triton 版本无法安全完成 64-lane permutation，第一阶段允许采用 3 个极轻量 butterfly Triton stage + 1 个 HiF4 QDQ kernel 的 ping-pong workspace 版本，通过性能门禁后再融合为单 kernel。禁止退回 Python / per-expert 调度。

## 5.2 correctness

BF16/FP16 输入至少测试：`[1,64]`、`[7,128]`、`[13,2048]`、`[5,4096], head_dim=128`。

reference：

```python
apply_r64_no_cross_head(x.float(), head_dim=...)
-> qdq_hif4_direct(..., BF16)
```

目标优先 bit-exact；若存在 Triton/torch BF16 rounding 平台差异，必须记录最大误差并证明只来自允许的 BF16 rounding。

---

# 6. Kernel K2：Dense Online DIAG/R64 + HiF4 QDQ

目标 API：

```python
hif4_online_dense_qdq_triton(
    x,
    d,
    *,
    use_r64: bool,
    rot_order: str,
    head_dim: int | None = None,
    out=None,
)
```

支持：

```text
E5: x * D -> HiF4 QDQ
E6: x * D -> R64 -> HiF4 QDQ
E7: R64(x) -> *D -> HiF4 QDQ
```

`d` 必须为 FP32 GPU tensor；runtime 不允许每个 forward 重新 `exp2(z)`。

## 6.1 sidecar 改造

materialize 时将 artifact 中的 `z_*` 转成 deployment-ready `D_* = exp2(z_*)`。sidecar 保存 FP32 D，并增加/锁死：

```text
runtime_schema_version
online_activation_scale
head_dim
hidden_size
moe_intermediate_size
num_experts
top_k
```

不要在 runtime 热路径里反复解析/exp2 原始 z。

## 6.2 dense prefix 映射

`hif4_runtime.py` 严格解析：

```text
model.layers.<L>.self_attn.q_proj -> D_q[L]
model.layers.<L>.self_attn.k_proj -> D_k[L]
model.layers.<L>.self_attn.v_proj -> D_v[L]
model.layers.<L>.self_attn.o_proj -> D_o[L], head_dim=128
```

其它 dense linear 若不在算法定义中，不允许误施加 Online DIAG。

## 6.3 device cache

sidecar 首次 load 可在 CPU，但每个 layer/projection 的 D 第一次使用时 lazily copy 到目标 GPU，并按 `(sidecar_path, device, layer_idx, proj)` 缓存。禁止每 token CPU->GPU copy。

---

# 7. Kernel K3：Routed Online DIAG/R64 + HiF4 QDQ

这是 E5–E7 的核心 kernel。

目标 API：

```python
hif4_online_routed_qdq_triton(
    x,
    route_expert_ids,
    expert_d,
    *,
    source_top_k: int,
    use_r64: bool,
    rot_order: str,
    out=None,
)
```

输入约定：

```text
x:
  gate/up 时 [M, K]
  down    时 [M*top_k, K]

route_expert_ids: [M*top_k]
expert_d: [num_experts, K]
source_top_k:
  gate/up = original top_k (8)
  down    = 1
```

每个 routed row：

```text
route_row = 0..M_route-1
expert = route_expert_ids[route_row]
source_row = route_row // source_top_k
```

## 7.1 E5

单 Triton kernel 完成 routed-expand + DIAG + QDQ：

```text
load x[source_row, G64]
load expert_d[expert, G64]
FP32 multiply
HiF4 QDQ
store out[route_row, G64]
```

## 7.2 E6

```text
load x
load D
FP32 x*D
R64
HiF4 QDQ
store
```

## 7.3 E7

```text
load x
R64
load D
FP32 *D
HiF4 QDQ
store
```

如果 R64 第一版采用 3-stage ping-pong，则 `diag_then_rot` 的首 stage 同时 source-route gather + D multiply，`rot_then_diag` 的最终 QDQ 前同时 D multiply，禁止额外 routed-expand kernel。

---

# 8. Online MoE grouped GEMM 设计

修改 `3rdparty/vllm/vllm/model_executor/layers/fused_moe/experts/hif4_emulation_moe.py`。

保留 Direct/Fusable 的当前 `_apply_hif4_triton_moe()`，新增独立 `_apply_hif4_online_triton_moe(...)`，禁止把 Direct 和 Online 大量条件分支揉进同一热路径。

## 8.1 routing layout

原：

```text
topk_ids     [M,8]
topk_weights [M,8]
```

Online exact path：

```text
route_ids     = topk_ids.reshape(-1,1)      # [M*8,1]
route_weights = topk_weights.reshape(-1,1)  # [M*8,1]
```

对 `route_ids` 只调用一次 `moe_align_block_size(route_ids, BLOCK_SIZE_M, num_experts)`。Gate / Up / Down 三个 grouped GEMM 复用同一 `sorted_ids/routed_experts/padded_count`。

## 8.2 Gate / Up

生成：

```text
A_gate [M*8,H]
A_up   [M*8,H]
```

第一版 exact 实现允许分别对 `w13[:, :I, :]` 与 `w13[:, I:, :]` 调 grouped GEMM。必须确认 slice 的 stride 可被现有 Triton grouped GEMM 正确消费；若不合适，只能在 weight load/post-load 阶段构造一次性 contiguous layout，禁止 forward 中 copy。

输出：

```text
gate [M*8,1,I]
up   [M*8,1,I]
```

随后用 fused `silu(gate)*up`。若现有 `torch.ops._C.silu_and_mul` 无法直接消费双 buffer，则新增简洁 `silu_mul(gate, up, out)` Triton kernel，禁止 forward 中 concat。

## 8.3 Down

```text
hidden
 -> routed Online D_down/R64/HiF4 QDQ
 -> A_down [M*8,I]
 -> grouped W2 GEMM, top_k=1
 -> route_weights 在 GEMM accumulator FP32 阶段乘
 -> C [M*8,1,H]
```

最后将同一 contiguous storage view 为 `[M,8,H]` 调 `ops.moe_sum`，不使用 Python scatter/index_add。

---

# 9. Online 权重离线 materialize

修改 `evaluation/moe_materialize.py`，数学 helper 建议放在 `core/moe_fold.py` 或新文件，避免 materializer 堆数学逻辑。

对每层：

```text
Attention:
W_q' = online_weight_transform_no_h(W_q, D_q, use_r64, rot_order)
W_k' = online_weight_transform_no_h(W_k, D_k, use_r64, rot_order)
W_v' = online_weight_transform_no_h(W_v, D_v, use_r64, rot_order)
W_o' = online_weight_transform_no_h(W_o, D_o, use_r64, rot_order, head_dim=128)

Experts e:
W_gate[e]' = online_weight_transform_no_h(W_gate[e], D_gate[e], ...)
W_up[e]'   = online_weight_transform_no_h(W_up[e],   D_up[e],   ...)
W_down[e]' = online_weight_transform_no_h(W_down[e], D_down[e], ...)
```

然后才执行 `qdq_hif4_direct(weight', BF16)`。Router / RMSNorm 不因为 Online expert DIAG 改写。

必须直接复用训练 `forward_student_*` 使用的 `online_weight_transform_no_h`，禁止另写近似公式。

---

# 10. vLLM dispatch

`UnquantizedFusedMoEMethod.apply()`：

```text
direct/fusable/r64/fusable_r64 -> existing apply_hif4_fused_moe
online -> new apply_hif4_online_fused_moe
```

Online runtime 从 sidecar/device cache 得到当前 layer 的 `D_gate [E,H]`、`D_up [E,H]`、`D_down [E,I]`、`use_r64`、`rot_order`。

FusedMoE layer 应在 setup 时解析/缓存 layer index，不要每 token regex。

Dense `LinearMethod.apply()` 通过 prefix dispatch K1/K2。

---

# 11. Workspace / allocator

Online 建议按最大需求申请少量 contiguous backing buffers，并用 view 切分：

```text
route_gate_q : M*topk*H
route_up_q   : M*topk*H
gate_out     : M*topk*I
up_out       : M*topk*I
hidden_q     : M*topk*I
down_out     : M*topk*H
R64 ping-pong scratch（若第一版需要）
```

可以复用生命周期不重叠的 buffer，但优先保证生命周期清晰；不要为了省一块 buffer 引入未验证的 CUDA stream alias hazard。

---

# 12. 必须新增的 correctness tests

## 12.1 R64 kernel

GPU 对 PyTorch reference，覆盖 contiguous G64 / per-head G64 / BF16 / FP16。

## 12.2 Online Dense

覆盖：`use_r64=false`、`diag_then_rot + R64`、`rot_then_diag + R64`，比较 `_student_activation(...)->qdq_hif4_direct`。

## 12.3 Online routed QDQ

构造 `M=5,E=8,top_k=2,K=64/128`，故意令 `D_gate[e] != D_up[e]`，逐 route 与 Python exact reference 比较。

## 12.4 Online weight materialize

关闭 QDQ 或用可控小矩阵验证：

```text
Linear(T(x,D), W_transformed) == Linear(x, W)
```

分别覆盖 E5/E6/E7 顺序与 `o_proj head_dim`。

## 12.5 Online MoE full small test

随机 `E=8,H=64/128,I=64,top_k=2`，CUDA grouped runtime 对当前 `forward_student_routed_moe(... use_ste=False)` 或独立 scalar reference，必须验证 gate/up 独立 DIAG、down DIAG、route weight 和 final output。

---

# 13. 真实模型 smoke gate

正式 E2–E7 fast eval 前，每种结构只做极短 generate，不直接跑 ARC/MMLU。

统一：

```text
TP=2
KV=BF16
enforce_eager=true
GPU 仅 0–3
```

顺序：E1 Direct -> E2 Direct+R64 -> E3 Fusable -> E4 Fusable+R64 -> E5 Online -> E6 Online diag->rot -> E7 Online rot->diag。

每个 smoke 记录：load success、首 token finite、prompt tok/s、generation tok/s、peak GPU memory、是否出现 Python expert-loop trace。

---

# 14. 性能门禁

## 14.1 硬门禁

以下任一出现都禁止 formal eval：

```text
Python expert loop
per-expert F.linear
forward 热路径 CPU<->GPU D copy
每 token torch.exp2(z)
.item()/.tolist() 触发 GPU sync
Online 退回普通 unfused HF MoE
```

## 14.2 相对性能门禁

以同一短 prompt 的 E1 为 HiF4 runtime 基准。第一轮宽松门禁用于发现执行范式 bug：

```text
E2/E4 >= 0.50 * E1 tok/s
E5    >= 0.25 * E1 tok/s
E6/E7 >= 0.15 * E1 tok/s
```

若低于门禁，先 profile kernel launch / routed activation traffic / GEMM occupancy，不允许直接跑几十小时 benchmark。

---

# 15. 第二阶段性能优化

第一版通过 correctness + smoke 后再做：

1. 将 3-stage R64 + QDQ 融合为单 Triton kernel。
2. 将 Gate/Up 两个 activation transform 合并为一个 dual-output routed transform kernel：同一次 load `x`，分别乘 `D_gate/D_up` 并各自 QDQ。
3. 若 Gate/Up 两次 grouped GEMM launch 成为明显瓶颈，再设计 branch-aware dual-input W13 grouped Triton kernel；仍保持 `D_gate != D_up` exact 语义。
4. 最后再考虑真正 packed HiF4 weight + inline decode GEMM；这属于最终部署性能优化，不阻塞当前算法精度实验。

---

# 16. 推荐实现顺序

严格按以下顺序闭环：

```text
Task K1
  R64 GPU transform + HiF4 QDQ
  -> unit test
  -> E2 short-generate smoke

Task K2
  Online weight offline materialize
  + deployment-ready D sidecar/device cache
  + Dense Online Q/K/V/O runtime
  -> unit test

Task K3
  Routed Online transform/QDQ
  -> exact route-level unit test

Task K4
  Online Gate/Up/Down grouped MoE
  -> small synthetic GPU parity

Task K5
  E5 real TP2 short-generate smoke

Task K6
  E6/E7 real TP2 short-generate smoke

Task K7
  tok/s gate + profile

Task K8
  通过后才恢复原 Task14 的 E2–E7 formal eval
```

任何 Task 出现算法语义不确定，必须停止实现并确认，不允许自行改变：DIAG sharing 范围、`diag_then_rot/rot_then_diag`、R64 placement、Router BF16 语义、HiF4 QDQ rounding、Q/K/V/O 或 gate/up/down 的独立性。
