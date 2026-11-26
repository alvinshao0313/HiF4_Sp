> 状态：已被 2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md 取代。
> 仅用于历史问题复盘，不再执行。

# E0 vLLM TP2 ↔ semantic NVFP4 parity 定位计划

> **状态：已停止执行。** semantic parity 不再作为正式长轨迹框架门禁；新的唯一主路径见 `2026-09-01-qwen3-30b-hif4-long-trajectory-real-vllm-hook-replan-cn.md`。本文仅用于历史问题复盘。

> 日期：2026-08-31  
> 前置 smoke：`results/long_trajectory_stability/trajectory_stability_smoke`  
> 当前门禁：E0 semantic top1 parity = 24/28 = 0.8571 < 0.99  
> 本计划目标：只定位 parity 来源，不启动 formal，不解释 H1/H2/H3，不修改生产 runtime。

---

# 1. 本轮拍板结论

**正式 E0 reference 继续保持 vLLM TP=2 native NVFP4 emulation。不要把单卡 E0 替换成正式 reference。**

原因：E1–E4 当前正式 runtime 都是 TP=2。若只把 E0 reference 换成单卡，会人为引入 E0/variant backend asymmetry，可能把 TP 数值路径误当成量化算法差异。

下一步先做三方诊断：

- A：vLLM E0 TP=2，现有 smoke 真值；
- B：vLLM E0 TP=1，同 prompt / greedy / KV BF16 / emulation；
- C：single-GPU `NativeQwen3MoELayerRuntime` semantic E0。

先回答：当前 4 个 low-margin parity mismatch 是不是主要由 **TP/GEMM/reduction 数值路径**造成。

---

# 2. 重要代码事实：不要先假定 max-scale 写错

当前 semantic loader 已经实现与 vLLM emulation 一致的 MoE activation global-scale collapse：

```text
semantic:
  a13 = max(all experts gate/up input_scale)
  a2  = max(all experts down input_scale)
  qdq global_scale = 1 / a13, 1 / a2

vLLM EMULATION:
  a13_scale = 1.0 / a13_scale.max()
  a2_scale  = 1.0 / a2_scale.max()
```

而 `qdq_native_nvfp4()` 直接调用 vLLM 自己的 `ref_nvfp4_quant_dequant()`。

因此当前**没有证据**证明 MoE `w13/w2 max-scale` 本身实现错误。不能直接修改这部分。

可能差异仍包括：

1. TP=2 partition + all-reduce / reduction 顺序；
2. Triton fused-MoE GEMM 与 `F.linear` 的 BF16 accumulation/rounding；
3. packed-weight on-the-fly dequant GEMM 与 semantic 先 dequant BF16 weight 再 GEMM；
4. dense QKV/O 的 TP column/row-parallel 数值路径；
5. SDPA/attention kernel 的小数值差。

本计划要按成本从低到高排除。

---

# 3. Task 1：只增加一个 E0 TP=1 free-run control

不改 `run_free_run_matrix.py` 的正式 TP2 默认。

使用现有 `launch_capture.py` 单独跑：

```bash
export GPU_POOL=0
export TP1_ROOT=Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_smoke_e0_tp1_control

python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/launch_capture.py \
  --variant E0 \
  --output_dir "$TP1_ROOT/free_run/E0" \
  --profile greedy \
  --max_samples 4 \
  --max_new_tokens 1024 \
  --tensor_parallel_size 1

python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/normalize_capture.py \
  --capture_dir "$TP1_ROOT/free_run/E0" \
  --output "$TP1_ROOT/E0_tp1.jsonl"
```

要求：

- 仍使用 native NVFP4 emulation；
- KV BF16；
- greedy；
- thinking 开启；
- 不改 chat template；
- exact prompt token IDs 必须与原 TP2 E0 四条逐 token 相同。

如果单卡 vLLM 因显存或 backend 限制不能启动，停止并报告，不改其它参数绕过。

---

# 4. Task 2：做 TP1 ↔ TP2 token 轨迹逐点对比

Cursor 在诊断目录新增一个很小的分析脚本即可，例如：

```text
experiments/long_trajectory_stability/compare_e0_tp_control.py
```

输入：

```text
原 smoke normalized/E0.jsonl        # TP2
TP1_ROOT/E0_tp1.jsonl               # TP1
原 semantic/E0/e0_semantic_parity.json
原 analysis/probe_plan.json
```

输出：

```text
TP1_ROOT/E0_TP1_TP2_SEMANTIC_TRIAGE.json
TP1_ROOT/E0_TP1_TP2_SEMANTIC_TRIAGE.md
```

对每条 prompt 记录：

- TP1 vs TP2 first divergence；
- 在 28 个 semantic probe position 上：
  - TP2 token；
  - TP1 token；
  - semantic argmax（parity 文件中 mismatch 点必须读出；match 点可标记 same-as-TP2）；
- 特别标记原 4 个 mismatch：decode 10/12/24/35。

至少输出如下表：

| prompt | decode | TP2 token | TP1 token | semantic top1 | semantic target rank | semantic margin |
|---|---:|---|---|---|---:|---:|

---

# 5. Task 3：严格按下面决策树停止/继续

## Case A：TP1 在原 4 个 mismatch 点全部或几乎全部等于 semantic top1

例如：

```text
TP2: the
TP1: this
semantic: this
```

并且 TP1↔TP2 的 first divergence 与这些 low-margin 点高度重合。

判定：

> **PRIMARY_TP_NUMERIC_PATH_DIFFERENCE**

这时：

1. 不修改 NVFP4 max-scale；
2. 不修改 checkpoint loader；
3. 不降低 0.99 gate；
4. 不把 TP1 E0 替换正式 TP2 reference；
5. 下一步实现一个**仅诊断目录使用的 TP2 semantic adapter**，目标是在 lazy replay 中模拟 vLLM TP2 的 partition/reduction 数值语义。

在完成 TP2 semantic adapter 之前 formal 仍不启动。

## Case B：TP1 与 TP2 一致，但 semantic 仍在原 4 个位置翻 top1

判定：

> **SEMANTIC_RUNTIME_NUMERIC_MISMATCH**

下一步进入逐算子 parity，见 Task 4。

## Case C：TP1、TP2、semantic 三者在不同点互相分叉

判定：

> **MIXED_NUMERIC_PATH_DIFFERENCE**

先做 Task 4，不能直接归因 TP。

## Case D：TP1 无法运行

判定：

> **TP1_CONTROL_UNAVAILABLE**

停止并汇报实际报错。不要因此直接改 gate 或启动 formal。

---

# 6. Task 4：仅 Case B/C 执行逐算子 NVFP4 parity

代码仍只能新增在：

```text
experiments/long_trajectory_stability/nvfp4_operator_parity/
```

禁止修改生产代码。

目标不是先比较完整 logits，而是找到**第一个产生不可忽略数值差的算子**。

## 4.1 Activation QDQ parity

在同一 tensor `x`、同一 scalar global scale 上比较：

```text
semantic qdq_native_nvfp4(x)
vs
vLLM ref_nvfp4_quant_dequant(x)
```

理论上当前两者调用的是同一实现，应 bitwise/近 bitwise 一致。

记录：

- max_abs；
- rel_L2；
- exact element fraction。

若不一致，检查实际调用路径是否一边进 Triton、一边进 Python fallback。

## 4.2 Dense packed-linear parity

对 q/k/v/o 分别使用**同一输入**，比较：

A. semantic：

```text
ref_nvfp4_quant_dequant
+ dequantize_to_dtype(weight)
+ F.linear
```

B. vLLM emulation primitive：

```text
run_nvfp4_emulations
```

先在 TP=1 语义下比较，不掺 TP partition。

输出每层/投影：

- output max_abs；
- rel_L2；
- cosine；
- top outlier channels。

优先检查 layer 0、1、以及能复现 low-margin mismatch 的前几层。

## 4.3 MoE W13/W2 parity

必须使用相同：

- routed token subset；
- selected expert ids；
- routing weights；
- packed weights/scales；
- `a13/a2` global scale。

分别比较：

```text
W13 output before SiLU*up
post activation hidden
W2 output before routed-weight accumulation
final MoE output
```

重点核对：

- semantic `a13/a2` 与 vLLM prepare 后的数值是否完全相同；
- `w13_global` 对 gate/up 的共享方式；
- packed on-the-fly dequant vs pre-dequant BF16 weight；
- fused kernel accumulation dtype。

**只有实际 tensor 数值证明 scale 不同，才允许提出 scale bug。**

## 4.4 TP partition numerical control

若 TP1 operator parity 很好、TP2 仍不同，则用同一 tensor 在单机显式模拟：

- column-parallel：按 vLLM shard 切输出维，分别 GEMM 后 concat；
- row-parallel：按输入维切分，分别 GEMM 后按 vLLM dtype/order 求和；
- MoE 对应 TP shard/reduction。

比较：

```text
full F.linear
vs
manual TP2 partition/reduction
```

若差异已经达到 low-margin token reversal 所需量级，说明无需修改 QDQ 定义。

---

# 7. 下一阶段 TP2 semantic adapter 的触发条件

只有 Case A，或 Task 4 明确证明主要差异来自 TP partition/reduction 后，才设计：

```text
experiments/long_trajectory_stability/tp2_semantic_adapter/
```

目标：

- 不修改 vLLM；
- 不修改 e2e training semantic runtime；
- 只为长轨迹诊断复现 TP2 数值语义；
- E0 parity 重新执行原来的 28 probes；
- 仍使用原门槛 `>=0.99`。

adapter 过门后，再恢复原长轨迹 smoke：E1–E4 semantic → summarize；smoke 全闭环后才 formal。

---

# 8. 明确禁止的捷径

本轮不允许：

1. 将 parity threshold 从 0.99 降低；
2. 因为 4 个 target 都 rank2 就直接忽略 mismatch；
3. 删除 divergence-neighborhood probes，只保留 uniform probes；
4. 把单卡 E0 当成正式 TP2 E0 reference；
5. 直接修改 `modelopt_moe_checkpoint.py` 的 max-scale；
6. 修改 vLLM NVFP4 emulation kernel；
7. 启动 formal；
8. 解释 H1/H2/H3。

---

# 9. Cursor 本轮最终只需要交付

生成：

```text
results/long_trajectory_stability/trajectory_stability_smoke_e0_tp1_control/E0_TP1_TP2_SEMANTIC_TRIAGE.md
```

固定回答：

1. TP1 是否成功使用同样的 4 个 prompt；
2. TP1↔TP2 first-divergence 在哪里；
3. 原 4 个 semantic mismatch 点上，TP1 token 更接近 TP2 还是 semantic；
4. Case A/B/C/D 属于哪一个；
5. 是否需要进入 operator parity；
6. 本轮未修改任何 production runtime。

做到这里停止，等待下一步。