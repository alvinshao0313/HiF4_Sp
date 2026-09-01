# Qwen3-30B-A3B NVFP4→HiF4 长轨迹稳定性诊断

本目录是一个**完全隔离的诊断实验包**，用于验证 Phase-A E0–E4 中出现的现象：R64 在 ARC 类固定前缀 likelihood 任务上较好，但在长 autoregressive reasoning 上可能更容易发生轨迹分叉；DIAG 与 R64 的局部重建收益也不一定能线性叠加为端到端收益。

本实验不修改现有 E0–E7 训练、materialize、vLLM HiF4 runtime 或正式评测代码。free-run 直接复用 Phase-A 已经 materialize 的 runtime ABI 3 checkpoint/sidecar；mechanism replay 复用现有 lazy Qwen3-MoE semantic runtime，并只在本进程内把 SDPA 切换为 causal。

## 1. 要区分的三个机制

### H1：长上下文 state drift

即使强制所有方案都吃 E0 的同一条 token trajectory，E2/E4 相对 E0 的 hidden/logit 误差仍随 decode position 持续扩大。

支持证据：

- teacher-forcing logit KL 随 `0–128 -> 128–512 -> 512–2k -> 2k–8k -> 8k+` 明显上升；
- final hidden rel-L2 同样随位置上升；
- 此时 free-run 较早分叉只是长上下文漂移的外显结果。

### H2：decision-margin / trajectory bifurcation

固定 E0 trajectory 后误差并没有随长度显著恶化，但 free-run 在 E0 的低 logit margin 位置发生首次 token 分叉；一旦 token 不同，后续状态空间自然分离。

支持证据：

- free-run E2/E4 首次分叉显著早于 E1/E3；
- teacher-forcing KL 不随 decode position 单调扩大；
- 首次分叉点的 E0 top1-top2 logit margin 显著小于统一探针点；
- 因而不应描述成“R64 随长度随机丢信息”，而应描述成量化误差方向改变了 decision boundary sensitivity。

### H3：MoE router 边界放大

量化误差先让 hidden/router input 偏移，再在 expert top-k 边界处离散放大。

支持证据：

- 首次 token 分叉附近，某些 layer 的 router top-k exact rate 明显低于统一位置；
- router top-k overlap 下降或边界 margin 变小；
- 若 E4 比 E2/E3 更明显，说明组合变换虽然可降低部分 block NMSE，却可能把误差移到 routing-sensitive 方向。

## 2. 方案定义

| 名称 | 语义 |
|---|---|
| E0 | 原生 NVFP4 |
| E1 | Direct HiF4 |
| E2 | R64-only |
| E3 | adopted fusable DIAG |
| E4 | adopted fusable DIAG + R64 |

E1–E4 free-run 启动时强制读取 `hif4_runtime_spec.pt` 并检查 `runtime_abi_version == 3`。不满足直接失败，禁止误用旧 E2 runtime。

## 3. 实验两条证据链

### A. Greedy free-run：真实轨迹何时分叉

使用 MMLU-Pro，`temperature=0`，同一批 prompt，E0–E4 分别真实 autoregressive 生成。greedy 是为了让“首次 token 分叉”成为确定性诊断量；它不是替代正式 MMLU-Pro accuracy 的新评测协议。

默认：

- samples = 64；
- max_new_tokens = 16384；
- TP=2；
- KV=BF16；
- enforce_eager=true；
- E1–E4 使用当前 Phase-A ABI-3 HiF4 runtime。

记录：

- exact input token ids；
- exact generated token ids；
- raw reasoning text；
- first divergence index；
- trajectory survival@128/512/2048/8192。

### B. E0-token causal replay：固定轨迹后误差是否随长度增长

从 E0 free-run 选 16 条轨迹：一半为最长轨迹，一半覆盖其余长度分布。每条轨迹统一取位置探针，并额外插入 E1–E4 各自首次分叉点 `[-2,-1,0,+1]`。

默认 decode bins：

- `[0,128)`；
- `[128,512)`；
- `[512,2048)`；
- `[2048,8192)`；
- `[8192,+∞)`。

默认每 bin 4 个均匀探针，最大 causal replay probe index=12287。

为了避免 full 30B BF16 semantic model 驻留显存，replay 采用 48 层 lazy materialization：一次只加载一层，完整 causal prefix 在层间以 BF16 hidden cache 传递，只保存稀疏 probe position 的统计量。

逐层记录：

- hidden rel-L2 / cosine / max-abs；
- router KL；
- router top-k overlap / exact；
- router top-k 边界 margin。

最终 logits 记录：

- full-vocab KL(E0||variant)；
- JS divergence；
- centered-logit cosine；
- top1 agreement；
- E0/variant top1-top2 margin；
- E0 trajectory target-token NLL / rank。

## 4. E0 semantic parity 硬门禁

lazy causal replay 与正式 vLLM 是两个实现路径，所以不能直接假定两者逐 token 等价。

E0 free-run 主实验固定为 greedy。semantic replay 会在所有 probe position 上检查：

`semantic E0 argmax == ABI/runtime E0 实际生成 token`

默认要求整体 top1 parity >= 0.99。低于 0.99 时 `causal_replay.py` 直接失败；后续 hidden/router/logit 机制统计禁止解释，Cursor 应先报告 parity mismatch，而不是继续跑 E1–E4。

## 5. GPU 调度

`gpu_pool.py` 与现有工程约束保持一致：

- 默认 `PROJECT_GPU_POOL=0,1,2,3`；
- 未显式指定 `GPU_POOL` 时，仅选 free ratio >= 0.90 且 utilization <= 10% 的 GPU；
- free-run 每个方案占 2 GPU，默认最多并行 2 个方案；
- semantic replay 每个方案占 1 GPU，E0 必须先单独完成 parity gate，之后 E1–E4 默认最多并行 2 个。

如需手工指定卡：

```bash
export GPU_POOL=0,1,2,3
```

## 6. 标准执行顺序

先定义一个全新的结果目录，不覆盖 Phase-A：

```bash
export RUN_ROOT=Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/trajectory_stability_run01
```

### Step 0：CPU helper tests

```bash
pytest -q Native_NVFP4_HiF4_Linear_Puncture/tests/long_trajectory_stability
```

### Step 1：E0–E4 greedy free-run

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/run_free_run_matrix.py \
  --run_root "$RUN_ROOT"
```

### Step 2：对齐 trajectory、计算首次分叉、构建 probe plan

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/prepare_analysis.py \
  --run_root "$RUN_ROOT"
```

### Step 3：E0 causal reference + parity gate，再并行 E1–E4 replay

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/run_semantic_matrix.py \
  --run_root "$RUN_ROOT"
```

### Step 4：自动汇总

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/summarize.py \
  --run_root "$RUN_ROOT"
```

所有 runner 默认支持断点续跑：目标结果文件已存在时跳过；只有明确传 `--force` 才重跑。

## 7. Smoke

正式长序列任务成本较高，Cursor 第一次执行必须先用小规模参数验证闭环：

```bash
python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/run_free_run_matrix.py \
  --run_root "${RUN_ROOT}_smoke" \
  --max_samples 4 \
  --max_new_tokens 1024 \
  --max_parallel 1

python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/prepare_analysis.py \
  --run_root "${RUN_ROOT}_smoke" \
  --num_samples 2 \
  --probes_per_bin 2 \
  --max_decode_index 511

python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/run_semantic_matrix.py \
  --run_root "${RUN_ROOT}_smoke" \
  --max_parallel 1

python Native_NVFP4_HiF4_Linear_Puncture/experiments/long_trajectory_stability/summarize.py \
  --run_root "${RUN_ROOT}_smoke"
```

Smoke 只验语义闭环，不用于机制结论。

## 8. 输出结构

```text
RUN_ROOT/
  free_run/
    E0..E4/
      capture.log
      capture_manifest.json
      <lighteval raw results/details>
  normalized/
    E0.jsonl ... E4.jsonl
    E0.meta.json ... E4.meta.json
  analysis/
    divergence_events.jsonl
    free_run_summary.json
    probe_plan.json
    long_trajectory_summary.json
    LONG_TRAJECTORY_STABILITY_REPORT.md
  semantic/
    E0/
      e0_semantic_parity.json
      causal_replay.log
      reference/*.pt
    E1..E4/
      causal_replay.log
      semantic_metrics.jsonl
      semantic_metrics.meta.json
```

## 9. 结果解释优先级

1. 先看 E0 semantic parity；不过门禁则停止。
2. 再看 free-run 首次分叉 survival curve；确定 E2/E4 是否真的更早分叉。
3. 再看固定 E0 trajectory 的 logit KL 是否随 decode bin 增长。
4. 再比较首次分叉点与统一 probe 的 E0 logit margin。
5. 最后定位 router mismatch enrichment 的 layer。

不得从单独一个指标下结论；尤其不得把 R64 描述为“本身随机丢失信息”。R64 在无量化下是正交等价变换，真正需要解释的是 `Q(XR)` / `Q(WR)` 或 `Q(XDR)` / `Q(WD^{-1}R)` 后的量化误差方向与长轨迹 decision boundary 的耦合。
