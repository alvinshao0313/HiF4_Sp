# Qwen3-8B 原生 NVFP4 → HiF4 逐层 DIAG/R64 端到端重建 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4` 上实现一套可复现的、逐 Transformer Block 联合优化 DIAG 的 NVFP4→HiF4 转换流程，支持 `fusable` / `online` 两类 DIAG、可选固定 R64、reasoning-aware calibration、逐层 progressive reconstruction、validation rollback，以及端到端 reasoning benchmark 对比。

**Architecture:** 复用现有 `NativeNVFP4SemanticLinear` 作为原生 NVFP4 teacher 数值语义，在新实验目录中升级为可切换 `native_nvfp4 / hif4_train_ste / hif4_eval` 的语义 Linear；每次只训练一个 Transformer Block 内的 DIAG，Teacher 和 Student 使用完全相同的 progressive HiF4 输入。权重始终从固定 NVFP4 反量化 master weight 出发，根据当前 DIAG/R64 实时做等价变换后重新执行 HiF4 QDQ；训练完成后使用 held-out validation 选择 best epoch，并强制执行整层 rollback gate。

**Tech Stack:** Python 3 / PyTorch / Transformers Qwen3 / Accelerate / HuggingFace Datasets / 当前仓库 HiF4 fake-QDQ oracle / `lm_eval` / 仓库内 `3rdparty/lighteval`。

## Global Constraints

- 所有 Python、pytest、训练命令必须在 `hif4` conda 环境执行。
- 第一阶段只实现 fake HiF4 QDQ 数值正确性与 E2E 精度验证；不要在本计划中同时实现 packed HiF4 kernel/vLLM 真量化部署。
- 目标模型固定为 `ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4`，36 个 decoder layers。
- 量化覆盖范围严格与原生 NVFP4 checkpoint 一致：每层 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` 共 7 个 Linear，总计 252 个；embedding、final norm、lm_head 不新增 HiF4 量化。
- 原生 checkpoint 已有每个 Linear 的在线 H16 (`forward_hadamard_matrix`)；native H16 永远保留。本计划中的 `rot` 只指新增固定 R64。
- R64 固定为现有 `diag_gradient/r64_transform.py` 中 `R4⊗R4⊗R4` 的 64×64 正交矩阵，不学习旋转参数。
- `fusable` 只允许 `DIAG -> native H16 -> R64`；禁止 `fusable + rot_then_diag`。
- `online` 允许 `native H16 -> DIAG -> R64` 与 `native H16 -> R64 -> DIAG`。
- `fusable` 每层四组 DIAG：`D_QKV`、`D_VO`、`D_GU`、`D_UD`；`online` 每层 7 个 Linear 各自独立 DIAG。
- 主训练粒度固定 `layer_joint`；`linear_independent` 只作为 `online` 消融，`fusable + linear_independent` 直接报错。
- Teacher 固定为原生 NVFP4 block；不使用 BF16/dequantized block 作为主 teacher。
- 第 l 层 Teacher 与 Student 必须吃同一个 progressive HiF4 输入 `X_l^H`；只传播最终 accepted Student 输出。
- 主 loss 默认 `block_delta_nmse`；attention/MLP auxiliary loss 支持但默认权重均为 0。
- DIAG 使用 `z=log2(d)`、`d=2^z`、`z=0` 初始化；默认单 CLI `--diag_log2_clamp=-4,4`。
- Adam 固定；默认 `lr=5e-3`、20 epochs、cosine scheduler；每层 scheduler 独立重启。
- Calibration 默认 `s1k_teacher_cot`；同时支持 `s1k_original`、`s1k_question`、`wikitext2`、`c4`。
- `s1k_teacher_cot` 默认保留 teacher 答错样本；支持同题重生成直到正确、或换题直到正确。
- Teacher CoT `max_new_tokens=32768`；实际缓存按真实生成长度。
- S1K 只做 batch 内动态 padding；padding 不参与 reconstruction loss。
- 不实现 calibration 与下游评测集之间的题目污染/去重门禁。
- Train/val calibration 默认 128/32，seed=42；同一 run 所有层使用同一批固定样本。
- 每层训练后若 best validation 不严格优于该结构的 `D=I` baseline，则整层 DIAG 回滚为 identity；开启 R64 时 rollback 后仍保留 R64。
- 不做单个 DIAG 局部 rollback。
- 任意 NaN/Inf、64 分组维度错误、未知 judge 输出、数学等价性失败、fold 一致性失败均直接 fail；禁止 silent fallback。
- 新实现必须以长期维护为目标：算法数学、数据构建、训练循环、评测逻辑、CLI/脚本分层，禁止把所有逻辑堆进一个 `train.py` 或 shell 脚本。
- 依赖方向固定：`core` 不依赖 `data/training/evaluation`；`data` 只依赖 `core`；`training` 可依赖 `core/data`；`evaluation` 只依赖 `core` 和 artifact；CLI 与 shell scripts 只做参数编排和调用，不承载算法公式。
- DIAG/R64 的矩阵顺序、weight transform、HiF4 runtime 语义必须只有一个实现源；fold、训练、评测统一复用，禁止复制三套近似相同公式。
- 配置默认值只能定义一次；shell preset 只覆盖 CLI，不再复制 Python 默认配置逻辑。
- 后续新增 calibration source、loss、评测任务或 transform 时，应优先扩展现有明确入口，而不是新增第二套 pipeline；同时禁止为了“以后可能用”预先搭建复杂 registry/plugin/factory 框架。
- 不新增文件哈希、tensor hash、SHA/checksum、rotation hash 或完整性校验框架；现有 `rotation_sha256` 不用于本端到端实现。
- 不堆叠重复 defensive code：不为同一 invariant 在多层重复校验，不用大段 try/except 重包 PyTorch/Transformers 已经清楚的错误，不增加 silent fallback。只保留会影响实验语义的必要检查：非法 CLI 组合、关键 shape/G64 约束、训练非有限值、rollback、fold 一致性。
- 数学等价性的大部分断言放到单元测试；正式训练热路径不每 step 重复执行等价性检查。fold gate 只在每层最终状态确认时执行一次。
- artifact/manifest 只保存恢复与复现实验所需的轻量 metadata 和参数，不保存 checksum，不做额外完整性扫描。
- 尽量只新增 `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/` 与对应测试；不要改变现有历史实验语义。

---

## 1. 数学定义与四组 fusable DIAG

### 1.1 原生 Teacher

原生 NVFP4 Linear：

\[
X_N = X H_{16},\qquad A_N=Q_{NVFP4}(X_N),\qquad Y_N=A_NW_N^T+b.
\]

`W_N` 是 packed NVFP4 权重反量化得到的固定 master weight。Student 每一步都必须从 `W_N` 重新生成 transformed weight，禁止从上一步已经 QDQ 的 HiF4 权重继续 fold。

### 1.2 Online 公式

`diag_then_rot`：

\[
X'=(XH)DR,\qquad W'=W_ND^{-1}R.
\]

`rot_then_diag`：

\[
X'=(XH)RD,\qquad W'=W_NR D^{-1}.
\]

无 R64 时令 `R=I`。

### 1.3 Fusable 四组

| DIAG | 维度 | 作用位置 | 最终融合 |
|---|---:|---|---|
| `D_QKV` | `hidden_size` | q/k/v native H16 前 | `input_layernorm.weight` + q/k/v 权重逆变换 |
| `D_VO` | `num_kv_heads*head_dim` | V 输出通道 | V 输出行缩放 + O 输入侧逆变换 |
| `D_GU` | `hidden_size` | gate/up native H16 前 | `post_attention_layernorm.weight` + gate/up 权重逆变换 |
| `D_UD` | `intermediate_size` | Up 输出通道 | Up 输出行缩放 + Down 输入侧逆变换 |

GQA 的 `D_VO` 只在 KV-head 空间学习。若 `r=num_attention_heads/num_key_value_heads`，每个 KV head 的 `head_dim` scale block 连续重复 r 次，形成 O 输入侧 `D_VO_expanded`。禁止复制后再给每个 Q head 学独立 scale。

通用 fusable 输入/输出等价变换，开启 R64 时：

\[
X'=(XD_{in})HR,
\]

\[
W'=D_{out}W_NH^TD_{in}^{-1}HR.
\]

于是无量化时：

\[
X'(W')^T=(XHW_N^T)D_{out}.
\]

具体：

- q/k：`D_in=D_QKV`, `D_out=I`。
- v：`D_in=D_QKV`, `D_out=D_VO`。
- gate：`D_in=D_GU`, `D_out=I`。
- up：`D_in=D_GU`, `D_out=D_UD`。
- o：输入已由 V/attention 带上 `D_VO_expanded`，wrapper 不重复乘 D；weight transform 使用该 D 的 inverse。
- down：输入已由 Up 分支带上 `D_UD`，wrapper 不重复乘 D；weight transform 使用 `D_UD` inverse。

MLP：

\[
\mathrm{SiLU}(g)\odot(uD_{UD})=[\mathrm{SiLU}(g)\odot u]D_{UD}.
\]

禁止在 SiLU 前对 gate 输出增加独立 `D_g`。

### 1.4 R64 位置

- q/k/v、gate/up：沿 hidden K 维连续 G64。
- down：沿 intermediate K 维连续 G64。
- o：沿 attention-output K 维，但必须断言 `head_dim % 64 == 0`，禁止 G64 跨 attention head。

---

## 2. Layer reconstruction 与 rollback

第 l 层输入：`X_l^H`。

\[
Y_l^N=F_l^{NVFP4}(X_l^H),\qquad Y_l^H=F_l^{HiF4}(X_l^H;D,R64).
\]

\[
\Delta_l^N=Y_l^N-X_l^H,\qquad \Delta_l^H=Y_l^H-X_l^H.
\]

默认：

\[
L_{block}=\frac{\sum M\|\Delta_l^H-\Delta_l^N\|_2^2}{\sum M\|\Delta_l^N\|_2^2+\epsilon}.
\]

支持：`block_delta_nmse`、`block_output_nmse`、`mse`。

可选 auxiliary：

\[
L=L_{block}+\lambda_A L_{attn}+\lambda_M L_{mlp},
\]

默认两权重均 0。若启用，只对齐 `self_attn` branch 输出和 `mlp` branch 输出；不默认增加 q/k/v 或 gate/up 局部 loss。

Validation baseline 是同一结构下 `D=I`：

- R64 off：Direct HiF4。
- R64 on：R64-only HiF4。

最终：

\[
D_l^{final}=D_l^{best}\quad\text{iff}\quad L_{best}<L_{identity},
\]

否则整层回 identity。只传播最终 accepted 状态生成 `X_{l+1}^H`。

---

## 3. Calibration 数据协议

`--calib_source`：

```text
{s1k_teacher_cot,s1k_original,s1k_question,wikitext2,c4}
```

| source | 输入 | reconstruction loss token |
|---|---|---|
| `s1k_teacher_cot` | S1K question prompt + teacher assistant continuation | 只算 generated continuation；question/prefill mask=0 |
| `s1k_original` | S1K 原始完整 `text` | 全部非 padding token |
| `s1k_question` | S1K question 输入 | 全部问题 token |
| `wikitext2` | 固定 window | 全部 window token |
| `c4` | 固定 window | 全部 window token |

Teacher CoT 固定 generation：`enable_thinking=True, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0`。开放：

```text
--teacher_max_new_tokens 32768
--teacher_trace_policy {all,regenerate_correct,replace_question_correct}
--teacher_max_attempts 4
```

Judge 使用同一个原生 NVFP4 teacher，输入 Question + Reference solution + Candidate response，`do_sample=False, max_new_tokens=8`，只接受 `CORRECT` / `INCORRECT`。

- `all`：错误与 truncated 均保留，但记录 metadata。
- `regenerate_correct`：同题最多总共 4 次；必须正常结束且 judge correct；失败直接报错，不换题。
- `replace_question_correct`：每候选题只生成一次，错则用 seeded 未使用题继续补；池耗尽报错。

S1K 不截断，只做 batch 内最长动态 padding。`--calib_seqlen 1024` 只对 Wiki/C4 生效。

---

## 4. 最终 CLI 与目录

统一训练入口：

主方案依赖 `core/config.py` 的唯一默认值，因此正式脚本只覆盖与默认不同的实验差异。例如默认就是 `fusable + layer_joint + s1k_teacher_cot + all + 20 epochs`，主方案只需显式开启 R64：

```bash
conda run -n hif4 python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli.train \
  --output_dir Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/[RUN_ID] \
  --use_r64
```

调试/消融脚本只追加它真正改变的参数；不要把所有默认值复制到每个 shell 文件。每次实际解析后的完整参数仍写入 `config.json`，所以不会损失可复现性。

CLI 精确定义：

```text
--model_path STR=ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4
--output_dir PATH                         # 必填，不自动生成隐式目录
--diag_mode {fusable,online}=fusable
--use_r64                                # 默认 false；主方案脚本显式开启
--rot_order {diag_then_rot,rot_then_diag}=diag_then_rot
--diag_train_scope {layer_joint,linear_independent}=layer_joint
--recon_loss {block_delta_nmse,block_output_nmse,mse}=block_delta_nmse
--attn_aux_loss_weight FLOAT=0
--mlp_aux_loss_weight FLOAT=0
--diag_lr FLOAT=5e-3
--diag_epochs INT=20
--diag_scheduler {cosine,constant}=cosine
--diag_log2_clamp {none|MIN,MAX}=-4,4
--calib_source {s1k_teacher_cot,s1k_original,s1k_question,wikitext2,c4}=s1k_teacher_cot
--calib_nsamples INT=128
--calib_val_nsamples INT=32
--calib_seed INT=42
--calib_seqlen INT=1024
--diag_batch_size INT=4
--teacher_trace_policy {all,regenerate_correct,replace_question_correct}=all
--teacher_max_attempts INT=4
--teacher_max_new_tokens INT=32768
--start_layer INT=0
--end_layer INT=35
```

强制：`fusable+rot_then_diag` 报错；`fusable+linear_independent` 报错；layer 范围必须合法；clamp 只能 `none` 或两个 finite 且 `MIN<MAX` 的数；所有 DIAG 共用同一个 clamp；S1K 不使用 `calib_seqlen` 截断；`start_layer>0` 要求同一 output_dir 前面所有 layer artifact 已完整存在，否则报错。

新增代码目录必须按职责分类，不允许全部平铺：

```text
Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/
  __init__.py
  core/
    __init__.py
    config.py              # dataclass/CLI defaults/合法组合
    transforms.py          # DIAG/R64/GQA 的唯一数学实现
    semantic_hif4.py       # native/HiF4 switchable Linear runtime
    fold.py                # fusable fold 与 final runtime 固化
    artifact.py            # compact overlay 读写；仅轻量 metadata
  data/
    __init__.py
    calibration.py         # 五种 calibration source
    teacher_traces.py      # teacher CoT + self-judge
    batching.py            # dynamic padding + length bucketing
  training/
    __init__.py
    layer_runtime.py       # Qwen3 单层重放 + progressive cache
    losses.py              # block/aux masked loss，统一数值定义
    trainer.py             # layer_joint / linear_independent 优化流程
  evaluation/
    __init__.py
    runner.py              # ARC/MMLU-Pro/AIME in-memory semantic eval
    summarize.py           # 汇总绝对分数与 vs NVFP4 drop
  cli/
    __init__.py
    train.py               # 只负责 orchestration
    evaluate.py            # 只解析评测 CLI 并调用 evaluation.runner
    summarize.py           # 只解析汇总 CLI
  scripts/
    common.sh              # 仅公共路径/启动函数，不放算法逻辑
    smoke/
      run_train_smoke.sh
      run_teacher_cot_smoke.sh
    train/
      run_full.sh
    ablation/
      run_phase1_matrix.sh
      run_calib_ablation.sh
      run_teacher_policy_ablation.sh
    eval/
      run_fast_eval.sh     # ARC-E/C + MMLU-Pro 300
      run_final_eval.sh    # finalist + AIME25 avg@5
  README.md

Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/
  test_config.py
  test_transforms.py
  test_semantic_hif4.py
  test_calibration.py
  test_layer_runtime.py
  test_trainer.py
  test_fold_artifact.py
```

测试也按该专项单独归档；不要继续向 `Native_NVFP4_HiF4_Linear_Puncture/tests/` 根目录堆新的 `test_e2e_*` 文件。

维护性约束：

- shell 脚本只组织 preset 和调用 Python CLI；禁止在 shell 中拼算法分支或处理模型 tensor。
- `core/transforms.py` 是 DIAG/R64 矩阵语义唯一 source of truth；`core/fold.py` 和 `core/semantic_hif4.py` 必须调用它，不重新手写公式。
- `training/losses.py` 是 reconstruction loss 唯一 source of truth；trainer 和 validation 都调用同一套 numerator/denominator 函数。
- `data/batching.py` 独立后，未来切 token-budget batching 时不改 calibration source 实现。
- `evaluation/runner.py` 只负责把已经构建好的 semantic model 接到 benchmark；模型转换逻辑仍由 `core` 提供。
- `cli/*.py` 保持薄：参数解析、加载配置、调用 domain 模块、保存结果；单个 CLI 文件不要承载核心公式。
- 各子包 `__init__.py` 只做包声明，默认保持空文件；禁止通过大量 re-export 制造隐式 API。
- 扩展点保持简单显式：新增 calibration source 时只在 `data/calibration.py` 增加 builder + 一个 dispatch 条目；新增 loss 时只在 `training/losses.py` 增加函数 + 一个 dispatch 条目；新增固定 transform 时只扩展 `core/transforms.py` 与配置 enum；新增 benchmark 时只扩展 `evaluation/runner.py` 和对应 eval preset。不要为这些扩展点额外引入插件系统。

结果：

```text
results/e2e_diag_reconstruction/[RUN_ID]/
  config.json
  calibration/train.pt
  calibration/val.pt
  calibration/train_manifest.json
  calibration/val_manifest.json
  calibration/teacher_traces/{train,val}.jsonl
  layers/layer_XX/{best_diag.pt,metrics.json,train_log.jsonl}
  checkpoint/final_model/{conversion_state.pt,manifest.json}
  eval/{arc,mmlu_pro,aime25}/
  summary.json
```

`conversion_state.pt` 是 compact overlay；不要为每个实验复制完整 8B BF16 checkpoint。

---

### Task 1: 配置 dataclass、CLI 默认值与必要参数校验

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/__init__.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/__init__.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/config.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py`

**Interfaces:**
- Produces: `parse_log2_clamp(text: str) -> tuple[float, float] | None`
- Produces: `E2ETrainConfig` dataclass
- Produces: `build_train_parser() -> argparse.ArgumentParser`
- Produces: `validate_train_config(cfg: E2ETrainConfig) -> None`

- [ ] **Step 1: 先写参数解析失败测试**

```python
import pytest
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
    E2ETrainConfig,
    parse_log2_clamp,
    validate_train_config,
)


def test_parse_log2_clamp():
    assert parse_log2_clamp("none") is None
    assert parse_log2_clamp("-4,4") == (-4.0, 4.0)
    assert parse_log2_clamp("-3,2") == (-3.0, 2.0)
    with pytest.raises(ValueError):
        parse_log2_clamp("-4")
    with pytest.raises(ValueError):
        parse_log2_clamp("4,-4")


def test_fusable_rejects_rot_then_diag():
    cfg = E2ETrainConfig.for_test(diag_mode="fusable", rot_order="rot_then_diag")
    with pytest.raises(ValueError, match="fusable"):
        validate_train_config(cfg)


def test_fusable_rejects_linear_independent():
    cfg = E2ETrainConfig.for_test(diag_mode="fusable", diag_train_scope="linear_independent")
    with pytest.raises(ValueError, match="linear_independent"):
        validate_train_config(cfg)
```

- [ ] **Step 2: 验证测试当前失败**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py -q
```

Expected: FAIL，因为新模块尚不存在。

- [ ] **Step 3: 实现唯一 CLI source of truth**

`E2ETrainConfig` 包含第 4 节全部 CLI 字段，`for_test()` 只用于测试构造默认值。不要在 `train.py` 复制默认值。

`parse_log2_clamp()` 固定实现：

```python
def parse_log2_clamp(text: str) -> tuple[float, float] | None:
    if text.strip().lower() == "none":
        return None
    parts = [x.strip() for x in text.split(",")]
    if len(parts) != 2:
        raise ValueError("--diag_log2_clamp must be 'none' or 'MIN,MAX'")
    lo, hi = map(float, parts)
    if not math.isfinite(lo) or not math.isfinite(hi) or not lo < hi:
        raise ValueError("--diag_log2_clamp requires finite MIN < MAX")
    return lo, hi
```

校验严格按本计划执行，不自动修复非法组合。

- [ ] **Step 4: 覆盖默认值与非法值测试**

至少覆盖默认 `5e-3/20/cosine/-4,4/128/32/42/batch4/32768`、非法 layer range、非正 sample、非正 batch、window dataset `seqlen<=0`。

- [ ] **Step 5: 运行测试**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py -q
```

Expected: PASS。

---

### Task 2: 固定 R64、online/fusable 数学变换与 GQA scale 扩展

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/transforms.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_transforms.py`
- Reuse: `Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/r64_transform.py`
- Reuse: `Native_NVFP4_HiF4_Linear_Puncture/src/rotation.py`

**Interfaces:**
- Produces: `apply_block_right_fp32(x, matrix, group_size)`
- Produces: `apply_r64(x)`
- Produces: `expand_vo_scale(d_vo, num_attention_heads, num_key_value_heads, head_dim)`
- Produces: `online_weight_transform(w_n, d, use_r64, rot_order)`
- Produces: `fusable_weight_transform(w_n, h16, d_in, d_out, use_r64)`

- [ ] **Step 1: 先写 online 无量化等价性测试**

FP32 随机 tensor，分别验证 `diag_then_rot` / `rot_then_diag`：

```python
x_native = apply_block_right_fp32(x, h16, 16)
y0 = x_native @ w.T
x_new = apply_r64(x_native * d)
w_new = online_weight_transform(w, d, use_r64=True, rot_order="diag_then_rot")
y1 = x_new @ w_new.T
assert relative_l2(y1, y0) < 1e-6
```

- [ ] **Step 2: 写 fusable generic 公式测试**

```python
y_expected = (apply_block_right_fp32(x, h16, 16) @ w.T) * d_out
x_new = apply_r64(apply_block_right_fp32(x * d_in, h16, 16))
w_new = fusable_weight_transform(w, h16, d_in, d_out, use_r64=True)
y_actual = x_new @ w_new.T
assert relative_l2(y_actual, y_expected) < 1e-6
```

- [ ] **Step 3: 写 GQA `D_VO` 扩展测试**

例如 `num_q_heads=4, num_kv_heads=2, head_dim=64`：

```python
d = torch.arange(128, dtype=torch.float32)
expanded = expand_vo_scale(d, 4, 2, 64)
assert expanded.shape == (256,)
assert torch.equal(expanded[:64], d[:64])
assert torch.equal(expanded[64:128], d[:64])
assert torch.equal(expanded[128:192], d[64:128])
assert torch.equal(expanded[192:256], d[64:128])
```

`num_q_heads % num_kv_heads != 0` 或 `head_dim % 64 != 0` 直接报错。

- [ ] **Step 4: 实现 weight transform，严格保持矩阵顺序**

`fusable_weight_transform`：

```text
W_N
 -> right multiply H^T block16
 -> column divide D_in
 -> right multiply H block16
 -> optional R64 block64
 -> optional row multiply D_out
```

禁止假设 checkpoint H16 对称而省掉转置。

`online_weight_transform`：

```text
diag_then_rot: W_N -> column divide D -> optional R64
rot_then_diag: W_N -> optional R64 -> column divide D
```

- [ ] **Step 5: 运行测试**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_transforms.py -q
```

Expected: PASS，所有数学 gate `<1e-6`。

---

### Task 3: 可切换 native/HiF4 的 252 Linear semantic runtime

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/semantic_hif4.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_semantic_hif4.py`
- Reuse: `Native_NVFP4_HiF4_Linear_Puncture/src/semantic_model.py`
- Reuse: `Native_NVFP4_HiF4_Linear_Puncture/src/formats.py`
- Reuse: `Native_NVFP4_HiF4_Linear_Puncture/src/native_nvfp4.py`

**Interfaces:**
- Produces: `LayerDiagState(nn.Module)`
- Produces: `SwitchableNVHiF4Linear(nn.Module)`
- Produces: `upgrade_semantic_model_inplace(model, cfg)`
- Produces: `set_layer_runtime_mode(layer, mode)`，mode=`native_nvfp4|hif4_train_ste|hif4_eval`
- Produces: `snapshot_layer_diag(layer_idx)` / `load_layer_diag_snapshot(...)`

- [ ] **Step 1: 写 `LayerDiagState` shape 测试**

Fusable：`z_qkv[hidden_size]`、`z_vo[num_kv_heads*head_dim]`、`z_gu[hidden_size]`、`z_ud[intermediate_size]`。Online：7 个 projection 的 z 长度等于各自 K 维。全部 z 初始化 0、FP32。

- [ ] **Step 2: 实现并测试 HiF4 STE**

```python
def qdq_hif4_ste_bf16(x: torch.Tensor) -> torch.Tensor:
    xf = x.to(torch.float32)
    with torch.no_grad():
        recon = qdq_hif4_direct(
            xf.detach(), output_dtype=torch.bfloat16
        ).to(torch.float32)
    y = xf + (recon - xf).detach()
    return y.to(torch.bfloat16)
```

Forward 数值必须等于 real HiF4 reconstruction；反向到 z 的梯度必须 finite/nonzero。真实 HiF4 quantizer 不需要梯度，因此不要让其内部 round/amax/scale 运算进入 autograd graph；STE 的唯一梯度路径是 `xf -> y` 的 identity path。

- [ ] **Step 3: 实现 `SwitchableNVHiF4Linear`，不复制 8B master weight**

构造时接管现有 `NativeNVFP4SemanticLinear` 的 `weight/bias/input_global_scale/rotation_matrix` 对象。Native mode 完全复刻现有 H16→NVFP4 activation QDQ→GEMM。

`hif4_train_ste`：每次 forward 根据当前 z 从固定 `W_N` 重新变换 weight，weight/activation 都走真实 HiF4 前向 + STE。

`hif4_eval`：无 STE，real `qdq_hif4_direct`；DIAG 固定后允许 frozen weight reconstruction cache。

- [ ] **Step 4: 写死 fusable 7 Linear wiring**

```text
q: input D_QKV explicit
k: input D_QKV explicit
v: input D_QKV explicit + output D_VO
o: input 已携带 D_VO_expanded；activation 不重复乘 D；weight 用 D_VO_expanded inverse
gate: input D_GU explicit
up: input D_GU explicit + output D_UD
down: input 已携带 D_UD；activation 不重复乘 D；weight 用 D_UD inverse
```

V/up 的 output D 必须作为量化前 weight row scaling 实现，而不是 Linear 后追加 online multiply。

- [ ] **Step 5: 实现 online 7 个独立 DIAG wiring**

每个 wrapper 只读取自己的 D，顺序严格按 Task 2。

- [ ] **Step 6: 权重量化计数测试**

Monkeypatch quantizer：一个 Linear 一次 forward 对 weight 只 QDQ 一次；更新 z 后下一次 forward 必须重新 QDQ，不能命中上一 step cache。

- [ ] **Step 7: Native regression**

升级前后 native semantic 输出一致；目标 wrapper coverage 必须仍为 252。

- [ ] **Step 8: 运行测试**

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_semantic_hif4.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_transforms.py -q
```

Expected: PASS。

---

### Task 4: S1K/C4/Wiki calibration、Teacher CoT 与 self-judge

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/data/__init__.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/data/calibration.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/data/teacher_traces.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/data/batching.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_calibration.py`
- Reference: `Block_Sparse/block_pruning/calibration.py`

**Interfaces:**
- `data/calibration.py` Produces: `CalibrationSample`
- `data/calibration.py` Produces: `build_or_load_calibration(cfg, tokenizer, native_model, out_dir)`
- `data/teacher_traces.py` Produces: `generate_teacher_trace(...)`
- `data/teacher_traces.py` Produces: `judge_teacher_trace(...)`
- `data/batching.py` Produces: `DynamicCalibrationCollator`
- `data/batching.py` Produces: `build_length_bucket_batches(samples, batch_size, seed)`

- [ ] **Step 1: 定义数据对象**

```python
@dataclass
class CalibrationSample:
    sample_id: str
    source_index: int | None
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    meta: dict[str, Any]
```

`input_ids/loss_mask` 都是 1-D CPU tensor、未 padding，长度不等直接 fail。

- [ ] **Step 2: 写五种 source mask 测试**

Teacher COT：prompt 5 token + generated 7 token -> `[0]*5+[1]*7`；original/question/Wiki/C4 对其有效 token 全 1。

- [ ] **Step 3: 校验 S1K schema，并实现固定 ID split**

加载 `simplescaling/s1K-1.1_tokenized` 后强制要求存在 `question`、`solution`、`text` 三个字段：`s1k_teacher_cot/s1k_question` 使用 `question`，self-judge 的 reference 使用 `solution`，`s1k_original` 只使用原始 `text`。任一字段缺失、为空或非字符串直接 fail；禁止从其它字段猜测替代。

```python
perm = random.Random(seed).sample(range(len(ds)), k=len(ds))
train_ids = perm[:cfg.calib_nsamples]
val_ids = perm[cfg.calib_nsamples:cfg.calib_nsamples + cfg.calib_val_nsamples]
```

`all/regenerate_correct/s1k_original/s1k_question` 使用同一组 base IDs。

- [ ] **Step 4: Teacher CoT generation**

```python
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": question}],
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=True,
    return_tensors="pt",
)
```

Teacher runtime 必须是 native NVFP4。每个 sample/attempt 的 deterministic seed：

```text
calib_seed + 100000*split_id + 1000*source_index + attempt_index
```

- [ ] **Step 5: Self-judge**

Prompt 包含 Question、Reference solution、Candidate response；greedy `max_new_tokens=8`；trim 后只接受 exact `CORRECT` / `INCORRECT`。

- [ ] **Step 6: 三种 trace policy 测试**

`all` 错题仍保留；`regenerate_correct` 同题最多 4 次；`replace_question_correct` 换 seeded 未用题，train/val 不重复。达到 generation 上限且无 EOS 记 `truncated`；只有 `all` 接受 truncated。

- [ ] **Step 7: 动态 padding collator**

输出 `input_ids[B,Tmax]`、`attention_mask[B,Tmax]`、`loss_mask[B,Tmax]`、`lengths[B]`、sample IDs；padding 同时 mask 掉 attention/loss。

- [ ] **Step 8: 固定 length bucketing**

每 epoch：长度排序 -> 每 `8*batch_size` 个为一个 bucket -> bucket 内按 epoch seed shuffle -> 切 batch -> shuffle batch 顺序。Validation 只按长度排序。

- [ ] **Step 9: Calibration 只生成一次并缓存**

写 `train.pt/val.pt`、轻量 manifest、teacher trace JSONL。manifest 只记录 source、sample IDs、seed、policy、token 长度等复现所需字段，不计算任何 hash/checksum。恢复运行时只检查会改变样本语义的核心字段（source、sample IDs/数量、seed、trace policy）；不要再做文件内容扫描、tensor checksum 或多层重复校验。

- [ ] **Step 10: 运行测试**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_calibration.py -q
```

Expected: PASS。

---

### Task 5: Qwen3 单层调用适配器与 progressive CPU BF16 cache

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/training/__init__.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/training/layer_runtime.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_layer_runtime.py`

**Interfaces:**
- Produces: `PreparedLayerCall`
- Produces: `capture_qwen3_pre_layer_call(model, input_ids, attention_mask)`
- Produces: `run_decoder_layer(layer, hidden_states, prepared_call)`
- Produces: `ProgressiveHiddenCache`
- Produces: `build_initial_hidden_cache(...)`

- [ ] **Step 1: 用 hook 捕获 Transformers 已准备好的 layer 调用参数**

不要复制/重写整个 `Qwen3Model.forward`。在 `model.model.layers[0]` 注册 `forward_pre_hook(with_kwargs=True)`，捕获 layer0 的 hidden state 与 Transformers 已准备好的 causal mask、position ids/embeddings 等参数，然后抛实验内部私有 exception 立即停止，避免执行 layer0。

- [ ] **Step 2: 实现 `run_decoder_layer`**

只替换 captured call 中的 hidden state；其余 args/kwargs 原样重放。不要硬编码一份容易随 Transformers 版本漂移的 layer 参数列表。

- [ ] **Step 3: 写完整模型一致性测试**

短 sequence 下：完整 native semantic model hook 捕获 layer0 input 与 layer1 input；再用 `capture_qwen3_pre_layer_call` + `run_decoder_layer(layer0, X0, prepared)` 重放。两者转 FP32 后 relative L2 必须 `<1e-6`；不一致时修 adapter，不加 fallback，也不通过放宽 tolerance 掩盖调用语义错误。

- [ ] **Step 4: 实现 `ProgressiveHiddenCache`**

每 sample 只保存真实长度 CPU BF16 `[T,H]`，不保存全局 padding。缓存中不保留所有历史层。

- [ ] **Step 5: 构建初始 X0**

对 calibration batch 运行到 layer0 前，按真实长度拆回 CPU BF16 hidden cache。这样 embedding/position preparation 与当前 Transformers 实现保持一致。

- [ ] **Step 6: 写死每层内存生命周期**

任意时刻只允许：

```text
current X_l CPU BF16 cache
+ current layer teacher delta CPU BF16 cache
+ aux teacher cache（仅 aux weight > 0）
```

训练结束先释放 teacher target，再用最终 accepted student 生成 `X_{l+1}`，然后释放 `X_l`。禁止累计 `X_0...X_35`。

- [ ] **Step 7: 运行测试**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_layer_runtime.py -q
```

Expected: PASS。

---

### Task 6: 逐层 `layer_joint` 训练、real-QDQ validation 与强制 rollback

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/training/losses.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/training/trainer.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py`

**Interfaces:**
- `training/losses.py` Produces: `masked_reconstruction_components(...)`
- `training/losses.py` Produces: `finalize_reconstruction_loss(total_num, total_den, loss_type)`
- `training/trainer.py` Produces: `evaluate_layer_real_qdq(...)`
- `training/trainer.py` Produces: `train_layer_joint(...) -> LayerTrainResult`
- `training/trainer.py` Produces: `propagate_accepted_layer(...)`

Train loss、epoch validation、identity baseline 都必须通过 `training/losses.py` 计算；`trainer.py` 禁止内联第二套 NMSE 公式。

- [ ] **Step 1: 写 masked loss 测试**

`block_delta_nmse`：

```python
mask3 = (loss_mask & attention_mask).to(torch.float32)[..., None]
num = ((delta_h.float() - delta_n.float()).pow(2) * mask3).sum()
den = (delta_n.float().pow(2) * mask3).sum().clamp_min(1e-30)
loss = num / den
```

Validation 跨 batch 必须累加全局 numerator/denominator 后做一次除法；禁止平均每 batch NMSE。

- [ ] **Step 2: 当前层 Teacher target 只计算一次**

Current layer 切 `native_nvfp4`，对 train/val 的同一个 `X_l^H` forward，缓存 `Delta_N=Y_N-X_l^H` 到 CPU BF16。若 aux weight >0，再通过 `self_attn` / `mlp` hooks 缓存 branch output；权重为 0 时不建立 aux cache。

- [ ] **Step 3: 计算 identity validation baseline**

全部 z=0；Student=`hif4_eval`；按当前 run 决定 R64 on/off；真实 `qdq_hif4_direct`，无 STE。记录 `identity_val_loss/num/den`。

- [ ] **Step 4: Adam 与 scheduler**

```python
optimizer = torch.optim.Adam(diag_parameters, lr=cfg.diag_lr, weight_decay=0.0)
```

Cosine 每层 `T_max=diag_epochs*number_of_train_batches, eta_min=0`；constant 不建 scheduler。不做 warmup、不做 gradient accumulation。

- [ ] **Step 5: 每个 train step 固定流程**

```text
load current X_l + teacher target to GPU
set current layer hif4_train_ste
zero_grad
full Block forward
masked reconstruction (+ optional aux)
backward
fail if any grad/loss non-finite
Adam step
project every z to global log2 clamp if configured
scheduler step
```

D 更新后下一 optimizer step 必须从 master `W_N` 重新 transformed + HiF4 QDQ。一个 Linear 一次 forward 只 weight-QDQ 一次，batch 内不按 sample 重复量化 weight。

- [ ] **Step 6: 每 epoch real-QDQ validation**

无 STE；保存 val loss 最低 epoch 的全部 z snapshot 到 CPU FP32。不要用最后 epoch 机械覆盖 best。

- [ ] **Step 7: Strict rollback**

```python
accepted = best_val_loss < identity_val_loss
```

False 时整层全部 z 清零。若 R64 on，R64 保持 on。

- [ ] **Step 8: 写 rollback 单测**

Fake layer 训练后 val 变差，断言最终 z 全零，下一层 input 来自 identity/R64-only student，而不是 bad state。

- [ ] **Step 9: 每层 telemetry**

`metrics.json` 至少保存：`layer_id, identity_val_loss, best_val_loss, best_epoch, accepted, rollback, recovery_vs_identity, final_val_loss, z_min/z_max, d_min/d_max, clamp_low_hit_ratio, clamp_high_hit_ratio, train_steps`。

`train_log.jsonl` 每 epoch 保存 train loss、real val loss、lr、scale extremes。

- [ ] **Step 10: Propagation**

最终 accepted/rollback state 切 `hif4_eval`，对全部 train/val forward 一次，只保留真实长度输出为下一层 CPU BF16 cache。

- [ ] **Step 11: 运行测试**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py -q
```

Expected: PASS。

---

### Task 7: fusable 真 fold、online runtime artifact 与 compact checkpoint

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/fold.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/artifact.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_fold_artifact.py`

**Interfaces:**
- Produces: `fold_fusable_layer_inplace(layer, diag_state, use_r64)`
- Produces: `freeze_online_layer_for_eval(layer, diag_state, use_r64, rot_order)`
- Produces: `save_conversion_artifact(...)`
- Produces: `load_and_apply_conversion_artifact(model, artifact_path)`

- [ ] **Step 1: 写 fusable fold 数学测试**

固定 fold：

```text
D_QKV -> input_layernorm.weight *= D_QKV
D_GU  -> post_attention_layernorm.weight *= D_GU
D_VO  -> v_proj transformed weight row scale + o_proj input inverse
D_UD  -> up_proj transformed weight row scale + down_proj input inverse
```

若 V/up 有 bias，输出 D 同时作用于该 bias；其它 bias 不变。

- [ ] **Step 2: 一次性物化 final transformed master weight**

通过 `core/transforms.py` 在 Task 2 定义的 `fusable_weight_transform(...)` 生成 7 个 `W_preQ_final`；`core/fold.py` 禁止重新手写矩阵乘法顺序。Fold 后 wrapper 进入 `folded` 状态：activation 只保留 final runtime 真正存在的 native H16 + optional R64，不再读取 fusable DIAG。

- [ ] **Step 3: fold 正确性放单测，正式流程只保留一次必要 gate**

FP32 无量化 fold 等价性 `<1e-6` 由本 Task 的单元测试覆盖，不在每层训练运行时重复执行。正式逐层流程只在最终 accepted/rollback 状态完成 fold 后，用 held-out validation 的真实 HiF4 QDQ 做一次 unfolded vs folded 输出比较，relative L2 `<1e-5`；失败则停止该 run。不要再叠加 hash、checksum、逐 tensor 扫描或重复 gate。

- [ ] **Step 4: Online final runtime**

Online 不 fold activation DIAG。Artifact 保存 7 个 fixed D；weight side按 fixed D 一次性 transformed + real HiF4 QDQ 后缓存 reconstructed weight，运行时仅保留 activation-side D/R64。

- [ ] **Step 5: Compact artifact schema**

```python
{
    "schema_version": 1,
    "source_model": "ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4",
    "diag_mode": diag_mode,
    "use_r64": use_r64,
    "rot_order": rot_order,
    "layers": layer_states,
}
```

`layer_states` 必须包含 accepted/rollback、best epoch、全部 learned z。Fusable 虽然 runtime 不再需要 D，也保留 z 用于审计/复现。

- [ ] **Step 6: 不复制完整模型**

`checkpoint/final_model/manifest.json + conversion_state.pt` 指向原始 source model。E2E evaluator 重新加载 source checkpoint 后 apply overlay。不要复制 embedding/lm_head/全部未变化权重。

- [ ] **Step 7: Round-trip 测试**

Save artifact -> fresh native semantic model -> apply artifact -> real HiF4 output 与保存前 final model 一致。

- [ ] **Step 8: 运行测试**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_fold_artifact.py -q
```

Expected: PASS。

---

### Task 8: `online + linear_independent` 消融

**Files:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/training/trainer.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py`

**Interfaces:**
- Produces: `train_layer_linear_independent(...) -> LayerTrainResult`

- [ ] **Step 1: 只支持 online**

Task 1 已拒绝 fusable；不要额外发明 fusable “四组独立训练”语义。

- [ ] **Step 2: Native Block 一次性捕获 7 个 Linear 局部数据**

通过 wrapper pre/forward hooks 保存各 Linear 的 raw pre-H input 与 native NVFP4 Linear output。o/down 局部输入必须来自 native 当前 Block 内部路径。

- [ ] **Step 3: 固定顺序逐 Linear 训练**

```text
q_proj -> k_proj -> v_proj -> o_proj -> gate_proj -> up_proj -> down_proj
```

每次只让当前 Linear z requires-grad，局部 objective 固定为 masked Linear output NMSE。

- [ ] **Step 4: 最终仍过 full-block rollback gate**

7 个局部 D 固定后，用 Task 6 的完整 Block real validation 与 identity baseline 比较；不严格更好则整层 7 个 D 全部回 identity。

- [ ] **Step 5: 单测**

验证训练某 Linear 不更新其它 z；full block worse 时全部 rollback。

---

### Task 9: 主训练入口、resume layer range 与 smoke 闭环

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/cli/__init__.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/cli/train.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/common.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_train_smoke.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/train/run_full.sh`

**Interfaces:**
- Consumes: Tasks 1–8
- Produces: 一次完整/部分 layer run 与 final conversion artifact

- [ ] **Step 1: `train.py` orchestration 固定顺序**

```text
parse + validate config
resolve local source snapshot
load tokenizer
load native semantic model
upgrade 252 target linears
build/load calibration once
build X0 progressive CPU BF16 cache
for layer in start_layer..end_layer:
    build native teacher targets on current progressive input
    run layer_joint or online linear_independent
    choose best epoch + strict rollback
    if fusable:
        execute the single final real-HiF4 fold gate and switch current layer to folded runtime
    else:
        freeze the accepted online D/runtime state
    write layer artifacts
    propagate the exact final runtime state to X_next
save compact conversion_state + manifest
write summary.json
```

- [ ] **Step 2: `start_layer>0` 行为写死**

同一 output_dir 必须已有所有 `[0,start_layer-1]` 的 layer artifact。程序从 X0 顺序 replay 这些 final accepted/rollback states 重新生成 `X_start_layer`。这里只检查 artifact 是否存在，以及 `diag_mode/use_r64/rot_order/calib_source/seed` 等会改变语义的核心配置是否一致；不做文件哈希、tensor checksum 或完整目录扫描。缺少必要 artifact/核心配置不一致时直接报错。禁止把 native/direct 输入当作中间 progressive 输入。

- [ ] **Step 3: 脚本分层与 `common.sh`**

`common.sh` 只允许定义项目根目录、Python module 名称、结果根目录、通用启动函数；禁止在这里写 DIAG/R64 算法分支、默认超参矩阵或数据处理。`scripts/train/run_full.sh` 只表达一套正式训练 preset；未来增加新实验 preset 时放到对应 `train/ablation/eval/smoke` 子目录，不继续往一个大脚本里追加 case。

- [ ] **Step 4: Smoke script**

固定两层、小 calibration，但保留完整算法路径：

```bash
CUDA_VISIBLE_DEVICES=[GPU_ID] conda run -n hif4 python -m \
  Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.cli.train \
  --output_dir Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/smoke_[RUN_ID] \
  --use_r64 \
  --diag_epochs 2 \
  --calib_source s1k_question \
  --calib_nsamples 8 \
  --calib_val_nsamples 4 \
  --diag_batch_size 2 \
  --start_layer 0 \
  --end_layer 1
```

Smoke 不生成 32K CoT，只验证 train/validation/rollback/fold/artifact 闭环。

- [ ] **Step 5: 运行 smoke**

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_train_smoke.sh
```

Expected：layer0/1 均有 `best_diag.pt/metrics.json/train_log.jsonl`；artifact 可重新加载；无 NaN/Inf；252 coverage 正确。

---

### Task 10: Fake-QDQ E2E 评估适配器

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/evaluation/__init__.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/evaluation/runner.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/cli/evaluate.py`
- Reference: `Block_Sparse/experiments/qwen35_4b_dynamic_input_sparse_e2e/configs/e2e.json`
- Reference: `main.py`
- Reference: `3rdparty/lighteval/src/lighteval/models/transformers/transformers_model.py`

**Interfaces:**
- `evaluation/runner.py` Produces: `build_eval_model(variant, artifact_path, device)`
- `evaluation/runner.py` Produces: `run_arc(...)`, `run_mmlu_pro_300(...)`, `run_aime25_avg5(...)`
- `cli/evaluate.py` Produces CLI: `--variant {native_nvfp4,direct_hif4,r64_only,artifact}`
- `cli/evaluate.py` Produces CLI: `--artifact_path`（variant=artifact 时必填）
- Produces task groups: `arc`, `mmlu_pro_300`, `aime25_avg5`

`cli/evaluate.py` 不允许复制 benchmark 实现；它只解析参数后调用 `evaluation/runner.py`。未来增加新 benchmark 时只改 runner + eval preset，不改 conversion core。

- [ ] **Step 1: 统一构建 in-memory semantic model**

```text
native_nvfp4 -> native semantic runtime
direct_hif4  -> D=I, R64 off, real HiF4
r64_only     -> D=I, R64 on, real HiF4
artifact     -> fresh source native model + apply conversion_state, real HiF4
```

E2E 时所有 D 固定；固定 transformed weight 的 HiF4 reconstruction 必须预缓存，禁止生成每个 token 时反复量化固定 weight。Activation 仍每次真实 fake-QDQ。

- [ ] **Step 2: ARC 保持 lm_eval 口径**

固定：

```text
arc_easy
arc_challenge
num_fewshot=0
batch_size=8
```

用 `lm_eval` HuggingFace adapter 包装已经在内存中的 semantic `PreTrainedModel`。不要把 compact artifact 冒充成 vLLM/HF model path。

- [ ] **Step 3: MMLU-Pro 300 用仓库 lighteval Transformers backend**

固定现有口径：

```text
task=mmlu_pro|0
max_samples=300
max_model_length=32768
max_new_tokens=32768
temperature=0.7
top_p=0.8
top_k=20
enable_thinking=False
```

用 `TransformersModel.from_model(model=semantic_model, config=...)`；随后显式：

```python
le_model.prompt_manager.enable_thinking = False
```

再 `Pipeline(model=le_model, ...)`。

- [ ] **Step 4: AIME25 avg@5**

固定：

```text
task=aime25_avg5
max_model_length=32768
max_new_tokens=32768
temperature=0.7
top_p=0.8
top_k=20
enable_thinking=True
```

显式 `le_model.prompt_manager.enable_thinking = True`。

- [ ] **Step 5: 统一保存绝对指标和相对 NVFP4 pp drop**

Summary row 至少：`method, arc_easy, arc_challenge, mmlu_pro_300, aime25_avg5` 以及每项 `delta_*_vs_nvfp4_pp`。

---

### Task 11: 结构实验与 calibration-source 消融

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/evaluation/summarize.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/cli/summarize.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_phase1_matrix.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_calib_ablation.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_teacher_policy_ablation.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/eval/run_fast_eval.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/eval/run_final_eval.sh`

- [ ] **Step 1: 固定脚本职责，禁止继续堆大一统脚本**

`scripts/ablation/*.sh` 只生成/调用不同训练 preset；`scripts/eval/run_fast_eval.sh` 只跑 ARC-E/C + MMLU-Pro 300；`scripts/eval/run_final_eval.sh` 只跑 finalist + AIME25。所有脚本通过 `scripts/common.sh` 调用 `cli/train.py`、`cli/evaluate.py`、`cli/summarize.py`，不解析模型 tensor、不读取内部 Python 对象、不复制结果汇总逻辑。

- [ ] **Step 2: Phase 1 固定 `s1k_teacher_cot + all`**

八组：

```text
E0 原生 NVFP4
E1 Direct HiF4
E2 HiF4 + R64 only
E3 fusable DIAG
E4 fusable DIAG -> R64
E5 online DIAG
E6 online DIAG -> R64
E7 online R64 -> DIAG
```

E0/E1/E2 不训练 DIAG；E3–E7 按主协议训练。

- [ ] **Step 3: 八组先只快速筛选**

```text
ARC-Easy
ARC-Challenge
MMLU-Pro 300
```

不对八组全部先跑 AIME avg@5。

- [ ] **Step 4: 选 best fusable / best online**

结构选择规则固定：先取 MMLU-Pro 300 分数最高者；若并列，再取 ARC-Challenge 更高者；仍并列再取 ARC-Easy 更高者。不要只按 reconstruction loss 选最终结构，也不要让脚本自行定义新的综合分数。

- [ ] **Step 5: Calibration source 消融只跑两个最佳结构**

```text
s1k_teacher_cot
s1k_original
s1k_question
wikitext2
c4
```

禁止 `5 sources × E3–E7` 全笛卡尔积。

- [ ] **Step 6: Teacher trace policy 消融**

最佳结构支持：`all / regenerate_correct / replace_question_correct`；主默认仍为 `all`。

- [ ] **Step 7: Finalists 才跑 AIME25 avg@5**

至少：native NVFP4、Direct HiF4、best fusable、best online。对 source ablation 仍按“MMLU-Pro 300 → ARC-Challenge → ARC-Easy”规则选每个结构的最佳 source；若最佳 source 不是默认 `s1k_teacher_cot`，则额外加入该 source variant 作为 finalist。

- [ ] **Step 8: `summarize.py` 中文总表**

至少输出：结构/source/policy、36 层 accepted/rollback 数、每层 recovery、任务绝对分数、vs NVFP4 pp drop、best fusable vs online、teacher CoT vs original S1K vs question-only vs Wiki/C4。

---

### Task 12: README、全量测试与最终验收

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/README.md`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_teacher_cot_smoke.sh`
- Modify only if tests expose real bug: Tasks 1–11 files

- [ ] **Step 1: README 只记录稳定协议**

必须说明 native H16 vs R64、online 两顺序、fusable 四 D 与 fold、source-specific mask、teacher generation/judge、progressive layer_joint、rollback、fake-QDQ 第一阶段限制，以及 smoke/full/eval 命令。README 还必须增加“扩展指南”小节，分别明确新增 calibration source、reconstruction loss、fixed transform、benchmark 时只需要修改哪些文件/dispatch 条目；不要写成需要理解整个 trainer 才能扩展的隐式流程。

- [ ] **Step 2: 跑新增测试**

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_transforms.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_semantic_hif4.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_calibration.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_layer_runtime.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_fold_artifact.py -q
```

Expected: 全 PASS。

- [ ] **Step 3: 跑专项全部回归测试**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests -q
```

Expected: 原有测试不回归。

- [ ] **Step 4: 跑两层 smoke**

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_train_smoke.sh
```

验收：Teacher/Student 同 progressive input；train loss finite；validation real QDQ；best epoch 可恢复；worse-than-identity rollback；fusable fold gate；compact artifact round-trip。

- [ ] **Step 5: 单独做一次 Teacher-COT generation smoke**

`scripts/smoke/run_teacher_cot_smoke.sh` 固定只用 2 train + 1 val，`teacher_max_new_tokens=512`，验证 thinking generation、真实长度缓存、question mask=0/generated mask=1、self-judge、`all` policy、动态 padding。该 smoke 不进入正式报告。

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_teacher_cot_smoke.sh
```

- [ ] **Step 6: 正式 run 启动时打印并落盘 resolved config**

至少：source snapshot、36/252 coverage、diag mode/R64/order/scope/loss、lr/epochs/scheduler/clamp、calib source/count/seed、teacher policy/attempts/32768、batch、layer range。只写可读配置，不生成 hash/checksum。

- [ ] **Step 7: 做一次维护性代码审查**

实现完成后人工检查：`core/transforms.py` 没有在 fold/trainer/eval 中被复制公式；shell 中没有 tensor/算法逻辑；`cli/*.py` 仅 orchestration；没有新增 hash/checksum/完整性框架；没有 broad `except Exception` 做兜底；没有为了未来扩展引入无实际用途的抽象基类/registry/plugin 系统。发现上述问题先简化代码，再进入正式实验。

---

## 5. 实现时禁止出现的偏差

1. 禁止在已经 HiF4 QDQ 的权重上继续 fold 新 D；每 step 从固定 `W_N` master 出发。
2. 禁止训练开始只量化一次 weight 后 20 epochs 复用；D 每 step 变，weight QDQ 必须随之重新算。
3. 禁止把 fusable D 放到 native H16 后再宣称可 fuse 到 RMSNorm。
4. 禁止 fusable q/k/v 分别学独立 pre-norm D。
5. 禁止 gate 在 SiLU 前增加独立 output D。
6. 禁止 GQA 展开后每个 Q head 独立学习 `D_VO`。
7. 禁止 O 的 R64 G64 跨 attention head。
8. 禁止 `s1k_teacher_cot` question token 进入 reconstruction loss。
9. 禁止把 S1K 统一截成 1024。
10. 禁止每层重新随机 calibration 或重新生成 teacher CoT。
11. 禁止 BF16/dequantized output 替代原生 NVFP4 teacher target 作为主目标。
12. 禁止 Teacher 用 `X_l^N`、Student 用 `X_l^H`；必须同输入。
13. 禁止 validation 用 STE 选择 best epoch。
14. 禁止训练后比 identity baseline 差仍进入下一层。
15. 禁止只看 reconstruction loss 宣布 E2E 更优；必须看任务分数。
16. 禁止第一版同时改 packed kernel/vLLM 并把工程问题混入算法结论。
17. 禁止加入未讨论的数据污染过滤；本计划明确不做。
18. 禁止 silent OOM fallback、自动减 batch、自动截长 CoT；资源不足显式报错，由用户手动调 batch。
19. 禁止新增 SHA/hash/checksum/tensor fingerprint 作为缓存、artifact 或模型一致性机制。
20. 禁止把 transform 数学分别复制到 trainer/fold/evaluator；必须统一调用 `core/transforms.py`。
21. 禁止把 reconstruction loss 分子/分母分别复制到 train/validation；必须统一调用 `training/losses.py`。
22. 禁止把所有 Python 文件平铺在实验根目录；严格保持 `core/data/training/evaluation/cli/scripts` 分类。
23. 禁止 shell 脚本包含模型 tensor 操作、Python 算法实现或结果解析算法；shell 只负责 preset 与启动。
24. 禁止为了“更安全”增加 broad try/except、重复 shape 检查、重复 artifact 扫描或未要求的 fallback；让底层明确错误自然暴露。
25. 禁止为了未来可能扩展预先引入抽象插件框架；只实现当前确实存在的简单显式扩展点。

---

## 6. 最终成功标准

1. 新增测试与原专项测试全部通过。
2. Native semantic runtime 升级前后不漂移。
3. Online/fusable 无量化数学 gate `<1e-6`。
4. Fusable folded vs unfolded real-HiF4 gate 通过。
5. 两层 smoke 完成 Teacher target、joint train、best epoch、rollback、propagation、artifact round-trip。
6. `s1k_teacher_cot` 可生成、judge、缓存、重载，question mask=0、generated mask=1。
7. 128/32 正式 calibration 只生成一次并跨 36 层复用。
8. 每层只保留当前 progressive CPU BF16 cache + 当前 teacher target，不累计所有层 activation。
9. 每层 metrics 明确记录 identity/best/final/accepted/rollback 与 DIAG 边界命中。
10. E0–E7 快速筛选能生成 ARC-E/C + MMLU-Pro 300 可比较结果。
11. Source 消融只跑 best fusable / best online，不产生不必要笛卡尔积。
12. Finalist 可跑 AIME25 avg@5，并统一报告相对原生 NVFP4 的 pp drop。
13. 最终目录符合 `core/data/training/evaluation/cli/scripts` 分类，shell preset 按 `smoke/train/ablation/eval` 分类。
14. DIAG/R64 数学只有 `core/transforms.py` 一处实现，loss 只有 `training/losses.py` 一处实现；fold/trainer/eval 无重复公式。
15. 新代码不包含 SHA/hash/checksum/tensor fingerprint，不包含额外完整性框架或 broad fallback 异常层。
16. README 明确写出“如何新增 calibration source / loss / fixed transform / benchmark”的最短修改入口，确保后续功能增加不需要重构整套 pipeline。

完成本计划后，再单独开下一份计划处理 packed HiF4 权重/scale 导出、真实 kernel/vLLM 部署与性能验证；不要在本计划中提前混入。
