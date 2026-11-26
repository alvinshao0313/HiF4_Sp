# Qwen3-8B Native NVFP4→HiF4 E2E 消融实验计划（暂不使用 Teacher-CoT）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 暂时完全绕开 Native NVFP4 Teacher 自回归生成 CoT / self-judge / regenerate/replace 轨迹链路，先使用现成可直接构造的 calibration 数据完成 NVFP4→HiF4 的核心 E2E 结构、DIAG、训练策略、loss、稳定性和 calibration-source 消融，并形成正式实验报告。

**Architecture:** 继续复用现有 `experiments/e2e_diag_reconstruction/` 的 `core/data/training/evaluation/cli/scripts` 分层，不新建第二套训练 pipeline。Native NVFP4 **仍保留为每一层 block reconstruction 的参考 Teacher**；本计划暂停的只是“Teacher 自回归生成 calibration trace”能力。当前实验默认 calibration 改为 `s1k_original`，因为它已经包含现成 reasoning/answer 内容，不需要在线生成轨迹。

**Tech Stack:** Python、PyTorch、Transformers、现有 HiF4 fake-QDQ oracle、lm_eval、lighteval、bash；Python/测试/训练命令使用 `hif4` conda 环境。

**Base Spec:** `Native_NVFP4_HiF4_Linear_Puncture/plans/2026-08-17-qwen3-8b-native-nvfp4-to-hif4-e2e-diag-r64-layerwise-reconstruction-plan-cn.md`

**Deferred Plan:** `Native_NVFP4_HiF4_Linear_Puncture/plans/2026-08-18-qwen3-8b-native-nvfp4-to-hif4-e2e-ablation-experiment-plan-cn.md` 中 Teacher-CoT / trace policy 相关阶段暂缓，不作为本计划的前置条件。

**Current implementation status:** 上一版消融计划已经部分落地：shared cache 参数、fusable component selector、teacher/progressive input mode、rollback on/off、eval seed、summary winner 逻辑、GPU pool 和大部分 ablation shell 都已经存在。本计划必须以当前代码为起点做增量修改，禁止把这些功能重新实现一遍。

## Global Constraints

- 当前计划**不运行** `s1k_teacher_cot`，不运行 Teacher autoregressive generation，不运行 self-judge，不运行 `regenerate_correct` / `replace_question_correct`。
- 不删除现有 `teacher_traces.py` 或相关代码；它们只是当前阶段不使用，后续可单独恢复。
- Native NVFP4 block teacher forward **必须保留**：对当前层 progressive input `X_l`，仍计算 `F_l^{NVFP4}(X_l)` 作为 reconstruction target。
- 正式默认 calibration：`s1k_original`，train=128，val=32，seed=42。
- S1K 原始数据必须继续使用现有 `row["text"]`，不重新拼 prompt，不调用 Teacher 生成。
- `s1k_original/s1k_question` 保持自然变长，不使用固定 `calib_seqlen` 截断；batch 内只 pad 到当前 batch 最长序列，padding token 同时从 attention mask 和 reconstruction loss mask 排除。`calib_seqlen=1024` 只作用于 WikiText2/C4。
- 正式主配置默认：`diag_mode=fusable`、`diag_train_scope=layer_joint`、`recon_loss=block_delta_nmse`、`diag_lr=5e-3`、`diag_epochs=20`、`diag_scheduler=cosine`、`diag_log2_clamp=-4,4`、`diag_batch_size=4`、`calib_input_mode=progressive_student`、`layer_rollback=on`。
- Transformer 量化覆盖固定为 36 层 × 7 Linear：q/k/v/o/gate/up/down；embedding/final norm/lm_head 保持原样。
- 第一阶段继续使用 fake HiF4 QDQ 做训练和 E2E 正确性评测，不把真实 packed HiF4/vLLM kernel 引入当前消融。
- Fast E2E：ARC-Easy、ARC-Challenge、MMLU-Pro 300；AIME25 avg@5 只跑 finalist。
- 不做 calibration/test 污染检测。
- 不新增 SHA/hash/checksum/tensor fingerprint。
- 不使用 broad exception fallback、silent OOM fallback、自动降低 batch size、自动截断 S1K。
- 同一比较组使用同一 calibration tensor、同一 batch size、同一 eval seed。
- 一张 GPU 同时只运行一个完整 train/eval run；阶段内独立 run 可多 GPU 并行。
- GPU 调度仅存在于 shell 层，不进入 trainer/model/quantizer。
- 当前工作区已有大量未提交改动；本计划不要求创建 git commit。

---

## 0. 当前阶段明确不做什么

当前 Cursor **禁止把时间继续花在以下问题上**：

```text
s1k_teacher_cot generation 卡住
Teacher max_new_tokens
Teacher chat template generation
Teacher answer correctness judge
regenerate_correct
replace_question_correct
Teacher trace policy ablation
Teacher-generated CoT cache producer
```

如果运行任何本计划脚本时进入 `generate_split_teacher_traces()`，视为配置错误，立即停止并检查 `calib_source`，而不是修 Teacher generation。

当前阶段允许出现的 calibration source 只有：

```text
s1k_original
s1k_question
wikitext2
c4
```

其中正式主线先使用：

```text
s1k_original
```

---

## 1. 本计划真正需要回答的研究问题

### Q1. NVFP4→HiF4 的直接 E2E 损失有多大？

比较：

```text
Native NVFP4
Direct HiF4
```

### Q2. DIAG 和 R64 各自贡献多少？二者是否互补？

比较：

```text
Direct HiF4
R64 only
Online DIAG
Online DIAG -> R64
Online R64 -> DIAG
Fusable DIAG
Fusable DIAG -> R64
```

### Q3. Fusable 四组 DIAG 中谁贡献最大？

拆：

```text
D_QKV
D_VO
D_GU
D_UD
Attention = QKV + VO
MLP = GU + UD
All
```

### Q4. Block joint optimization 是否优于逐 Linear 独立优化？

比较：

```text
layer_joint
linear_independent
```

仅在 online 结构上做。

### Q5. Progressive student input 是否必要？

比较：

```text
progressive_student
teacher
```

这里的 `teacher` 指**逐层输入传播来自 Native NVFP4 block**，不是 Teacher 生成 CoT。

### Q6. `block_delta_nmse` 是否比 output NMSE / MSE 更合适？

比较：

```text
block_delta_nmse
block_output_nmse
mse
```

### Q7. Rollback / clamp 是否是必要稳定机制？

比较：

```text
rollback on / off
clamp -4,4 / none
```

### Q8. 不依赖 Teacher-CoT 时，哪种 calibration 最好？

比较：

```text
s1k_original
s1k_question
wikitext2
c4
```

核心要看：

```text
S1K original reasoning vs question-only
reasoning/task-aware data vs generic text
```

---

# Part I：先让当前实现彻底脱离 Teacher-CoT 阻塞

### Task 1: 将当前 E2E 主实验默认 calibration 改为 `s1k_original`

**Files:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/config.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/train/run_full.sh`
- Keep fast smoke: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_train_smoke.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_s1k_original_smoke.sh`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py`

**Required behavior:**

`E2ETrainConfig` 默认：

```python
calib_source: str = "s1k_original"
```

CLI：

```python
p.add_argument(
    "--calib_source",
    choices=CALIB_SOURCES,
    default="s1k_original",
)
```

不要从 `CALIB_SOURCES` 删除 `s1k_teacher_cot`；只是默认不再进入它。

- [ ] **Step 1: 修改 config 测试**

断言：

```python
cfg = E2ETrainConfig.for_test()
assert cfg.calib_source == "s1k_original"
```

- [ ] **Step 2: 修改默认值，并让 standalone full 脚本显式写明 source**

只改 config/CLI default，不改现有 source builder。`scripts/train/run_full.sh` 额外显式传 `--calib_source s1k_original`，这样正式 standalone run 的配置无需依赖读者知道 Python 默认值。

- [ ] **Step 3: 保留现有快速 smoke，并新增专门的 `s1k_original` smoke**

现有 `run_train_smoke.sh` 已显式使用 `s1k_question`，继续保留为轻量框架 smoke，不要因为主 calibration 改成 `s1k_original` 而让通用 smoke 变重。

新增 `run_s1k_original_smoke.sh`，固定：

```text
calib_source=s1k_original
calib_nsamples=2
calib_val_nsamples=1
diag_epochs=1
diag_batch_size=1
start_layer=0
end_layer=0
use_r64=true
```

这个 smoke 只验证完整 S1K 原始样本、自然变长、动态 padding、当前 block teacher target、HiF4 training/fold/artifact 能闭环，不追求统计意义。

- [ ] **Step 4: 删除 `test_config.py` 中“配置字典不得出现 checksum/sha256 字符串”的测试断言**

当前已有：

```python
assert "checksum" not in dumped.lower()
assert "sha256" not in dumped.lower()
```

这类“证明没有某种安全代码”的测试没有算法价值，删除即可；不要用新的反向安全测试替代。

- [ ] **Step 5: 跑 config/calibration 测试**

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_calibration.py -q
```

- [ ] **Step 6: 跑两个非 Teacher-CoT smoke**

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_train_smoke.sh
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_s1k_original_smoke.sh
```

验收：两者均不进入 Teacher generation；`s1k_original` smoke 至少完成 layer0 的训练、真实 QDQ validation、rollback/fold 和 artifact 保存。

---

### Task 2: 暂停 Teacher-CoT 脚本进入当前正式实验入口

**Files:**
- Do not modify algorithm: `data/teacher_traces.py`
- Do not run: `scripts/smoke/run_teacher_cot_smoke.sh`
- Do not run: `scripts/ablation/run_teacher_policy_ablation.sh`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/README.md`

- [ ] **Step 1: README 加“当前实验阶段”说明**

明确：

```text
当前正式 ablation 默认使用 s1k_original。
s1k_teacher_cot 功能保留但暂不进入正式实验矩阵。
Teacher trace smoke / policy ablation 不属于当前计划验收项。
```

- [ ] **Step 2: 不修、不重构 Teacher generation**

本 Task 不允许修改 `teacher_traces.py` 来解决当前卡住问题。

---

# Part II：补齐非 Teacher-CoT 消融需要的最小控制接口

### Task 3: 复用现有 shared cache 能力，只新增无 Teacher 模型的 calibration 预构建入口

**Current state already present，禁止重复实现：**

- `calib_cache_dir` 已经存在于 config/CLI。
- `cli/train.py` 已经把 shared calibration root 与 run `output_dir` 分开。
- `build_or_load_calibration()` 已经具备 cache load/write 逻辑。

**Files:**
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/cli/prepare_calibration.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/data/calibration.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/common.sh`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_calibration.py`

**CLI:**

```text
--calib_cache_dir PATH
```

默认 `""`：保持当前 `output_dir/calibration/`。

显式路径：所有 run 共享该目录下的 calibration 文件。

当前计划 cache 目录固定：

```text
results/e2e_diag_reconstruction/shared_calibration/
  s1k_original_n128_v32_seed42/
  s1k_question_n128_v32_seed42/
  wikitext2_n128_v32_seed42_len1024/
  c4_n128_v32_seed42_len1024/
```

不要增加 hash/checksum。

- [ ] **Step 1: 简化 non-Teacher cache manifest**

当前 `_core_calib_fields()` 对所有 source 都写入 Teacher trace 参数。改成只在 `calib_source == "s1k_teacher_cot"` 时保存 `teacher_trace_policy/teacher_max_attempts/teacher_max_new_tokens`；`s1k_original/s1k_question/wikitext2/c4` 只保存真正影响它们的 source/sample/seed/seqlen 字段。旧 manifest 多出的字段不需要迁移。

- [ ] **Step 2: 允许 non-Teacher builder 在 `native_model=None` 下运行**

`build_or_load_calibration()` 参数顺序不变。四个 non-Teacher source 禁止访问 `native_model`；只有进入 `s1k_teacher_cot` 分支前才要求其非空。

- [ ] **Step 3: 创建 `prepare_calibration.py`**

只允许 `s1k_original/s1k_question/wikitext2/c4`，明确拒绝 `s1k_teacher_cot`。只加载 tokenizer，不加载 8B Native NVFP4 model。

parser 固定只暴露：`model_path/calib_source/calib_nsamples/calib_val_nsamples/calib_seed/calib_seqlen/calib_cache_dir`。构造 `E2ETrainConfig` 时显式提供 `model_path` 与 `output_dir=calib_cache_dir`，再覆盖 calibration 字段；不要假设 dataclass 只需要 calibration 参数。

完成后打印：train/val 样本数、总 token 数、max/P50/P95 sequence length，并把同一信息写到 `<calib_cache_dir>/calibration/stats.json`。`stats.json` 只保存计数/长度统计，不保存 tensor 副本。统计使用 Python 内置排序即可，不新增 NumPy。P50/P95 固定使用 nearest-rank：对升序长度 `v`，百分位 `p` 取 `v[min(n-1, ceil(p*n)-1)]`。Phase D 报告直接读取这个文件，不从终端日志反推。

- [ ] **Step 4: `common.sh` 增加统一 helper**

新增 `E2E_MODULE_PREPARE_CALIB` 和 `e2e_prepare_calib()`；正式阶段脚本只调用 helper，不重复 Python module path。

- [ ] **Step 5: 测试预构建与共享复用**

验证 `s1k_original` 在 `native_model=None` 下能构建；第二次读取相同 cache 不重新构建；不同 consumer 得到完全相同的 sample IDs/input_ids/loss_mask；prepare CLI 对 `s1k_teacher_cot` 明确报错。

---

### Task 4: 验收现有 Fusable DIAG component 接口，不重复实现

**Current state already present：**

- `FUSABLE_DIAG_COMPONENTS/FUSABLE_COMPONENT_MAP` 已在 `core/config.py`。
- `LayerDiagState.configure_fusable_components()` 已按 preset 设置 `requires_grad`。
- `upgrade_semantic_model_inplace()` 已调用该配置。
- artifact 已保存 `fusable_diag_components`。
- online 已拒绝 partial fusable components。

因此 Cursor **禁止重新设计或重写这一套**。

- [ ] **Step 1: 只跑现有 config/semantic tests，确认 7 个 preset 和 `all` 默认行为**

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_semantic_hif4.py -q
```

- [ ] **Step 2: 如果测试通过，不修改 component 数学或 state 结构**

正式实验直接使用现有 `--fusable_diag_components`。不要新增第二套 component selector。

---

### Task 5: 验收现有 `progressive_student` vs `teacher` 输入传播，不重复实现

**Current state already present：**

- `CALIB_INPUT_MODES` 与 CLI 已存在。
- `propagate_native_layer()` 已在 `training/layer_runtime.py`。
- `cli/train.py` 已在进入当前 layer 训练前计算 `native_next`，并在 teacher-input 模式使用它传播到下一层。
- `start_layer>0` prefix replay 已区分 teacher/progressive 两种模式。
- artifact/resume 已记录/检查 `calib_input_mode`。

严格语义保持：

```text
progressive_student:
    下一层 X_{l+1} 来自最终 accepted/rollback HiF4 student layer l。

teacher:
    下一层 X_{l+1} 来自 Native NVFP4 layer l。
    当前层 Teacher/Student 仍然共享同一个 X_l。
```

这里的 `teacher` 只是 Native NVFP4 block propagation，与 Teacher-CoT generation 无关。

- [ ] **Step 1: 跑现有 layer_runtime/trainer tests**

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_layer_runtime.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py -q
```

- [ ] **Step 2: 测试通过则直接复用现有实现，不重写 propagation**

---

### Task 6: 验收现有 rollback ON/OFF，并清理一处不可达死代码

**Current state already present：**

- `--layer_rollback {on,off}` 已存在。
- `apply_rollback_decision()` 已实现 `would_rollback/rollback_applied/accepted_vs_identity`。
- joint 与 linear-independent 两条训练路径已使用统一 rollback 语义。
- `test_rollback_off_keeps_worse_z` 已存在。

- [ ] **Step 1: 删除 `rollback_metric_fields()` 中 `return` 之后的不可达代码**

当前函数在返回 dict 后还残留一段遍历 `layer.modules()` 并调用 `SwitchableNVHiF4Linear.clear_weight_cache()` 的代码，永远不会执行，而且 `layer` 在该函数作用域中也不存在。直接删除这段死代码，不引入新的 cache defensive layer。

- [ ] **Step 2: 跑 trainer tests**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py -q
```

测试通过后不再修改 rollback 算法。

---

### Task 7: 验收现有 eval seed / summarize，不重复实现

**Current state already present：**

- `cli/evaluate.py` 已有 `--eval_seed=42`。
- `evaluation/runner.py` 已统一设置 Python/Torch CPU/CUDA seed。
- `evaluation/summarize.py` 已读取结构、训练、calibration、layer recovery 与 E2E 字段。
- `test_evaluation.py` 已测试 seed 和 Phase A winner 三级选择规则。

- [ ] **Step 1: 跑 evaluation tests**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_evaluation.py -q
```

- [ ] **Step 2: 保持 winner 规则不变**

```text
1. MMLU-Pro 300
2. ARC-Challenge
3. ARC-Easy
```

不新增综合分数。

- [ ] **Step 3: 所有后续 stage 的 `run_map.json` 必须额外引用 Phase A 的 `E0_native_nvfp4`**

原因：当前 `summarize_runs()` 只有在 run map 里能找到 `E0_native_nvfp4` 时，才会计算 `delta_*_vs_nvfp4_pp`。Component/loss/input/calibration 等 stage 不能只放本阶段方法，否则最终报告里的“相对 Native NVFP4 pp drop”会为空。

后续每个 stage 统一加入：

```text
E0_native_nvfp4 -> <PHASE1_DIR>/E0_native_nvfp4
```

不复制 E0 结果目录。

---

### Task 8: 验收现有 GPU pool，多卡调度不重复实现

**Current state already present：**

- `scripts/utils/gpu_pool.sh` 已实现 GPU 检测与 wave wait。
- `scripts/common.sh` 已 source GPU pool，并提供 stage 初始化 helper。
- 当前实现已经支持人工 `GPU_POOL` 与自动空闲卡检测。

因此本 Task 只验收，不重写 GPU scheduler。

固定环境变量：

```text
GPU_POOL=""              # 显式指定，如 0,2,3
GPU_MIN_FREE_RATIO=0.90
GPU_MAX_UTIL=10
MAX_PARALLEL_JOBS=""
DIAG_BATCH_SIZE=4
STAMP
```

自动检测：

```bash
nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
```

自动选择条件：

```text
free / total >= 0.90
utilization <= 10%
```

`GPU_POOL` 非空时仅校验 GPU ID 存在，不再次按 utilization 过滤。

对外函数固定：

```bash
detect_available_gpus
resolve_parallel_slots
wait_gpu_wave
```

调度采用简单 wave：一张 GPU 一个完整 run；一波全部结束后再启动下一波。

不要实现 GPU daemon、lock service、job database、自动 retry。

- [ ] **Step 1: 不改算法，只做 shell syntax check**
- [ ] **Step 2: dry-run 只打印 available GPUs / slots，不启动正式训练**
- [ ] **Step 3: 若 `detect_available_gpus` 返回至少 2 张卡，则用两个轻量 smoke job 在两张不同 GPU 上并行，输出目录不得冲突；若当前仅 1 张可用卡，则只完成 dry-run 与单卡 smoke，不占用其它正在工作的 GPU**

---

# Part III：正式实验阶段

## Phase A：核心结构消融——第一优先级

### Task 9: 跑 E0–E7 结构矩阵

**Script:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_phase1_matrix.sh`

**Calibration 固定：**

```text
calib_source=s1k_original
calib_nsamples=128
calib_val_nsamples=32
calib_seed=42
shared cache=s1k_original_n128_v32_seed42
```

**实验矩阵：**

| ID | Method | Train |
|---|---|---:|
| E0 | Native NVFP4 | No |
| E1 | Direct HiF4 | No |
| E2 | Direct HiF4 + R64 | No |
| E3 | Fusable DIAG | Yes |
| E4 | Fusable DIAG→R64 | Yes |
| E5 | Online DIAG | Yes |
| E6 | Online DIAG→R64 | Yes |
| E7 | Online R64→DIAG | Yes |

**执行顺序：**

- [ ] **Step 1: 在任何 36 层训练前，用 `prepare_calibration` 单独构建/复用 `s1k_original` shared cache**

`run_phase1_matrix.sh` 必须删除当前 `run_teacher_cot_cache_4gpu.sh` / `ensure_shared_teacher_cot_cache` 依赖，改成先同步执行：

```bash
CALIB="$(shared_calib_dir_for s1k_original)"
e2e_prepare_calib \
  --calib_source s1k_original \
  --calib_nsamples 128 \
  --calib_val_nsamples 32 \
  --calib_seed 42 \
  --calib_cache_dir "${CALIB}"
require_shared_calib_cache "${CALIB}"
```

只有 cache 完整后才启动训练。

- [ ] **Step 2: E3–E7 按 GPU pool 分波次并行训练**

五个训练 run 都必须显式传：

```text
--calib_source s1k_original
--calib_cache_dir <同一个 s1k_original shared cache>
```

不要仅依赖 Python 默认值。正式脚本里显式写 source，方便后续审计实验口径。

- [ ] **Step 3: E0–E7 按 GPU pool 做 fast eval**

```text
ARC-Easy
ARC-Challenge
MMLU-Pro 300
eval_seed=42
```

- [ ] **Step 4: 输出结构总表**

直接比较：

```text
E1 -> E2 : R64 only
E1 -> E5 : Online DIAG
E5 -> E6 : DIAG 后加 R64
E6 vs E7 : 顺序
E3 vs E5 : fusable freedom cost
E4 vs E6 : fusable freedom cost with R64
```

- [ ] **Step 5: 固定 winner**

输出：

```text
best_fusable_preset
best_online_preset
best_overall_preset
```

此后所有实验只使用 Phase A winner，不允许 Cursor 凭主观换结构。

---

## Phase B：Fusable 四个 DIAG 的贡献

### Task 10: Fusable component ablation

**Script:**
- Modify existing: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_fusable_component_ablation.sh`

固定：

```text
diag_mode=fusable
use_r64=false
calib_source=s1k_original
```

矩阵：

```text
B1 qkv
B2 vo
B3 gu
B4 ud
B5 attn
B6 mlp
B7 all
```

B7 直接复用 Phase A E3，不重训。

当前脚本的 `CALIB` 仍指向 `s1k_teacher_cot`，必须改成：

```bash
CALIB="$(shared_calib_dir_for s1k_original)"
```

B1–B6 每个 `e2e_train` 都显式传 `--calib_source s1k_original`。`run_map.json` 额外加入 `E0_native_nvfp4 -> PHASE1_DIR/E0_native_nvfp4`，用于统一计算 vs NVFP4 pp drop。

- [ ] **Step 1: B1–B6 多卡 wave 训练**
- [ ] **Step 2: fast eval**
- [ ] **Step 3: 统计 layer recovery / rollback**
- [ ] **Step 4: 输出 Attention vs MLP 归因结论**

---

## Phase C：训练机制消融

**Phase C 共用脚本修正：** 当前 `run_train_scope_ablation.sh`、`run_input_mode_ablation.sh`、`run_loss_ablation.sh`、`run_rollback_ablation.sh`、`run_clamp_ablation.sh` 都已经存在，但其 `CALIB` 仍指向 `s1k_teacher_cot`。五个脚本统一改为 `shared_calib_dir_for s1k_original`，所有新增训练命令显式传 `--calib_source s1k_original`，并在各自 `run_map.json` 中加入 Phase A 的 `E0_native_nvfp4` 路径。不要新建第二套脚本。

### Task 11: Joint vs Linear-independent

**Script:**
- Modify existing: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_train_scope_ablation.sh`

仅 best online：

```text
layer_joint              # Phase A baseline，复用
linear_independent       # 新跑
```

- [ ] 新跑 1 个 training run
- [ ] fast eval
- [ ] 比较 E2E + layer recovery

---

### Task 12: Progressive input vs Native teacher input

**Script:**
- Modify existing: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_input_mode_ablation.sh`

对：

```text
best fusable
best online
```

各比较：

```text
progressive_student  # Phase A baseline
teacher              # 新跑
```

新跑两个 `teacher` input mode run，可两卡并行。

重点看最终 E2E，而不是只看 local val loss。

---

### Task 13: Loss ablation

**Script:**
- Modify existing: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_loss_ablation.sh`

对 best fusable + best online：

```text
block_delta_nmse    # Phase A baseline
block_output_nmse
mse
```

新增 4 个 training run，可多卡 wave。

报告：

```text
E2E accuracy
accepted layer count
rollback count
mean/median layer recovery
```

---

### Task 14: Rollback ablation

**Script:**
- Modify existing: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_rollback_ablation.sh`

只 best fusable：

```text
rollback=on    # reuse Phase A
rollback=off   # new
```

报告：

```text
would_rollback_count
rollback_applied_count
最终 ARC-E/C
MMLU-Pro300
逐层 final_val_loss
```

---

### Task 15: Clamp ablation

**Script:**
- Modify existing: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_clamp_ablation.sh`

只 best fusable：

```text
-4,4  # reuse Phase A
none  # new
```

报告：

```text
z_min/z_max
d_min/d_max
E2E
accepted/rollback
```

不做更多 clamp grid search。

---

## Phase D：不使用 Teacher-CoT 的 Calibration Source 消融

### Task 16: 四种 calibration source

**Script:**
- Modify existing: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_calib_ablation.sh`

当前阶段**只允许**：

```text
s1k_original
s1k_question
wikitext2
c4
```

当前脚本虽然 `SOURCES` 已经不包含 Teacher-CoT，但仍有两个错误必须修：

1. 它会重新训练 `s1k_original`，而本计划要求直接复用 Phase A 的两个 `s1k_original` winner baseline。
2. `run_map.json` 仍把 Phase A baseline 错标为 `fusable_s1k_teacher_cot/online_s1k_teacher_cot`。

固定改法：脚本新增训练 source 只保留 `s1k_question/wikitext2/c4`；Phase A baseline 映射名改为 `fusable_s1k_original/online_s1k_original`；同时加入 `E0_native_nvfp4` 路径。Python 对 `s1k_teacher_cot` 的 source support 不删除。

对：

```text
best fusable
best online
```

共比较 4 source × 2 structure。

其中 `s1k_original` 两个 baseline 复用 Phase A，因此新增 6 个 training run：

```text
s1k_question × 2
wikitext2 × 2
c4 × 2
```

**cache 规则：先预构建，再训练，不让正式训练进程承担 cache 构建职责。**

在启动任何 Phase D 训练前，脚本顺序调用三次 `e2e_prepare_calib`，source 分别为 `s1k_question/wikitext2/c4`，cache 路径全部通过 `shared_calib_dir_for` 获取；Wiki/C4 额外显式传 `--calib_seqlen 1024`。每次命令完成后调用 `require_shared_calib_cache`。

三个 cache 全部完成后，6 个新增 training run 不再有 producer/consumer 依赖，可直接按 GPU pool 分波次并行。每个训练命令必须显式传自己的 `--calib_source`，不能因为 cache 路径正确就省略 source。

分析固定围绕：

```text
s1k_question vs s1k_original
    比较“question-only 原始文本”与“完整 S1K 样本（含原格式/reasoning/answer）”的实用校准效果

s1k_original vs wikitext2/c4
    比较 task/reasoning-oriented 完整样本与 generic text window 的实用校准效果
```

**口径限制必须写进报告：** 四种 source 固定的是 sample 数，不是总 token budget。`s1k_original/question` 自然变长，而 Wiki/C4 是 1024-token windows，因此 source 之间总 token 数可能不同。`prepare_calibration` 输出的 train/val total tokens、max/P50/P95 length 必须进入 calibration 表。当前阶段把它解释为“不同实用 calibration recipe 的比较”，不能严格归因成纯数据域效果；先不为这一项增加 token-budget sampler。

不要讨论 Teacher-generated trajectory，因为当前没有这项实验。

Phase D `summary.json` 直接复用现有 `best_fusable` / `best_online` 完整 row 作为 calibration winner；Finalist 脚本从这两个 row 读取 `calib_source` 和 `run_dir`。不要只传 `best_fusable_preset/best_online_preset`，因为 Phase D 比较的是相同结构下的不同 calibration source，preset 本身无法区分 winner 数据源。

---

# Part IV：Finalist E2E

### Task 17: AIME25 avg@5 只跑 finalist

**Scripts:**
- Modify single-run helper: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/eval/run_final_eval.sh`
- Create orchestrator: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/eval/run_finalists.sh`

Finalist：

1. Native NVFP4。
2. Direct HiF4。
3. Phase A best fusable。
4. Phase A best online。
5. Phase D best fusable calibration（若不是 s1k_original）。
6. Phase D best online calibration（若不是 s1k_original）。

去重后每个 artifact 只跑一次。

全部：

```text
eval_seed=42
AIME25 avg@5
```

当前 `run_final_eval.sh` 会把 `arc,mmlu_pro_300,aime25_avg5` 全部重跑。把它保持为**单模型 helper**，只调用 `--groups aime25_avg5`；ARC/MMLU 直接复用 fast-eval 已有 metrics 文件。单模型接口仍保持 `OUT_DIR VARIANT [ARTIFACT_PATH]`，不要让它承担 winner 选择。

新增 `run_finalists.sh PHASE1_DIR CALIB_ABLATION_DIR` 作为薄 orchestration 层：

1. 从 Phase A summary 取得 best fusable / best online 的 `run_dir`。
2. 从 Phase D summary 的 `best_fusable` / `best_online` 完整 row 取得 calibration winner 的 `run_dir` 与 `calib_source`；只有 source 不是 `s1k_original` 时追加该 run。
3. 固定加入 E0 Native NVFP4 与 E1 Direct HiF4。
4. 按规范化绝对 `run_dir` 去重，不用 hash。
5. 对 baseline 分别调用 `run_final_eval.sh <dir> native_nvfp4` / `direct_hif4`；对转换 run 调用 `run_final_eval.sh <dir> artifact <conversion_state.pt>`。
6. 使用现有 GPU pool 按 wave 并行。

这样 winner 选择、列表去重和多卡调度只存在 orchestrator，单模型评测 helper 保持简单。

---

# Part V：结果汇总

### Task 18: 生成当前阶段正式报告

**Output:**

`Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/ABLATION_NO_TEACHER_COT_REPORT_2026-08-18.md`

主表固定 5 张：

1. **结构消融 E0–E7**。
2. **Fusable DIAG component**。
3. **训练机制**：joint / input mode / loss。
4. **稳定性**：rollback / clamp。
5. **Calibration source + Finalist E2E**：S1K original/question/Wiki/C4 + AIME25。

每张表至少报告：

```text
method/config
calib_source
accepted / rollback / would_rollback
mean/median layer recovery
ARC-Easy
ARC-Challenge
MMLU-Pro300
AIME25（若执行）
vs Native NVFP4 pp drop
```

正文按：

```text
研究问题
实验设置
观察结果
可支持结论
不能过度推断的内容
```

书写。

---

# Part VI：实验复用和并行规则

## 复用规则

```text
Phase A E3 -> Phase B all baseline
Phase A best online -> joint/progressive/delta-loss baseline
Phase A best fusable -> progressive/delta-loss/rollback-on/clamp baseline
Phase A best fusable + best online -> s1k_original calibration baselines
```

相同配置禁止重复训练。

## GPU 并行规则

```text
Phase A:
  default calibration cache ready
  -> E3-E7 stage variants 多卡 wave

Phase B:
  B1-B6 多卡 wave

Phase C:
  各阶段内部独立 variant 多卡 wave

Phase D:
  先用 prepare_calibration 顺序预构建 s1k_question / wikitext2 / c4 三个 cache
  -> 6 个新增 training run 直接按 GPU pool 多卡 wave

Final:
  finalist AIME 多卡 wave
```

不要同时启动两个独立 stage scheduler 抢同一 GPU pool。默认按 Part III 顺序逐阶段启动，但阶段内部尽可能并行。

---

# Part VII：当前计划的预计算量

不考虑 Teacher-CoT 后，正式新增的 36-layer training run 约为：

```text
Phase A: 5
Phase B: 6
C1: 1
C2: 2
C3: 4
C4: 1
C5: 1
Phase D: 6
----------------
总计: 26 个新增 training run
```

E0/E1/E2 不训练。

由于多 GPU wave 调度，墙钟时间由可用卡数显著缩短；但 Cursor 不允许一次性无依赖地提交全部 26 个任务。

---

# Part VIII：执行顺序

```text
Task 1   默认 calibration 改 s1k_original + 增加专用 smoke
Task 2   Teacher-CoT 当前阶段脱钩
Task 3   只新增 prepare_calibration；复用现有 shared cache
Task 4   验收现有 fusable components
Task 5   验收现有 calib_input_mode
Task 6   验收 rollback + 删除死代码
Task 7   验收 eval seed/summary + 统一 E0 引用
Task 8   验收现有 GPU pool
  ↓
正式实验 preflight：
  全专项 pytest
  bash syntax check
  fast s1k_question smoke
  1-layer s1k_original smoke
  prepare s1k_original 128/32 shared cache
  GPU pool dry-run
  ↓
Phase A  E0-E7
  ↓
选 best fusable / best online / best overall
  ↓
Phase B  component
  ↓
Phase C1-C5
  ↓
Phase D  4-source calibration
  ↓
Finalist AIME25 avg@5
  ↓
最终报告
```

---

# Final Acceptance Criteria

1. 默认 E2E 训练不再进入 `s1k_teacher_cot`，但 Python 对该 source 的旧支持未被删除。
2. 原 `run_train_smoke.sh` 继续完成快速 `s1k_question` smoke；新增 `run_s1k_original_smoke.sh` 能以 2 train / 1 val / 1 epoch / layer0 闭环完成。
3. 当前正式实验没有 Teacher autoregressive generation / self-judge / trace policy 依赖。
4. Native NVFP4 block reconstruction target 保持不变，没有因为“暂缓 Teacher-CoT”而误删 teacher block forward。
5. `prepare_calibration` 能在不加载 Native NVFP4 8B model 的情况下预构建四种 non-Teacher cache；每个 cache 有 `stats.json`，shared cache 可被多 run 复用，不使用 hash/checksum。
6. 正式实验前全专项 pytest、shell syntax check、两个 non-Teacher smoke、shared-cache preflight 和 GPU pool dry-run 全部通过。
7. Phase A 完成 E0–E7 并确定 best fusable / best online / best overall。
8. Fusable QKV/VO/GU/UD component 消融完整。
9. joint、progressive input、delta-NMSE、rollback、clamp 均有对应对照。
10. Calibration source 只比较 `s1k_original/s1k_question/wikitext2/c4`，并同时报告各 source 的 total tokens / max / P50 / P95 length，避免把 sample-count 对照误写成等 token-budget 对照。
11. 所有后续 stage 的 `run_map.json` 都引用 Phase A E0，统一得到相对 Native NVFP4 的 pp drop。
12. 所有同阶段独立 run 能通过 GPU pool 按可用卡并行执行，一 GPU 一 run。
13. Finalist 复用已有 ARC-E/C、MMLU-Pro300，只新增 AIME25 avg@5，不重复跑 fast eval。
14. 最终形成 `ABLATION_NO_TEACHER_COT_REPORT_2026-08-18.md`，能够回答当前 NVFP4→HiF4 E2E 转换方案的主要结构和优化问题。
