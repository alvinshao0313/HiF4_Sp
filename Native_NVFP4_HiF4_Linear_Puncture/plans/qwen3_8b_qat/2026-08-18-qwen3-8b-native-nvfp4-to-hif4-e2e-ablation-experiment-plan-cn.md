# Qwen3-8B Native NVFP4→HiF4 E2E DIAG/R64 消融实验执行计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 `e2e_diag_reconstruction` 端到端框架上，以最少新增代码补齐必要的消融开关，并按严格的阶段顺序完成结构、DIAG 自由度、训练策略、校准数据和稳定性消融，最终形成可直接用于技术报告的 E2E 对比结果。

**Architecture:** 不新建第二套训练/评测 pipeline。所有消融继续复用现有 `core/data/training/evaluation/cli/scripts` 结构；新增能力仅通过 `core/config.py` 的显式配置进入现有训练流程。所有正式实验按阶段执行，前一阶段的最佳结构和已有 run 必须复用，禁止全笛卡尔积和无意义重复训练。

**Tech Stack:** Python、PyTorch、Transformers、现有 HiF4 fake-QDQ oracle、lm_eval、lighteval、bash；所有 Python/测试/训练命令使用 `hif4` conda 环境。

**Spec:** `Native_NVFP4_HiF4_Linear_Puncture/plans/2026-08-17-qwen3-8b-native-nvfp4-to-hif4-e2e-diag-r64-layerwise-reconstruction-plan-cn.md`

## Global Constraints

- 不修改 `Native_NVFP4_HiF4_Linear_Puncture` 之外的算法代码；评测只复用仓库已有 lm_eval/lighteval 能力。
- 不改变现有 HiF4 quantizer、native H16、R64 数学、4 个 fusable DIAG 定义、STE、progressive layerwise reconstruction 的主方案语义。
- 正式比较默认固定：`calib_nsamples=128`、`calib_val_nsamples=32`、`calib_seed=42`、`diag_lr=5e-3`、`diag_epochs=20`、`diag_scheduler=cosine`、`diag_log2_clamp=-4,4`、`diag_batch_size=4`、`recon_loss=block_delta_nmse`、`calib_input_mode=progressive_student`、`layer_rollback=on`。
- 默认 calibration 为 `s1k_teacher_cot + teacher_trace_policy=all`；结构/训练策略消融必须使用完全同一份 calibration token 序列。
- 增加共享 calibration cache；禁止每个结构重新生成同一批 Teacher CoT。
- 同一消融组必须使用相同 `diag_batch_size`。若 batch=4 因显存无法完成，整组统一改为相同较小 batch 后重新执行；禁止只改单个 variant。
- Fast E2E 固定：ARC-Easy、ARC-Challenge、MMLU-Pro 300；AIME25 avg@5 只在最终 finalist 上运行。
- 评测增加固定 `eval_seed=42`；所有对比方法使用相同 eval seed。
- 不做 calibration/test 数据污染过滤。
- 不新增 SHA/hash/checksum/tensor fingerprint/完整性扫描。
- 不加入 silent OOM fallback、自动截断 S1K、自动切换 calibration source。
- shell 脚本只负责 preset、路径和 Python CLI 调用；不在 shell 中实现模型算法或解析 tensor。
- 不新增大一统 `run_all_ablation.sh`；每个阶段有独立脚本，Cursor 按依赖关系执行。
- 增加 GPU pool 自动检测与阶段内并行调度：一张 GPU 同时只运行一个完整训练/评测 run；最多并发数默认等于检测到的可用 GPU 数，允许通过 `GPU_POOL` / `MAX_PARALLEL_JOBS` 显式收紧。
- 并行只允许发生在依赖已经满足的实验之间；Phase A 的结构选择必须先完成。默认由每个阶段脚本在自己内部并行多个独立 variant，阶段脚本本身仍按本计划顺序启动，避免多个独立调度器同时抢同一张 GPU；Phase F 必须等待 A/D/E 的 finalist 信息完成后再启动。
- 共享 calibration cache 同一目录只允许一个 producer 写入；其它并行 run 必须等该 cache 完整生成后再作为 consumer 读取，禁止并发写同一 cache 目录。
- 当前工作区已有大量未提交改动；本计划不要求 Cursor 创建 git commit。

---

## 1. 当前能力与本计划只需要补齐的能力

现有代码已经直接支持：

```text
结构：fusable / online
R64：off / on
online 顺序：diag_then_rot / rot_then_diag
训练粒度：layer_joint / linear_independent
loss：block_delta_nmse / block_output_nmse / mse
calibration：s1k_teacher_cot / s1k_original / s1k_question / wikitext2 / c4
Teacher trace：all / regenerate_correct / replace_question_correct
clamp：-4,4 / none / 任意 MIN,MAX
diag_lr / epochs / scheduler
```

现有正式消融前只补以下四项；不要顺手新增其它框架：

1. `--fusable_diag_components {all,qkv,vo,gu,ud,attn,mlp}`：选择性训练 fusable 四组 DIAG。
2. `--calib_input_mode {progressive_student,teacher}`：比较 progressive student input 与纯 native teacher input。
3. `--layer_rollback {on,off}`：仅用于 rollback 必要性消融，默认 `on`。
4. `--calib_cache_dir PATH`：不同 run 共用同一份 calibration cache。

评测侧再补一个实验控制参数：

```text
--eval_seed INT=42
```

---

### Task 1: 先确认当前主实现仍然闭环，不直接开始正式消融

**Files:**
- Read only unless test exposes a real bug: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/**`
- Read: `Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/smoke_impl_20260818T024039Z/summary.json`
- Read: `Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/teacher_cot_smoke_20260818T025405Z/summary.json`

- [ ] **Step 1: 跑现有专项测试**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction -q
```

Expected: 全部 PASS。若失败，只修复真实回归问题；不要为消融先改数学语义。

- [ ] **Step 2: 跑现有两层训练 smoke**

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_train_smoke.sh
```

验收：layer0/1 均产生 `metrics.json`、`best_diag.pt`、`train_log.jsonl`，并产生 compact artifact；fusable fold gate 通过。

- [ ] **Step 3: 跑现有 Teacher-CoT smoke**

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_teacher_cot_smoke.sh
```

验收：question token mask=0、generated token mask=1、动态 padding、Teacher self-judge 和缓存均正常。

- [ ] **Step 4: 只有 Task 1 全通过后进入消融开关实现**

不要在 smoke 失败时启动 36 层正式实验。

---

### Task 2: 增加共享 calibration cache，确保所有结构看到完全相同的数据

**Files:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/config.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/cli/train.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/common.sh`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_calibration.py`

**Interfaces:**
- Add config field: `calib_cache_dir: str = ""`
- Add CLI: `--calib_cache_dir PATH`
- Empty path preserves current behavior: cache under current run `output_dir/calibration/`.
- Non-empty path means `build_or_load_calibration()` reads/writes `<calib_cache_dir>/calibration/`.

- [ ] **Step 1: 写 config 单测**

验证默认 `calib_cache_dir=""`，显式路径能进入 `E2ETrainConfig.to_dict()`，不增加 hash/checksum。

- [ ] **Step 2: 修改训练入口的 calibration root**

固定逻辑：

```python
calib_root = Path(cfg.calib_cache_dir) if cfg.calib_cache_dir else output_dir
train_samples, val_samples = build_or_load_calibration(
    cfg, tokenizer, model, calib_root
)
```

不要复制 calibration 文件到每个 run；run 的 `config.json` 只记录共享路径。

- [ ] **Step 3: `scripts/common.sh` 增加共享缓存路径函数**

只构造可读目录名，不做 hash。固定约定：

```text
${RESULTS_ROOT}/shared_calibration/
  s1k_teacher_cot_all_n128_v32_seed42/
  s1k_original_n128_v32_seed42/
  s1k_question_n128_v32_seed42/
  wikitext2_n128_v32_seed42_len1024/
  c4_n128_v32_seed42_len1024/
  s1k_teacher_cot_regenerate_correct_n128_v32_seed42/
  s1k_teacher_cot_replace_question_correct_n128_v32_seed42/
```

- [ ] **Step 4: 用一个极小 cache 做复用测试**

第一次构建 2 train + 1 val；第二个不同 `output_dir` 指向同一 `calib_cache_dir`，确认直接加载相同 sample IDs/input_ids/loss_mask，不重新调用 Teacher generation。

- [ ] **Step 5: 跑测试**

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_calibration.py -q
```

---

### Task 3: 增加 fusable 四组 DIAG 选择性启用开关

**Files:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/config.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/semantic_hif4.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/artifact.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_semantic_hif4.py`

**Interfaces:**

```text
--fusable_diag_components {all,qkv,vo,gu,ud,attn,mlp}=all
```

固定映射：

```python
all  -> {qkv, vo, gu, ud}
qkv  -> {qkv}
vo   -> {vo}
gu   -> {gu}
ud   -> {ud}
attn -> {qkv, vo}
mlp  -> {gu, ud}
```

- [ ] **Step 1: config 校验规则写死**

`diag_mode=online` 时只允许 `fusable_diag_components=all`；其它值直接报错，避免产生无意义配置。

- [ ] **Step 2: `LayerDiagState` 增加显式启用函数**

例如：

```python
def configure_fusable_components(self, preset: str) -> None:
    active_map = {
        "all": {"qkv", "vo", "gu", "ud"},
        "qkv": {"qkv"},
        "vo": {"vo"},
        "gu": {"gu"},
        "ud": {"ud"},
        "attn": {"qkv", "vo"},
        "mlp": {"gu", "ud"},
    }
    active = active_map[preset]
    self.z_qkv.requires_grad_("qkv" in active)
    self.z_vo.requires_grad_("vo" in active)
    self.z_gu.requires_grad_("gu" in active)
    self.z_ud.requires_grad_("ud" in active)
```

未启用参数必须保持初始化 `z=0`，即 `D=1`。不要删除参数、不要为每种组合创建新的 State 类。

- [ ] **Step 3: 模型 upgrade 时只调用一次 component 配置**

trainer 仍然使用：

```python
params = [p for p in layer.diag_state.parameters() if p.requires_grad]
```

不要在 train step 中反复切 `requires_grad`。

- [ ] **Step 4: artifact/manifest 仅额外记录 preset 名称**

运行时 fold 仍只依赖最终 z；inactive z 为 0。因此不要改变 fold 数学，也不需要 schema hash。

- [ ] **Step 5: 单测 7 个 preset**

每个 preset 检查 trainable z 名称和 inactive z=0；`all` 必须与当前默认行为完全一致。

- [ ] **Step 6: 跑测试**

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_config.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_semantic_hif4.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_fold_artifact.py -q
```

---

### Task 4: 增加 `progressive_student` vs `teacher` calibration input 消融

**Files:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/config.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/training/layer_runtime.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/cli/train.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/artifact.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_layer_runtime.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py`

**Interfaces:**

```text
--calib_input_mode {progressive_student,teacher}=progressive_student
```

严格语义：

```text
progressive_student:
X_0 相同；训练完 layer l 后，用最终 accepted/rollback HiF4 student 生成 X_{l+1}。

teacher:
X_0 相同；在训练/修改 layer l 之前，用 native NVFP4 当前层在 X_l 上生成 X_{l+1}^{N}；
当前 layer 的 Teacher 与 Student 都仍然吃同一个 X_l；训练完成后下一层使用预先缓存的 X_{l+1}^{N}，不使用 student 输出。
```

- [ ] **Step 1: 增加 `propagate_native_layer` 函数**

实现位置放在 `training/layer_runtime.py`，函数签名固定为：

```python
def propagate_native_layer(
    *,
    model: nn.Module,
    layer: nn.Module,
    samples: list[CalibrationSample],
    collator: DynamicCalibrationCollator,
    x_cache: ProgressiveHiddenCache,
    device: torch.device,
    batch_size: int,
) -> ProgressiveHiddenCache:
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.semantic_hif4 import (
        set_layer_runtime_mode,
    )

    set_layer_runtime_mode(layer, "native_nvfp4")
    layer.eval()
    nxt = ProgressiveHiddenCache()
    for batch in build_validation_batches(samples, batch_size):
        packed = collator(batch)
        prepared = capture_qwen3_pre_layer_call(
            model,
            packed["input_ids"].to(device),
            packed["attention_mask"].to(device),
        )
        sample_ids = [sample.sample_id for sample in batch]
        hidden, _ = x_cache.assemble(sample_ids, device)
        with torch.no_grad():
            y = run_decoder_layer(layer, hidden, prepared)
        for i, sample in enumerate(batch):
            n = int(packed["lengths"][i].item())
            nxt.store(sample.sample_id, y[i, :n], n)
    return nxt
```

函数只负责：当前 layer 设为 `native_nvfp4` → `torch.no_grad()` forward → 按真实长度写入新的 CPU BF16 `ProgressiveHiddenCache`。实现时复用现有 layer-call preparation/动态 batching 辅助函数，不复制第二套 Qwen3 layer forward。

- [ ] **Step 2: 修改主训练循环顺序**

当 `calib_input_mode=teacher`：

```text
进入 layer l
先从当前 X_l 生成 native_next_cache
再训练 layer l
再做 rollback/fold/freeze
下一层 x_cache = native_next_cache
```

当 `progressive_student`：保持现有逻辑不变。

- [ ] **Step 3: 修正 `start_layer>0` replay**

`teacher` 模式 prefix replay 必须按以下顺序：

```text
for prefix layer i:
    用仍处于 native 状态的 layer i 生成 teacher X_{i+1}
    再加载该 layer 已保存的 student DIAG artifact 并 fold/freeze，恢复最终模型状态
    x_cache = teacher X_{i+1}
```

禁止先 fold 再尝试把同一个 layer 切回 native。

`progressive_student` 继续使用现有 student artifact propagation。

- [ ] **Step 4: 加单测**

构造两层 toy runtime：第一层 student 输出故意与 native 不同；断言 progressive 模式第二层收到 student output，而 teacher 模式第二层收到 native output。两种模式当前层 Teacher/Student 输入必须相同。

- [ ] **Step 5: artifact/config 记录 `calib_input_mode`**

用于实验审计和 resume 核心配置比较；不影响最终 E2E runtime。

- [ ] **Step 6: 跑 input-mode 相关测试**

```bash
conda run -n hif4 pytest \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_layer_runtime.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py -q
```

---

### Task 5: 增加 rollback ON/OFF 消融，同时保留“本来是否应该 rollback”的统计

**Files:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/config.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/training/trainer.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/core/artifact.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py`

**Interfaces:**

```text
--layer_rollback {on,off}=on
```

- [ ] **Step 1: 统一 joint / linear-independent 的判断语义**

固定：

```python
would_rollback = best_val_loss >= identity_val_loss
rollback_applied = (cfg.layer_rollback == "on") and would_rollback
accepted_vs_identity = not would_rollback
```

若 `rollback_applied=True`，最终 z 清零；否则保留 best z，即使它比 identity 差。

- [ ] **Step 2: metrics 字段固定**

每层至少新增：

```text
rollback_enabled
would_rollback
rollback_applied
accepted_vs_identity
```

保留旧字段兼容：

```text
accepted = accepted_vs_identity
rollback = rollback_applied
```

- [ ] **Step 3: rollback off 单测**

构造训练后 validation 变差的 layer：`off` 时 z 不清零、`would_rollback=True`、`rollback=False`；`on` 时保持当前回滚行为。

- [ ] **Step 4: artifact 记录 `layer_rollback`**

仅用于复现实验；runtime 仍只读取最终保存 z。

- [ ] **Step 5: 跑 rollback 相关测试**

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_trainer.py -q
```

---

### Task 6: 固定评测随机种子并扩展消融结果汇总字段

**Files:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/cli/evaluate.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/evaluation/runner.py`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/evaluation/summarize.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_evaluation.py`

**Interfaces:**

```text
--eval_seed INT=42
```

- [ ] **Step 1: evaluation 入口固定 seed**

在每次 `run_eval_groups()` 开始时统一设置 `random.seed(eval_seed)`、`torch.manual_seed(eval_seed)`、`torch.cuda.manual_seed_all(eval_seed)`。不要为了这一项新增 NumPy 依赖，也不要改变当前 MMLU-Pro/AIME generation 参数。

- [ ] **Step 2: eval summary 保存 seed**

`eval/eval_summary.json` 记录 `eval_seed`。

- [ ] **Step 3: `collect_run_row()` 增加消融配置字段**

至少读取：

```text
diag_train_scope
recon_loss
diag_log2_clamp
fusable_diag_components
calib_input_mode
layer_rollback
diag_lr
calib_source
teacher_trace_policy
```

- [ ] **Step 4: layer_stats 增加分析字段**

从 `summary.json` 汇总：

```text
accepted
rollback
would_rollback_count
mean_recovery
median_recovery
```

不要从日志推断新的复杂综合分数。

- [ ] **Step 5: 最佳结构选择规则保持固定**

```text
1. MMLU-Pro 300 高者优先
2. 完全并列时 ARC-Challenge 高者优先
3. 仍并列时 ARC-Easy 高者优先
```

`summarize_runs()` 额外输出规范化结构名 `structure_preset`，取值固定为：`native_nvfp4`、`direct_hif4`、`r64_only`、`fusable`、`fusable_r64`、`online`、`online_diag_then_r64`、`online_r64_then_diag`。E0/E1/E2 由 method 名固定映射到前三个 baseline preset；E3–E7 根据 `diag_mode/use_r64/rot_order` 映射。Phase A summary 顶层固定保存 `best_fusable_preset`、`best_online_preset`、`best_overall_preset`：前两者分别只在 E3/E4 与 E5/E6/E7 内选择，`best_overall_preset` 再对这两个 winner 使用同一 `MMLU-Pro300 -> ARC-C -> ARC-E` 规则选择。后续训练脚本只接收五个可训练 preset 名称，不接收任意自由 flags。

- [ ] **Step 6: 跑 evaluation/summarize 单测**

`test_evaluation.py` 至少验证：相同 `eval_seed` 设置相同 Torch 随机序列；`structure_preset` 映射无歧义；构造固定分数的 rows 后，`best_fusable_preset/best_online_preset/best_overall_preset` 严格按既定三级规则选择。

```bash
conda run -n hif4 pytest Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_evaluation.py -q
```

---

### Task 7: 整理脚本公共结构 preset，并新增各阶段消融脚本

**Files:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/common.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/utils/gpu_pool.sh`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_phase1_matrix.sh`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_calib_ablation.sh`
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_teacher_policy_ablation.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_fusable_component_ablation.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_train_scope_ablation.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_input_mode_ablation.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_loss_ablation.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_rollback_ablation.sh`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_clamp_ablation.sh`

**Interfaces:**

`common.sh` 增加唯一 structure preset 映射，使用 bash array，不使用字符串 flags 的二次 word splitting。函数固定写成：

```bash
set_structure_args() {
  local preset="$1"
  STRUCTURE_ARGS=()
  case "${preset}" in
    fusable)
      STRUCTURE_ARGS=(--diag_mode fusable)
      ;;
    fusable_r64)
      STRUCTURE_ARGS=(--diag_mode fusable --use_r64)
      ;;
    online)
      STRUCTURE_ARGS=(--diag_mode online)
      ;;
    online_diag_then_r64)
      STRUCTURE_ARGS=(--diag_mode online --use_r64 --rot_order diag_then_rot)
      ;;
    online_r64_then_diag)
      STRUCTURE_ARGS=(--diag_mode online --use_r64 --rot_order rot_then_diag)
      ;;
    *)
      echo "unknown structure preset: ${preset}" >&2
      return 2
      ;;
  esac
}

phase1_run_name() {
  case "$1" in
    fusable) echo "E3_fusable" ;;
    fusable_r64) echo "E4_fusable_r64" ;;
    online) echo "E5_online" ;;
    online_diag_then_r64) echo "E6_online_diag_then_r64" ;;
    online_r64_then_diag) echo "E7_online_r64_then_diag" ;;
    *) return 2 ;;
  esac
}
```

后续脚本的位置参数固定，不允许再用 `BEST_FLAGS` 这种自由字符串：

```text
run_fusable_component_ablation.sh PHASE1_DIR
run_train_scope_ablation.sh       PHASE1_DIR BEST_ONLINE_PRESET
run_input_mode_ablation.sh        PHASE1_DIR BEST_FUSABLE_PRESET BEST_ONLINE_PRESET
run_loss_ablation.sh              PHASE1_DIR BEST_FUSABLE_PRESET BEST_ONLINE_PRESET
run_rollback_ablation.sh          PHASE1_DIR BEST_FUSABLE_PRESET
run_clamp_ablation.sh             PHASE1_DIR BEST_FUSABLE_PRESET
run_calib_ablation.sh             PHASE1_DIR BEST_FUSABLE_PRESET BEST_ONLINE_PRESET
run_teacher_policy_ablation.sh    PHASE1_DIR BEST_OVERALL_PRESET
```

这些 preset 必须逐字来自 Phase A `summary.json` 的 `best_*_preset` 字段。baseline run 路径统一用 `PHASE1_DIR/$(phase1_run_name PRESET)` 获取。

- [ ] **Step 1: 修改现有 phase1 script 使用公共 preset**

避免 E3–E7 在多个脚本重复手写 flags。

- [ ] **Step 2: 每个 stage 脚本必须完成“训练 → fast eval → run_map → summary”闭环**

目录名固定采用：

```text
phase1_${STAMP}/
fusable_components_${STAMP}/
train_scope_${STAMP}/
input_mode_${STAMP}/
loss_${STAMP}/
rollback_${STAMP}/
clamp_${STAMP}/
calib_ablation_${STAMP}/
teacher_policy_${STAMP}/
```

每个目录内部保存本阶段各 variant 子目录，以及统一的 `run_map.json`、`summary.json`、`report.md`。脚本只写 method→run_dir 的 JSON 映射，然后调用现有 `cli.summarize`；结果解析仍在 Python。

- [ ] **Step 3: 增加轻量 GPU pool 自动检测，不把调度逻辑塞进训练 Python**

新增 `scripts/utils/gpu_pool.sh`，只负责资源检测和 shell job 调度。固定环境变量：

```text
GPU_POOL=""                 # 可选；例如 "0,2,3"。非空时直接使用该显式卡池，不再自动挑其它 GPU
GPU_MIN_FREE_RATIO=0.90     # auto 模式下至少 90% 显存空闲
GPU_MAX_UTIL=10             # auto 模式下当前 GPU utilization <=10%
MAX_PARALLEL_JOBS=""       # 空时=可用 GPU 数；非空时取 min(该值, 可用 GPU 数)
STAMP
DIAG_BATCH_SIZE=4
```

自动模式只调用一次：

```bash
nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
```

对每行计算 `memory.free / memory.total >= GPU_MIN_FREE_RATIO` 且 `utilization.gpu <= GPU_MAX_UTIL`，满足者加入当前阶段 GPU pool。检测结果必须在阶段开始时打印，例如：

```text
available_gpus=0,2,3
parallel_slots=3
```

若没有可用 GPU，阶段脚本直接退出并报告“no available GPU”；不要 fallback 到 CPU，也不要自动降低 batch size。

`GPU_POOL` 是人工显式覆盖，传入后只检查 ID 能被 `nvidia-smi` 识别，不再按 utilization/free ratio 二次过滤。这样用户已经确认某些卡可用时不会被启发式规则误删。

`gpu_pool.sh` 对外只暴露三个函数，名称固定：

```bash
detect_available_gpus   # stdout: 每行一个 physical GPU id
resolve_parallel_slots  # argv: n_gpus；stdout: 一个正整数 slot 数
wait_gpu_wave           # argv: pid...；任一失败则 return 1
```

`common.sh` 只 `source scripts/utils/gpu_pool.sh`，不要复制检测公式。阶段脚本统一用：

```bash
mapfile -t AVAILABLE_GPUS < <(detect_available_gpus)
PARALLEL_SLOTS="$(resolve_parallel_slots "${#AVAILABLE_GPUS[@]}")"
```

- [ ] **Step 4: 同一阶段使用“按 GPU 数分波次”的简单并行，不实现复杂后台调度器**

每个完整 train/eval run 只绑定一张 GPU：

```bash
CUDA_VISIBLE_DEVICES="${gpu_id}" <one complete command> &
pids+=("$!")
```

一波最多启动 `parallel_slots` 个 run；该波全部 `wait` 完成且 exit code=0 后，再启动下一波。任一 run 失败时，先等待同一波已经启动的其它 run 退出，然后整个阶段返回非 0；不自动重跑、不换卡、不改 batch。

该实现刻意采用 wave scheduling，而不是额外写守护进程、GPU lock service 或任务数据库，保证后期容易维护。

正式实验允许的资源环境变量统一为：

```text
GPU_POOL
GPU_MIN_FREE_RATIO
GPU_MAX_UTIL
MAX_PARALLEL_JOBS
STAMP
DIAG_BATCH_SIZE
```

默认正式 `DIAG_BATCH_SIZE=4`。同一消融比较组不得给不同 method 设置不同 batch size。

- [ ] **Step 5: calibration cache 由 stage 脚本显式传入**

结构、component、scope、input、loss、rollback、clamp 消融全部共用：

```text
shared_calibration/s1k_teacher_cot_all_n128_v32_seed42
```

Calibration source 和 teacher policy 消融按 source/policy 使用对应共享目录。

- [ ] **Step 6: 做 shell 语法检查、GPU pool dry-run 并重新跑两层 smoke**

```bash
bash -n \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/common.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/utils/gpu_pool.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_phase1_matrix.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_fusable_component_ablation.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_train_scope_ablation.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_input_mode_ablation.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_loss_ablation.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_rollback_ablation.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_clamp_ablation.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_calib_ablation.sh \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/ablation/run_teacher_policy_ablation.sh
```

随后执行 GPU pool dry-run：

```bash
source Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/utils/gpu_pool.sh
mapfile -t GPUS < <(detect_available_gpus)
printf 'detected=%s\n' "${GPUS[*]}"
resolve_parallel_slots "${#GPUS[@]}"
```

Expected: 能打印当前可用 GPU 列表和正整数 slot；不启动任何训练进程。然后重新执行 `scripts/smoke/run_train_smoke.sh`，确认新增调度代码没有改变单 run 主路径行为。

---

## 2. 正式消融执行顺序

> Phase A 是唯一硬前置阶段：必须先得到 best fusable / best online / best overall。每个阶段内部的独立 variant 应尽可能使用检测到的多张 GPU 并行；共享 cache producer/consumer 关系除外。默认仍由阶段脚本逐个阶段启动，避免多个独立调度器同时抢卡。

按当前复用设计，若所有阶段全部执行，需要新增约 **30 个 36 层训练 run**：Phase A 5 个、Phase B 6 个、C1 1 个、C2 2 个、C3 4 个、C4 1 个、C5 1 个、Phase D 8 个、Phase E 2 个；E0/E1/E2 只评测不训练。这个数字只用于算力预算。并行上限不是 30，而是当前阶段检测到的 `parallel_slots`；其余 run 排到下一 wave。

---

### Task 8 / Phase A: 核心结构消融 E0–E7

**Script:** `scripts/ablation/run_phase1_matrix.sh`

**固定 calibration:** `s1k_teacher_cot + all`，共享 128/32 cache。

**实验：**

| ID | 配置 | 训练？ | 主要问题 |
|---|---|---:|---|
| E0 | Native NVFP4 | 否 | Teacher/reference |
| E1 | Direct HiF4 | 否 | 直接转换损失 |
| E2 | Direct HiF4 + R64 | 否 | R64 单独收益 |
| E3 | Fusable DIAG | 是 | 可融合 DIAG 收益 |
| E4 | Fusable DIAG→R64 | 是 | fusable 主候选 |
| E5 | Online DIAG | 是 | 最大自由度 DIAG 基线 |
| E6 | Online DIAG→R64 | 是 | Linear 实验当前最强顺序 |
| E7 | Online R64→DIAG | 是 | 顺序消融 |

- [ ] **Step 1: 由 E3 第一次运行创建共享 Teacher-CoT cache**

不新增 standalone calibration generator。`run_phase1_matrix.sh` 必须先运行 E3，并给 E3 传入共享 `--calib_cache_dir`；E3 第一次创建 128/32 Teacher-CoT cache。E4–E7 随后全部指向同一目录并直接加载。若共享 cache 已由此前相同配置 run 创建，则 E3 直接复用。

- [ ] **Step 2: E3 完成后，将 E4–E7 按 GPU pool 并行完成 36 层训练**

E3 是默认 Teacher-CoT cache 的唯一 producer，因此先完成 E3。确认共享 cache 完整存在后，E4/E5/E6/E7 不再有相互依赖，使用 `parallel_slots` 分波次启动：若有 4 张可用 GPU，则四个 run 同时跑；若只有 2 张，则先跑两个、完成后再跑两个。每个 run 必须有 36 个 layer metrics 和 final artifact。

- [ ] **Step 3: E0–E7 全部按 GPU pool 跑 fast eval**

```text
ARC-Easy
ARC-Challenge
MMLU-Pro 300
eval_seed=42
```

- [ ] **Step 4: 自动汇总结构表**

至少计算以下直接比较：

```text
E1 -> E2 : R64-only gain
E1 -> E5 : Online DIAG gain
E5 -> E6 : R64 after DIAG gain
E6 vs E7 : rotation order
E3 vs E5 : fusable freedom cost without R64
E4 vs E6 : fusable freedom cost with R64
```

- [ ] **Step 5: 固定选择 best fusable / best online**

Fusable 只在 E3/E4 中按 `MMLU-Pro300 -> ARC-C -> ARC-E` 选择。
Online 只在 E5/E6/E7 中按同一规则选择。

把选出的 preset 名称写进 `phase1/summary.json` 和 `report.md`；后续阶段严格使用这两个结果，不由 Cursor 自行换结构。

---

### Task 9 / Phase B: Fusable 四组 DIAG 自由度贡献消融

**Script:** `scripts/ablation/run_fusable_component_ablation.sh`

**固定结构:** `fusable`、R64 off。这样只研究四组 D 本身，不让 R64 混入归因。

**实验：**

| ID | `fusable_diag_components` |
|---|---|
| B1 | qkv |
| B2 | vo |
| B3 | gu |
| B4 | ud |
| B5 | attn = qkv+vo |
| B6 | mlp = gu+ud |
| B7 | all |

B7 不重新训练，直接复用 Phase A 的 E3。

- [ ] **Step 1: 只训练 B1–B6**

全部使用与 E3 相同 calibration、loss、progressive input、rollback、batch size、seed。

- [ ] **Step 2: B1–B7 全部 fast eval**

- [ ] **Step 3: 汇总两级结论**

```text
单组贡献：QKV / VO / GU / UD
子模块贡献：Attention-only / MLP-only
全模型：All
```

同时报告每个 variant 的 accepted/rollback 数和 mean/median layer recovery，判断收益来自哪些 Transformer 子结构。

---

### Task 10 / Phase C1: `layer_joint` vs `linear_independent`

**Script:** `scripts/ablation/run_train_scope_ablation.sh`

只在 Phase A 的 **best online preset** 上做，因为 `linear_independent` 只定义于 online。

- [ ] **Step 1: joint baseline 直接复用 Phase A best online run**

不要重新训练。

- [ ] **Step 2: 新跑一个 `--diag_train_scope linear_independent`**

其它参数完全一致。

- [ ] **Step 3: fast eval + 汇总**

主要回答：逐 Linear 局部最优是否弱于完整 Transformer Block 联合优化。

---

### Task 11 / Phase C2: Progressive student input vs Teacher input

**Script:** `scripts/ablation/run_input_mode_ablation.sh`

同时验证 best fusable 与 best online，避免结论只依赖一种 DIAG 部署形态。

- [ ] **Step 1: progressive baseline 复用 Phase A 两个 best run**

- [ ] **Step 2: 新跑两个 teacher-input run**

```text
best fusable + --calib_input_mode teacher
best online  + --calib_input_mode teacher
```

- [ ] **Step 3: fast eval + layer reconstruction 汇总**

重点比较最终 E2E 分数，而不是只看当前层 reconstruction。报告 teacher input 是否因为忽略 upstream HiF4 distribution shift 导致后续层 E2E 较差。

---

### Task 12 / Phase C3: Reconstruction loss 消融

**Script:** `scripts/ablation/run_loss_ablation.sh`

对 best fusable 与 best online 各比较三种 loss：

```text
block_delta_nmse   # baseline，复用 Phase A
block_output_nmse
mse
```

- [ ] **Step 1: baseline 不重跑**

- [ ] **Step 2: 新跑 4 个 run**

```text
best fusable + block_output_nmse
best fusable + mse
best online  + block_output_nmse
best online  + mse
```

- [ ] **Step 3: fast eval + accepted/rollback 统计**

报告 delta-NMSE 是否同时带来更好的层级稳定性和 E2E 精度。

---

### Task 13 / Phase C4: Rollback 必要性消融

**Script:** `scripts/ablation/run_rollback_ablation.sh`

只在 **best fusable** 上做，避免为纯稳定性机制扩大实验量。

- [ ] **Step 1: rollback ON 复用 Phase A best fusable**

- [ ] **Step 2: 新跑 rollback OFF**

```text
--layer_rollback off
```

- [ ] **Step 3: 汇总**

必须同时报告：

```text
would_rollback_count
rollback_applied_count
最终 ARC-E/C、MMLU-Pro300
每层 final_val_loss
```

如果 OFF 允许负收益层进入后续 progressive input，观察误差是否逐层累积。

---

### Task 14 / Phase C5: Clamp 必要性消融

**Script:** `scripts/ablation/run_clamp_ablation.sh`

只在 **best fusable** 上比较：

```text
-4,4   # baseline，复用 Phase A
none   # 新跑
```

- [ ] **Step 1: 新跑 no-clamp**

- [ ] **Step 2: fast eval**

- [ ] **Step 3: 汇总 scale telemetry**

报告 `z_min/z_max`、`d_min/d_max`、E2E 分数、accepted/rollback。该实验只回答 clamp 是否是必要稳定约束，不做多范围网格搜索。

---

### Task 15 / Phase D: Calibration source 消融

**Script:** `scripts/ablation/run_calib_ablation.sh`

只在 Phase A 的 best fusable 和 best online 上跑：

```text
s1k_teacher_cot
s1k_original
s1k_question
wikitext2
c4
```

- [ ] **Step 1: 每种 source 指定唯一 cache producer，避免并发写竞态**

两个结构必须读同一个 source cache。`s1k_teacher_cot` 已由 Phase A 生成。对于 `s1k_original/s1k_question/wikitext2/c4`，固定由 **best fusable** run 作为 producer；四个 source 使用四个不同 cache 目录，因此这四个 producer 可以按 GPU pool 并行。

- [ ] **Step 2: Teacher-CoT baseline 复用 Phase A 两个 best run**

不要重跑默认 source。

- [ ] **Step 3: 四个 producer 完成后，并行启动对应 best online consumer**

这样其余 4 source × 2 structure 仍然得到 8 个 run，但每个 cache 始终只有一个 writer。Producer wave 和 consumer wave 内部都按 `parallel_slots` 多卡执行。

- [ ] **Step 4: 全部 fast eval 按 GPU pool 分波次执行**

- [ ] **Step 5: 按以下三条关系分析**

```text
s1k_question -> s1k_teacher_cot
    是否需要 reasoning trajectory，而不是只有题目

s1k_original -> s1k_teacher_cot
    是否需要匹配当前 Native NVFP4 Teacher 自己的推理分布

WikiText2/C4 -> S1K family
    reasoning-aware calibration 是否优于通用文本
```

- [ ] **Step 6: 分别选 best fusable source / best online source**

仍按 `MMLU-Pro300 -> ARC-C -> ARC-E`，不新造综合分数。

---

### Task 16 / Phase E: Teacher trace correctness policy 消融

**Script:** `scripts/ablation/run_teacher_policy_ablation.sh`

只选 Phase A 中 fast-eval 更高的 **单一 best structure**，source 固定 `s1k_teacher_cot`。

比较：

```text
all
regenerate_correct
replace_question_correct
```

- [ ] **Step 1: `all` 复用对应 Phase A run**

- [ ] **Step 2: 分别建立 regenerate/replace 的共享 Teacher-CoT cache**

两个 policy 使用不同 cache 目录，因此不存在共享写冲突。

- [ ] **Step 3: 新跑 2 个 policy run，并按 GPU pool 并行**

若至少 2 张可用 GPU，则 `regenerate_correct` 与 `replace_question_correct` 同时启动；只有 1 张则顺序运行。

- [ ] **Step 4: fast eval 按 GPU pool 执行**

重点回答：量化校准是否需要“Teacher 必须答对”，还是保留 Teacher 真实错误轨迹反而更匹配部署 distribution。

---

## 3. 不作为核心消融的超参数检查

不要做 `lr × epoch × scheduler × batch` 笛卡尔积。

正式主报告不新增大规模超参实验。只从 Phase A 的现有日志检查：

```text
每层 best_epoch 分布
clamp hit ratio
train/val loss 曲线
```

如果主方案明显无法收敛，先按 bug/实现问题排查；不要直接通过大量超参搜索掩盖算法问题。

---

### Task 17 / Phase F: Finalist AIME25 avg@5 正式评测

**Script:** `scripts/eval/run_final_eval.sh`

Fast eval 完成后只对以下模型跑 AIME25 avg@5：

1. E0 Native NVFP4。
2. E1 Direct HiF4。
3. Phase A best fusable。
4. Phase A best online。
5. 若 Phase D 的 best fusable source 不是 `s1k_teacher_cot`，加入该 source variant。
6. 若 Phase D 的 best online source 不是 `s1k_teacher_cot`，加入该 source variant。
7. 若 Phase E 最佳 teacher policy 不是 `all`，加入该 policy variant。

同一个 artifact 若同时满足多条，只评测一次。

- [ ] **Step 1: 用同一 `eval_seed=42` 按 GPU pool 并行跑 AIME25 avg@5**

Finalist 之间互不依赖，每个 artifact 占一张 GPU，按 `parallel_slots` 分波次执行；同一 artifact 若重复出现在 finalist 列表中只运行一次。

- [ ] **Step 2: 将 AIME 写回已有 run 的 eval 目录**

不要复制 checkpoint。

- [ ] **Step 3: 重新执行统一 summarize**

最终所有 finalist 行同时拥有 ARC-E/C、MMLU-Pro300、AIME25 avg@5 和 vs Native NVFP4 的 pp drop。

---

### Task 18: 生成最终消融报告，不只留下 raw JSON

**Files:**
- Modify: `Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/evaluation/summarize.py`
- Create: `Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/ABLATION_REPORT_2026-08-18.md` only after formal experiments finish

最终报告固定包含 6 张主表：

1. **结构消融 E0–E7**：Direct / R64 / DIAG / 顺序 / Fusable-vs-Online。
2. **Fusable DIAG 组件消融**：QKV / VO / GU / UD / Attention / MLP / All。
3. **训练策略消融**：joint vs independent、progressive vs teacher、3 种 loss。
4. **稳定性消融**：rollback ON/OFF、clamp vs none。
5. **Calibration source 消融**：Teacher CoT / Original CoT / Question / Wiki / C4。
6. **Teacher trace policy + Finalist E2E**：all/regenerate/replace + AIME25。

每张表都必须同时给：

```text
关键配置
accepted / rollback 或 would_rollback
mean/median layer recovery
ARC-Easy
ARC-Challenge
MMLU-Pro 300
AIME25（若该阶段运行）
相对 Native NVFP4 的 pp drop
```

报告正文按“问题 → 对照设置 → 结果 → 可以得出的结论 → 不能过度解释的部分”书写，不只列分数。

---

## 4. 实验结果复用规则：禁止重复训练

Cursor 执行时严格遵守：

```text
Phase A E3 复用为 Phase B all baseline
Phase A best online 复用为 joint / progressive / delta-loss baseline
Phase A best fusable 复用为 progressive / delta-loss / rollback-on / clamp baseline
Phase A 两个 best 的 teacher_cot run 复用为 Calibration source baseline
Phase A overall best 的 teacher_cot/all run 复用为 Teacher policy=all baseline
```

任何已有 run 只要配置完全相同且产物完整，就在 `run_map.json` 中直接引用原路径；不要复制目录，也不要重新训练。

---

## 5. 每个正式 run 的最小完成标准

训练 run 必须存在：

```text
config.json
summary.json
36 个目录 `layers/layer_00/` 到 `layers/layer_35/`
checkpoint/final_model/conversion_state.pt
checkpoint/final_model/manifest.json
```

其中 `summary.json.layers` 必须为 36 条；不存在 NaN/Inf；fusable run 的最终 fold gate 通过。

Fast eval run 必须存在：

```text
eval/arc/metrics.json
eval/mmlu_pro/metrics.json
eval/eval_summary.json
```

Finalist 再增加：

```text
eval/aime25/metrics.json
```

检查只基于这些明确产物和 JSON 字段，不新增 checksum/hash。

---

## 6. 推荐的实际执行顺序

```text
Task 1   当前实现测试 + smoke
  ↓
Task 2   shared calibration cache
Task 3   fusable component 开关
Task 4   calib_input_mode
Task 5   rollback on/off
Task 6   eval seed + summary 字段
Task 7   ablation scripts
  ↓
重新跑专项测试 + 2-layer smoke
  ↓
Phase A  E0-E7 结构消融（E3 cache producer 完成后，E4-E7 多卡并行）
  ↓ 选 best fusable / best online / best overall
Phase B  Fusable 组件（variant 多卡分波次）
  ↓
Phase C1 joint vs independent（新 run 按可用卡执行）
Phase C2 progressive vs teacher（2 个新 run 可并行）
Phase C3 loss（4 个新 run 多卡分波次）
Phase C4 rollback
Phase C5 clamp
  ↓
Phase D  calibration source（4 producer 并行 -> 4 consumer 并行）
  ↓
Phase E  teacher trace policy（2 个新 policy 可并行）
  ↓
Phase F  finalist AIME25 avg@5（finalist 多卡分波次）
  ↓
最终统一报告
```

不要并行跳过 Phase A 的结构选择。多卡并行的粒度是“当前阶段已经就绪的独立 run”，不是跨依赖盲目启动全部 30 个任务。

---

## 7. 最终验收标准

1. 原 `e2e_diag_reconstruction` 测试全部通过，新增三个核心消融开关有明确单测。
2. `fusable_diag_components=all`、`calib_input_mode=progressive_student`、`layer_rollback=on` 与修改前默认行为一致。
3. 同一消融组实际读取完全相同的 shared calibration sample IDs/token tensors。
4. Phase A 得到完整 E0–E7 fast-E2E 表，并唯一确定 best fusable / best online。
5. Phase B 能回答四个 fusable DIAG 分别贡献多少，以及 Attention/MLP 哪侧收益更大。
6. Phase C 能回答 joint、progressive、delta-NMSE、rollback、clamp 是否必要。
7. Phase D 能回答 reasoning-aware calibration 与通用文本的差别。
8. Phase E 能回答错误 Teacher trace 是否应过滤。
9. Finalist 完成 AIME25 avg@5，统一报告相对 Native NVFP4 的精度下降。
10. 全部阶段没有无意义的全笛卡尔积、重复 Teacher-CoT 生成或重复相同配置训练。
11. GPU pool 能在 auto 模式正确列出当前空闲卡，并能通过 `GPU_POOL` 人工指定卡池；同一阶段至少用 2 个轻量 smoke job 验证不同 `CUDA_VISIBLE_DEVICES` 可并行执行且输出目录互不覆盖。
12. 新代码仍保持 `core/data/training/evaluation/cli/scripts` 边界；GPU 调度只存在于 `scripts/utils/gpu_pool.sh`/阶段脚本，不进入训练算法模块，也不引入插件框架、hash/checksum 或大一统脚本。
