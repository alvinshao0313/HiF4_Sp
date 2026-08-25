# Native NVFP4 → HiF4 逐层 DIAG / R64 端到端重建

第一阶段只做 fake HiF4 QDQ 的数值正确性与精度。不导出 packed kernel，也不走 vLLM 真量化。

目标模型：`ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4`（36 层 × 7 个 Linear = 252）。embedding / final norm / lm_head 不新增 HiF4。

## 当前实验阶段

当前正式 ablation **默认使用 `s1k_original`**（128 train / 32 val / seed=42）。

`s1k_teacher_cot` 代码保留，但**暂不进入正式实验矩阵**。Teacher trace smoke / policy ablation 不属于当前计划验收项。不要用 `run_teacher_cot_smoke.sh` 或 `run_teacher_policy_ablation.sh` 作为当前阶段入口。

## 协议要点

- 原生 checkpoint 里每个 Linear 已有在线 H16。本实验的 `rot` 只指新增固定 R64 = `R4⊗R4⊗R4`。
- Teacher 永远是原生 NVFP4 block。第 l 层 Teacher 与 Student 吃同一个 progressive HiF4 输入 `X_l^H`。
- 权重每一步都从固定 NVFP4 反量化 master `W_N` 出发，按当前 DIAG/R64 重新变换再做 HiF4 QDQ。
- 主 loss 默认 `block_delta_nmse`。validation 用真实 QDQ（无 STE）选 best epoch。
- 若 best validation 不严格优于同一结构的 `D=I` baseline，整层 DIAG 回滚为 identity；R64 若开启则保留。

### online

- `diag_then_rot`：`X'=(XH)DR`，`W'=W D^{-1} R`
- `rot_then_diag`：`X'=(XH)RD`，`W'=W R D^{-1}`
- 每层 7 个 Linear 各自独立 DIAG。

### fusable

只允许 `DIAG -> native H16 -> R64`。每层四组：`D_QKV`、`D_VO`（GQA 只在 KV-head 空间学，再重复到 Q-head）、`D_GU`、`D_UD`。

最终 fold：

- `D_QKV` → `input_layernorm.weight`
- `D_GU` → `post_attention_layernorm.weight`
- `D_VO` / `D_UD` 留在 V/Up 输出行缩放与 O/Down 输入逆变换

禁止 `fusable + rot_then_diag`，禁止 `fusable + linear_independent`。

### Calibration

| source | 输入 | loss token |
|---|---|---|
| `s1k_teacher_cot` | question 的 chat prompt + teacher continuation | 只算 generated；question mask=0 |
| `s1k_original` | 原始 `text` | 全部非 padding |
| `s1k_question` | question | 全部问题 token |
| `wikitext2` / `c4` | 固定 window | 全部 window token |

Teacher CoT：`enable_thinking=True`，`temperature=0.6, top_p=0.95, top_k=20`，`max_new_tokens=32768`。Judge 只接受 `CORRECT` / `INCORRECT`。S1K 只做 batch 内动态 padding，不截成 1024。NVFP4 snapshot 若没有 `chat_template`，必须挂上本地官方 `Qwen/Qwen3-8B` 模板；没有该模板就直接 fail，禁止改成 raw tokenize。

## 命令

主训练只覆盖与默认不同的项。默认已是 `fusable + layer_joint + s1k_original + 20 epochs`：

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/train/run_full.sh
```

两层 smoke（`s1k_question`，不生成 CoT）：

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_train_smoke.sh
```

`s1k_original` 单层 smoke：

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_s1k_original_smoke.sh
```

Teacher-CoT smoke 当前阶段不跑：

```bash
bash Native_NVFP4_HiF4_Linear_Puncture/experiments/e2e_diag_reconstruction/scripts/smoke/run_teacher_cot_smoke.sh
```

快速评测（ARC-E/C + MMLU-Pro 300）与 finalist + AIME25：

```bash
bash .../scripts/eval/run_fast_eval.sh OUT_DIR artifact CONVERSION_STATE_PT
bash .../scripts/eval/run_final_eval.sh OUT_DIR artifact CONVERSION_STATE_PT
```

结构消融见 `scripts/ablation/`。E0–E7 先只跑 ARC + MMLU-Pro 300；选 best fusable / best online 的规则是：先比 MMLU-Pro 300，再 ARC-Challenge，再 ARC-Easy。

## 扩展指南

不要加 plugin / registry。按现有入口加一项即可：

- 新 calibration source：只改 `data/calibration.py`（builder + 一个 dispatch）
- 新 reconstruction loss：只改 `training/losses.py`（函数 + 一个 dispatch）
- 新固定 transform：只扩展 `core/transforms.py` 和配置 enum；`fold.py` / `semantic_hif4.py` 必须调用它
- 新 benchmark：只扩展 `evaluation/runner.py` 和对应 `scripts/eval` preset

DIAG/R64 矩阵顺序只有 `core/transforms.py` 一处实现。loss 分子分母只有 `training/losses.py` 一处实现。
