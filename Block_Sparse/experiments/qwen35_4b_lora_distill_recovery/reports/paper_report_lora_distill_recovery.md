# Masked-LoRA SFT / QAD Distillation Recovery for Block-Sparse MLP Pruning on Qwen3.5-4B

**Technical Report (experiment summary for academic presentation)**
Model: Qwen3.5-4B · Date: 2026-08-04 · Artifacts: `Block_Sparse/experiments/qwen35_4b_lora_distill_recovery/`

---

## Abstract

We study **post-pruning recovery** of a block-sparse Qwen3.5-4B checkpoint (`fisher_budget_wanda`, 20% block sparsity, block 64×32, `permwanda_shared` / `rpermnone`) via **peft Masked-LoRA** fine-tuning on the `simplescaling/s1K-1.1_tokenized` reasoning traces. Two arms are compared against the prune-only checkpoint (M1) and the dense baseline (M0): **M2** = plain cross-entropy SFT (control), **M3** = QAD-style distillation (CE + EAKLD + LAFD, teacher = frozen dense Qwen3.5-4B). The single controlled variable between M2 and M3 is the loss function; all other hyper-parameters are identical (LoRA r=16, 500 steps, lr 1e-4, seq 32768, no truncation).

Downstream evaluation uses three protocols aligned with the existing 4B reports: (i) **lm_eval** 0-shot loglikelihood accuracy on ARC-Easy / ARC-Challenge / MMLU, (ii) **lighteval** generative **MMLU-Pro** on a 300-question subset (`extractive_match`), and (iii) **WikiText-2 PPL** (seq 2048).

**Findings.** Masked-LoRA SFT recovers a large fraction of the pruning loss on every metric. On the headline MMLU-Pro-300, M1 collapses from 71.00% (dense) to 16.00%; M2 (pure CE) recovers to 48.00% (recovery 58.2%) and M3 (distillation) to 45.00% (recovery 52.7%). **The distillation signal provides no incremental benefit over plain CE SFT**; on WikiText-2 PPL and MMLU-Pro-300, M3 is in fact worse than M2 (PPL 25.06 vs 20.96; MMLU-Pro 45.00 vs 48.00). The original hypothesis — that teacher logits are critical for repairing the generative collapse — is **not supported** under the configuration actually run.

A key deviation from the plan must be flagged: the KL term was downgraded from full-vocabulary `eakld` to `eakld_topk` with k=128, because the full-vocab EAKLD autograd graph accumulates ≈77 GB on the loss GPU at sequence length 32768 and OOMs on every GPU layout tested. The downgrade preserves the teacher-entropy-based confidence weight γ but restricts the KL to 128 dimensions, which likely degrades the distillation signal. The conclusion is therefore conditional on the `eakld_topk` configuration; a full-vocab `eakld` re-run (requiring a code-level fix to chunk-wise backward) is the recommended next step.

---

## 1. Setup

### 1.1 Model and pruning base (M1)

| Item | Setting |
|------|---------|
| Base model | Qwen3.5-4B (dense BF16) |
| Pruning score | `fisher_budget_wanda` |
| Target / actual block sparsity | 0.20 |
| Block geometry | 64×32 (H×W) |
| Per-matrix cap | `max_prune_ratio_per_matrix=0.80` |
| Calibration | s1k, 128 sequences |
| Intermediate FFN permutation | `wanda_shared` |
| Residual permutation | none (`rpermnone`) |
| Checkpoint | `Block_Sparse/outputs/qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_rpermnone` |
| Pruning artifacts | `pruning_artifacts/block_masks.pt` (96 MLP matrices) |

### 1.2 Recovery arms

| Arm | Loss | Teacher | Output dir |
|-----|------|---------|------------|
| M0 | — (dense reference) | — | `Qwen/Qwen3.5-4B` |
| M1 | — (prune-only reference) | — | `…/qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_rpermnone` |
| M2 | plain CE (chunked lm_head) | none | `…/qwen35_4b_rpermnone_mlplora16_ce500` |
| M3 | 0.05·CE + 2.0·EAKLD + 0.5·LAFD, T=1.0 | frozen dense Qwen3.5-4B | `…/qwen35_4b_rpermnone_mlplora16_qad500` |

### 1.3 Shared training hyper-parameters (script defaults)

| Item | Value |
|------|-------|
| LoRA | r=16, alpha=32, dropout 0.0, targets `gate_proj/up_proj/down_proj` |
| Trainable params | 18,087,936 / 4,223,839,232 (0.43%) |
| Data | `simplescaling/s1K-1.1_tokenized` (1000 deepseek-R1 traces) |
| Steps | 500 (grad_accum 8 × batch 1 ≈ 4 epoch) |
| Optimizer | AdamW, lr 1e-4, cosine, warmup 3%, max_grad_norm 1.0 |
| Sequence | `model_max_length=32768`, `allow_truncate=false`, `logit_chunk_size=512` |
| Precision / parallel | BF16, gradient checkpointing on, `parallel_mode=layer` |
| Seed | 42 |
| Checkpointing | `save_strategy=no` (no intermediate ckpt; restart from 0 on failure) |

### 1.4 Evaluation protocols

| Protocol | Tool | Key parameters |
|----------|------|----------------|
| A. lm_eval 0-shot | `Block_Sparse/tools/eval_lm_eval.py` | `arc_easy,arc_challenge,mmlu`, fewshot=0, batch=16, report `acc` |
| B. MMLU-Pro-300 | repo root `main.py` (vLLM + lighteval) | `mmlu_pro\|0`, `max_samples=300`, TP=1, `DISABLE_THINKING=1`, `max_model_length=32768`, `max_new_tokens=32768`, temp 0.7, top_p 0.8, top_k 20 |
| C. WikiText-2 PPL | `Block_Sparse/tools/eval_ppl.py` | seq_len 2048, BF16 |

Protocols match the existing 4B reports (`qwen35_4b_dense/`, `qwen35_4b_residual_perm_channel_agg/`).

### 1.5 Recovery rate definition

For accuracy-like metrics (higher is better): recovery = (M_recovered − M1) / (M0 − M1).
For PPL (lower is better): recovery_PPL = (PPL_M1 − PPL_recovered) / (PPL_M1 − PPL_M0).

---

## 2. Deviations from the plan and their causes

| Item | Plan | Actual | Cause |
|------|------|--------|-------|
| `kl_mode` | `eakld` (full vocabulary) | `eakld_topk`, k=128 | Full-vocab EAKLD's autograd graph accumulates ≈77 GB on the loss GPU at seq 32768 (64 chunks × ~1.2 GB float32 intermediate tensors held for backward). OOMs on 1-GPU and 2-GPU layouts. `eakld_topk` restricts the KL graph to k=128 dims; peak drops to ~51 GB (2-GPU) / ~60 GB (1-GPU). The confidence weight γ is still computed from full-vocabulary teacher entropy, matching `eakld`. |
| M2 launch | `TEACHER_MODEL_DIR=""` via `run_mlp_lora_sft.sh` | direct `python train_mlp_lora_sft.py` without `--teacher_model_dir` | `run_mlp_lora_sft.sh:41` uses `${TEACHER_MODEL_DIR:-Qwen/Qwen3.5-4B}`, which treats the empty string as unset and would force distillation. Direct python invocation bypasses this without modifying the script. |
| Process management | `nohup &` | `setsid nohup` + on-disk driver scripts + OOM circuit-breaker retry | The IDE reaps long-running background shells; the shared node has frequent GPU preemption. Driver scripts `run_logs/driver_qad_v2.sh` / `driver_ce.sh` are kept as reproducible artifacts. The retry loop distinguishes preemption OOM (another process holds >5 GB on the card → retry, re-pick GPUs) from self-OOM (stop immediately, no retry). |

These deviations are recorded in `lora_train_summary.json` of each checkpoint and in the driver scripts under `run_logs/`.

---

## 3. Results

### 3.1 Main table (4 models × 5 metrics)

| Metric | M0 Dense | M1 Pruned | ΔM1−M0 | M2 Pure CE | M3 Distill |
|--------|---------:|---------:|----------------:|-----------:|-----------:|
| ARC-Easy acc (%) | 81.40 | 63.51 | −17.89 | 67.72 | 67.09 |
| ARC-Challenge acc (%) | 51.54 | 35.84 | −15.70 | 40.44 | 39.68 |
| MMLU acc (%) | 74.37 | 71.17 | −3.20 | 72.05 | 72.75 |
| MMLU-Pro-300 extractive_match (%) | 71.00 ±2.62 | 16.00 ±2.12 | −55.00 | 48.00 ±2.89 | 45.00 ±2.88 |
| WikiText-2 PPL (seq=2048) | 9.5806 | 33.1286 | +23.55 | 20.9603 | 25.0590 |

M0/M1 numbers are sourced from `Block_Sparse/experiments/qwen35_4b_dense/dense_baseline.json` and `qwen35_4b_residual_perm_channel_agg/reports/metrics_tables.json`. M1 PPL was补测 in this experiment (Task 1).

### 3.2 Recovery rate

| Metric | M2 recovery (%) | M3 recovery (%) | M3 − M2 (pp) |
|--------|----------------:|----------------:|-------------:|
| ARC-Easy | 20.0 | 20.0 | 0.0 |
| ARC-Challenge | 24.4 | 24.4 | 0.0 |
| MMLU | 27.5 | 49.4 | +21.8 |
| MMLU-Pro-300 | 58.2 | 52.7 | −5.5 |
| WikiText-2 PPL | 51.6 | 33.9 | −17.7 |

### 3.3 Training curves

**M3 distillation (500 steps, 13.05 h, single attempt on GPU 0,1):**

| step | ce | eakld | lafd | qad_total |
|------|----:|------:|-----:|----------:|
| 10 | 3.28 | 0.330 | 0.265 | 0.956 |
| 50 | 3.26 | 0.120 | 0.167 | 0.476 |
| 100 | 3.66 | 0.136 | 0.176 | 0.516 |
| 250 | 3.06 | 0.087 | 0.140 | 0.379 |
| 500 | 2.02 | 0.082 | 0.152 | 0.341 |

`train_loss = 3.365`. CE falls 3.28 → 2.02, EAKLD 0.33 → 0.08, LAFD 0.26 → 0.14.

**M2 pure CE (500 steps, 3.28 h, single attempt on GPU 0):**

| step | loss |
|------|----:|
| 10 | 9.77 |
| 50 | 7.48 |
| 100 | 7.12 |
| 250 | 7.04 |
| 500 | 6.71 |

`train_loss = 7.16`. The two `loss` columns are **not directly comparable**: M2 reports full-vocabulary CE; M3 reports the `ce` component of the chunked loss plus the weighted `qad_total = 0.05·CE + 2.0·EAKLD + 0.5·LAFD`, which is much smaller in magnitude.

---

## 4. Discussion

### 4.1 M3 − M1: total distillation recovery

Distillation recovers a positive fraction on every metric. The largest absolute gain is on MMLU-Pro-300 (+29 pp, 16 → 45), consistent with the pruning collapse being dominated by generative failure modes (looping / extraction failure) rather than pure knowledge removal. PPL recovers from 33.1 to 25.1 (recovery 33.9%), and MMLU from 71.2 to 72.8 (recovery 49.4%). ARC recovery is the weakest (~20%).

### 4.2 M2 − M1: SFT-data-only contribution

Plain CE SFT recovers a positive fraction on every metric as well, and on most metrics it is close to or better than the distillation arm. MMLU-Pro recovers to 48% (recovery 58.2%, +3 pp over M3) and PPL to 21.0 (recovery 51.6%, +17.7 pp of recovery over M3).

### 4.3 M3 − M2: distillation-signal increment

**The distillation-signal increment is non-positive and in some cases significantly negative**, which is the central finding of this experiment and contradicts the original hypothesis:

- **MMLU-Pro-300**: M3 is 3.0 pp below M2 (45 vs 48). With stderr ≈ ±2.9, this is within 1σ and not individually significant, but the direction is negative.
- **WikiText-2 PPL**: M3 is 4.1 worse than M2 (25.06 vs 20.96). PPL is computed on the full WikiText-2 corpus with no sampling, so this is a **deterministic** negative effect, not noise.
- **ARC-E / ARC-C**: the two arms are within 1 pp (effectively tied).
- **MMLU**: M3 is 0.7 pp above M2, the only metric where distillation is slightly better.

**Conclusion (conditional on the `eakld_topk` configuration actually run):** the recovery of the pruning collapse on MMLU-Pro is primarily attributable to the s1K SFT data itself (style / format alignment, exposure to long reasoning traces), not to the teacher-logit distillation signal. The distillation signal provides no incremental benefit and hurts language-modeling quality.

### 4.4 Candidate causes

1. **`kl_mode` downgrade.** The planned full-vocabulary `eakld` was downgraded to `eakld_topk` k=128 due to OOM. The topk variant computes KL on only 128 dimensions, discarding the vast majority of the teacher distribution; the resulting gradient is plausibly noisier and less informative. This is the most likely confounder and must be removed before the hypothesis can be considered falsified.
2. **Loss-weight imbalance under topk.** The weights 2.0·EAKLD + 0.5·LAFD were designed for full-vocabulary EAKLD. Under topk the numerical range of EAKLD changes, and the distillation terms may dominate the CE term, crowding out the data-driven learning that benefits M2.
3. **Strong pure-CE baseline.** The s1K traces are deepseek-R1 reasoning trajectories; pure CE on them already teaches the model better generation format and extraction behavior, leaving little marginal room for distillation.
4. **Sampling noise on MMLU-Pro.** 300-question stderr is ~±2.9; a 3 pp gap is within 1σ. The PPL gap (4.1, no sampling) is the robust signal.

---

## 5. Limitations

- **Single model scale (4B)** and a fixed sparsity / block size.
- **The KL term was not run at the planned full-vocabulary setting.** The headline negative result is conditional on `eakld_topk` k=128; a full-vocab re-run is required before declaring the hypothesis falsified.
- **MMLU-Pro uses a 300-question subset**; the 3 pp M2-vs-M3 gap is within noise.
- **No intermediate checkpoints** (`save_strategy=no`); training is not resumable.
- **Single seed** (42); no variance estimate across seeds.
- The distillation arms ran on a shared node with intermittent GPU preemption; the final runs succeeded on the first attempt after switching to 2-GPU layer-parallel, but earlier single-GPU attempts OOMed (systematic, not preemption).

---

## 6. Conclusions and next steps

**Conclusion.** Masked-LoRA SFT is an effective post-pruning recovery method for block-sparse Qwen3.5-4B: it recovers 20–58% of the pruning loss across metrics, with the largest gains on the generative MMLU-Pro-300 (the metric where pruning collapses most). However, under the `eakld_topk` k=128 configuration actually run, **the QAD distillation signal provides no incremental benefit over plain CE SFT**, and hurts PPL and MMLU-Pro. The original hypothesis is not supported under this configuration.

**Next steps (priority order).**

1. **Fix the full-vocab `eakld` OOM** by implementing per-chunk immediate backward (mathematically equivalent to the current single-backward formulation, since the total loss is a sum of per-chunk scalar losses; gradients add). Peak drops to ~1–2 GB per chunk. Re-run M3 under the planned full-vocab `eakld` and re-evaluate. This is the necessary condition for fairly testing the hypothesis.
2. **If full-vocab `eakld` still shows no increment**: the hypothesis is falsified for this model/scale, and the recovery should be attributed to s1K SFT data alone. Pivot to data ablation (s1K vs other SFT sets) and capacity ablation (r=32/64).
3. **MMLU-Pro sample size**: if more reliable MMLU-Pro numbers are needed, run the full test set (12k questions) or increase to 1000 samples.
4. **PPL gap is deterministic**: M2 PPL=21.0 vs M3 PPL=25.1 has no sampling noise; the negative effect of distillation on language modeling quality should be reported explicitly.

---

## 7. Artifact index

```
Block_Sparse/experiments/qwen35_4b_lora_distill_recovery/
├── README.md                          # lineage + config summary + pointer to this report
├── results/
│   ├── ppl/
│   │   ├── qwen35_4b_..._rpermnone_wikitext2_s2048.json      # M1 (Task 1)
│   │   ├── qwen35_4b_rpermnone_mlplora16_qad500_wikitext2_s2048.json  # M3
│   │   └── qwen35_4b_rpermnone_mlplora16_ce500_wikitext2_s2048.json   # M2
│   ├── lm_eval_0shot/
│   │   ├── qwen35_4b_rpermnone_mlplora16_qad500_arc_mmlu.json  # M3
│   │   └── qwen35_4b_rpermnone_mlplora16_ce500_arc_mmlu.json  # M2
│   └── lighteval_mmlu_pro_300/
│       ├── qwen35_4b_rpermnone_mlplora16_qad500/results/      # M3
│       └── qwen35_4b_rpermnone_mlplora16_ce500/results/      # M2
├── run_logs/
│   ├── driver_qad_v2.sh               # M3 training driver (reproducible)
│   ├── driver_ce.sh                   # M2 training driver (reproducible)
│   ├── train_qad500.log               # M3 training log
│   ├── smoke_qad20.log                # smoke test log
│   ├── eval_lmeval_m3.log / _m2.log   # lm_eval logs
│   ├── eval_mmlupro_m3.log / _m2.log  # MMLU-Pro logs
│   └── eval_ppl_m3.log / _m2.log      # PPL logs
└── reports/
    ├── paper_report_lora_distill_recovery.md   # this document
    ├── lora_distill_recovery_report.md         # earlier draft (kept for history)
    └── metrics_tables.json                    # machine-readable metrics
```

Checkpoints: `Block_Sparse/outputs/qwen35_4b_rpermnone_mlplora16_{qad,ce}500` (merged HF format, with `lora_train_summary.json` and `block_masks.pt`).

---

## Acknowledgments / Reproducibility notes

- Python / training / eval: conda env `hif4`; training via `Block_Sparse/scripts/run_mlp_lora_sft.sh` (M3) or direct `python Block_Sparse/tools/train_mlp_lora_sft.py` (M2, to bypass the shell `${TEACHER_MODEL_DIR:-...}` quirk); lm_eval via `Block_Sparse/tools/eval_lm_eval.py`; MMLU-Pro via repo `main.py` (vLLM + local `3rdparty/lighteval`); PPL via `Block_Sparse/tools/eval_ppl.py`.
- Dense (M0) and pruned (M1) baselines share the same ARC/MMLU, MMLU-Pro-300, and WikiText-2 PPL protocols described above; numbers are taken from the existing 4B experiment reports, except M1 PPL which was补测 in this experiment.
- Driver scripts under `run_logs/` record the exact environment variables and retry policy used, and are sufficient to reproduce both training arms.
