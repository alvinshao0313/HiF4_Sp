# Residual Hidden Permutation for Block-Sparse MLP Pruning on Qwen3.5-4B

**Technical Report (experiment summary for academic presentation)**  
Model: Qwen3.5-4B · Date: 2026-07-29 · Artifacts: `Block_Sparse/experiments/qwen35_4b_residual_perm_channel_agg/`

---

## Abstract

We study a **global residual-stream channel permutation** \(\pi\) applied before MLP block pruning, with the goal of reducing the pruned-block score loss under a fixed block-sparsity budget. On Qwen3.5-4B with **20% block sparsity** (`fisher_budget_wanda`, block size \(64\times32\)), we ablate the \(\pi_0\) **channel-score aggregation** used to initialize \(\pi\) (no local search; `search_steps=0`). Downstream evaluation uses (i) **lm_eval** 0-shot loglikelihood accuracy on ARC-Easy / ARC-Challenge / MMLU, and (ii) **lighteval** generative **MMLU-Pro** on a 300-question subset (`extractive_match`).

**Findings.** Relative to a prune-only baseline without residual permutation (`rpermnone`), most aggregations hurt lm_eval MMLU. The only aggregation that improves lm_eval MMLU is **raw Wanda summation without L1 normalization** (`raw_wanda`: \(71.93\%\) vs \(71.17\%\)). On generative MMLU-Pro-300, residual-permutation variants outperform `rpermnone` (best: `sparsity_raw_wanda` at \(20.33\%\)), but **all pruned models collapse from a dense baseline of \(71.00\%\)** (\(\approx 50\)–\(55\) absolute points). Metric rankings are **inconsistent** across lm_eval MMLU, ARC, and MMLU-Pro; residual permutation does not yield a uniformly superior recipe under our MLP-only block pruning setup.

---

## 1. Setup

### 1.1 Model and pruning

| Item | Setting |
|------|---------|
| Base model | Qwen3.5-4B (dense BF16) |
| Pruning score | `fisher_budget_wanda` |
| Target block sparsity | \(0.20\) |
| Block geometry | \(64\times32\) (\(H\times W\)) |
| Per-matrix cap | `max_prune_ratio_per_matrix=0.80` |
| Calibration | s1k, \(128\) sequences |
| Intermediate FFN permutation | `wanda_shared` (orthogonal to residual \(\pi\)) |
| Residual permutation | `block_loss` (global \(\pi\) on residual width \(2560\)) |
| Search | `search_steps=0` (use \(\pi_0\) only), unless noted |

Residual \(\pi\) is absorbed into embeddings, RMSNorms, attention projections, MLP, and `lm_head` (same mounting policy as the 27B pipeline). GDN-specific parameters are not remapped.

### 1.2 \(\pi_0\) channel aggregation (ablation factor)

Let \(s_m^{\mathrm{raw}}(c)\) be the Wanda-style residual-axis channel mass of MLP matrix \(m\) on channel \(c\). Aggregated score:

\[
S(c)=\sum_m w_m\cdot \tilde s_m(c).
\]

| Name | \(\tilde s_m\) | Weight \(w_m\) |
|------|----------------|----------------|
| `equal` | L1-normalize \(s_m^{\mathrm{raw}}\) | \(1\) |
| `layer_fisher` | L1 | layer Fisher total |
| `matrix_fisher` | L1 | matrix Fisher total |
| `raw_wanda` | raw | \(1\) |
| `sparsity_raw_wanda` | raw | \(\rho_m=K_m/N_m\) (Fisher prune rate) |
| `density_raw_wanda` | raw | \(1-\rho_m\) |

### 1.3 Evaluation protocols

**A. lm_eval (0-shot, loglikelihood).** Tasks: `arc_easy`, `arc_challenge`, `mmlu`. Reported metric: `acc` (not `acc_norm`).

**B. lighteval MMLU-Pro (300).** Task `mmlu_pro|0`, `max_samples=300`, thinking disabled, `max_new_tokens=32768`, temperature \(0.7\), top-\(p=0.8\), top-\(k=20\), TP\(=1\). Metric: `extractive_match`. Protocol matched to the §12 setting in the 27B wiki report.

**C. WikiText-2 PPL** (optional diagnostic): sequence length \(2048\).

---

## 2. Dense baseline (unpruned)

Source: `Block_Sparse/experiments/qwen35_4b_dense/`.

| Metric | Value |
|--------|------:|
| WikiText-2 PPL (seq=2048) | 9.58 |
| ARC-Easy (`acc`) | 81.40% |
| ARC-Challenge (`acc`) | 51.54% |
| MMLU (`acc`) | 74.37% |
| MMLU-Pro-300 (`extractive_match`) | **71.00%** \(\pm 2.62\%\) |

---

## 3. Results

### 3.1 lm_eval 0-shot after pruning

Common prune recipe: `fisher_budget_wanda`, \(s=0.20\), \(b=64\times32\), s1k, `wanda_shared`.  
Reference prune-only control: **`rpermnone`**.

| Setting | ARC-E (%) | ARC-C (%) | MMLU (%) | \(\Delta\)MMLU vs `rpermnone` |
|---------|----------:|----------:|---------:|------------------------------:|
| Dense (unpruned) | 81.40 | 51.54 | 74.37 | — |
| **`rpermnone`** | **63.51** | 35.84 | 71.17 | 0.00 |
| `equal`, steps=2000 | 62.29 | 35.67 | 68.59 | −2.58 |
| `equal`, steps=0 | 61.36 | 34.98 | 68.61 | −2.56 |
| `layer_fisher`, steps=0 | 62.08 | 35.32 | 67.00 | −4.17 |
| `matrix_fisher`, steps=0 | 62.29 | 34.47 | 64.59 | −6.58 |
| **`raw_wanda`, steps=0** | 62.21 | 34.73 | **71.93** | **+0.76** |
| `sparsity_raw_wanda`, steps=0 | 62.08 | 35.84 | 70.82 | −0.35 |
| `density_raw_wanda`, steps=0 | 62.88 | **36.26** | 70.77 | −0.40 |

Artifacts: `results/lm_eval_0shot/*_arc_mmlu.json`.

### 3.2 lighteval MMLU-Pro (300 questions)

| Setting | `extractive_match` (%) | stderr | vs dense (pt) |
|---------|-----------------------:|-------:|--------------:|
| **Dense** | **71.00** | \(\pm 2.62\) | 0.00 |
| `rpermnone` | 16.00 | \(\pm 2.12\) | −55.00 |
| `raw_wanda` | 18.67 | \(\pm 2.25\) | −52.33 |
| `density_raw_wanda` | 18.67 | \(\pm 2.25\) | −52.33 |
| **`sparsity_raw_wanda`** | **20.33** | \(\pm 2.33\) | −50.67 |

Artifacts: `results/lighteval_mmlu_pro_300/<ckpt>/{results,details}/`.

### 3.3 Residual permutation alone (no pruning)

Applying residual \(\pi_0\) (`block_loss`, steps=0) to the **dense** model changes WikiText-2 PPL from \(9.5806\) to \(9.5820\) (\(\Delta\mathrm{PPL}\approx +0.0014\)). This indicates the permutation itself is nearly lossless under the language-modeling proxy, so downstream degradation after pruning is dominated by **sparsity + score/geometry mismatch**, not by \(\pi\) alone.

Artifact: `results/ppl/ppl_residual_perm_only_qwen35_4b.json`.

---

## 4. Discussion

1. **Proxy mismatch.** \(\pi\) is optimized (or initialized) for MLP block-score geometry, yet the residual stream is shared with attention. Global \(\pi\) can therefore trade MLP-friendly layouts against attention / head coupling—consistent with frequent ARC/MMLU regressions vs `rpermnone`.

2. **Aggregation matters for \(\pi_0\).** L1 + equal weighting and Fisher reweighting after L1 are harmful on lm_eval MMLU. Removing L1 (`raw_wanda`) is the only setting that improves MMLU over prune-only. Sparsity/density weights on raw Wanda change the ranking on generative MMLU-Pro but do not dominate lm_eval MMLU.

3. **Protocol gap.** lm_eval MMLU (loglikelihood MCQ) and lighteval MMLU-Pro (long generative extraction) are not interchangeable. Dense MMLU-Pro is high (\(71\%\)), while pruned models drop to \(\sim 16\)–\(20\%\), suggesting **generation / extraction failure modes** (including looping under large `max_new_tokens`) beyond knowledge removal alone.

4. **Search does not rescue equal aggregation.** For `equal`, `search_steps=2000` vs `0` yields nearly identical MMLU; local channel swaps under the current \(L(\pi)\) proxy are weak.

5. **Practical recommendation (4B, this recipe).** If optimizing lm_eval MMLU under residual perm: prefer **`raw_wanda`**. If reporting generative MMLU-Pro-300 among pruned models: **`sparsity_raw_wanda`** is best in-group, but the absolute gap to dense remains large. **Defaulting to `rpermnone` remains competitive** on lm_eval ARC-E and overall stability.

---

## 5. Limitations

- Single model scale (4B) and a fixed sparsity / block size.
- MMLU-Pro uses a 300-question subset, not the full suite; generative variance and looping are not fully audited here.
- Residual search objective is a block-score surrogate, not end-task loss.
- Attention–residual coupling is not explicitly regularized.

---

## 6. Artifact index

```
Block_Sparse/experiments/
├── qwen35_4b_dense/                          # dense baseline (separate experiment)
└── qwen35_4b_residual_perm_channel_agg/      # this experiment (single copy)
    ├── README.md
    ├── results/
    │   ├── lm_eval_0shot/*_arc_mmlu.json
    │   ├── lighteval_mmlu_pro_300/<setting>/{results,details}/
    │   ├── ppl/ppl_residual_perm_only_qwen35_4b.json
    │   └── run_logs/
    └── reports/
        ├── paper_report_residual_perm_channel_agg.md   # this document
        ├── metrics_tables.json
        └── notes_residual_channel_agg_draft.md
```

Checkpoints: `Block_Sparse/outputs/qwen35_4b_fisher_budget_wanda_s0.20_b64x32_s1k_permwanda_shared_*`.

---

## Acknowledgments / Reproducibility notes

- Python / inference: conda env `hif4`; pruning via `Block_Sparse/scripts/prune_mlp.sh`; lm_eval via `Block_Sparse/tools/eval_lm_eval.py`; MMLU-Pro via repo `main.py` (vLLM + local `3rdparty/lighteval`).
- Dense baselines and pruned residual-perm runs share the same ARC/MMLU and MMLU-Pro-300 protocols described above.
