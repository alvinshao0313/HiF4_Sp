# Experiment cleanup record — 2026-09-20

> 2026-09-20 的清理记录。文中的运行状态、容量和保留决定均指当时，后续操作须重新核实；本次文档归类没有再次删除实验产物。参见 [目录保护经验](../../lessons/workspace_protection.md)。

This document records the local experiment cleanup performed in this checkout. It is intended to preserve the provenance of removed folders and checkpoints while keeping active experiments untouched.

## Boundary and protected data

- The progressive error cancellation matrix is still running from `Native_NVFP4_HiF4_Linear_Puncture/results/progressive_error_cancellation/20260920T054500Z`. Its launcher and worker processes were observed at cleanup time; this directory was not modified.
- The staged/modified progressive error cancellation source files were not changed.
- `internal_error_accumulation` remains at the existing S4 review boundary and was not modified.
- The completed `kl_direction_reuse` cleanup manifest and its protected formal/capture/verification roots were not modified.
- Progressive smoke results `progressive_error_cancellation_smoke_20260920` through `v6` were retained. Their protocol hashes differ, so they were not merged by name alone.

## Removed or deduplicated items

### E2E reconstruction temporary duplicate

Removed:

`Native_NVFP4_HiF4_Linear_Puncture/results/e2e_diag_reconstruction/shared_vllm_qwen3_30b/x`

Reason and evidence:

- `x/artifact` and `E3_fusable/artifact` had identical contents for all 61 common files, including the model index and model tensors.
- `x/r64_only` and `E2_r64_only/r64_only` had identical contents for all 61 common files.
- The retained E2/E3 directories contain four additional evaluation parquet files per variant; those were not present in `x` and remain available in the canonical directories.
- No repository reference or running process used the `x` path.

Approximate space reclaimed: 114G.

### Non-equivalent reconstruction resume checkpoints

Removed the four completed resume checkpoints:

- `linear_top_partial/resume.pt`
- `linear_top_mass/resume.pt`
- `group_top_partial/resume.pt`
- `group_top_mass/resume.pt`

Each manifest showed completion of layers 0–47. Final model exports, initialization artifacts, evaluations, and `comparison.json` were retained.

Approximate space reclaimed: 25.7G.

### QAD smoke deduplication

Used `smoke_teacher_cpu` as the canonical copy and removed:

`QAD/.result/smoke_grad_check`

The three files in both directories had identical SHA256 values:

- `s0_scales.pt`: `a64d64e924a486d9a5bee251e98d553ca23ae1e534975345d56a42d9556082d2`
- `s0_scales_meta.json`: `c2ca95c767c36a2151f1c59f2f9983a36f0280604f9e964aa00befd83eaacfc2`
- `wrapped_modules.json`: `e6885f0dfbacf3a105d6db0e95bf07e41d261d5a06a75f7b699b40aa138d2d3f`

Approximate space reclaimed: 829M. The other QAD smoke directories were retained because their scale tensors have different hashes.

### Empty failed QAD attempts

Removed the metadata-only failed-attempt directories:

- `QAD/.result/fit_test_frozen_b_3gpu`
- `QAD/.result/fit_test_frozen_b_4gpu`

Their recorded metadata was summarized here before removal. No active process referenced either directory.

## Retained experiment record

- E2E reconstruction E1–E7 and candidate exports remain under `results/e2e_diag_reconstruction/shared_vllm_qwen3_30b`.
- Non-equivalent final exports and downstream comparison remain under `results/non_equivalent_reconstruction/full48_20260915T084300Z`.
- Current progressive training and all smoke protocol variants remain under `results/progressive_error_cancellation*`.
- Internal-error-accumulation S4 reports, captures, holdout artifacts, objective outputs, and handoffs remain unchanged.
- Long-trajectory stability evidence and its protocol-specific reports remain unchanged.
- HiFloat4 and NVFP4 reports and model-specific artifacts remain unchanged.

## Post-cleanup checks

Post-cleanup verification at 2026-09-20T06:30Z:

1. The progressive matrix still has its launcher and five worker-related processes using `20260920T054500Z`; its output path was unchanged.
2. The retained E2/E3 model indexes and final exports are present.
3. The non-equivalent comparison and all five final export directories are present; no `resume.pt` remains.
4. The removed paths are absent and the canonical QAD smoke output is present.
5. Two broken symlinks were found under `long_trajectory_stability`; both predate this cleanup and point to a missing E1 model-view cache. They were outside the removal targets and were not changed.
6. `/home` reports about 1.7T available after cleanup, compared with about 1.6T before it.

This cleanup did not stop or modify any pre-existing experiment process.

## Progressive smoke retention policy

The raw smoke checkpoints are temporary validation artifacts, not final model results. The v2–v5 directories were complete but used different dataset protocol hashes, so their records are retained while their raw checkpoint directories are removed. The latest v6 smoke remains as the current reference because it contains direct, JVP, and shuffled runs.

| version | direct protocol hash | JVP protocol hash | completed layers | action |
|---|---|---|---|---|
| v2 | `0a6dc4d2...81e065` | `33a69834...f64516` | direct 0–1, JVP 0 | remove raw directory; record retained here |
| v3 | `d082fd7a...dbd188b` | `21e7773c...fff3db` | direct 0–1, JVP 0 | remove raw directory; record retained here |
| v4 | `537ab556...debfebf` | `de45f6e8...7a8334` | direct 0–1, JVP 0 | remove raw directory; record retained here |
| v5 | `465f60ee...3dca40d` | `9174d5fd...df571f` | direct 0–1, JVP 0 | remove raw directory; record retained here |
| v6 | `5516d05a...21cc994` | `42a9c05f...899f2d9` | direct/JVP/shuffled 0–1 | retain latest smoke reference |

This policy removed 8,703,606,212 bytes (about 8.1 GiB) of repeated smoke checkpoints while retaining the protocol differences and completion status needed to interpret the runs.

The protocol-only `progressive_error_cancellation_smoke_20260920` directory was also removed; it contained no checkpoint and its existence is recorded by this cleanup entry.

## QAD state discrepancy

At approximately 2026-09-20T06:53Z, the entire `QAD/.result` directory was absent. The cleanup targets in this record did not include the remaining QAD result directories. No active QAD process, deleted-file handle, or local trash/snapshot copy was found during the follow-up check. No further QAD cleanup or restoration was attempted.

## Mainline reorganization

The current research scope is limited to learnable diagonal transforms, non-equivalent transforms, and optimization objectives. The following source directories were moved under `archive/legacy/` after checking imports from the current Native code:

- `experiments/activation_3d_viz/` → `archive/legacy/native_qwen3_8b/activation_3d_viz/`
- `experiments/h4_block_rotation/` → `archive/legacy/native_qwen3_8b/h4_block_rotation/`
- obsolete files from `experiments/diag_gradient/` → `archive/legacy/native_qwen3_8b/diag_gradient/`
- `HiFloat4/rotation/` → `archive/legacy/hifloat4/rotation/`
- `QAD/` → `archive/legacy/qad/QAD/`
- `ScaleTuning/` → `archive/legacy/qad/ScaleTuning/`

`experiments/diag_gradient/r64_transform.py` and its package initializer stayed at the original path because `e2e_diag_reconstruction` and `non_equivalent_reconstruction` import them. `HiFloat4/hif4_scale_threshold_optimization/src/` and `NVFP4/` also stayed because the current Native and vLLM paths import them. `long_trajectory_stability/` source and results stayed because the optimization-objective code and its default diagnostic entrances still reference them.

The existing user deletion state under `HiFloat4/permutation_optimization/` was preserved; its now-empty directory was removed without restoring any file. The missing `QAD/.result/` state was not changed.

## Cache cleanup and editor exclusions

Removed 228 Python/editor cache directories found before this pass, excluding the active progressive experiment source and result tree. Updated `.vscode/settings.json` to exclude generated results, `.result`, third-party sources, model/data/output/log/checkpoint trees, caches, Git metadata, and `archive/` from file watching, search, Explorer, and Python analysis. The three active research source lines remain visible and analyzable.

The formal progressive matrix was still running during this pass. Its launcher, workers, source files, checkpoints, and result directory were not moved or deleted; the two current `resume.pt` files remain protected.
