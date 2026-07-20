# MLP Shared Wanda Permutation Before Block Pruning Implementation Plan

> **Generated on:** 2026-07-20


> **For agentic workers:** Execute this plan task by task with tests first. Track progress using the checkboxes below.

**Goal:** Add an optional, one-time MLP intermediate-dimension permutation that uses Wanda importance to concentrate important FFN neurons before block scoring and pruning.

**Architecture:** For each Transformer MLP layer, compute one shared permutation over the FFN intermediate dimension from `up_proj`, `gate_proj`, and `down_proj` Wanda neuron scores. Apply it to `up/gate` rows and the matching `down` columns, then execute all existing Fisher, Wanda, mask, and export logic in the permuted coordinate system.

**Tech stack:** Python, PyTorch, Hugging Face model modules, pytest, and the existing `Block_Sparse` infrastructure.

## Global constraints

- Work only inside `Block_Sparse`.
- Run all Python commands in conda environment `hif4`.
- Existing `fisher`, `magnitude`, `random`, and `fisher_budget_wanda` behavior must remain unchanged when permutation is disabled.
- Do not introduce runtime gather/scatter or online permutation.
- Do not permute the residual hidden dimension or attention modules.
- Do not use different permutations for `up_proj` and `gate_proj`.
- Do not unpermute the final pruned checkpoint.
- Compute and apply the permutation once on the dense model, before mask initialization.
- Do not recompute the permutation after pruning begins.
- No learned permutation, clustering, Sinkhorn optimization, local swap search, or post-pruning fine-tuning is included.

---

## 1. Mathematical definition

For one SwiGLU MLP:

\[
h=\operatorname{SiLU}(W_gx)\odot(W_ux),\qquad y=W_dh.
\]

For a permutation matrix \(P\) over the FFN intermediate dimension:

\[
W_u'=PW_u,\qquad W_g'=PW_g,\qquad W_d'=W_dP^\top.
\]

Because SiLU and the Hadamard product are element-wise:

\[
\operatorname{SiLU}(PW_gx)\odot(PW_ux)
=P\left[\operatorname{SiLU}(W_gx)\odot(W_ux)\right].
\]

Therefore:

\[
W_dP^\top P\left[\operatorname{SiLU}(W_gx)\odot(W_ux)\right]=y.
\]

Given an index vector `perm` where new position `k` contains old neuron `perm[k]`, apply:

```python
with torch.no_grad():
    up.weight.copy_(up.weight.detach().index_select(0, perm_up_device))
    gate.weight.copy_(gate.weight.detach().index_select(0, perm_gate_device))
    down.weight.copy_(down.weight.detach().index_select(1, perm_down_device))
```

The implementation must preserve parameter identity and dtype.

## 2. Wanda neuron importance

Reuse `collect_mlp_input_rms()` from `Block_Sparse/block_pruning/wanda_scorer.py`.

For intermediate neuron index \(k\):

### Up projection

\[
s_k^u=\sum_j|W^u_{k,j}|a_j^u.
\]

```python
up_score = up.weight.detach().float().abs().matmul(up_input_rms.float())
```

### Gate projection

\[
s_k^g=\sum_j|W^g_{k,j}|a_j^g.
\]

```python
gate_score = gate.weight.detach().float().abs().matmul(gate_input_rms.float())
```

### Down projection

The FFN neurons are input channels of `down_proj`:

\[
s_k^d=a_k^d\sum_i|W^d_{i,k}|.
\]

```python
down_score = down.weight.detach().float().abs().sum(dim=0) * down_input_rms.float()
```

Normalize each projection independently:

\[
\hat{s}_k^p=\frac{s_k^p}{\sum_t s_t^p},\qquad p\in\{u,g,d\}.
\]

Combine with equal projection contribution:

\[
s_k=\hat{s}_k^u+\hat{s}_k^g+\hat{s}_k^d.
\]

Create a stable descending permutation:

```python
perm = torch.argsort(combined_score, descending=True, stable=True)
```

Ties preserve the original neuron order. The first implementation must not add tunable projection coefficients.

## 3. Required pipeline order

```text
load model
→ collect MLP targets
→ build calibration batches when required
→ collect dense-model RMS statistics once
→ compute one shared permutation per layer
→ apply all permutations once
→ initialize all-one block masks
→ run existing pruning rounds
→ export the permuted-and-pruned checkpoint
```

Rules:

- Mask initialization happens after permutation.
- Fisher scoring happens after permutation.
- Block-Wanda scoring happens after permutation.
- Pre-permutation `down_proj` RMS is used only to build the permutation.
- Round-level Block-Wanda recollects RMS from the permuted model.
- Multi-round pruning never recomputes or reapplies the permutation.
- All saved masks and block coordinates refer to the permuted layout.

## 4. Current reusable code

Reuse:

- `Block_Sparse/block_pruning/wanda_scorer.py`
  - `InputRMSRecord`
  - `collect_mlp_input_rms()`
  - `collect_wanda_block_scores()`
- `Block_Sparse/block_pruning/mlp_registry.py`
  - `MLPLinearTarget`
- `Block_Sparse/scripts/score_and_prune_mlp.py`
  - current `fisher_budget_wanda` and multi-round logic
- `Block_Sparse/block_pruning/serialization.py`
  - current pruning artifact patterns

No permutation implementation currently exists under `Block_Sparse/block_pruning/`.

## 5. File map

### Create

- `Block_Sparse/block_pruning/mlp_permutation.py`
- `Block_Sparse/tests/test_mlp_permutation.py`
- `Block_Sparse/tests/test_permutation_pruning_pipeline.py`

### Modify

- `Block_Sparse/block_pruning/config.py`
- `Block_Sparse/scripts/score_and_prune_mlp.py`
- `Block_Sparse/block_pruning/serialization.py`
- `Block_Sparse/scripts/prune_mlp.sh`
- `Block_Sparse/scripts/run_baselines.sh`
- `Block_Sparse/README.md`

---

### Task 1: Group complete MLP projection triplets

**Files:**
- Create: `Block_Sparse/block_pruning/mlp_permutation.py`
- Create: `Block_Sparse/tests/test_mlp_permutation.py`

**Produces:**

```python
@dataclass(frozen=True)
class MLPProjectionTriplet:
    layer_index: int
    gate: MLPLinearTarget
    up: MLPLinearTarget
    down: MLPLinearTarget
    intermediate_size: int
```

```python
def group_mlp_projection_triplets(
    targets: list[MLPLinearTarget],
) -> list[MLPProjectionTriplet]:
    ...
```

**Validation:**

- exactly one gate, up, and down target per layer;
- gate and up weight shapes are identical;
- gate/up shape is `[d_ff, d_model]`;
- down shape is `[d_model, d_ff]`;
- transpose-compatible dimensions match;
- no target is silently ignored;
- returned triplets are sorted by layer index.

- [ ] Write a failing test with two complete layers and verify grouping order and module identity.
- [ ] Add failing tests for missing projections, duplicate projections, and shape mismatches.
- [ ] Run:

```bash
conda run -n hif4 --no-capture-output python -m pytest Block_Sparse/tests/test_mlp_permutation.py -v
```

Expected: fail because the grouping API does not exist.

- [ ] Implement the dataclass and grouping function.
- [ ] Run focused tests and confirm pass.
- [ ] Commit:

```bash
git add Block_Sparse/block_pruning/mlp_permutation.py Block_Sparse/tests/test_mlp_permutation.py
git commit -m "feat: group mlp projection triplets"
```

---

### Task 2: Implement pure Wanda neuron-score functions

**Files:**
- Modify: `Block_Sparse/block_pruning/mlp_permutation.py`
- Modify: `Block_Sparse/tests/test_mlp_permutation.py`

**Produces:**

```python
def compute_up_or_gate_neuron_score(
    weight: torch.Tensor,
    input_rms: torch.Tensor,
) -> torch.Tensor:
    ...
```

```python
def compute_down_neuron_score(
    weight: torch.Tensor,
    input_rms: torch.Tensor,
) -> torch.Tensor:
    ...
```

```python
def normalize_projection_score(
    score: torch.Tensor,
    layer_index: int,
    projection_type: str,
) -> torch.Tensor:
    ...
```

**Required behavior:**

- validate weight and RMS ranks;
- validate RMS length against the relevant weight dimension;
- compute in float32;
- return CPU float64 vectors;
- reject non-finite or negative values;
- reject a non-finite or non-positive normalization total;
- include layer and projection context in errors;
- do not retain a full element-level importance matrix.

- [ ] Add a hand-computed up/gate score test.
- [ ] Add a hand-computed down score test.
- [ ] Add validation tests for rank mismatch, channel mismatch, non-finite score, negative score, and zero total.
- [ ] Run focused tests and confirm failure.
- [ ] Implement the three functions.
- [ ] Run focused tests and confirm pass.
- [ ] Commit:

```bash
git add Block_Sparse/block_pruning/mlp_permutation.py Block_Sparse/tests/test_mlp_permutation.py
git commit -m "feat: compute mlp wanda neuron scores"
```

---

### Task 3: Build deterministic shared permutation records

**Files:**
- Modify: `Block_Sparse/block_pruning/mlp_permutation.py`
- Modify: `Block_Sparse/tests/test_mlp_permutation.py`

**Produces:**

```python
@dataclass
class MLPIntermediatePermutationRecord:
    layer_index: int
    gate_module_name: str
    up_module_name: str
    down_module_name: str
    intermediate_size: int
    gate_score: torch.Tensor
    up_score: torch.Tensor
    down_score: torch.Tensor
    normalized_gate_score: torch.Tensor
    normalized_up_score: torch.Tensor
    normalized_down_score: torch.Tensor
    combined_score: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
```

```python
def compute_mlp_shared_wanda_permutations(
    triplets: list[MLPProjectionTriplet],
    input_rms_records: dict[str, InputRMSRecord],
) -> dict[int, MLPIntermediatePermutationRecord]:
    ...
```

All score tensors are CPU float64. Permutation tensors are CPU int64.

**Invariants:**

```python
sorted(record.permutation.tolist()) == list(range(record.intermediate_size))
```

```python
record.inverse_permutation[record.permutation] == torch.arange(record.intermediate_size)
```

`record.combined_score[record.permutation]` is monotonically non-increasing.

- [ ] Add a test where up/gate/down favor different neurons and verify equal projection normalization.
- [ ] Add a stable-tie test expecting identity order.
- [ ] Add tests for missing RMS records and mismatched RMS dimensions.
- [ ] Run tests and confirm failure.
- [ ] Implement record construction and stable sorting.
- [ ] Run tests and confirm pass.
- [ ] Commit:

```bash
git add Block_Sparse/block_pruning/mlp_permutation.py Block_Sparse/tests/test_mlp_permutation.py
git commit -m "feat: build shared mlp permutations"
```

---

### Task 4: Apply and undo the strict shared mapping

**Files:**
- Modify: `Block_Sparse/block_pruning/mlp_permutation.py`
- Modify: `Block_Sparse/tests/test_mlp_permutation.py`

**Produces:**

```python
def apply_mlp_intermediate_permutations(
    triplets: list[MLPProjectionTriplet],
    records: dict[int, MLPIntermediatePermutationRecord],
) -> None:
    ...
```

```python
def undo_mlp_intermediate_permutations(
    triplets: list[MLPProjectionTriplet],
    records: dict[int, MLPIntermediatePermutationRecord],
) -> None:
    ...
```

**Requirements:**

- use `torch.no_grad()`;
- preserve `nn.Parameter` identity and weight dtype;
- move index tensors to each weight device independently;
- support sharded models;
- validate permutation length, dtype, range, uniqueness, and layer coverage;
- undo with the inverse permutation on the same axes;
- production export must not call undo.

- [ ] Add index-coded row/column mapping tests.
- [ ] Add apply-then-undo exact identity test.
- [ ] Add parameter-object identity test.
- [ ] Add invalid permutation tests.
- [ ] Run tests and confirm failure.
- [ ] Implement apply and undo.
- [ ] Run tests and confirm pass.
- [ ] Commit:

```bash
git add Block_Sparse/block_pruning/mlp_permutation.py Block_Sparse/tests/test_mlp_permutation.py
git commit -m "feat: apply shared mlp permutations"
```

### Task 5: Prove SwiGLU functional equivalence

**Files:**
- Modify: `Block_Sparse/tests/test_mlp_permutation.py`

Construct a float32 toy SwiGLU MLP without bias:

```python
hidden = torch.nn.functional.silu(gate(x)) * up(x)
out = down(hidden)
```

- [ ] Verify output before and after a nontrivial shared permutation.
- [ ] Verify apply then undo restores the original output.
- [ ] Add negative controls showing output changes when:
  - only up rows are permuted;
  - up/gate rows are permuted without down columns;
  - up and gate use different permutations.
- [ ] Run:

```bash
conda run -n hif4 --no-capture-output python -m pytest Block_Sparse/tests/test_mlp_permutation.py -v
```

Expected: all tests pass.

- [ ] Commit:

```bash
git add Block_Sparse/tests/test_mlp_permutation.py
git commit -m "test: verify mlp permutation equivalence"
```

---

### Task 6: Add configuration and CLI control

**Files:**
- Modify: `Block_Sparse/block_pruning/config.py`
- Modify: `Block_Sparse/scripts/score_and_prune_mlp.py`
- Modify: `Block_Sparse/tests/test_mlp_permutation.py`

Add:

```python
mlp_permutation: str = "none"  # none | wanda_shared
```

Accepted values:

```text
none
wanda_shared
```

Update calibration requirement:

```python
def requires_calibration(self) -> bool:
    return (
        self.score_type in {"fisher", "fisher_budget_wanda"}
        or self.mlp_permutation == "wanda_shared"
    )
```

Do not enable gradient checkpointing solely because permutation is enabled.

Add CLI option:

```text
--mlp_permutation {none,wanda_shared}
```

Default remains `none`.

- [ ] Test valid and invalid modes.
- [ ] Test calibration requirements for magnitude, Fisher, and hybrid configurations.
- [ ] Implement config and parser changes.
- [ ] Run focused tests.
- [ ] Commit:

```bash
git add Block_Sparse/block_pruning/config.py Block_Sparse/scripts/score_and_prune_mlp.py Block_Sparse/tests/test_mlp_permutation.py
git commit -m "feat: configure mlp wanda permutation"
```

---

### Task 7: Integrate permutation before mask initialization

**Files:**
- Modify: `Block_Sparse/scripts/score_and_prune_mlp.py`
- Create: `Block_Sparse/tests/test_permutation_pruning_pipeline.py`

Add:

```python
def prepare_mlp_permutation(
    model,
    batches,
    targets,
    config,
):
    """Compute and apply the one-time dense-model MLP permutation."""
```

Return:

```python
triplets, dense_input_rms_records, permutation_records
```

Required order in `main()`:

```python
model, _text_helper = load_model_and_tokenizer(config)
targets = collect_mlp_linears(model, config.block_height, config.block_width)
batches = build_calibration_batches(...) if config.requires_calibration() else None

permutation_state = None
if config.mlp_permutation == "wanda_shared":
    permutation_state = prepare_mlp_permutation(
        model=model,
        batches=batches,
        targets=targets,
        config=config,
    )

current_masks = initialize_all_one_masks(
    targets,
    config.block_height,
    config.block_width,
)
```

Then run the existing pruning loop without changing its scoring semantics.

**Critical integration rules:**

- `prepare_mlp_permutation()` rejects missing calibration batches.
- It calls `collect_mlp_input_rms()` exactly once.
- It groups triplets, computes records, then applies records in that order.
- It returns records for serialization.
- It does not initialize or transform masks.
- Disabled mode performs no permutation work.

- [ ] Add a monkeypatch call-order test:

```text
collect RMS → compute permutation → apply permutation → initialize masks → score/prune
```

- [ ] Add a three-round test proving compute/apply each run once.
- [ ] Add a disabled-path regression test.
- [ ] Add an error test when permutation is requested without calibration batches.
- [ ] Run focused tests and confirm failure.
- [ ] Implement the helper and integration.
- [ ] Run:

```bash
conda run -n hif4 --no-capture-output python -m pytest Block_Sparse/tests/test_permutation_pruning_pipeline.py Block_Sparse/tests/test_wanda_scorer.py Block_Sparse/tests/test_module_budget_allocator.py -v
```

Expected: all pass.

- [ ] Commit:

```bash
git add Block_Sparse/scripts/score_and_prune_mlp.py Block_Sparse/tests/test_permutation_pruning_pipeline.py
git commit -m "feat: permute mlp before block pruning"
```

---

### Task 8: Add block-concentration diagnostics

**Files:**
- Modify: `Block_Sparse/block_pruning/mlp_permutation.py`
- Modify: `Block_Sparse/tests/test_mlp_permutation.py`

Measure whether sorting concentrates importance into contiguous groups aligned with block axes.

For group size \(g\):

\[
G_q=\sum_{k=qg}^{(q+1)g-1}s_k.
\]

Use:

```text
up/gate group size = block_height
down group size = block_width
```

For original and permuted order, compute:

- number of groups;
- minimum, maximum, mean, and standard deviation of group mass;
- coefficient of variation;
- bottom-20%-group importance ratio;
- top-20%-group importance ratio.

Use:

```python
num_selected = max(1, math.ceil(0.2 * num_groups))
```

These are research diagnostics, not universal pass/fail assertions.

**Produces:**

```python
@dataclass(frozen=True)
class GroupConcentrationStats:
    group_size: int
    num_groups: int
    mass_min: float
    mass_max: float
    mass_mean: float
    mass_std: float
    coefficient_of_variation: float
    bottom20_mass_ratio: float
    top20_mass_ratio: float
```

```python
def compute_group_concentration_stats(
    score: torch.Tensor,
    order: torch.Tensor,
    group_size: int,
) -> GroupConcentrationStats:
    ...
```

- [ ] Add hand-computed group-mass tests.
- [ ] Add coefficient-of-variation and top/bottom ratio tests.
- [ ] Require intermediate size divisible by group size; do not pad.
- [ ] Define coefficient of variation as `std / mean`; reject zero or non-finite mean.
- [ ] Implement pure diagnostic helpers.
- [ ] Run focused tests.
- [ ] Commit:

```bash
git add Block_Sparse/block_pruning/mlp_permutation.py Block_Sparse/tests/test_mlp_permutation.py
git commit -m "feat: measure permutation concentration"
```

---

### Task 9: Serialize permutation artifacts

**Files:**
- Modify: `Block_Sparse/block_pruning/serialization.py`
- Modify: `Block_Sparse/scripts/score_and_prune_mlp.py`
- Modify: `Block_Sparse/tests/test_mlp_permutation.py`

Add:

```python
def save_mlp_permutation_artifacts(
    output_dir,
    records,
    config,
) -> None:
    ...
```

Save once per run:

```text
pruning_artifacts/mlp_permutations.pt
pruning_artifacts/mlp_permutation_report.csv
pruning_artifacts/mlp_permutation_summary.json
```

The PT file stores:

- layer index and exact module names;
- raw up/gate/down scores;
- normalized up/gate/down scores;
- combined score;
- permutation and inverse permutation;
- original and permuted concentration statistics for both relevant group sizes.

CSV fields:

```text
layer_index
gate_module_name
up_module_name
down_module_name
intermediate_size
up_gate_group_size
down_group_size
combined_score_min
combined_score_median
combined_score_mean
combined_score_max
original_up_gate_group_cv
permuted_up_gate_group_cv
original_up_gate_bottom20_mass_ratio
permuted_up_gate_bottom20_mass_ratio
original_up_gate_top20_mass_ratio
permuted_up_gate_top20_mass_ratio
original_down_group_cv
permuted_down_group_cv
original_down_bottom20_mass_ratio
permuted_down_bottom20_mass_ratio
original_down_top20_mass_ratio
permuted_down_top20_mass_ratio
```

Summary metadata includes:

```json
{
  "generated_on": "2026-07-20",
  "permutation_type": "wanda_shared",
  "permutation_axis": "mlp_intermediate_dimension",
  "up_mapping": "row",
  "gate_mapping": "row",
  "down_mapping": "column",
  "projection_normalization": "l1_equal_projection",
  "applied_once_before_pruning": true,
  "export_coordinate_system": "permuted"
}
```

Also add to existing pruning summaries:

```text
mlp_permutation
permutation_applied_before_pruning
```

- [ ] Add temporary-directory tests for exact files, fields, row counts, and stored array types.
- [ ] Verify multi-round pruning writes permutation artifacts only once.
- [ ] Implement serialization and call it immediately after permutation preparation.
- [ ] Run focused tests.
- [ ] Commit:

```bash
git add Block_Sparse/block_pruning/serialization.py Block_Sparse/scripts/score_and_prune_mlp.py Block_Sparse/tests/test_mlp_permutation.py
git commit -m "feat: save mlp permutation artifacts"
```

---

### Task 10: Protect existing pruning behavior

**Files:**
- Modify only tests or permutation-related code if a regression is found.

With `mlp_permutation=none`, verify unchanged behavior for:

- Fisher;
- magnitude;
- random;
- Fisher-budget + Wanda;
- shared and independent up/gate masks;
- one and multiple pruning rounds.

With `mlp_permutation=wanda_shared`, verify:

- masks are initialized after permutation;
- masks are not independently transformed;
- Fisher reference and final Wanda masks use the same permuted coordinates;
- `verify_masks_and_weights()` passes;
- exported zero blocks match saved masks.

Run:

```bash
conda run -n hif4 --no-capture-output python -m pytest Block_Sparse/tests/test_global_allocator.py Block_Sparse/tests/test_module_budget_allocator.py Block_Sparse/tests/test_apply_mask.py Block_Sparse/tests/test_wanda_scorer.py Block_Sparse/tests/test_mlp_permutation.py Block_Sparse/tests/test_permutation_pruning_pipeline.py -v
```

- [ ] Fix only permutation-related regressions.
- [ ] Commit regression fixes separately with precise messages.

---

### Task 11: Update scripts and documentation

**Files:**
- Modify: `Block_Sparse/scripts/prune_mlp.sh`
- Modify: `Block_Sparse/scripts/run_baselines.sh`
- Modify: `Block_Sparse/README.md`

Add a shell setting:

```bash
MLP_PERMUTATION=wanda_shared  # none | wanda_shared
```

Pass the selected value to:

```text
--mlp_permutation [PERMUTATION_MODE]
```

Include the permutation mode in output directory tags so enabled and disabled runs never overwrite each other.

README heading:

```markdown
## MLP 中间维共享重排
```

Document:

- exact equivalence equations;
- shared up/gate row and down column mapping;
- Wanda neuron-score equations;
- projection-wise normalization;
- stable ordering;
- one-time pre-pruning execution;
- distinction from `share_up_gate_mask`;
- no online overhead;
- permanent permuted export;
- artifact files and diagnostics;
- recommended ablations.

- [ ] Update both shell scripts.
- [ ] Update README.
- [ ] Verify shell syntax:

```bash
bash -n Block_Sparse/scripts/prune_mlp.sh
bash -n Block_Sparse/scripts/run_baselines.sh
```

Expected: exit code 0.

- [ ] Commit:

```bash
git add Block_Sparse/scripts/prune_mlp.sh Block_Sparse/scripts/run_baselines.sh Block_Sparse/README.md
git commit -m "docs: add mlp permutation workflow"
```

---

### Task 12: Full verification

- [ ] Run the complete suite:

```bash
conda run -n hif4 --no-capture-output python -m pytest Block_Sparse/tests/ -v
```

Expected: all tests pass.

- [ ] Verify CLI help includes:

```text
--mlp_permutation {none,wanda_shared}
```

- [ ] Review the diff and reject unrelated refactors.
- [ ] Confirm permutation code only mutates MLP weights before mask initialization.
- [ ] Confirm production code never calls the undo helper.
- [ ] Confirm the output checkpoint is intentionally kept in the permuted coordinate system.

---

### Task 13: Dense-model equivalence diagnostic

Before pruning experiments, verify the real model with permutation only.

Create a diagnostic script only if required:

```text
Block_Sparse/scripts/verify_mlp_permutation.py
```

It must:

1. load the model and a small fixed calibration subset;
2. capture logits before permutation;
3. compute and apply shared permutations;
4. capture logits after permutation;
5. report maximum absolute difference, mean absolute difference, and relative L2 difference;
6. reject non-finite outputs;
7. avoid saving model weights.

BF16 computation may produce small numerical differences due to changed accumulation order. Record observed values first instead of inventing an unsupported strict threshold.

- [ ] Run on one or two short samples.
- [ ] Record exact numerical differences.
- [ ] Do not commit generated outputs.

---

### Task 14: Real-model pruning smoke test

Use a low-cost configuration:

```text
score type = fisher_budget_wanda
MLP permutation = wanda_shared
block size = 128
target sparsity = 0.01
one pruning round
small fixed calibration subset
```

Verify:

- permutation artifacts exist;
- hybrid pruning artifacts exist;
- every permutation is bijective;
- shared up/gate/down mapping is consistent;
- final per-matrix counts equal Fisher budgets;
- saved masks match zero blocks;
- exported checkpoint loads through the existing evaluation path.

Do not commit generated checkpoint weights.

---

### Task 15: Controlled ablation experiment

Use identical calibration, pruning, and evaluation settings:

| Variant | Permutation | Budget | Block selection |
|---|---|---|---|
| Magnitude baseline | none | magnitude | magnitude |
| Direct Fisher | none | Fisher | Fisher |
| Hybrid baseline | none | Fisher | Block-Wanda |
| Proposed | wanda_shared | Fisher | Block-Wanda |

Primary comparison:

```text
fisher_budget_wanda + none
versus
fisher_budget_wanda + wanda_shared
```

Also compare magnitude with and without permutation to isolate the packing effect from Fisher budget allocation.

Keep fixed:

- model;
- block size;
- target sparsity;
- per-matrix constraints;
- calibration data and sample count;
- seed;
- pruning rounds;
- evaluation configuration.

Report:

- actual global block sparsity;
- PPL;
- downstream score when used;
- layer-wise and projection-wise sparsity;
- original/permuted group CV;
- top/bottom group importance ratios;
- Fisher/Wanda mask IoU.

Do not claim success from heatmaps alone. Success requires lower degradation at identical sparsity or consistently lower importance mass in pruned blocks.

---

### Task 16: Calibration stability experiment

Evaluate at least three calibration seeds or disjoint subsets.

Measure:

- Spearman correlation of combined neuron scores;
- top-10% neuron overlap;
- bottom-10% neuron overlap;
- overlap of first and last block-sized neuron groups;
- variance of pruned-model PPL.

This task is experimental only. Do not add automatic ensembling or smoothing to the first implementation.

---

## Failure diagnosis order

If permutation worsens performance, check in this order:

1. Verify the exact shared mapping: up rows, gate rows, and down columns.
2. Verify the permutation is computed on dense, unpruned weights.
3. Verify the permutation is applied before mask initialization.
4. Verify the permutation is applied exactly once.
5. Verify the down-neuron score uses the real pre-permutation `down_proj` input RMS.
6. Verify projection-wise normalization occurs before score summation.
7. Verify each permutation is stable, bijective, and descending by combined score.
8. Verify round-level Block-Wanda recollects RMS after permutation.
9. Verify Fisher reference masks, final masks, and zero blocks all use permuted coordinates.
10. Inspect block-concentration diagnostics layer by layer.
11. Inspect calibration-subset stability.

Only after these checks should profile clustering or local swap optimization be considered in a separate plan.

## Completion criteria

The work is complete only when:

- `mlp_permutation=none|wanda_shared` is supported;
- one permutation is produced per Transformer MLP layer;
- up and gate share exactly the same row order;
- down uses the matching column order;
- toy SwiGLU equivalence passes;
- negative controls demonstrate why strict sharing is necessary;
- permutation runs once before masks and all block scoring;
- final checkpoints remain in the permuted coordinate system;
- disabled mode preserves existing pruning behavior;
- permutation artifacts and concentration diagnostics are saved;
- all `Block_Sparse` tests pass in `hif4`;
- dense equivalence and pruning smoke tests are complete;
- no-permutation and shared-permutation ablations are complete.

## Out of scope

- residual hidden-dimension permutation;
- attention permutation;
- arbitrary independent row/column sorting;
- different up and gate permutations;
- online runtime permutation;
- learned or differentiable permutation;
- balanced clustering;
- local swap optimization;
- permutation recomputation after pruning rounds;
- projection-weight hyperparameter search;
- post-pruning fine-tuning;
- SparseGPT compensation;
- sparse CUDA kernels.

