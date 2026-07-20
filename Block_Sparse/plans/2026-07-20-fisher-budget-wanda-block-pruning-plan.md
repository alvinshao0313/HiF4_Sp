# Fisher-Budget + Block-Wanda MLP Block Pruning Implementation Plan

> **Generated on:** 2026-07-20  
> **Workspace:** `/home/shaoyuantian/program/HiF4_Sp`  
> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development` while implementing each task, and use `superpowers:verification-before-completion` before claiming completion. Execute tasks in order and track progress with the checkboxes below.

## Goal

Add a new MLP block-pruning method named `fisher_budget_wanda` that uses Fisher/gradient information only to determine the exact number of blocks pruned from each MLP `nn.Linear`, then uses Block-Wanda scores only to select the concrete block coordinates inside that matrix.

The method must preserve the current global target sparsity and all existing matrix constraints, while preventing Fisher scores from directly selecting fine-grained block positions.

## Method Definition

For each target MLP matrix \(m\), let \(b\) index structured blocks.

### Stage 1: Fisher budget allocation

Reuse the current Fisher scorer and current constrained global allocator to produce a temporary reference mask:

\[
M^{F}=\operatorname{GlobalAllocate}(\{F_{m,b}\}, \rho).
\]

The temporary Fisher mask is not applied to the model. It is used only to obtain the cumulative pruning budget of each matrix:

\[
K_m = \sum_b \mathbb{I}[M^{F}_{m,b}=0].
\]

### Stage 2: Block-Wanda position selection

For matrix

\[
W_m\in\mathbb{R}^{d_{\mathrm{out}}\times d_{\mathrm{in}}},
\]

collect the RMS of its actual input activation channels:

\[
a_{m,j}=\sqrt{\frac{1}{T}\sum_{t=1}^{T} X_{m,t,j}^2}.
\]

The element-level Wanda importance is

\[
s_{m,ij}=|W_{m,ij}|a_{m,j}.
\]

The structured block importance is

\[
S^{\mathrm{Wanda}}_{m,b}
=\sum_{i\in R_b}\sum_{j\in C_b}|W_{m,ij}|a_{m,j}.
\]

For each matrix independently, prune the \(K_m\) currently active blocks with the smallest Block-Wanda scores.

### Non-negotiable separation of responsibilities

- Fisher decides **how many blocks each matrix prunes**.
- Wanda decides **which coordinates are pruned inside that matrix**.
- Do not add Fisher and Wanda scores together.
- Do not globally rank Wanda scores across different matrices.
- Do not normalize Wanda scores across matrices in the first implementation.
- Do not apply the temporary Fisher mask to model weights.

## Existing Code Context

The current implementation is under `Block_Sparse/`.

Relevant files:

- `Block_Sparse/block_pruning/block_utils.py`
  - Existing block-reduction utilities.
- `Block_Sparse/block_pruning/gradient_scorer.py`
  - Existing Fisher, magnitude, and random block scorers.
  - `BlockScoreRecord` is the current shared score container.
- `Block_Sparse/block_pruning/mask_allocator.py`
  - Existing global constrained allocator.
- `Block_Sparse/block_pruning/config.py`
  - Current public method and pruning configuration.
- `Block_Sparse/block_pruning/calibration.py`
  - Current causal-LM calibration batch construction.
- `Block_Sparse/block_pruning/serialization.py`
  - Current masks, score records, reports, and summary serialization.
- `Block_Sparse/scripts/score_and_prune_mlp.py`
  - Current scoring, allocation, mask application, artifact export, and HF save pipeline.
- `Block_Sparse/scripts/prune_mlp.sh`
  - Single-method execution script.
- `Block_Sparse/scripts/run_baselines.sh`
  - Baseline batch execution script.
- `Block_Sparse/tests/`
  - Existing block math, Fisher, allocator, mask, and model-registry tests.

## Global Constraints

- All Python commands and tests must run in conda environment `hif4`.
- The implementation must remain limited to dense MLP projections:
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- Do not add attention pruning, SparseGPT compensation, fine-tuning, or sparse CUDA kernels.
- Existing `fisher`, `magnitude`, and `random` behavior must not change.
- Block dimensions must remain strictly divisible by weight dimensions; do not add padding.
- Unreachable pruning budgets must raise a clear exception; do not silently reduce sparsity.
- Hooks must always be removed in `finally` blocks.
- Activation statistics must not store full activations.
- Accumulate activation statistics on CPU in `float64`.
- Preserve deterministic behavior under the existing `seed`.
- `pruning_rounds=1` is the first validation target, but the implementation must correctly support the current multi-round cumulative semantics.

## Planned File Structure

### Create

- `Block_Sparse/block_pruning/wanda_scorer.py`
  - Collect per-module input RMS and calculate Block-Wanda scores.
- `Block_Sparse/tests/test_wanda_scorer.py`
  - Test Block-Wanda math and activation-statistics collection.
- `Block_Sparse/tests/test_module_budget_allocator.py`
  - Test Fisher budget extraction and exact per-module Wanda allocation.

### Modify

- `Block_Sparse/block_pruning/block_utils.py`
  - Add pure Block-Wanda reduction utility.
- `Block_Sparse/block_pruning/config.py`
  - Register `fisher_budget_wanda`.
- `Block_Sparse/block_pruning/mask_allocator.py`
  - Add budget extraction and exact per-module allocation.
- `Block_Sparse/block_pruning/serialization.py`
  - Save both Fisher-reference and final Wanda artifacts.
- `Block_Sparse/scripts/score_and_prune_mlp.py`
  - Add the two-stage hybrid execution path.
- `Block_Sparse/scripts/prune_mlp.sh`
  - Expose the new method in comments and output naming.
- `Block_Sparse/scripts/run_baselines.sh`
  - Permit batch comparison with the new method.
- `Block_Sparse/README.md`
  - Document the method, artifacts, and recommended experiment.

---

## Task 1: Add the pure Block-Wanda block-reduction primitive

**Files:**

- Modify: `Block_Sparse/block_pruning/block_utils.py`
- Test: `Block_Sparse/tests/test_wanda_scorer.py`

**Produces:**

```python
def reduce_weight_wanda_to_blocks(
    weight: torch.Tensor,
    input_rms: torch.Tensor,
    block_height: int,
    block_width: int,
) -> torch.Tensor:
    """Return block sums of abs(weight) weighted by input-channel RMS."""
```

**Required behavior:**

- `weight` must be rank 2.
- `input_rms` must be rank 1.
- `input_rms.numel()` must equal `weight.shape[1]`.
- Existing `validate_weight_divisible()` must enforce block divisibility.
- Compute in `float32`:

```python
element_score = weight.float().abs() * input_rms.float().unsqueeze(0)
```

- Reshape to the existing block layout and sum over within-block axes.
- Return shape:

```text
[d_out // block_height, d_in // block_width]
```

- The function must not inspect masks, modules, hooks, or configuration.

- [ ] **Step 1.1: Create a failing hand-computed math test**

Create `Block_Sparse/tests/test_wanda_scorer.py` with a test using a `4 x 4` weight, a length-4 RMS vector, and `2 x 2` blocks.

Use this exact fixture:

```python
weight = torch.tensor(
    [
        [1.0, -2.0, 3.0, -4.0],
        [5.0, -6.0, 7.0, -8.0],
        [9.0, -10.0, 11.0, -12.0],
        [13.0, -14.0, 15.0, -16.0],
    ]
)
input_rms = torch.tensor([1.0, 2.0, 3.0, 4.0])
```

Expected block scores:

```python
expected = torch.tensor(
    [
        [1 * 1 + 2 * 2 + 5 * 1 + 6 * 2,
         3 * 3 + 4 * 4 + 7 * 3 + 8 * 4],
        [9 * 1 + 10 * 2 + 13 * 1 + 14 * 2,
         11 * 3 + 12 * 4 + 15 * 3 + 16 * 4],
    ],
    dtype=torch.float32,
)
```

Assert exact shape and `torch.testing.assert_close()`.

- [ ] **Step 1.2: Add failing validation tests**

Add tests asserting `ValueError` for:

- rank-2 `input_rms`;
- channel-count mismatch;
- non-divisible weight shape.

The error messages must contain the conflicting shapes or dimensions.

- [ ] **Step 1.3: Run the focused test and confirm failure**

Run:

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest Block_Sparse/tests/test_wanda_scorer.py -v
```

Expected result: import or attribute failure because `reduce_weight_wanda_to_blocks` does not exist.

- [ ] **Step 1.4: Implement the minimal pure function**

Add the function to `block_utils.py`. Do not add unrelated helpers.

- [ ] **Step 1.5: Run the focused test and confirm pass**

Run the same command.

Expected result: all Task 1 tests pass.

- [ ] **Step 1.6: Commit Task 1**

```bash
git add Block_Sparse/block_pruning/block_utils.py \
        Block_Sparse/tests/test_wanda_scorer.py
git commit -m "feat: add block wanda score reduction"
```

---

## Task 2: Collect input-channel RMS for each MLP projection

**Files:**

- Create: `Block_Sparse/block_pruning/wanda_scorer.py`
- Modify: `Block_Sparse/tests/test_wanda_scorer.py`

**Consumes:**

- `MLPLinearTarget` from `mlp_registry.py`.
- Calibration batches with `input_ids` and `attention_mask`.
- `move_batch_to_device()` and `resolve_model_input_device()`.

**Produces:**

```python
@dataclass
class InputRMSRecord:
    module_name: str
    layer_index: int
    projection_type: str
    num_tokens: int
    channel_square_sum: torch.Tensor
    input_rms: torch.Tensor
```

```python
def collect_mlp_input_rms(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    targets: list[MLPLinearTarget],
) -> dict[str, InputRMSRecord]:
    ...
```

**Required data path:**

1. Resolve the model input device with `resolve_model_input_device(model)`.
2. Set `model.eval()`.
3. Disable cache through `model.config.use_cache = False` when available.
4. Register one `forward_pre_hook` per target module.
5. For each hook invocation:
   - Require a positional input tensor.
   - Read `x = inputs[0]`.
   - Require `x.shape[-1] == target.module.weight.shape[1]`.
   - Flatten all leading dimensions:

```python
x2d = x.detach().float().reshape(-1, x.shape[-1])
```

   - Accumulate `x2d.square().sum(dim=0).double().cpu()`.
   - Accumulate the flattened token/position count.
6. Run model forward under `torch.no_grad()` using only:

```python
model(
    input_ids=batch_dev["input_ids"],
    attention_mask=batch_dev["attention_mask"],
    use_cache=False,
)
```

7. Remove every hook in a `finally` block.
8. Require every target to have at least one hook invocation and `num_tokens > 0`.
9. Compute:

```python
input_rms = torch.sqrt(channel_square_sum / num_tokens)
```

10. Return CPU `float64` statistics.

**Important:**

- Do not pass `labels`; no loss or backward is required.
- Do not store `x`, `x2d`, or per-batch activations after each hook returns.
- Collect the actual `down_proj` input after the SwiGLU multiplication by hooking `down_proj` itself.

- [ ] **Step 2.1: Add a failing hook-statistics test**

Create a tiny model with two linear modules that are invoked in forward. Construct an `MLPLinearTarget` list manually.

Use deterministic inputs where channel RMS is hand-computable. Verify:

- each module has the expected `num_tokens`;
- each module has the expected `channel_square_sum`;
- each module has the expected `input_rms`;
- the two modules can receive different inputs.

- [ ] **Step 2.2: Add failure-path tests**

Add tests for:

- empty calibration batch list;
- a target module never invoked by forward;
- target input last dimension not matching `d_in`;
- hook input missing a tensor.

Each case must raise, rather than skip the module.

- [ ] **Step 2.3: Run tests and confirm failure**

Run:

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest Block_Sparse/tests/test_wanda_scorer.py -v
```

Expected result: failures because `InputRMSRecord` and `collect_mlp_input_rms` do not exist.

- [ ] **Step 2.4: Implement `wanda_scorer.py` input-statistics collection**

Keep hook construction in a private helper so each closure captures the correct module name and expected input dimension.

Required internal accumulator structure:

```python
accumulators[module_name] = {
    "square_sum": torch.zeros(d_in, dtype=torch.float64),
    "num_tokens": 0,
    "num_calls": 0,
}
```

- [ ] **Step 2.5: Run tests and confirm pass**

Run the focused test command.

Expected result: all Task 1 and Task 2 tests pass.

- [ ] **Step 2.6: Commit Task 2**

```bash
git add Block_Sparse/block_pruning/wanda_scorer.py \
        Block_Sparse/tests/test_wanda_scorer.py
git commit -m "feat: collect mlp input rms for wanda"
```

---

## Task 3: Build Block-Wanda score records compatible with the pruning pipeline

**Files:**

- Modify: `Block_Sparse/block_pruning/wanda_scorer.py`
- Modify: `Block_Sparse/block_pruning/gradient_scorer.py`
- Modify: `Block_Sparse/tests/test_wanda_scorer.py`

**Design decision:**

Extend the existing `BlockScoreRecord` with an explicit optional `wanda` tensor instead of storing Wanda in the misleading `fisher` field.

Required dataclass field:

```python
wanda: torch.Tensor | None = None
```

Required `primary_score()` behavior:

```python
if score_type == "fisher":
    return self.fisher
if score_type == "magnitude":
    return self.fisher
if score_type == "random":
    return self.fisher
if score_type == "wanda":
    if self.wanda is None:
        raise ValueError(...)
    return self.wanda
```

Do not alter existing `fisher`, `magnitude`, or `random` score construction.

**Produces:**

```python
def collect_wanda_block_scores(
    model: nn.Module,
    batches: list[dict[str, torch.Tensor]],
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, BlockScoreRecord]:
    ...
```

**Required behavior:**

- Call `collect_mlp_input_rms()` once.
- For each target, call `reduce_weight_wanda_to_blocks()`.
- Convert scores to CPU `float64` for consistency with existing allocator records.
- Preserve `current_mask` in each record.
- Do not multiply active scores by zero. Instead, leave numeric Wanda scores intact and rely on the allocator to enumerate only active blocks.
- Fill unused legacy fields with zero tensors of the correct shape.

- [ ] **Step 3.1: Update test record factories first**

Existing tests instantiate `BlockScoreRecord` directly. Add `wanda=None` to their factories only if the dataclass change requires it. Prefer a default field value to avoid unnecessary edits.

- [ ] **Step 3.2: Add a failing end-to-end scorer test**

Use a tiny model and known calibration inputs. Verify that:

```text
collect_wanda_block_scores
= collect_mlp_input_rms
+ abs(weight) * input_rms
+ block sum
```

for at least two modules.

- [ ] **Step 3.3: Add a `primary_score("wanda")` test**

Verify:

- returns the Wanda tensor when present;
- raises a clear `ValueError` when absent.

- [ ] **Step 3.4: Run tests and confirm failure**

Run:

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest Block_Sparse/tests/test_wanda_scorer.py -v
```

- [ ] **Step 3.5: Implement score-record integration**

Modify only the dataclass, its score selector, the internal `_make_record()` signature/default, and Wanda score construction.

- [ ] **Step 3.6: Run tests and confirm pass**

Run the focused test command.

- [ ] **Step 3.7: Run existing scorer and allocator tests**

Run:

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest \
  Block_Sparse/tests/test_block_reduce.py \
  Block_Sparse/tests/test_fisher_grad_offload.py \
  Block_Sparse/tests/test_global_allocator.py \
  -v
```

Expected result: existing behavior remains green.

- [ ] **Step 3.8: Commit Task 3**

```bash
git add Block_Sparse/block_pruning/gradient_scorer.py \
        Block_Sparse/block_pruning/wanda_scorer.py \
        Block_Sparse/tests/test_wanda_scorer.py
git commit -m "feat: expose block wanda score records"
```

---

## Task 4: Extract exact per-module budgets from the Fisher reference mask

**Files:**

- Modify: `Block_Sparse/block_pruning/mask_allocator.py`
- Create: `Block_Sparse/tests/test_module_budget_allocator.py`

**Produces:**

```python
def extract_module_prune_budgets(
    reference_masks: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Return cumulative pruned-block count for every module."""
```

Implementation:

```python
return {
    module_name: int((~mask).sum().item())
    for module_name, mask in reference_masks.items()
}
```

Required validation:

- input dictionary must not be empty;
- every mask must have boolean dtype;
- every mask must be rank 2;
- every budget must be between zero and `mask.numel()`.

- [ ] **Step 4.1: Add a failing exact-count test**

Create three reference masks representing gate/up/down and assert exact extracted counts.

- [ ] **Step 4.2: Add a global-count identity test**

Assert:

```python
sum(budgets.values()) == sum(int((~m).sum()) for m in reference_masks.values())
```

- [ ] **Step 4.3: Add validation tests**

Cover empty input, non-boolean masks, and non-2D masks.

- [ ] **Step 4.4: Run tests and confirm failure**

Run:

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest Block_Sparse/tests/test_module_budget_allocator.py -v
```

- [ ] **Step 4.5: Implement the minimal extractor**

Do not couple this function to `MaskAllocationResult` or score records.

- [ ] **Step 4.6: Run tests and confirm pass**

- [ ] **Step 4.7: Commit Task 4**

```bash
git add Block_Sparse/block_pruning/mask_allocator.py \
        Block_Sparse/tests/test_module_budget_allocator.py
git commit -m "feat: extract fisher module pruning budgets"
```

---

## Task 5: Allocate exact per-module budgets using local Wanda ranking

**Files:**

- Modify: `Block_Sparse/block_pruning/mask_allocator.py`
- Modify: `Block_Sparse/tests/test_module_budget_allocator.py`

**Produces:**

```python
def allocate_masks_by_module_budget(
    score_records: dict[str, BlockScoreRecord],
    target_pruned_per_module: dict[str, int],
    config: GradientBlockPruningConfig,
    current_masks: dict[str, torch.Tensor],
    ranking_score_type: str = "wanda",
) -> MaskAllocationResult:
    ...
```

### Independent-mask semantics

When `config.share_up_gate_mask` is false:

For every module independently:

1. Validate module sets are identical across:
   - `score_records`
   - `target_pruned_per_module`
   - `current_masks`
2. Let:

```python
current_pruned = int((~current_mask).sum().item())
target_pruned = target_pruned_per_module[module_name]
additional_needed = target_pruned - current_pruned
```

3. Require:

```text
0 <= current_pruned <= target_pruned <= max_prunable
```

where `max_prunable` is calculated with the existing `_max_prunable()`.

4. Enumerate only active coordinates from `current_mask`.
5. Sort candidates stably by:

```text
(score, out_block, in_block)
```

6. Prune exactly `additional_needed` candidates.
7. Never reopen an already pruned block.

### Result accounting

Return a `MaskAllocationResult` where:

- `num_total_blocks` is the total physical blocks over all matrices;
- `num_pruned_blocks` is the final cumulative count;
- `actual_block_sparsity` is cumulative final sparsity;
- `newly_pruned` is the number added by this call;
- `target_pruned` is `sum(target_pruned_per_module.values())`.

After allocation, assert for every module:

```python
int((~result.masks[name]).sum().item()) == target_pruned_per_module[name]
```

- [ ] **Step 5.1: Add a failing two-module exact-budget test**

Use:

- module A budget: 2 blocks;
- module B budget: 3 blocks;
- deliberately make module A have more globally low scores than module B.

Verify that A still prunes exactly 2 and B exactly 3. This proves there is no global Wanda ranking.

- [ ] **Step 5.2: Add a stable-tie test**

Set equal scores and verify lower `(out_block, in_block)` coordinates are selected first.

- [ ] **Step 5.3: Add multi-round monotonicity test**

Start with pre-pruned masks. Increase cumulative module targets. Verify:

- old zero blocks remain zero;
- only the exact additional count is pruned;
- the final count equals the cumulative target.

- [ ] **Step 5.4: Add unreachable-budget tests**

Cover:

- target below already-pruned count;
- target above `max_prune_ratio_per_matrix` limit;
- target violating `min_keep_blocks_per_matrix`;
- module-key mismatch;
- score-shape and mask-shape mismatch.

All must raise with the module name in the message.

- [ ] **Step 5.5: Run tests and confirm failure**

Run:

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest Block_Sparse/tests/test_module_budget_allocator.py -v
```

- [ ] **Step 5.6: Implement independent per-module allocation**

Reuse existing private helpers where correct. Do not alter `allocate_block_masks()` behavior.

- [ ] **Step 5.7: Run tests and confirm pass**

- [ ] **Step 5.8: Run current global allocator regression tests**

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest Block_Sparse/tests/test_global_allocator.py -v
```

- [ ] **Step 5.9: Commit Task 5**

```bash
git add Block_Sparse/block_pruning/mask_allocator.py \
        Block_Sparse/tests/test_module_budget_allocator.py
git commit -m "feat: allocate wanda masks by fisher module budget"
```

---

## Task 6: Support shared `up_proj`/`gate_proj` masks

**Files:**

- Modify: `Block_Sparse/block_pruning/mask_allocator.py`
- Modify: `Block_Sparse/tests/test_module_budget_allocator.py`

### Required shared-pair semantics

When `config.share_up_gate_mask` is true:

1. Use the existing `_pair_key()` naming logic.
2. Require every pair to have both `up` and `gate` modules.
3. Require:

```python
target_pruned_per_module[up_name] == target_pruned_per_module[gate_name]
```

4. Require current up/gate masks to be identical. Do not repair mismatched masks.
5. Require score-grid shapes to match.
6. Form joint score:

\[
S^{\mathrm{pair}}_b=S^{\mathrm{Wanda}}_{\mathrm{up},b}+S^{\mathrm{Wanda}}_{\mathrm{gate},b}.
\]

7. Locally select the lowest-scoring active pair coordinates until the pair reaches its cumulative target.
8. Apply every selected coordinate to both masks.
9. Treat each pair coordinate as two physical blocks in global counts.
10. Allocate `down_proj` independently using Task 5 semantics.

- [ ] **Step 6.1: Add a failing shared-pair selection test**

Construct up/gate scores where their independent minima differ. Verify the result uses the sum and produces identical masks.

- [ ] **Step 6.2: Add validation tests**

Cover:

- unequal up/gate target budgets;
- unequal current masks;
- incomplete pair;
- unequal score shapes.

- [ ] **Step 6.3: Add exact physical-block accounting test**

Verify a pair target of 2 coordinates contributes 4 pruned physical blocks to `num_pruned_blocks`.

- [ ] **Step 6.4: Run tests and confirm failure**

- [ ] **Step 6.5: Implement the shared-pair branch**

Keep independent and shared paths in separate private functions for readability.

- [ ] **Step 6.6: Run tests and confirm pass**

- [ ] **Step 6.7: Commit Task 6**

```bash
git add Block_Sparse/block_pruning/mask_allocator.py \
        Block_Sparse/tests/test_module_budget_allocator.py
git commit -m "feat: support shared up gate wanda allocation"
```

---

## Task 7: Register the new public method in configuration and CLI

**Files:**

- Modify: `Block_Sparse/block_pruning/config.py`
- Modify: `Block_Sparse/scripts/score_and_prune_mlp.py`
- Test: `Block_Sparse/tests/test_module_budget_allocator.py` or a new focused CLI-config test only if needed

**Required public method name:**

```text
fisher_budget_wanda
```

**Configuration behavior:**

- Add `fisher_budget_wanda` to the accepted `score_type` set.
- Treat it as requiring calibration batches.
- Do not introduce a second public selector such as `budget_score_type` in the first version.
- Internally, the method is fixed to:

```text
budget score = fisher
position score = wanda
```

**CLI behavior:**

Add the method to `--score_type` choices and help text.

- [ ] **Step 7.1: Add a failing config-validation test**

Verify `GradientBlockPruningConfig(score_type="fisher_budget_wanda").validate()` succeeds.

- [ ] **Step 7.2: Add a failing CLI parse test only if existing project style supports it**

If no CLI tests exist, do not introduce a subprocess-heavy test solely for argument choices. Directly cover config validation and rely on the final smoke command.

- [ ] **Step 7.3: Implement config and parser registration**

- [ ] **Step 7.4: Run focused tests**

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest \
  Block_Sparse/tests/test_module_budget_allocator.py \
  Block_Sparse/tests/test_global_allocator.py \
  -v
```

- [ ] **Step 7.5: Commit Task 7**

```bash
git add Block_Sparse/block_pruning/config.py \
        Block_Sparse/scripts/score_and_prune_mlp.py \
        Block_Sparse/tests/test_module_budget_allocator.py
git commit -m "feat: register fisher budget wanda pruning"
```

---

## Task 8: Integrate the two-stage method into each pruning round

**Files:**

- Modify: `Block_Sparse/scripts/score_and_prune_mlp.py`
- Modify: `Block_Sparse/block_pruning/serialization.py`
- Modify: `Block_Sparse/tests/test_module_budget_allocator.py`

### Required round-level control flow

For ordinary `fisher`, `magnitude`, and `random`, preserve the current loop exactly.

For `fisher_budget_wanda`, each round must execute:

```python
# 1. Fisher scores on the current masked model.
fisher_records = collect_mlp_block_scores(
    model=model,
    batches=batches,
    targets=targets,
    config=config,
    current_masks=current_masks,
)

# 2. Temporary Fisher reference allocation at this round's cumulative target.
fisher_reference_allocation = allocate_block_masks(
    score_records=fisher_records,
    config=config,
    current_masks=current_masks,
    cumulative_target_sparsity=cumulative_target,
    ranking_score_type="fisher",  # implement explicit override or equivalent local API
)

# 3. Extract cumulative per-module targets.
module_budgets = extract_module_prune_budgets(
    fisher_reference_allocation.masks
)

# 4. Wanda scores on the same current model before applying this round's new mask.
wanda_records = collect_wanda_block_scores(
    model=model,
    batches=batches,
    targets=targets,
    config=config,
    current_masks=current_masks,
)

# 5. Exact local allocation using Fisher budgets.
final_allocation = allocate_masks_by_module_budget(
    score_records=wanda_records,
    target_pruned_per_module=module_budgets,
    config=config,
    current_masks=current_masks,
    ranking_score_type="wanda",
)

# 6. Apply only the final Wanda-selected masks.
current_masks = final_allocation.masks
apply_mlp_block_masks(...)
```

### Required allocator API adjustment

The current `allocate_block_masks()` reads `config.score_type`, which would equal `fisher_budget_wanda` and is not a score tensor name.

Add an explicit optional argument:

```python
ranking_score_type: str | None = None
```

Inside the existing allocator:

```python
score_type = ranking_score_type or config.score_type
```

Use the same rule in independent and shared-up/gate paths.

Existing callers must remain unchanged and preserve their behavior.

### Cross-stage invariants

Before applying masks, assert:

```python
sum(module_budgets.values()) == fisher_reference_allocation.num_pruned_blocks
```

and:

```python
final_allocation.num_pruned_blocks == fisher_reference_allocation.num_pruned_blocks
```

For every module:

```python
int((~final_allocation.masks[name]).sum().item()) == module_budgets[name]
```

Do not apply `fisher_reference_allocation.masks` to weights.

- [ ] **Step 8.1: Add a failing allocator override test**

Create a `config.score_type="fisher_budget_wanda"` record set containing Fisher scores. Verify `allocate_block_masks(..., ranking_score_type="fisher")` uses Fisher without changing config.

- [ ] **Step 8.2: Add a pipeline helper instead of embedding all logic in `main()`**

Create a focused helper in `score_and_prune_mlp.py`:

```python
def allocate_hybrid_round(
    model,
    batches,
    targets,
    config,
    current_masks,
    cumulative_target_sparsity,
):
    """Return Fisher records, Wanda records, budgets, reference allocation, final allocation."""
```

Returning all intermediate objects is required for serialization and diagnostics.

- [ ] **Step 8.3: Add a small unit test for cross-stage accounting**

Avoid loading the real 27B model. Patch or inject scorer results and verify:

- Fisher reference coordinates differ from Wanda final coordinates;
- per-module counts are identical;
- only final masks are returned for application.

- [ ] **Step 8.4: Run tests and confirm failure**

- [ ] **Step 8.5: Implement the hybrid round helper and main-loop branch**

Ordinary methods must remain on the old `score_blocks() -> allocate_block_masks()` path.

- [ ] **Step 8.6: Run focused and regression tests**

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest \
  Block_Sparse/tests/test_module_budget_allocator.py \
  Block_Sparse/tests/test_global_allocator.py \
  Block_Sparse/tests/test_apply_mask.py \
  -v
```

- [ ] **Step 8.7: Commit Task 8**

```bash
git add Block_Sparse/block_pruning/mask_allocator.py \
        Block_Sparse/scripts/score_and_prune_mlp.py \
        Block_Sparse/tests/test_module_budget_allocator.py
git commit -m "feat: integrate fisher budget wanda pruning rounds"
```

---

## Task 9: Save complete hybrid diagnostics

**Files:**

- Modify: `Block_Sparse/block_pruning/serialization.py`
- Modify: `Block_Sparse/scripts/score_and_prune_mlp.py`
- Modify: `Block_Sparse/tests/test_module_budget_allocator.py`

### Required hybrid artifacts

For a single-round run, save under `pruning_artifacts/`:

```text
fisher_block_scores.pt
wanda_block_scores.pt
fisher_reference_masks.pt
block_masks.pt
module_prune_budget.csv
hybrid_per_matrix_report.csv
pruning_summary.json
```

For multiple rounds, use the existing round suffix convention:

```text
_round0
_round1
...
```

and still save unsuffixed final artifacts at the end.

### Required score serialization

Update `save_score_records()` so it saves the optional Wanda field when present:

```python
"wanda": None if rec.wanda is None else rec.wanda.cpu()
```

Do not remove or rename existing keys.

### Required module budget report

Add:

```python
def save_module_prune_budget_report(
    path: str | Path,
    targets: list[MLPLinearTarget],
    current_masks_before: dict[str, torch.Tensor],
    fisher_reference_masks: dict[str, torch.Tensor],
    final_masks: dict[str, torch.Tensor],
) -> None:
    ...
```

Required CSV fields:

```text
module_name
layer_index
projection_type
num_total_blocks
current_pruned_blocks
fisher_target_pruned_blocks
newly_pruned_blocks
final_pruned_blocks
final_block_sparsity
```

Required identity:

```text
fisher_target_pruned_blocks == final_pruned_blocks
```

### Required hybrid per-matrix report

For each module, save:

```text
module_name
layer_index
projection_type
num_blocks
fisher_target_pruned_blocks
final_pruned_blocks
fisher_score_min
fisher_score_median
fisher_score_mean
fisher_score_max
wanda_score_min
wanda_score_median
wanda_score_mean
wanda_score_max
fisher_wanda_mask_overlap_blocks
fisher_wanda_mask_union_blocks
fisher_wanda_mask_iou
```

Define the sets as the cumulative pruned coordinates in the Fisher reference and final masks.

If the union is empty, define IoU as `1.0`.

### Required summary metadata

For the hybrid method, include:

```json
{
  "score_type": "fisher_budget_wanda",
  "budget_score": "fisher",
  "selection_score": "wanda",
  "budget_granularity": "mlp_linear_module"
}
```

Do not change ordinary-method summaries except where optional fields are absent.

- [ ] **Step 9.1: Add serialization tests using a temporary directory**

Verify exact filenames, CSV headers, row counts, and core identities.

- [ ] **Step 9.2: Add a test proving both masks are saved**

Load the saved tensors and verify Fisher coordinates can differ from final coordinates while counts match.

- [ ] **Step 9.3: Run tests and confirm failure**

- [ ] **Step 9.4: Implement dedicated hybrid serialization functions**

Do not overload `save_round_artifacts()` with ambiguous positional arguments. Either:

- add a separate `save_hybrid_round_artifacts()`, or
- add explicit keyword-only hybrid arguments.

Preferred interface:

```python
def save_hybrid_round_artifacts(
    output_dir: str | Path,
    fisher_records: dict[str, BlockScoreRecord],
    wanda_records: dict[str, BlockScoreRecord],
    current_masks_before: dict[str, torch.Tensor],
    fisher_reference_allocation: MaskAllocationResult,
    final_allocation: MaskAllocationResult,
    targets: list[MLPLinearTarget],
    config: GradientBlockPruningConfig,
    round_idx: int | None = None,
) -> None:
    ...
```

- [ ] **Step 9.5: Run focused tests and confirm pass**

- [ ] **Step 9.6: Commit Task 9**

```bash
git add Block_Sparse/block_pruning/serialization.py \
        Block_Sparse/scripts/score_and_prune_mlp.py \
        Block_Sparse/tests/test_module_budget_allocator.py
git commit -m "feat: save hybrid pruning diagnostics"
```

---

## Task 10: Update scripts and user documentation

**Files:**

- Modify: `Block_Sparse/scripts/prune_mlp.sh`
- Modify: `Block_Sparse/scripts/run_baselines.sh`
- Modify: `Block_Sparse/README.md`

### Script changes

In `prune_mlp.sh`:

```bash
SCORE_TYPE=fisher_budget_wanda  # fisher | magnitude | random | fisher_budget_wanda
```

Do not force the default to change unless explicitly desired during implementation review. The script must at least document the new valid value.

Output naming remains:

```bash
qwen35_27b_${SCORE_TYPE}_s${SPARSITY}_b${BLOCK_SIZE}
```

In `run_baselines.sh`, update the documented method list and permit:

```bash
METHODS=(magnitude fisher fisher_budget_wanda)
```

Do not require `random` in the recommended accuracy comparison.

### README changes

Document:

1. Why direct Fisher block selection can overfit the calibration set.
2. The exact two-stage method.
3. The fact that Fisher reference masks are not applied.
4. The Block-Wanda equation.
5. Per-matrix rather than global Wanda ranking.
6. Shared up/gate joint score semantics.
7. Multi-round cumulative semantics.
8. New artifact files.
9. Recommended first experiment:

```text
model: Qwen/Qwen3.5-27B
block_size: 128
target sparsity: 0.20
calibration dataset: s1k
calibration samples: 128
sequence length: 0
pruning rounds: 1
```

10. Required comparison:

```text
magnitude
fisher
fisher_budget_wanda
```

- [ ] **Step 10.1: Update shell comments and method arrays**

- [ ] **Step 10.2: Add a dedicated README section**

Suggested heading:

```markdown
## Fisher 预算 + Block-Wanda 选块
```

- [ ] **Step 10.3: Verify shell syntax**

Run:

```bash
bash -n Block_Sparse/scripts/prune_mlp.sh
bash -n Block_Sparse/scripts/run_baselines.sh
```

Expected result: no output and exit code 0.

- [ ] **Step 10.4: Commit Task 10**

```bash
git add Block_Sparse/scripts/prune_mlp.sh \
        Block_Sparse/scripts/run_baselines.sh \
        Block_Sparse/README.md
git commit -m "docs: add fisher budget wanda workflow"
```

---

## Task 11: Full verification before real-model execution

**Files:** none unless a discovered defect requires a focused fix.

- [ ] **Step 11.1: Run the full Block_Sparse test suite**

```bash
conda run -n hif4 --no-capture-output \
  python -m pytest Block_Sparse/tests/ -v
```

Expected result: all tests pass.

- [ ] **Step 11.2: Verify ordinary-method CLI help remains valid**

```bash
conda run -n hif4 --no-capture-output \
  python Block_Sparse/scripts/score_and_prune_mlp.py --help
```

Expected result: help lists:

```text
fisher
magnitude
random
fisher_budget_wanda
```

- [ ] **Step 11.3: Inspect git diff for scope control**

The diff must contain only:

- the new Wanda scorer;
- the new budget allocator path;
- hybrid serialization;
- tests;
- CLI/script/docs integration.

Reject unrelated refactors.

- [ ] **Step 11.4: Verify no test-only dependency or stub was added**

All tests must use installed `hif4` environment dependencies.

- [ ] **Step 11.5: Commit any verification-only corrections separately**

Use a precise message describing the defect, not a generic cleanup commit.

---

## Task 12: Real-model smoke run

**Precondition:** Tasks 1-11 complete and all tests pass.

**Purpose:** Verify model loading, activation hooks, Fisher scoring, Wanda scoring, allocation, serialization, mask application, and HF export on the real project path.

Use a low-cost smoke configuration first. Do not evaluate PPL until the checkpoint exports successfully.

- [ ] **Step 12.1: Run a reduced calibration smoke test**

Example command:

```bash
conda run -n hif4 --no-capture-output \
  python Block_Sparse/scripts/score_and_prune_mlp.py \
  --model_path Qwen/Qwen3.5-27B \
  --output_dir Block_Sparse/outputs/qwen35_27b_fisher_budget_wanda_smoke \
  --score_type fisher_budget_wanda \
  --target_block_sparsity 0.01 \
  --max_prune_ratio_per_matrix 0.10 \
  --block_size 128 \
  --calibration_dataset wikitext2 \
  --calibration_samples 2 \
  --sequence_length 128 \
  --pruning_rounds 1 \
  --seed 42 \
  --dtype bfloat16 \
  --device cuda
```

The exact visible GPUs must be set by the operator according to available hardware.

- [ ] **Step 12.2: Verify artifacts**

Check that:

- all required hybrid artifacts exist;
- global Fisher reference count equals final count;
- every per-module Fisher target equals final count;
- Fisher and Wanda masks are allowed to differ in coordinates;
- exported zero blocks match `block_masks.pt`;
- the exported directory contains a valid HF model and tokenizer.

- [ ] **Step 12.3: Verify saved model loads through the existing evaluation path**

Use the project evaluation command with a minimal sample limit if supported. The purpose is loading verification, not accuracy reporting.

- [ ] **Step 12.4: Record smoke configuration in the output summary**

Do not commit generated model weights or large artifacts.

---

## Task 13: Controlled PPL comparison

**Purpose:** Test the research hypothesis, not just implementation correctness.

Use identical settings for all methods:

```text
model = Qwen/Qwen3.5-27B
block_size = 128
target_block_sparsity = 0.20
max_prune_ratio_per_matrix = same value
min_keep_blocks_per_matrix = same value
share_up_gate_mask = same value
calibration_dataset = s1k
calibration_samples = 128
sequence_length = 0
seed = 42
pruning_rounds = 1
dtype = bfloat16
```

Methods:

```text
magnitude
fisher
fisher_budget_wanda
```

- [ ] **Step 13.1: Generate all three checkpoints**

Use `run_baselines.sh` after updating `METHODS` or execute the Python entry explicitly.

- [ ] **Step 13.2: Verify equal actual global sparsity**

Read each `pruning_summary.json` and require identical `num_total_blocks` and `num_pruned_blocks`.

- [ ] **Step 13.3: Evaluate PPL with the same script and dataset**

Use the repository's existing `Block_Sparse/scripts/eval_ppl.py` or `eval_ppl.sh` path without changing evaluation data between methods.

- [ ] **Step 13.4: Compare module budgets**

Use `module_prune_budget.csv` to inspect:

- layer-wise sparsity;
- projection-wise sparsity;
- whether `down_proj` receives excessive budgets;
- whether early or late Transformer layers receive extreme budgets.

- [ ] **Step 13.5: Compare Fisher/Wanda coordinate IoU**

Use `hybrid_per_matrix_report.csv` to identify matrices where Fisher and Wanda select very different coordinates.

- [ ] **Step 13.6: State results without overclaiming**

Primary expected inequality:

\[
\operatorname{PPL}_{\mathrm{FisherBudget+Wanda}}
<
\operatorname{PPL}_{\mathrm{Fisher}}.
\]

Stronger desired result:

\[
\operatorname{PPL}_{\mathrm{FisherBudget+Wanda}}
\le
\operatorname{PPL}_{\mathrm{Magnitude}}.
\]

If the stronger result fails, do not immediately add score mixing or heuristics. First inspect budget stability and activation-statistics correctness.

---

## Failure Diagnosis Order

If `fisher_budget_wanda` does not improve over direct Fisher selection, investigate in this order:

1. Confirm the temporary Fisher mask was never applied before Wanda scoring.
2. Confirm `down_proj` hooks observe the actual post-SwiGLU activation.
3. Confirm RMS denominator counts all flattened token/position rows exactly once.
4. Confirm Block-Wanda uses `abs(W) * input_rms`, not squared weights or squared RMS.
5. Confirm Wanda ranking is local to each matrix.
6. Confirm final per-module counts exactly equal Fisher budgets.
7. Confirm shared up/gate masks use the joint score and identical coordinates.
8. Inspect whether Fisher gives extreme matrix sparsity distributions.
9. Measure Fisher budget stability across calibration seeds or subsets.
10. Only after the above checks, consider a separate research plan for budget smoothing. Do not add smoothing in this implementation.

## Completion Criteria

The implementation is complete only when all conditions below are met:

- `fisher_budget_wanda` is accepted by config and CLI.
- Fisher reference allocation determines exact per-module cumulative budgets.
- The Fisher reference mask is not applied to weights.
- Block-Wanda uses actual input-channel RMS for every target matrix.
- Final Wanda selection is local to each matrix.
- Final per-module counts exactly equal Fisher budgets.
- Existing constraints and shared up/gate semantics are enforced.
- Multi-round masks are monotonic and reach exact cumulative budgets.
- Hybrid artifacts expose both reference and final decisions.
- Existing methods pass regression tests unchanged.
- Full `Block_Sparse/tests/` passes in conda environment `hif4`.
- A real-model smoke checkpoint exports and loads.
- Magnitude, Fisher, and hybrid PPL are evaluated under identical settings.

## Out of Scope

Do not implement any of the following as part of this plan:

- Fisher/Wanda weighted score fusion;
- per-layer budget smoothing;
- hand-designed lower or upper sparsity bounds beyond existing matrix constraints;
- calibration-set ensembling;
- activation-aware gradient score variants;
- SparseGPT reconstruction compensation;
- post-pruning fine-tuning;
- attention pruning;
- MoE expert pruning;
- block-sparse CUDA kernels;
- automatic hyperparameter search.
