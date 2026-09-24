# Progressive error cancellation

This directory is an independent 48-layer experiment. It does not read or
write the existing `kl_direction_reuse` or `non_equivalent_reconstruction`
result directories.

The formal matrix is:

```text
baseline, direct_l010, direct_l030, jvp_l010, jvp_l030, shuffled_l030
```

All runs use `group` G64 matrices, absolute hidden-output MSE, `top_mass`
router auxiliary loss, and the same E4/R64 initialization. A completed layer
freezes its progressive hidden boundary. The next layer receives either the
same-sample cumulative residual (`direct`), its native fixed-route STE JVP
(`jvp`), or a deterministic sample permutation (`shuffled`). The direction is
detached and written once per layer boundary.

Mathematical checks:

```bash
conda run --no-capture-output -n hif4 python -m \
  Native_NVFP4_HiF4_Linear_Puncture.experiments.progressive_error_cancellation.smoke
```

Prepare and run a formal configuration explicitly:

```bash
conda run --no-capture-output -n hif4 python -m \
  Native_NVFP4_HiF4_Linear_Puncture.experiments.progressive_error_cancellation.train \
  --output_dir /absolute/new/run \
  --method direct --lambda_direction 0.1
```

Use `--resume` only for an accounted run. The real short smoke is explicit:

```bash
conda run --no-capture-output -n hif4 python -m \
  Native_NVFP4_HiF4_Linear_Puncture.experiments.progressive_error_cancellation.smoke \
  --real --output_dir /absolute/new/smoke
```

JVP failures are fatal. The implementation never silently falls back to a
finite difference or an unverified derivative path.
