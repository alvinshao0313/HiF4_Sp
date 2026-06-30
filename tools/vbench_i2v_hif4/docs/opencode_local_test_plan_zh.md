# 本地电脑使用 opencode 的测试方案

目标：在不依赖 Slurm、不依赖真实 GPU、不依赖真实 VBench 的情况下，先让 opencode 帮你检查工具包结构、CLI、基础输入构建逻辑；然后再切到真实 VBench 环境做集成测试。

---

## 阶段 A：纯本地静态/烟雾测试

在本地电脑解压最新提交包后。如果已经把 `tools/vbench_i2v_hif4/` 合入 HiF4_Sp 仓库，也可以直接进入该目录执行：

```bash
cd hif4_vbench_i2v_submit_assets_checklist/tools/vbench_i2v_hif4
python -m venv .venv
source .venv/bin/activate
pip install -e .
bash scripts/local_smoke_test.sh
```

预期：

```text
VBENCH_PREFLIGHT_OK
BUILD_EVAL_INPUTS_OK
VALIDATE_CASE_INPUT_OK
TOTAL_OK=1
TOTAL_EXPECTED=1
TOTAL_MISSING=0
LOCAL_SMOKE_TEST_OK
```

这个测试不需要真实 VBench 包，也不需要 GPU。

---

## 阶段 B：让 opencode 做代码审查

可以给 opencode 的任务：

```text
请审查 tools/vbench_i2v_hif4 工具包。目标是让原 HiF4/Wan 视频生成仓库能兼容 VBench-I2V 官方 10 维评测。重点检查：
1. CLI 参数是否一致；
2. 是否有硬编码用户路径；
3. 是否默认依赖 Slurm；
4. build_eval_inputs / repair_exact_repeats 是否严格按完整文件名复制 exact repeats，不能把单个 base-0.mp4 复制成 base-1..4.mp4；
5. run_eval_10dim 是否把 VBench 真实调用隔离在核心 runner 中；
6. scan_missing / retry_missing 是否能支持部分结果重试；
7. README 中是否清楚区分 strict same-env 与 split-env。
请直接修改代码和 README。
```

---

## 阶段 C：真实 VBench 依赖预检

在有 VBench 环境的机器上：

```bash
pip install -e .
python -m hif4_vbench_i2v.preflight \
  --vbench-root /path/to/vbench2_beta_i2v \
  --full-info-json /path/to/vbench2_i2v_full_info.json \
  --image-folder /path/to/vbench2_beta_i2v/data/crop/16-9 \
  --model-root /path/to/VBench/models \
  --scratch-dir .scratch_vbench_i2v
```

预期：

```text
VBENCH_PREFLIGHT_OK
```

---

## 阶段 D：小样本真实集成测试

先不要跑 10 维全量。建议只构造一个 seed，一个维度：

```bash
python -m hif4_vbench_i2v.run_eval_10dim \
  --case-input evaluation_inputs/hif4_seed42 \
  --output-dir evaluation_results/hif4_seed42/quant \
  --image-folder /path/to/vbench2_beta_i2v/data/crop/16-9 \
  --mode quant \
  --dims i2v_subject \
  --device cuda:0 \
  --continue-on-error
```

再扫描：

```bash
python -m hif4_vbench_i2v.scan_missing \
  --out-base evaluation_results \
  --cases hif4_seed42 \
  --modes quant \
  --dims i2v_subject
```

---

## 阶段 E：多 GPU 并行测试

```bash
python -m hif4_vbench_i2v.parallel.run_parallel_cases \
  --cases hif4_seed42 hif4_seed43 \
  --input-base evaluation_inputs \
  --out-base evaluation_results \
  --image-folder /path/to/vbench2_beta_i2v/data/crop/16-9 \
  --mode quant \
  --dims i2v_subject \
  --devices 0,1 \
  --max-workers 2
```

---

## 阶段 F：全量 10 维测试

```bash
python -m hif4_vbench_i2v.parallel.run_parallel_cases \
  --cases hif4_seed42 hif4_seed43 hif4_seed44 \
  --input-base evaluation_inputs \
  --out-base evaluation_results \
  --image-folder /path/to/vbench2_beta_i2v/data/crop/16-9 \
  --mode quant \
  --dims all \
  --devices 0,1 \
  --max-workers 2

python -m hif4_vbench_i2v.scan_missing \
  --out-base evaluation_results \
  --cases hif4_seed42 hif4_seed43 hif4_seed44 \
  --modes quant \
  --dims all
```

成功标准：

```text
TOTAL_OK=30
TOTAL_EXPECTED=30
TOTAL_MISSING=0
```
