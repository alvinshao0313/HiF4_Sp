# HiF4 VBench-I2V 兼容评测工具包

本目录为 HiF4 / Wan 视频生成结果补齐一层可复用的 **VBench-I2V 兼容评测层**。它不实现新的量化算法，也不声称 packed 4-bit GEMM 或推理速度提升；它解决的是：生成仓库已经产出 mp4 后，如何稳定整理输入、预检环境、运行官方 VBench-I2V 10 维评测、扫描失败维度并重试。

## 设计原则

- **不绑定 Slurm**：核心逻辑均为 `python -m hif4_vbench_i2v.*`，Slurm 只是 `examples/slurm_ustc_template/` 下的示例 wrapper。
- **不隐式修改 VBench 源码**：兼容处理集中在 `hif4_vbench_i2v/compat/`，运行时会打印提示；需要时可用 `--no-compat-patches` 关闭。
- **不提交模型、视频和结果大文件**：本目录只包含工具代码、文档、最小 smoke test 和依赖说明。
- **环境复杂性显式化**：VBench-I2V 依赖较脆弱，推荐使用独立评测环境；完整环境快照和资产表放在提交包根目录的 `reproducibility_archive/`，不混入核心工具目录。

## 推荐目录位置

在 HiF4_Sp 仓库中建议放置为：

```text
HiF4_Sp/
  tools/
    vbench_i2v_hif4/
      hif4_vbench_i2v/
      scripts/
      configs/
      docs/
      examples/slurm_ustc_template/
      requirements-vbench.txt
      requirements-vbench-key-pinned.txt
      pyproject.toml
```

安装为 editable 包：

```bash
cd /path/to/HiF4_Sp
pip install -e tools/vbench_i2v_hif4
```

## 环境建议

最稳妥的方式是 split-env：

```text
生成环境：原 HiF4 / Wan 环境，只负责生成 mp4。
评测环境：独立 vbench_i2v_official 环境，只负责 VBench-I2V。
```

原因是 VBench-I2V 会牵涉 `torch / torchvision / transformers / peft / dreamsim / timm / open_clip / decord / cv2` 等依赖，和生成环境共用时容易发生版本冲突。

安装建议：

```bash
conda create -n vbench_i2v_official python=3.10 -y
conda activate vbench_i2v_official
pip install -r tools/vbench_i2v_hif4/requirements-vbench.txt
pip install -e tools/vbench_i2v_hif4
```

若当前机器环境与已验证环境差异较大，优先参考提交包根目录：

```text
reproducibility_archive/env_snapshots/
reproducibility_archive/asset_manifests/
```

其中包含 key pinned requirements、pip/conda 快照、导入报告和 VBench 资产表。路径已脱敏，主要用于对照版本和资产完整性，而不是要求逐字复现绝对路径。

## 最短流程

### 1. 预检 VBench 环境、图片和外部模型

```bash
python -m hif4_vbench_i2v.preflight \
  --vbench-root /path/to/vbench2_beta_i2v \
  --full-info-json /path/to/vbench2_i2v_full_info.json \
  --image-folder /path/to/vbench2_beta_i2v/data/crop/16-9 \
  --model-root /path/to/VBench/models \
  --scratch-dir .scratch_vbench_i2v \
  --json-report .scratch_vbench_i2v/preflight.json
```

成功标记：

```text
VBENCH_PREFLIGHT_OK
```

### 2. 构建 HiF4 评测输入

VBench-I2V 官方采样语义是：**每个 image-prompt pair 真实采样 5 个视频**，命名为：

```text
<prompt>-0.mp4, <prompt>-1.mp4, <prompt>-2.mp4, <prompt>-3.mp4, <prompt>-4.mp4
```

因此本工具的输入构建器只做“整理/复制”，不负责生成视频，也不会把一个 `<prompt>-0.mp4` 复制成 5 个 repeat。假设已有旧 case 模板和 HiF4 新生成的 exact-repeat 视频：

```text
outputs/evaluation_inputs/empty_seed42/        # 旧 BF16/W4A4 的 VBench 输入模板
outputs/hif4_seed42_exact_repeats/*.mp4        # 新生成视频，必须已经包含 prompt-0..prompt-4
```

执行：

```bash
python -m hif4_vbench_i2v.build_eval_inputs \
  --template-case outputs/evaluation_inputs/empty_seed42 \
  --generated-dir outputs/hif4_seed42_exact_repeats \
  --out-case outputs/evaluation_inputs/hif4_seed42 \
  --copy-mode physical
```

如果源目录缺少任何一个模板要求的 exact repeat，例如只有 `base-0.mp4` 而没有 `base-1.mp4` 到 `base-4.mp4`，命令会直接失败。这是刻意设计，用来避免非官方的“复制一个视频凑 5 个 repeat”。

默认会禁止 symlink。仅在确认所有评测节点都能访问同一路径时才使用：

```bash
--copy-mode symlink --allow-symlink
```

### 3. 重建或确认 exact repeats

`repair_exact_repeats` 只会从已经真实生成的同名 repeat 文件中重建 quant 目录；它不会补造缺失 repeat。若缺少 `-1` 到 `-4`，应回到生成阶段按不同 seed/index 补生成。

```bash
python -m hif4_vbench_i2v.repair_exact_repeats \
  --template-case outputs/evaluation_inputs/empty_seed42 \
  --case-input outputs/evaluation_inputs/hif4_seed42 \
  --generated-dir outputs/hif4_seed42_exact_repeats
```

验收：

```bash
python -m hif4_vbench_i2v.validate_case_input \
  --case-input outputs/evaluation_inputs/hif4_seed42 \
  --expected-sb 200 \
  --expected-camera 100 \
  --expected-repeats 5 \
  --forbid-symlink
```

成功标记：

```text
VALIDATE_CASE_INPUT_OK
```

默认验收还会检查同一 prompt 的 5 个 repeat 是否 SHA256 完全相同；如果完全相同，会判定为疑似复制凑数并失败。极少数确有理由允许完全相同文件时，才显式添加 `--allow-identical-repeat-files`。

### 4. 运行官方 10 维评测

```bash
python -m hif4_vbench_i2v.run_eval_10dim \
  --case-input outputs/evaluation_inputs/hif4_seed42 \
  --output-dir outputs/evaluation_results/hif4_seed42/quant \
  --image-folder /path/to/vbench2_beta_i2v/data/crop/16-9 \
  --mode quant \
  --dims all \
  --device cuda:0 \
  --continue-on-error
```

`run_eval_10dim` 的 full_info 选择规则是显式的：优先选择文件名包含当前维度的 `*full_info*.json`；如果目录只有一个 full_info，则使用它；如果多个候选无法判断，会报错。确需回退时加：

```bash
--allow-full-info-fallback
```

### 5. 扫描缺失维度

```bash
python -m hif4_vbench_i2v.scan_missing \
  --out-base outputs/evaluation_results \
  --cases hif4_seed42 hif4_seed43 hif4_seed44 \
  --modes quant \
  --dims all
```

成功标准：

```text
TOTAL_MISSING=0
```

### 6. 只重试缺失维度

```bash
python -m hif4_vbench_i2v.retry_missing \
  --missing-tsv outputs/evaluation_results/_scan/missing_jobs.tsv \
  --input-base outputs/evaluation_inputs \
  --out-base outputs/evaluation_results \
  --image-folder /path/to/vbench2_beta_i2v/data/crop/16-9 \
  --devices 0,1 \
  --max-rounds 5
```

优化后每轮都会重新调用 `scan_missing`，不会反复跑已经修好的旧 missing 项。

## 单机多 GPU 并行

```bash
python -m hif4_vbench_i2v.parallel.run_parallel_cases \
  --cases hif4_seed42 hif4_seed43 hif4_seed44 \
  --input-base outputs/evaluation_inputs \
  --out-base outputs/evaluation_results \
  --image-folder /path/to/vbench2_beta_i2v/data/crop/16-9 \
  --mode quant \
  --devices 0,1 \
  --max-workers 2
```

默认不会让多个 VBench worker 共享同一 GPU。确需超售 GPU 时显式添加：

```bash
--allow-device-oversubscribe
```

## 本地 smoke test

不需要 GPU，也不需要真实 VBench：

```bash
bash tools/vbench_i2v_hif4/scripts/local_smoke_test.sh
```

预期输出：

```text
LOCAL_SMOKE_TEST_OK
```

## 文档

```text
docs/generation_contract_zh.md        VBench-I2V 生成阶段输入契约
docs/troubleshooting_zh.md              常见故障
docs/opencode_local_test_plan_zh.md     本地审查/测试方案
docs/reproducibility_archive_zh.md      环境快照和资产表说明
examples/slurm_ustc_template/           Slurm 示例 wrapper，需按集群修改
```

## 外部资源核对表

VBench-I2V 首次运行可能触发外部模型和图像资产下载。为了便于离线/半离线复现，本提交包保留了：

```text
tools/vbench_i2v_hif4/docs/vbench_asset_checklist_zh.md
reproducibility_archive/asset_manifests/vbench_required_assets_minimal.tsv
reproducibility_archive/asset_manifests/vbench_data_files.tsv
reproducibility_archive/asset_manifests/vbench_model_files.tsv
```

其中 `vbench_required_assets_minimal.tsv` 是最小人工核对表，两个大 TSV 是已跑通环境的逐文件清单。
