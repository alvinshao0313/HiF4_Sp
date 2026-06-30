# VBench-I2V 外部资源核对表

VBench-I2V 评测通常会在首次运行时尝试读取或下载外部模型、第三方代码和图像资源。为了让 HiF4 分支提交可复现，本提交包保留了两层资产核对材料：

```text
reproducibility_archive/asset_manifests/vbench_required_assets_minimal.tsv
reproducibility_archive/asset_manifests/vbench_assets_manifest_summary.txt
reproducibility_archive/asset_manifests/vbench_data_files.tsv
reproducibility_archive/asset_manifests/vbench_model_files.tsv
```

其中 `vbench_required_assets_minimal.tsv` 是给人看的最小核对表；`vbench_data_files.tsv` 和 `vbench_model_files.tsv` 是从已跑通环境导出的逐文件清单，用于离线机器上逐项比对。

## 最小必须核对项

| 资源 | 作用 | 已跑通环境中的规模 | 核对方式 |
|---|---|---:|---|
| `vbench2_i2v_full_info.json` | 官方 prompt/image 元信息 | 357K / 1 file | `preflight --full-info-json` |
| `data/crop/16-9` | VBench-I2V 官方输入图像 | 359M / 355 files | `preflight --image-folder` |
| `models/checkpoints/cotracker2.pth` | camera motion / motion 相关评测 | 195M / 1 file | `preflight --model-root` |
| `models/facebookresearch_co-tracker_main` | CoTracker 本地代码 | 36M / 87 files | `preflight --model-root` |
| DINO / OpenCLIP / CLIP 权重 | subject/background/aesthetic/semantic 等指标 | 见 `vbench_model_files.tsv` | 对照模型清单 |
| `cache/torch` | torch hub/cache 中的模型文件 | 388M / 94 files | 对照模型清单 |

## 推荐预检命令

```bash
python -m hif4_vbench_i2v.preflight \
  --vbench-root /path/to/vbench2_beta_i2v \
  --full-info-json /path/to/vbench2_beta_i2v/vbench2_i2v_full_info.json \
  --image-folder /path/to/vbench2_beta_i2v/data/crop/16-9 \
  --model-root /path/to/VBench/models \
  --scratch-dir .scratch_vbench_i2v \
  --json-report .scratch_vbench_i2v/preflight.json
```

看到 `VBENCH_PREFLIGHT_OK` 说明核心 import、full_info、图像目录、scratch 写入和已知 CoTracker 依赖通过。若只出现模型 warning，不一定立即失败，但说明后续某些维度可能触发联网下载或中途报错。

## 逐文件清单格式

两个 `.tsv` 原始清单的格式为：

```text
<size_bytes>    <mtime>    <path>
```

路径已经脱敏为 `${PROJECT_ROOT}`、`${CONDA_PREFIX}`、`${USER_HOME}` 等占位符。它们不是要求新机器使用相同绝对路径，而是用于确认：

1. 关键目录是否存在；
2. 文件数量是否大致一致；
3. 大模型文件大小是否明显缺失或为 0；
4. 离线环境是否已经避免运行时联网下载。

## 注意事项

本提交包只包含核对表和环境快照，不包含 VBench 图像资产、外部模型权重和 torch cache 本体。这样做是为了符合 GitHub 分支提交习惯，避免提交大文件、权重和缓存目录。完整复现时，用户仍需按 VBench 官方说明准备资源；本表用于判断资源是否已经准备齐全。
