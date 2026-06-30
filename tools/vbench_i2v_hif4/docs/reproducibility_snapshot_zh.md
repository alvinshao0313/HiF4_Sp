# 复现环境与资产快照说明

本目录用于说明本工具包在一次已跑通的 VBench-I2V 官方 10 维评测中所使用的环境、数据与模型资产。它不是把 VBench 数据集或外部模型权重打包进 GitHub；GitHub 提交包只包含依赖清单、路径清单、资产清单和代码 patch 摘要。

## 1. 已包含的环境快照

路径：

```text
reproducibility_archive/env_snapshots/vbench_i2v_official_20260630_160840/
```

关键文件：

```text
conda_env_full.yml
conda_env_from_history.yml
conda_list_explicit.txt
requirements.freeze.txt
requirements-vbench-key-pinned.txt
vbench_env_report.txt
vbench_paths_check.txt
vbench_paths_snapshot.txt
pip_check.txt
```

该环境快照记录的关键版本包括：

```text
Python 3.10.20
torch 2.7.1+cu118
torchvision 0.22.1+cu118
vbench 0.1.5
dreamsim 0.2.1
timm 1.0.12
open-clip-torch 2.24.0
transformers 4.33.2
peft 0.5.0
```

`vbench_env_report.txt` 中记录的 VBench-I2V API 形式为：

```text
VBenchI2V.__init__(self, device, full_info_dir, output_path)
VBenchI2V.evaluate(self, videos_path, name, dimension_list=None, local=False, read_frame=False, custom_prompt=False, resolution='1-1', **kwargs)
```

## 2. 已包含的资产清单

路径：

```text
reproducibility_archive/asset_manifests/
```

包含：

```text
vbench_assets_manifest_summary.txt
vbench_data_files.tsv
vbench_model_files.tsv
```

它们记录了已跑通环境中 VBench 数据和模型的大致规模，例如：

```text
vbench2_i2v_full_info.json: 357K
image_folder data/crop/16-9: 359M, 355 files
VBench code: 4.3G
CoTracker checkpoint: 195M
CoTracker local repo: 36M
Torch cache: 388M
```

注意：这些是清单，不是模型/数据本体。完整复现仍需要用户按路径准备 VBench 数据、图片和外部模型权重。

## 3. 已包含的代码状态快照

路径：

```text
reproducibility_archive/git_diffs/
```

包含：

```text
HiF4_Sp_git_status_and_diff.txt
VBench_git_status_and_diff.txt
Wan2.2-I2V-A14B-W4A4_git_status_and_diff.txt
```

其中 Wan 仓库记录到了实际修改状态；HiF4_Sp / VBench 的文件若只有标题，说明当时命令没有捕获到 git hash 或 diff。若需要严格论文级复现，请在原机器上再次导出对应仓库的 commit hash、remote URL 和 patch diff。

## 4. 已包含的结果表

路径：

```text
reproducibility/result_tables/i2v10_four_blocks_raw_and_signed_deltas.csv
```

该表按照四大块组织：

```text
原始值
相对 BF16
相对 W4A4-empty
相对 W4A4-searched
```

所有 delta 均带正负号，方便直接放入实验报告。

## 5. 当前已知非阻塞问题

`pip_check.txt` 中记录：

```text
open-clip-torch 2.24.0 requires sentencepiece, which is not installed.
decord 0.6.0 is not supported on this platform
```

实际评测已经在该环境中跑通。为了迁移更稳，建议新环境中额外安装：

```bash
python -m pip install sentencepiece
```

如果 decord 在新机器上读视频失败，可重装：

```bash
python -m pip uninstall -y decord
python -m pip install decord==0.6.0
```

或使用 conda-forge：

```bash
conda install -c conda-forge decord -y
```

## 6. 建议的重建顺序

优先使用 conda 环境文件：

```bash
cd reproducibility_archive/env_snapshots/vbench_i2v_official_20260630_160840
grep -v '^prefix:' conda_env_full.yml > conda_env_full_noprefix.yml
conda env create -n vbench_i2v_rebuild -f conda_env_full_noprefix.yml
conda activate vbench_i2v_rebuild
python print_vbench_env_report.py
```

若 PyTorch CUDA wheel 解析失败，建议分步安装 torch：

```bash
conda create -n vbench_i2v_rebuild python=3.10 -y
conda activate vbench_i2v_rebuild
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
grep -vE '^(torch|torchvision|torchaudio)==' requirements.freeze.txt > requirements.no_torch.txt
python -m pip install -r requirements.no_torch.txt
python -m pip install sentencepiece
python print_vbench_env_report.py
```

## 7. 提交包层面的结论

本提交包已经包含工具包代码、中文说明、VBench 兼容环境快照、资产清单、代码状态摘要和最终结果表。它足以作为 GitHub 提交材料。

它没有也不应该包含 VBench 数据集、外部模型权重和缓存目录本体；这些资产需要用户按 README 和本文件中的路径说明自行准备。
