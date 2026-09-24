# 历史无用数据清理记录（2026-09-24）

范围：`/home/shaoyuantian/program/HiF4_Sp`。依据本轮用户“清理一下历史无用数据”的要求执行；未扩大到 VAELLM、home 缓存或共享模型。

执行状态：删除已完成，目标清单如下。

## 已完成训练的冗余恢复状态

运行目录：`Native_NVFP4_HiF4_Linear_Puncture/results/residual_lora_compensation/20260922T012547Z/`。

三组 manifest 均包含完整的 0–47 层，summary 均为 `complete=true`、`smoke_only=false`。使用 `hif4` 的 CPU mmap 读取确认：每份 resume 仅含 `layer=48`、`next_epoch=0` 和终层 `hidden`，无额外选中权重或优化器状态。144 份 `layers/NNN/selected.pt` 与三份 `initialization.pt` 均可读取。

已检查 `materialize.py`：后续导出读取 manifest、初始化和逐层 selected 权重，不读取 resume。训练完成不代表 materialize、TP2 验证或下游评测已完成；这些结果未在本次清理中补跑。

| 删除文件（相对上述运行目录） | 文件占用字节（分配块） | 原因 |
| --- | ---: | --- |
| `attention/resume.pt` | 6414880768 | 已完成 48 层，终层训练隐状态不再被导出使用 |
| `moe/resume.pt` | 6414880768 | 已完成 48 层，终层训练隐状态不再被导出使用 |
| `both/resume.pt` | 6414880768 | 已完成 48 层，终层训练隐状态不再被导出使用 |

保留三组各自的 manifest、summary、train.log、initialization、全部逐层 selected 权重、每轮指标和同输入 baseline 对照；保留初始模型、共享校准数据、E4 初始化和 group_top_mass 对照。清理后不能再从已删除的终层 resume 入口恢复，但原训练已完成且完整训练权重仍在。

## 无用测试缓存

只删除下列五处可重建的测试/已结束训练缓存；第三方目录、progressive 整个源码与结果目录、活动运行共享模块的缓存保持原状。

- `.pytest_cache`：5 个文件，20480 字节。
- `Native_NVFP4_HiF4_Linear_Puncture/tests/__pycache__`：1 个文件，4096 字节。
- `Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/__pycache__`：1 个文件，28672 字节。
- `Native_NVFP4_HiF4_Linear_Puncture/experiments/residual_lora_compensation/__pycache__`：7 个文件，94208 字节。
- `Native_NVFP4_HiF4_Linear_Puncture/experiments/non_equivalent_reconstruction/tests/__pycache__`：1 个文件，20480 字节。

## 删除前核对与保留边界

- 目标全部位于本项目，无符号链接越界、无 Git 跟踪文件、无多重硬链接；删除前再次核对 inode、大小和修改时间。未发现目标的符号链接引用或可见进程打开句柄。
- 全仓库配置、脚本和文档检索只发现 LoRA PLAN 对运行目录的记录，未发现额外消费这些终层 resume 的入口；后续导出需求明确保留。
- progressive launcher `3658198` 及 worker `2251529` / `2251530` 在盘点时仍运行，所在 `20260920T054500Z` 完整保护。
- E3/E4 candidate 仍有机制方案引用；KL captures、verification 有既有保护记录；正式对照模型、internal-error 证据和当前训练产物继续保留，不因日期较早删除。
- 多处模型文件已有硬链接（观察到 link count=25）；目录逻辑大小不能直接相加当作可释放空间。本次目标均为单链接文件。

## 证据标识

| 分支 | manifest SHA256 | summary SHA256 |
| --- | --- | --- |
| attention | `fe9cfeb8a0fda2290ff38c442fb2936b2035c48b00a9c4f0044e8ec4517bb649` | `e8709eade1534626425cf94446d5911a61b9c017c9189f5f199f3af78a77f411` |
| moe | `ff77a16761a2de46214f6e9f9e36f080f818f741165a49953ccbf76f174b7284` | `fe39dccd5e1c8d0bfe719ce1280700ed23eb0764a35c88b053821acd77db5d09` |
| both | `6b716bdcf8355a307cbb7dd0caed5f51bda87f7ad5746bd98063ac23025e6294` | `a14be6d67f1b6f5cb9a2d9ce7f4f423b29e3b273398005cf710c7bc731ec699b` |

## 删除后检查

- 8 个精确目标均已移除，共 18 个文件；删除文件逻辑大小 19,244,741,294 字节，原分配块占用 **19,244,810,240 字节（约 17.923 GiB）**。这是目标文件统计，不把共享磁盘其他任务的变化算成本次收益。
- 三组保留文件共 3471 份（含 144 份 selected 权重），均未改变 inode、大小或修改时间；小型记录另外核对 SHA256。
- progressive 源码哈希未变，launcher/两名训练 worker 仍在运行；未停止任何既有进程，未改动其结果目录。
- Git 暂存区条目哈希未变；vLLM、lighteval、NVFP4 和共享 R64 源码仍在。
- 本次仅做产物完整性与文档检查，没有启动训练、导出或下游评测。

相关经验：[目录归档与活动实验保护](../../lessons/workspace_protection.md)。此前清理见 [2026-09-20 记录](2026-09-20-experiment-cleanup.md)，其释放量不计入本次。
