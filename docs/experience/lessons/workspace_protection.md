# 目录归档与活动实验保护

证据状态：来自 [2026-09-20 清理记录](../records/maintenance/2026-09-20-experiment-cleanup.md) 与 [归档入口](../../../archive/legacy/README.md)。所列运行和容量属于当时，不代表现在仍相同。

## 目录名称不能决定是否可移除

旧实验目录可能含当前主线的共享实现。既有清理保留了 `experiments/diag_gradient/r64_transform.py`、`experiments/long_trajectory_stability/real_vllm_hooks/`、`HiFloat4/hif4_scale_threshold_optimization/src/` 和 `NVFP4/`；旧 8B 的其他源码才进入 `archive/legacy/`。

整理前检查当前 import、脚本引用、实际进程命令和后续任务。活动 progressive 的源码、配置、结果与 checkpoint 作为整体保护，不能因其属于某个历史父目录而一起移动。

## IDE 排除与物理删除分开

`3rdparty/vllm` 和 `3rdparty/lighteval` 是共享依赖，VS Code 的搜索或 watcher 排除不代表它们已从磁盘删除。目录不可见时先检查磁盘和配置，再解释状态。

## 原始证据与授权保留边界

历史清理释放空间的数字只描述对应操作，不能再计为本次收益。HiF4_Sp 默认保留已有结果；不把 VAELLM 的自动清理授权带入本项目。源码、结果是否继续使用必须基于当前证据判断，旧计划或旧清理记录不是新删除授权。

2026-09-24 的文档整理阶段只迁移正文和修复导航，未清理实验产物。同日后续获用户明确要求的清理另见下文。

## 完成训练后的恢复状态与空间统计

[2026-09-24 清理记录](../records/maintenance/2026-09-24-history-cleanup.md) 核实 residual LoRA 三组均完成 48 层；终层 `resume.pt` 仅保存下一层编号、轮次和 hidden，后续导出从初始化及逐层 `selected.pt` 读取参数。因此本轮在保留全部训练权重、指标和日志后，删除了三份冗余恢复状态。

这个判断依赖实际文件内容、完整的完成记录和导出调用链，不能推广成按文件名删除所有 resume。未完成训练、约定的续训或其他消费方仍需要的状态必须保留；训练完成也不代表下游评测完成。

盘点还发现部分模型文件已有多重硬链接。计算清理收益时要核对 inode、link count 和实际分配块；删除一个仍有其他硬链接的副本不会释放对应数据块。共享磁盘剩余空间会随其他任务变化，不能直接把 `df` 差值当成本次删除量。
