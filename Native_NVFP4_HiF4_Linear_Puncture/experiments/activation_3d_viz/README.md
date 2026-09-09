# Qwen3-8B QAT：激活 3D 可视化历史实验

**模型归属：`ISTA-DASLab/Qwen3-8B-FPQuant-QAT-NVFP4`。**

本目录读取旧 Qwen3-8B Linear puncture 保存的 post-rotation activation capture，用于观察 HiF4 group/sub-group 内的数据分布和局部结构。

默认 source capture 为 `20260812T103800Z_native_nvfp4_hif4_linear_puncture`。该实验属于旧 8B 机制分析，不属于当前 `nvidia/Qwen3-30B-A3B-NVFP4` MoE 主线。

执行入口：`run_plot.sh`；核心绘图逻辑：`plot_activation_3d.py`。
