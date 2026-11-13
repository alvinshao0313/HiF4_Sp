# H-only 闭式代理的 CPU 单 Linear 验证（2026-11-05）

状态：代码完成，CPU真实Linear诊断完成且产物核验通过；没有启动GPU、完整decoder层或正式下游评测。
本次实现闭式代理与真实QDQ回溯，未实现ADMM。用户先确认H-only独立实现，再将当前运行范围缩小为CPU单Linear。

## 实际配置与证据

- 模型：`nvidia/Qwen3-30B-A3B-NVFP4`，snapshot `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`。
- 第0层`q_proj`真实4096×2048权重，32个G64矩阵；原生norm/基础权重、activation R64固定。
- 既有隔离校准池最长4条train与4条validation，每条均匀取64个token，256+256；训练3 code+1 math，验证4 code。
  层0的embedding/RMSNorm逐token独立，本次输入无需执行上下文；不外推为更深层或专家投影的采样方法。
- H-only Adam20步、lr=3e-4、无weight decay、固定学习率；闭式代理2次顺序扫块，阻尼0.01，真实QDQ回溯。
  公式、精度及量化轴见[实现说明](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/matrix_only_analytic/README.md)。
- `hif4` Python3.11.13、PyTorch2.10.0+cu128，CPU两线程，seed=42；CUDA未初始化。
- RUN：`Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z`，PID2232623，退出码0。
  [生成报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z/REPORT.md)、
  [完整指标/核验](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z/report.json)、
  [源码与输入摘要](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z/manifest.json)、
  [stdout/stderr](../../../Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z.log)。

实际启动命令（仓库根目录、调用者已激活hif4，输出同时经tee保存到上述log）：

```bash
CUDA_VISIBLE_DEVICES='' python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_only_analytic.run \
  --output Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/cpu_linear_20261105T085702Z
```

目录已存在，复现时使用新输出路径。未提交源码以manifest中逐文件SHA256绑定，不伪造commit。

## 结论与限制

闭式代理两扫块分别接受32/32和23/32个块。末步训练NMSE下降46.60%，验证NMSE却比R64初值升高5.26%，
验证选优因此保留初始H；Adam末步训练NMSE下降20.63%，验证NMSE下降12.47%，选中第20步。
这证实当前解析更新能降低真实QDQ后的局部训练误差，但未证明能提供有效的留出集补偿。

解析总计5.687秒、Adam22.590秒，包含不同次数的测量；扣除测量后为3.738秒与8.343秒，约2.23倍。
解析时间包含0.119秒统计构建和0.055秒特征分解。公共数据准备1.704秒，运行内总墙钟30.726秒，
不含Python/依赖导入。仅为本机单次CPU诊断，不能作为等质量加速、GPU加速或20小时转换目标达成证据。

训练256个token中仅131个不同token ID；本层输入逐token独立，所以完整2048维输入Gram的秩至多131。
这一有限覆盖与训练改善/验证退化的现象符合小样本过拟合解释，但尚未隔离数据覆盖、阻尼及代理更新的贡献。
不据此断言闭式法普遍无效，也不推断改用ADMM便会改善泛化；后续若研究，先区分覆盖与约束的作用。

四项CPU回归已通过：非对称H融合及G64量化轴、缓存损失和跨块残差恒等式、秩亏Gram的正则Sylvester求解、
真实QDQ的Adam/解析路径及选优保存重载。真实q_proj运行另外核验基础权重/norm/输入不变、H逐位重载、
重载验证指标一致、源文件未变化；完整曲线和末步H均保留，未只保存选优初值。

缓存实数乘积训练MSE与最终BF16 Linear MSE有小差别，符合已明确的输出舍入边界；验证退化发生在真实BF16输出上，
不是仅看代理分数。相关经验并入[量化与解析代理的边界](../lessons/fake_quant_boundaries.md)。
