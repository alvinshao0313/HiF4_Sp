# H-only 解析更新与 CPU Linear 验证

独立验证可融合的 G64 块矩阵，不修改现有转换/训练入口，不优化 norm。当前入口只在 CPU
运行原生 Qwen3-30B-A3B-NVFP4 的第0层 `q_proj`，不加载整个 MoE、不启动 GPU 或下游评测。
这落实了用户2026-11-05的“先准备代码，CPU测试一个Linear”范围；此前完整第0层代表投影短测暂不执行。

## 求解定义

存储权重 `W` 为 `[out,in]`，原融合矩阵为 `M`，此处优化 `H=M.T`；初值为 `R64.T`。
固定 `A=Q_h(X R64)`、`B=W.T`，教师 `Y` 是同输入的原生 NVFP4 Linear 输出，包含原生激活量化、
QKV全局scale合并与最终BF16舍入。学生 `V=Q_h((HB).T).T` 按每个输出的输入维G64量化。

一次累计完整 `G=A.T A/N`、`C=A.T Y/N`；维护 `S=GV-C`，包含跨块相关性。
冻结当前量化残差，只求第i块增量的二次代理：

```text
min_delta  || A V - Y + A_i delta_H_i B_i ||_F^2 / N + lambda_i ||delta_H_i||_F^2

G_ii delta_H_i K_i + lambda_i delta_H_i = -S_i B_i.T
K_i = B_i B_i.T
lambda_i = 0.01 * mean(diag(G_ii)) * mean(diag(K_i))
```

两次64维特征分解给出闭式解；Gram/求解使用FP64，H保存FP32，量化操作数和Linear输出为BF16。
零输入/权重能量块保持原值；其余块正阻尼允许Gram秩亏。
候选按 `1, 1/2, 1/4, 1/8, 1/16` 回溯，每次重新执行真实HiF4权重量化。接受第一个下降的候选：

```text
delta_V_i = Q_h((H_i + alpha delta_H_i) B_i) - V_i   # 仍按上述转置后的输入维量化
delta_loss = 2 <delta_V_i, S_i> + <delta_V_i, G_ii delta_V_i>
S += G[:,i] delta_V_i
```

缓存损失对应BF16量化操作数的实数乘积，尚未包含GEMM输出BF16舍入；每扫块结束均用实际BF16
Linear输出验证，初始R64也参与选优。原含量化目标非凸，本实现不声称其全局闭式解，也未实现ADMM。

## 固定的 CPU 诊断

- 原生权重4096×2048，32个64×64块；norm和基础权重保持原生值。
- 从已有隔离校准池选最长4条train、4条validation，每条均匀取64个有效token位置，共256+256。
  第0层attention之前的embedding/RMSNorm逐token独立，无需为这些Linear输入执行完整上下文。
  这不适用于更深层或专家投影，亦不是完整序列校准的替代证据。
- Adam 20步、lr=3e-4、无weight decay、固定学习率；解析法2次顺序扫块。两者同输入、目标和初始化。
- seed=42、CPU两线程。每步/扫块分别记录训练与验证MSE/NMSE，验证MSE最小者选优。
  验证集参与选优，不是独立测试集。两种固定预算不是等墙钟或等收敛预算。
- 分别记录数据准备、统计构建、特征分解、含测量总耗时和扣除测量的求解耗时。
  两者测量次数不同，CPU时间不可外推GPU或整模型转换速度。

在仓库根目录，由调用者激活 `hif4` 后执行；输出目录必须尚不存在：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_only_analytic.test_solver -v
CUDA_VISIBLE_DEVICES='' python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_only_analytic.run \
  --output Native_NVFP4_HiF4_Linear_Puncture/results/matrix_only_analytic/my_cpu_run
```

`manifest.json`保存模型revision、实际读取权重/scale及选中embedding摘要、样本ID/token位置、代码哈希和配置。
`report.json`与`progress.jsonl`保留完整曲线，`adam.pt`/`analytic.pt`保留选优和末步H。
运行末尾核验基础权重/norm/输入未改变、H保存重载逐位一致、重载后验证指标一致、CUDA未初始化。
`REPORT.md`给出简表，`status.json`记录PID、命令、状态和退出码。

这是HiF4 W4A4 QDQ的数值诊断，不是packed HiF4计算核、全层蒸馏或五任务精度结论。
长期目标和历史结果见[总体计划](../matrix_norm_efficiency/PLAN.md)。

首轮[CPU实测记录](../../../docs/experience/records/matrix_only_analytic_20261105.md)：解析训练误差下降，
验证误差退化而选优保留初值；不能据该运行声称有效替代Adam或完成转换加速。
