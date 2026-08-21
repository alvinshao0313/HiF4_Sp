# AX4：Scale System 与 Payload 交叉因子

## 1. 实验目标

区分 HiF4 激活损失主要来自 Scale System 还是 Payload Grid。

## 2. 实验设置

- 真实组合：NN / HH
- 诊断 Hybrid：HN / NH，并做 raw 与 range-matched
- Hybrid 标记 `is_valid_hardware_format=false`
- 样本行数：4032

## 3. 实验结果

- HH: 平均 R_Y=0.0000
- HN: 平均 R_Y=-0.6319
- NH: 平均 R_Y=-38.2299
- NN: 平均 R_Y=1.0000

## 4. 实验分析

若 NH（NVFP4 Scale + HiF4 Payload）明显更好，说明 Scale System 是主因；若 HN 更好，则 Payload Grid 更关键。raw 与 range-matched 若结论不同，需把动态范围与网格形状分开讨论。

## 5. 实验结论

- Payload 主导
- NH 聚合 R_Y≈-38.2299；HN 聚合 R_Y≈-0.6319

## 6. 对算法设计的启示

下一步应优先优化结论指向的部件，而不是同时大改 Scale 与 Payload。
