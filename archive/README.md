# 归档入口

当前维护实现和新实验入口在 [fp4_convert](../fp4_convert/README.md)。这里用于历史参考与论文证据。

| 目录 | 用途 |
| --- | --- |
| [legacy/](legacy/README.md) | 已退出主线的历史实现和实验定义 |
| [reference/](reference/README.md) | 本地上游示例与跨硬件参考导航 |
| `native_nvfp4_evidence/` | 27 个论文及历史报告引用的冻结文件，仅保留本地，不能当作完整旧包运行 |

旧 `Native_NVFP4_HiF4_Linear_Puncture/experiments` 链接到 `native_nvfp4_evidence/`，使论文中的旧路径继续读取原字节。此处 `plans`、`results` 也是本地链接，分别指向旧目录的计划和实验产物，用于保持冻结说明文件里的相对链接；没有复制或移动这些计划、模型及运行结果。三个链接及证据目录均不纳入 Git。

完整 786 个冻结源码文件另存于[源码归档](../results/repository_organization/20261126T015013Z/frozen_source_20261120.tar.gz)，哈希见[归档清单](../results/repository_organization/20261126T015013Z/frozen_source_manifest.json)。历史复现须先恢复完整版本；当前开发不导入这里的证据片段。

整理边界及验证见[仓库目录指南](../docs/guides/repository_layout.md)。
