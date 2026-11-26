# 仓库目录与文件归属

返回[仓库首页](../../README.md)。仓库主线是 **NVFP4→HiF4 / MXFP4 的实现、GPU 验证及可复现实验**。本页说明迁移后的职责与历史路径保留边界；盘点与验证证据见[整理记录](../experience/records/maintenance/2026-11-18-repository-organization.md)。

## 本地与远程分别需要什么

“远程”指 Git 仓库应分发的内容。本次已获准调整暂存与跟踪清单，但尚未提交或推送。“本地保留”不等于允许删除；原始数据、权重和研究证据独立于 Git 管理。

| 内容 | 本地 | Git 远程 | 处理原则 |
| --- | --- | --- | --- |
| `fp4_convert/{src,experiments,tests,configs,scripts}` | 必需 | 必需 | 两条路线及共享实现；按[实验导航](../../fp4_convert/experiments/README.md)查找 |
| `archive/native_nvfp4_evidence/` 与旧 `experiments` 链接 | 集中保存引用证据，完整版本另有归档 | 不再分发 | 27 个文件保持原字节和旧访问路径；旧运行入口须先恢复完整冻结源码 |
| `Native_NVFP4_HiF4_Linear_Puncture/plans/` | 论文与历史协议需要 | 必需 | 24 份计划原位保留，避免改动论文引用和现有未暂存修改 |
| `NVFP4/` 的解码、QDQ、脚本和测试 | 必需 | 必需 | 源格式及 GPU emulation 依赖，不能因名称旧而移除 |
| `HiFloat4/hif4_gpu`、`hif4gptq`、阈值量化器源码 | 必需 | 必需 | 当前转换及 vLLM 的共享依赖 |
| `3rdparty/vllm`、`3rdparty/lighteval` 的源码、构建配置、许可证 | 必需 | 必需 | 保留项目适配；构建生成文件另行忽略 |
| `main.py`、`tasks/`、`tools/`、安装入口 | 必需 | 必需 | 评测、评分、模型适配及维护工具 |
| 实验 `PLAN*.md`、`plans/`、配置、指南、经验和结果摘要 | 必需 | 应收录 | 解释算法和协议，不能全局忽略 JSON / YAML / 方案目录 |
| `results/`、checkpoint、模型视图、预测缓存、原始数据、日志 | 按实验需要保留 | 不直接放 Git | 保留 manifest、版本、哈希和评分证据；发布结果时按主题整理可审阅摘要 |
| `LaTeX/` | 论文需要 | 按既有忽略规则留本地 | 本轮不修改、不移动，也不重写它引用的原始证据 |
| `.vscode/`、`.cursor/`、`.codex` | 客户端按需 | 不收录 | 已退掉清单中的客户端跟踪项，保留本地配置和标记 |
| `.ai-bridge/` | 历史保留 | 不收录 | 原位保留审计、补丁和交接；旧审计依赖的 IPC 工程已不在工作区，当前主线不导入它 |
| `archive/legacy/` | 历史参考 | 可保留源码与文档 | 标明历史用途，不把已归档等同于无用 |
| `HiFloat4/hifx4_npu/` | 跨硬件参考 | 保留参考源码 | 恢复原实体目录以消除 Git 路径冲突；参考区仅导航，不存副本 |
| `archive/reference/llm_compressor_examples/` | 本地上游示例参考 | 不收录 | 137 个文件原样归档；旧 `NVFP4/llm-compressor` 的残缺 gitlink 已退出跟踪 |
| `__pycache__`、pytest 缓存、可再生编译中间文件 | 按实际使用情况 | 不需要 | 未检查活动引用前不清理；已安装扩展和第三方构建依赖不能一并删除 |
| `scripts/eval/` | 旧手动评测配置 | 可保留 | 已按模型与配置命名；原 `test.sh`、`scripts/test.sh` 通过符号链接保留原命令 |
| 根 pytest 缓存、空 `report/`、空 `.agents/`、`plans/.tmp_ignore` | 已按本次授权清理 | 不需要 | 核对无进程引用后删除，记录精确路径和大小；不推广为其他同名目录的删除授权 |

## 代码与产物的分界

源码中的 `data/`、`models/` 是常见模块名，不应全局忽略。例如 `experiments/e2e_diag_reconstruction/data/` 包含校准、batching 和 teacher trace 实现，训练与恢复流程直接依赖它。vLLM 的 `model_executor/models/` 则存放模型实现。

同样，JSON / YAML 既可能是配置，也可能是结果；应根据所在目录判断。新产物放在已有 `results/`、`outputs/`、`logs/`、`checkpoints/`、`cache/` 等产物位置，方法代码和有效配置放在源码侧。不移动历史结果来追求形式统一。

Git 忽略规则只影响未跟踪文件。改变规则后仍须单独审阅新增文件；此前已提交的客户端状态也不会自动消失。源码许可、配置与小型验证证据不能用“清理产物”一起移除。

## 2026-11-20 主包迁移与旧实验恢复

用户已选择迁到 `fp4_convert/`，并保留本地冻结源码。786 个源码、配置、测试和说明文件迁入新包，后续命令把模块前缀 `Native_NVFP4_HiF4_Linear_Puncture.` 改为 `fp4_convert.`；CLI 参数不变。示例：

```bash
python -m fp4_convert.experiments.thesis_format_conversion_supplement.workflow --help
python -m fp4_convert.experiments.qwen3_mxfp4.entry --help
python -m fp4_convert.experiments.paper_baselines.entry --help
```

迁移时原包源码保留本地并退出 Git 跟踪，供当时两组实验恢复。历史 `results/`、校准缓存、模型视图、默认输出位置与 `plans/` 保持原位；没有重写历史 manifest 或源码哈希。计划继续跟踪，论文工作区没有修改。新旧源码使用独立实体文件，没有目录符号链接或导入转发。

已冻结的旧 run 必须用原源码接续。新包用于新 run；即使算法一致，包名与源码路径变化也会改变版本哈希，不能覆盖旧 `source_hashes.json` 让校验放行。两组任务经用户要求恢复后触发通信超时，随后建立过独立接续队列；这些队列又于 2026-11-20 按用户要求停止，并非五轮完成。历史 PID、日志与失败证据见[恢复与剩余阶段接续记录](../experience/records/maintenance/2026-11-18-repository-organization.md#2026-11-20-恢复与剩余阶段接续)。

`NVFP4`、`HiFloat4` 和第三方 editable 安装路径保持稳定。实验内部共享代码沿用原有相对层级，HiF4、MXFP4 入口由实验导航分类，不在本次目录迁移中改算法或拆分运行时。

## 2026-11-26 旧源码清理后的边界

旧目录保留 `results/`、24 份 `plans/`、说明用途的 `README.md` 和一个 `experiments` 链接。27 个被论文/历史报告引用的文件实体已集中到 `archive/native_nvfp4_evidence/`；旧链接读取的字节和文件哈希不变。这些片段只供查证，不组成可运行的旧包。后续开发和新实验继续使用 `fp4_convert/`；它的部分配置仍引用旧目录中的校准数据、模型和结果，所以不能删除整个旧目录。

归档的 `archive/plans`、`archive/results` 分别链接到旧目录对应位置，以保持冻结说明的相对链接。只对这三个未跟踪的本地入口使用目录链接；已跟踪计划和维护代码的父目录仍为实体目录，避免再次出现 Git `beyond a symbolic link`。分类入口见[archive/README.md](../../archive/README.md)。

原 786 个冻结文件已完整保存为[本地源码归档](../../results/repository_organization/20261126T015013Z/frozen_source_20261120.tar.gz)，逐文件 SHA256 与迁移时一致。[归档清单](../../results/repository_organization/20261126T015013Z/frozen_source_manifest.json)记录压缩包及各成员的哈希，[保留清单](../../results/repository_organization/20261126T015013Z/protected_source_paths.json)记录 27 个文件的引用来源。归档和清理证据仅保留本地，不放入 Git。

历史脚本若仍导入旧包，需要先把归档解压到独立目录，按清单核对后恢复所需的完整原路径；仅凭保留的源码片段不能重跑旧实验。恢复源码不等于恢复实验授权，两组接续队列仍保持停止。清理详情及验证见[整理记录](../experience/records/maintenance/2026-11-18-repository-organization.md#2026-11-26-旧源码归档与无用文件清理)。
