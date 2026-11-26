# 非 GPU 主线的参考实现

本目录保留有参考价值、当前 GPU 转换调用链不依赖的实现。与 [legacy](../legacy/README.md) 中按实验归档的历史代码分开；不把参考实现当作已在当前机器验证的后端。

## HiF4 NPU

NPU 源码保留在 [HiFloat4/hifx4_npu](../../HiFloat4/hifx4_npu/) 实体目录，28 个文件内容不变。来源与许可证沿用 [HiFloat4](../../HiFloat4/README.md) 和 [LICENSE](../../HiFloat4/LICENSE)。

此前把已跟踪目录替换为符号链接，会使编辑器按旧子文件路径暂存时报 `beyond a symbolic link`。已撤回该目录迁移，恢复实体路径；本目录仅提供分类导航，不保留源码副本。原构建脚本需要 Ascend / `torch_npu`，本次未安装依赖或运行 NPU 验证。

## llm-compressor 上游示例（仅本地）

`llm_compressor_examples/` 保存原 `NVFP4/llm-compressor/` 下的 137 个示例文件，文件内容不变。旧 Git 条目指向 `vllm-project/llm-compressor` 的 commit `1a20a616d6ea7cd395ca923a8c8387e189b8cefd`，但工作区没有完整子模块元数据；仅据此记录来源，不声称本地示例逐文件等同该 commit。

示例不是当前 GPU 运行依赖，不随主仓库分发；不存在时不影响转换入口。原始文档保留原样。已移除没有 `.gitmodules` 的旧 gitlink，不新增子模块或安装依赖。

具体路径、Git 状态和校验见[整理记录](../../docs/experience/records/maintenance/2026-11-18-repository-organization.md)。
