# vLLM 构建与 torch ABI

证据状态：历史安装记录的归纳；本次未重装或验证 GPU 推理。当前操作入口是 [安装指南](../../guides/install.md)，依赖或环境变更仍遵守根目录 [AGENTS.md](../../../AGENTS.md)。

## 问题与证据

历史 v0.17.0 / v0.19.0 预编译安装出现 `vllm._C` 的 `c10::MessageLogger` undefined symbol。记录对比了 wheel 实际来源和本地 torch 的符号，归因为扩展与 torch 的 C++ ABI 不匹配。完整证据与尝试保留在 [安装排错记录](../history/environment/vllm_install_notes.md)。

## 有效做法与边界

- 先核对实际导入路径、torch 和扩展构建来源，不能只看包版本字符串。项目源码应来自 `3rdparty/vllm`，Python 来自 `hif4`。
- 当前仓库安装入口采用 editable source build；旧 `VLLM_USE_PRECOMPILED=1` 尝试不能当作现行推荐命令。源码构建与所用 torch 应配套，具体版本由安装指南和当前脚本决定。
- conda CUDA toolkit 不能替代内核驱动；import 通过与实际 GPU 推理通过是不同证据。
- 上述错误只支持对应历史 ABI 判断，不能据此认定之后所有 import 错误都来自相同根因。新的错误应先查实际日志与依赖来源。

[v0.17 源码构建记录](../history/environment/vllm_0_17_source_build_guide.md) 保留迁移过程，不构成切回旧版本的建议。
