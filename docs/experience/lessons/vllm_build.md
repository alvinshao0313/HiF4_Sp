# vLLM 构建与 torch ABI

证据状态：历史 ABI 安装记录，以及 2026-11-02 MiniMax 回移的 GPU 核/真实单层实测，
包含 GPU 0/1 的 TP2 单层通信，以及 2026-11-03 的 TP2 × PP3 整模型生成短测；正式下游准确率仍待验证。当前操作入口是 [安装指南](../../guides/install.md)，依赖或环境变更仍遵守根目录 [AGENTS.md](../../../AGENTS.md)。

## 问题与证据

历史 v0.17.0 / v0.19.0 预编译安装出现 `vllm._C` 的 `c10::MessageLogger` undefined symbol。记录对比了 wheel 实际来源和本地 torch 的符号，归因为扩展与 torch 的 C++ ABI 不匹配。完整证据与尝试保留在 [安装排错记录](../history/environment/vllm_install_notes.md)。

## 有效做法与边界

- 先核对实际导入路径、torch 和扩展构建来源，不能只看包版本字符串。项目源码应来自 `3rdparty/vllm`，Python 来自 `hif4`。
- 当前仓库安装入口采用 editable source build；旧 `VLLM_USE_PRECOMPILED=1` 尝试不能当作现行推荐命令。源码构建与所用 torch 应配套，具体版本由安装指南和当前脚本决定。
- conda CUDA toolkit 不能替代内核驱动；import 通过与实际 GPU 推理通过是不同证据。
- 上述错误只支持对应历史 ABI 判断，不能据此认定之后所有 import 错误都来自相同根因。新的错误应先查实际日志与依赖来源。

[v0.17 源码构建记录](../history/environment/vllm_0_17_source_build_guide.md) 保留迁移过程，不构成切回旧版本的建议。

## Python 模型回移的运行时接口

2026-11-02 在未更改依赖的 `hif4` / vLLM 0.19 / A800 环境中，MiniMax M3.1 的 CPU
配置、契约检查和 SM80 离线编译均通过，但 GPU 实跑仍发现两处接口缺失：新版
`is_arch_support_pdl` 在旧平台类上不存在；旧 FusedMoE 加载器通过量化类名中的
`ModelOpt` 选择该格式的 scale 加载规则。新增子类若丢失该名称会错误进入通用加载分支。
采用旧版现有 capability API，并将新增子类命名为 `MiniMaxM3ModelOptNvFp4FusedMoE` 后，
4 组稀疏核对照与包含 1,562 个真实权重张量的第 5 层 prefill/decode 通过。

经验边界：离线编译只覆盖核编译，CPU 契约检查也不能替代真实权重加载、GPU 主循环。
回移时应沿用目标版本的加载器和 worker 初始化约定；无需为 Python 接口问题升级公共依赖。
单层手动运行还须初始化引擎正常提供的 workspace manager，这属于测试入口的准备工作。
具体命令、误差及原始报告见 [MiniMax 回移指南](../../guides/minimax_m3.md)。

同日后续在 GPU 0/1 实跑真实 layer 5 的 TP2 权重切分、NCCL 归约、129-token prefill
和三步 decode，输出/残差在两 rank 一致，独立分片参考的 MoE NMSE 约 `9.95e-6`。
这补充了单层多卡证据，不能替代整模型/PP3 验证。减少评测题数不会减少需要加载的
模型权重；该模型仅 packed 专家数值已超过两张 80-GiB 卡的容量，不能靠部分 checkpoint
生成正式准确率。来源见上述指南链接的 TP2 报告。

2026-11-03 使用完整 checkpoint、正式上下文/批容量的 TP2 × PP3 六卡生成短测通过，
覆盖两题真实生成、连续 decode、有限 logprob 和算术答案；补足整模型与 PP 通信证据，
不代表五项任务已评分。随后 GSM8K 初始化的失败来自 lighteval 参数 schema，不能归因于
底层 CUDA/量化核，详见上述指南中的阶段状态和修复记录。
