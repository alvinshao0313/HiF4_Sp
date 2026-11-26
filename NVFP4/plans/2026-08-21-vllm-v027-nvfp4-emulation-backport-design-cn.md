# vLLM v0.27.0 NVFP4 Emulation 局部回移设计说明

## 1. 目标

在当前仓库 vendored `3rdparty/vllm` 的基础上，局部回移 vLLM v0.27.0 中与 NVFP4 emulation 直接相关的功能，使 A800 等不具备原生 NVFP4 Tensor Core 的 GPU 可以直接加载原生 NVFP4 checkpoint，并以 vLLM upstream 定义的 emulation 数值语义完成：

- NVFP4 Dense Linear W4A4 emulation；
- NVFP4 FusedMoE expert W4A4 emulation；
- ModelOpt NVFP4 checkpoint；
- compressed-tensors NVFP4 checkpoint；
- 可选使用 checkpoint 对应的量化 KV cache；
- 保留 packed NVFP4 权重常驻，不将全模型/全 expert 权重展开为 BF16 常驻。

本工作用于**精度/功能评估**，不宣称 A800 上获得原生 FP4 Tensor Core 性能。

## 2. 固定 upstream 基线

功能和数值语义以 **vLLM tag `v0.27.0`** 为唯一基线，不跟随滚动 `main`。历史上 MoE NVFP4 emulation 的核心能力由 upstream PR `#44667` 引入；该 PR 的 merge commit `552a9db` 只用于追溯，不作为最终逐文件基线。若 PR 后续到 v0.27.0 之间对相关文件有修正，应以 v0.27.0 最终内容为准。

禁止：

- 整体替换 `3rdparty/vllm`；
- 升级整个 PyTorch/CUDA/FlashInfer/Transformers 依赖栈来迁就 v0.27.0；
- 从 v0.27.0 以后的 `main` 随机摘取实现；
- 对 upstream NVFP4 数学规则做“精度修正”。

回移策略是：**只回移 NVFP4 dense + MoE emulation 所需的最小依赖闭包，并适配当前 vendored vLLM 的公共接口。**

## 3. 支持范围

### 3.1 checkpoint 范围

必须支持：

1. NVIDIA ModelOpt 原生 NVFP4，当前 vLLM quant method 为 `modelopt_fp4`；
2. compressed-tensors NVFP4 pack-quantized checkpoint。

不支持：

- 任意自定义“NVFP4-like”格式；
- 缺失 upstream 必需 scale/metadata 的 checkpoint；
- 通过猜测 tensor 名称兜底加载。

### 3.2 模型范围

不做 Qwen3 专用层名判断。只要模型架构已经能被当前 vLLM 加载，并且对应 Dense Linear / FusedMoE 量化方法属于上述两种 NVFP4 checkpoint 范围，就应复用同一套 emulation backend。

`nvidia/Qwen3-30B-A3B-NVFP4` 仅作为真实模型验收样例，不作为代码中的特殊分支。

## 4. 数值语义

### 4.1 总原则

**完全遵循 vLLM v0.27.0 的 NVFP4 emulation 逻辑。**

包括但不限于：

- FP4 payload 为 E2M1 packed representation；
- block group size = 16；
- block scale 使用 upstream 定义的 FP8 E4M3 语义；
- weight global scale / input global scale 的 reciprocal 关系按 v0.27.0 loader/kernel 处理；
- activation QDQ 的 clipping、rounding、zero/boundary 行为按 upstream；
- MoE `a13_scale` / `a2_scale` 在 emulation backend 中的 `max()` 合并与倒数逻辑保持 upstream 原样；
- 不自行改成 per-expert scale，即使其与某些 native backend 的内部行为存在差异；
- gated W13 的 scale 合并规则保持 upstream 原样。

因此，本项目 correctness 的第一定义是：

> 当前仓库回移后的结果与 vLLM v0.27.0 NVFP4 emulation 语义一致。

而不是重新定义一个“理论上更严格”的 NVFP4。

### 4.2 Dense Linear

Dense NVFP4 不继续以当前旧式 `NvFp4LinearBackend.EMULATION + 环境变量` 作为最终接口，而应对齐 v0.27.0 的 linear kernel backend 体系：

- 对外可显式选择 `linear_backend="emulation"`；
- NVFP4 packed weight 保持 checkpoint 原始低比特形式；
- emulation apply 使用 upstream v0.27.0 的 NVFP4 emulation 数值函数；
- 不重新量化 checkpoint weight；
- bias 与输出 dtype 处理按 upstream；
- 当前旧环境变量/旧路径若仍被仓库其它代码依赖，只做必要兼容，不作为新实验入口。

### 4.3 FusedMoE

MoE 必须对齐 v0.27.0 的 `Nvfp4QuantizationEmulationTritonExperts` 结构：

- `NvFp4MoeBackend` 增加 `EMULATION`；
- `moe_backend="emulation"` 能显式选择该 backend；
- W13/W2 packed weight 不全量 materialize 成 BF16 常驻；
- Triton expert GEMM 在计算中对命中 expert 的 packed NVFP4 weight 按 upstream 逻辑解码；
- W13 输入和激活后的 W2 输入按 upstream emulation 做 NVFP4 QDQ；
- 必须防止 activation 被重复 QDQ；
- router/top-k/router weight/reduce 顺序保持当前 vLLM 模型语义；
- shared expert 按 checkpoint/vLLM 当前 quant method 决定，不人工强制 A4。

## 5. backend 选择策略

不修改 vLLM 默认 `auto` 行为。新能力通过 upstream 风格的显式 backend 参数开放：

- `linear_backend="emulation"`
- `moe_backend="emulation"`

项目根目录 `main.py` 需要把这两个参数透传给 Python `vllm.LLM(...)`。正式 A800 NVFP4 W4A4 实验脚本必须同时显式设置两者为 `emulation`。

这样避免改变其它现有任务，也避免实验中因自动 backend 选择而意外落到 Marlin W4A16。

验收时必须证明：所有目标 NVFP4 Dense Linear 和 NVFP4 FusedMoE 层分别选择了 emulation backend；如果目标 NVFP4 层无法使用显式 emulation，应直接报错，不允许为了跑通而静默改为 Marlin。

注意：checkpoint 中本来就被 `exclude_modules` 排除、或者本来不是 NVFP4 的模块，不要求强制进入 emulation。

## 6. KV cache 开关

不实现新的 fake-KV kernel，复用 vLLM v0.27.0 的 `kv_cache_dtype` 机制。

项目侧提供两个明确实验模式：

1. **checkpoint KV 模式**：`kv_cache_dtype="auto"`，由 vLLM/model config/quant config 决定 KV cache dtype；对于声明 FP8 KV 的目标 checkpoint，应实际得到 FP8；
2. **BF16 KV 模式**：`kv_cache_dtype="bfloat16"`，显式关闭 FP8 KV cache，用于单独测 W4A4 损失。

不要增加第三套 `fake_kv_quant` 语义来模拟这两个模式。已有项目自定义 KV fake quant 功能保持不变，但不得与本次 native NVFP4 emulation 验收混用。

## 7. 与现有本地功能的关系

当前仓库已有 `NVFP4/torch_fake.py`、`NVFP4/triton_fake.py` 和 `main.py --fake_act_quant nvfp4`。这条路径主要服务于反量化/转换后的普通 Linear + sidecar activation scales，不等同于“直接加载原生 packed NVFP4 checkpoint”。

本次工作不得删除它，也不得把它作为原生 checkpoint emulation 的实现基础。新能力应落在 vendored vLLM 自己的 quantization/kernel backend 中。

## 8. 代码组织原则

目标结构应尽量接近 vLLM v0.27.0：

- `vllm/config/kernel.py`：`LinearBackend` / `MoEBackend` 暴露 `emulation`；
- `vllm/model_executor/kernels/linear/nvfp4/`：新版 NVFP4 linear kernel abstraction 和 emulation kernel；
- `vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py`：共享 NVFP4 emulation 数值原语；
- `vllm/model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py`：MoE emulation Triton experts；
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`：backend 注册、选择、kernel-format conversion；
- ModelOpt / compressed-tensors quant method：只做调用新 backend 所需适配，不复制一份数学逻辑。

不要在 `main.py`、模型文件或 Qwen3 类里塞 NVFP4 数学实现。

## 9. 验收模型

真实验收 checkpoint 固定为：

`/home/shaoyuantian/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3/`

最低真实验收包含：

- A800，TP=1 smoke（若运行时真实显存证明 TP=1 无法初始化，才允许在报告中记录原因并改成最小可运行 TP；不得预先假定必须多卡）；
- `linear_backend=emulation`；
- `moe_backend=emulation`；
- 一组 `kv_cache_dtype=bfloat16`；
- 一组 `kv_cache_dtype=auto`，并确认该 checkpoint 实际解析为 FP8 KV；
- fixed prompt 可正常 prefill + decode，无 NaN/Inf；
- 日志/测试证明目标 NVFP4 dense 和 MoE 没有落到 Marlin W4A16。

## 10. 非目标

本次不做：

- A800 NVFP4 性能优化；
- 与 Blackwell native kernel bitwise 对齐承诺；
- 修改 upstream 的 scale 聚合策略；
- 自动为所有非 Blackwell GPU 强制 emulation；
- 整体升级 vLLM；
- 修改 router、attention 算法或 MoE routing；
- 将权重离线转 BF16 checkpoint；
- 为单个 Qwen3 模型硬编码兼容逻辑。
