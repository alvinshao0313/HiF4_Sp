# NVFP4 emulation 与 native_nvfp4_linear 算子差异核查

核查日期：2026-11-10（北京时间）。本次用户明确比较 `native_nvfp4_linear` 研究参考路径与 vLLM 原生 packed NVFP4 emulation，关注准确率。仅做源码核对和有限算子诊断，未修改生产算法、启动完整推理或重算下游准确率。

## 结论与适用范围

当前 `native_nvfp4_linear` 直接复用 vLLM 的 `ref_nvfp4_quant_dequant`；A800/SM80 下该 QDQ 走 PyTorch reference，不调用 `NVFP4/triton_fake.py`。两条 Linear 路径的主要数学操作相同，但研究参考路径周围的 MoE router 权重精度、W2 加权时机、专家求和及 TP 路径不同。因此不能由单 Linear 相同推出整个模型的 logits 或自由生成结果相同。

本轮 6 个固定真实 W2 输入上，直接调用 `native_nvfp4_linear` 与 `run_nvfp4_emulations` 的输出逐元素相同。packed 权重与预解码 BF16 权重送入相同形状/配置的融合 GEMM，仅 2/12,288 个输出不同，最大局部 NMSE 为 2.1833e-8。路由权重精度和加权位置的局部差异明显更大，见下表。这些证据定位了实际算子差异，不证明它们解释全部准确率差距，也不证明哪条实现下游更好。

## 当前调用链

| 环节 | vLLM packed NVFP4 emulation | native_nvfp4_linear 所属研究路径 |
| --- | --- | --- |
| 激活 QDQ | `ref_nvfp4_quant_dequant`，group16 | `qdq_native_nvfp4` 直接调用同一函数 |
| 权重解码 | 同一 E2M1/block scale/global scale；普通 Linear 每次解码，MoE 在核内解码到 BF16 | `modelopt_moe_checkpoint._decode_weight` 调用同一解码器，存 FP32，在 GEMM 前转 BF16 |
| scale 合并 | QKV global scale 取 max；MoE 激活 scale 跨专家取 max 后倒数；W13 使用 gate weight global scale | 当前 `modelopt_moe_checkpoint.py` 明确复现上述规则 |
| 普通 GEMM | `torch.matmul` | `F.linear`；默认 TP1 |
| MoE GEMM | Triton BF16 operands / FP32 accumulator | 每个专家分别调用 `F.linear`，输出先转 BF16 |
| router 权重 | `fused_topk` 返回 FP32 | Transformers `Qwen3MoeTopKRouter` 在归一化后转为 router logits 的 dtype，本例 BF16 |
| W2 加权 | 在 FP32 accumulator 上乘 FP32 router 权重，然后转 BF16 | W2 的 BF16 输出乘 BF16 router 权重，再写 BF16 |
| 专家求和 | 当前 top8 经 `moe_sum` 的默认分支调用 `at::sum_out` | 按 expert id 排序循环，BF16 `index_add_`，存在逐次舍入 |
| TP | 按实际 TP 切分 W13/W2 并归约 | `native_nvfp4_linear` 默认 tp_size=1；`NativeMoESemanticTeacher.routed_moe` 的 W2 调用未传 TP 参数；O-proj 可选 TP1/2 |

两条路径的 SiLU 均按 BF16 SiLU 结果再乘 BF16 up，本轮 48 个真实 gate/up 对共 36,864 个元素完全一致。若自行把 SiLU 和乘法合并到 FP32、末尾仅舍入一次，会改变数值；这个额外变体在本轮有 9,966 个元素不同，但它不是当前参考路径。

如果比较整个 teacher block，参考 attention 还使用 SDPA；不能把完整模型差异全部归到 NVFP4 GEMM。本次没有扩展到 attention 或全模型重评。

## 本轮 GPU 证据

输入来自既有 [frozen-input manifest](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/e0_nvfp4_operator_parity/frozen_inputs_manifest.json)：2 个 prompt，层 0/24/47，每个 prompt 取各层最早保存的 decode 位置（10 或 12），共 6 个真实参考路径激活包。它们是历史保存的真实激活，不是本轮新生成的完整模型轨迹。所用权重为 `nvidia/Qwen3-30B-A3B-NVFP4` revision `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`。

环境为 `hif4`、PyTorch `2.10.0+cu128`、CUDA 12.8、A800；固定保存输入，无随机输入或采样种子。仅短时使用 GPU 7，未停止既有进程。主检查 PyTorch 峰值分配约 204.6 MiB，目标路径补测约 47.0 MiB；不将该分配量当作整个 CUDA context 的占用。

单专家 W2 测试保持真实 `K=768, N=2048`，使用完整 TP1 Qwen3 MoE 的 tuning lookup 形状（128 专家、top8），仅提取一个实际被路由专家进行计算。它隔离算子差异，不是完整 MoE 或真实多卡运行。实际选中配置保存在 JSON；使用后端原有默认配置，不人为更改计算条件。

| 隔离比较 | 不同输出数 / 总数 | 最大局部 NMSE | 含义 |
| --- | ---: | ---: | --- |
| `native_nvfp4_linear` vs 普通 Linear emulation | 0 / 12,288 | 0 | 当前同 QDQ、同权重的普通 GEMM 对齐 |
| packed vs 预解码 BF16，同融合 GEMM 配置及 FP32 路由权重 | 2 / 12,288 | 2.1833e-8 | 解码融合与编译调度带来的小数值差异，未进一步判定具体指令原因 |
| Triton vs torch，不乘路由权重 | 3 / 12,288 | 2.0008e-8 | GEMM 实现的局部差异 |
| 仅将路由权重从 FP32 换成 BF16 | 2,681 / 12,288 | 1.3837e-5 | 两边都在 accumulator 上乘权重 |
| 仅改变 W2 加权前后的 BF16 舍入位置 | 3,222 / 12,288 | 8.3661e-6 | 使用同一 BF16 路由权重隔离时机差异 |
| 生产式加权 vs 研究参考加权 | 4,101 / 12,288 | 1.4455e-5 | 局部综合结果，不是整模型误差 |

同一批 router logits 上，6/6 个 top8 专家及顺序相同。FP32 softmax/topk 权重在两个后端间最大绝对差为 2.98e-8；参考路径转 BF16 后，48/48 个权重不同，最大绝对差为 4.8150e-4。

另固定每个专家的 BF16 contribution，仅比较 top8 一次 `sum` 与按 expert id 顺序逐次 BF16 加法：最大绝对差 0.125、最大局部 NMSE 2.8099e-5。这里直接隔离相同 contribution 的求和方式，没有重新执行完整专家前向。

原始数据、源码/输入/权重哈希和逐例结果：[results.json](results.json)、[native_diagnosis.json](native_diagnosis.json)。三个脚本均退出 0，日志和退出码保留于本目录。

## 单独的 fake-act Triton 发现

用户明确函数前，另核对了 `--fake_act_quant nvfp4` 使用的 `NVFP4/triton_fake.py`，它与上述研究参考路径不同。

- 在相同输入、相同 global scale 下，66 个激活张量的 86,016 个元素中，257 个 BF16 QDQ 输出不同；最大单张量 NMSE 为 6.3874e-5。
- 受影响的 7 个张量上，PyTorch fake helper 与原生 reference 的 block scales 相同；普通 FP8 cast 与手写 FP8 cast 也相同。
- 至少一个已定位反例来自量化前倒数表达式的舍入：同一元素经原生 `1/(s*(1/g))` 得到 `-3.5000002384`，经 `g/s` 得到 `-3.4999997616`，使 E2M1 编码从 `-4` 变为 `-3`。
- PyTorch fake helper 与实际 Triton QDQ 另有 255 个不同输出；本轮未逐个分解其编译后除法和最终 BF16 舍入，因此不将全部 257 个差异归到同一个原因。
- 该通用开关只挂接普通 unquantized Linear，不覆盖 fused MoE；它也不能直接替代整个 Qwen3 MoE 的 W4A4 路径。

详见 [qdq_diagnosis.json](qdq_diagnosis.json)。这些事实不能被用作 `native_nvfp4_linear` 路径已经调用自定义 fake-act Triton 的证据。

## 既有证据与局限

既有 [E0 算子报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/long_trajectory_stability/e0_nvfp4_operator_parity/E0_NVFP4_OPERATOR_PARITY_REPORT.md)在当时版本固定输入下，QDQ、权重解码、普通 Linear 和单专家对齐；首次差异出现在 fused MoE，手工 TP2 也有差异。本轮没有将该历史报告当作当前整模型验证，新增检查只覆盖本报告所列当前函数和算子。

此次完成的是算子差异核对，尚未拿到用户产生准确率差距的两份完整运行配置/结果，不能量化各因素对准确率的贡献。局部小差异可能改变临界路由或后续采样轨迹，但大量输出位不相同不等于准确率按同一比例下降。

若后续要求评估精度因果，应在同一输入 token/prefix、模型、TP 和生成/评分口径下，隔离上述 router 权重、加权舍入和求和差异，再评估 logits 及生成结果。这里没有修改研究算法或把普通浮点误差设为正式阻断条件。

技术经验归并到 [fake quant 边界经验](../../../docs/experience/lessons/fake_quant_boundaries.md)。

## 复现

调用者先激活 `hif4`。本次实际运行的命令为以下三条；stdout/stderr 分别保存到 `run.log`、`qdq_diagnosis.log`、`native_diagnosis.log`。脚本拒绝覆盖各自已有 JSON，复查应复制整个诊断目录到独立新目录再执行，不能覆盖此证据。

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u NVFP4/reports/operator_flow_audit_20261110T092657Z/audit.py
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u NVFP4/reports/operator_flow_audit_20261110T092657Z/diagnose_qdq.py
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u NVFP4/reports/operator_flow_audit_20261110T092657Z/diagnose_native.py
```

GPU 编号仅记录本次小规模验证，不构成之后的资源授权。
