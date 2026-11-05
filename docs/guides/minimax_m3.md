# MiniMax-M3.1-Flash 文本推理回移

2026-11-03 状态：独立文本推理与 lighteval 入口已加入；此前 63 项 CPU 回归、4 组 GPU
稀疏索引/注意力对照、layer 5 单卡和 TP2 前向，以及 TP2 × PP3 六卡整模型生成短测已通过。
**五项正式下游评测尚未启动；整模型生成短测不能作为准确率结果。**
初次权重下载于 05:12 UTC 因连接中断退出；已为下述六卡评测恢复下载，并切换为
ModelScope 国内直连四路断点续传。
权重已于北京时间 19:06（11:06 UTC）下载完成，73 个文件均就绪，49 个权重文件
全部通过完整 SHA256 校验；前序任务已退出，六卡整模型验证与评测初始化情况见下文。

GPU 0/1 的 TP2 单层检查已实跑通过，见
[双卡报告](../../results/minimax_m3/tp2_layer_20261102T093649Z/report.json)：
两 rank 均加载 1,562 个真实 checkpoint 张量，使用 32Q/2KV、128 专家和 top-4，
完成 129 token prefill + 3 步 decode；归约后两卡输出/残差一致，MoE 分片参考 NMSE
约 `9.95e-6`。参考使用实际 TP 分片上的逐 token QDQ 和必要的全局归约。
这是单层通信/计算验证，不是完整模型的文本生成或下游准确率。
当前全权重驻 GPU 的方式不能用两张 A800 加载整模型：仅 57 个 MoE 层的 packed
专家数值就约 206.6 GB，已超过两卡合计 160 GiB，尚未计入 scale、其他权重和 KV。
CPU offload 不在当前已验证范围内。

下载提速检查见 [2026-11-02 测速与断点校验报告](../../results/minimax_m3/download_speed_20261102T095650Z/speed_report.json)。
当前机器访问 ModelScope 国内站时，64-MiB 单路直连约 34.1 MB/s，四路共 256 MiB 的
短测合计约 60.7 MB/s；原 Hugging Face 代理下载约 2 MB/s。`hf-mirror.com` 对该权重
仍重定向到 Hugging Face，未提供独立加速路径。ModelScope 国内站经代理的小块测试明显
慢于直连，因此提速方案只在下载命令使用 `--noproxy '*'`，不改变全局代理配置。

ModelScope 两个站点公布的 49 个权重文件大小和 SHA256 均与本次 HF 固定 revision
相同；另复制原断点、补齐 `model-00005` 的 12,556,552 字节尾段，完整 8.5-GB 分片的
SHA256 校验通过，原断点未修改。用户明确授权后，于 10:10 UTC 停止原下载控制器
PID 3176572 及其 curl 子进程，保留断点，启动四路直连下载 PID 3300125；六卡评测
控制器保持运行，其下载进程记录已更新。切换后 10 秒实际落盘约 655.8 MB，合计
65.6 MB/s，采样结束时权重已落盘约 35.3 / 252.4 GB（含未完成分片），3 个完整文件
已通过 SHA256 校验。过程见
[切换记录](../../results/minimax_m3/continuation_20261102T084837Z/download_switch.json)、
[下载日志](../../results/minimax_m3/continuation_20261102T084837Z/download_modelscope.log) 和
[推进采样](../../results/minimax_m3/download_speed_20261102T095650Z/post_switch_progress.json)。
持续速度及完成时间仍受后续带宽影响；已落盘字节不等于所有分片均完整且通过校验。

下载最终状态为 `complete`、退出码 0，详见
[下载状态](../../results/minimax_m3/continuation_20261102T084837Z/download_status.json) 和
[49 个完整权重校验记录](../../results/minimax_m3/continuation_20261102T084837Z/modelscope_transfer_results.json)。
原 TP2 单层测试、六卡评测配置与新下载器使用相同的 HF snapshot 目录和固定 revision；
仅远端传输来源改为 ModelScope，仍写入原 `blobs` 并建立原 snapshot 文件链接。

## 范围与来源

- 使用 `hif4` 和现有 editable vLLM `0.19.2.dev0+gb1388b1fb.d20260430`，未升级依赖或替换编译扩展。
- 上游稀疏注意力与索引 Triton 核来自 vLLM `v0.33.0`，commit
  `4ba787ec9959e566d3b587a796dff07673324da5`。
- M3.1 权重及激活约定参考上游 [PR #64111](https://github.com/vllm-project/vllm/pull/64111)，
  merge commit `dbd49af2223553d077eb839f610ad1aa84677391`。
- 目标 checkpoint：`MiniMaxAI/MiniMax-M3.1-Flash`，revision
  `2f33a8b9ae49ce7b51c994df97166a7fd059bc2e`。
- 新代码位于 `3rdparty/vllm/vllm/model_executor/models/minimax_m3/`。
  通过 `tools/minimax_m3.py` 在当前进程及其 spawn worker 注册。
  原 `main.py`、Qwen/HiF4/NVFP4 emulation 与 Ban 源码不变；直接运行旧入口不会自动加载本适配。

GPU 实跑补齐两处旧版接口适配：PDL 支持通过 v0.19 的 capability API 查询；
NVFP4 子类保留 `ModelOpt` 名称，以满足旧专家加载器按类名分派的约定。
改动均在新增 MiniMax 模块中，没有替换原量化类或公共引擎实现。

本轮覆盖文本、eager、BF16 KV/index cache、MXFP8 dense 和 packed NVFP4 routed experts。
图像/视频和 DSpark 不属于本次支持范围。底层计算为软件量化/反量化及 BF16 GEMM，
不代表 A800 提供原生 FP4/FP8 硬件加速，也没有实测吞吐或下游准确率结论。

## 关键适配

1. 专家 `weight_packed` 保持 uint8；`weight_global_scale` 是量化倍率，转换为已有
   ModelOpt ABI 的反量化 scale 时取倒数。gate/up 的全局 scale 不一致时明确失败。
2. M3.1 的专家 FC1/FC2 使用逐 token NVFP4 激活量化；checkpoint 中的
   `input_global_scale=1.0` 是占位值。新增专家类复用现有 packed GEMM，独立保留这个边界，
   没有改写原 Qwen 的 scalar QDQ。SwiGLU-OAI 使用 `[gate; up]` 布局。
3. MXFP8 dense 权重及 E8M0 block scale 保持序列化格式，通过已有 BF16 emulation 执行。
   新量化方法只注册为 `minimax_m3_mixed`，不替换已有的 `mxfp8` 或其他量化类。
4. 为适配旧引擎，将主 K/V 和共享 index key 放在同一页分配中，保持页号一致；
   底层复用上游 block score、top-k、稀疏 prefill/decode 核。这里使用旧引擎的
   `MLAAttentionSpec` 描述单向量页存储，并没有将模型注意力改成 MLA。
5. 下游仍复用本仓库 lighteval 和任务注册；MiniMax 推理标签为
   `<mm:think>…</mm:think>`。MMLU-Pro 未指定样本数时使用 300。

## 使用

由调用者激活 `hif4`。以下路径固定到本次下载 revision：

```bash
MODEL=/home/shaoyuantian/.cache/huggingface/hub/models--MiniMaxAI--MiniMax-M3.1-Flash/snapshots/2f33a8b9ae49ce7b51c994df97166a7fd059bc2e
python tools/minimax_m3.py check --model_path "$MODEL"
```

`check` 只验证真实模型配置与架构注册，不加载权重，也不证明能生成。
生成/评测入口在启动引擎前核对 index 中的全部主模型分片是否存在；下载不完整时停止。
TP 必须显式指定，不自动使用所有可见 GPU。

以下为待执行的 TP8 短测配置，需等待 8 张 GPU 空闲；本次未运行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/minimax_m3.py generate \
  --model_path "$MODEL" --tensor_parallel_size 8 \
  --max_model_len 4096 --max_new_tokens 128 \
  --prompt 'What is 2 + 2?'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/minimax_m3.py evaluate \
  --model_path "$MODEL" --tensor_parallel_size 8 --datasets gsm8k \
  --max_samples 2 --max_model_len 4096 --max_new_tokens 128 \
  --output_dir results/minimax_m3/gsm8k_short
```

短测分数不解释为准确率证据。正式评测需使用确认后的完整上下文、生成长度、样本数和采样协议。
新入口的上下文与生成长度默认均为 32768，采样默认采用模型推荐的 temperature=1.0、top_p=0.95。

## GPU 2–7 的五项评测接续

用户已授权接在当前 GPU 2–7 的任务之后。使用 TP2 × PP3，共六张卡；64 个 Q head
不能直接使用 TP6。新增 `--pipeline_parallel_size` 默认 1，原命令保持原行为。
当前 [接续 RUN 配置](../../results/minimax_m3/continuation_20261103T015711Z/config.json) 与
[复用的输入准备结果](../../results/minimax_m3/continuation_20261103T015711Z/preparation.json)
已落盘。原 RUN 的六卡整模型生成已通过两题算术、连续 decode 和有限 logprob 检查。

| 数据集 | 题数 | 每题回答数 | 指标 | 最大生成 token |
| --- | ---: | ---: | --- | ---: |
| GSM8K | 1,319 | 1 | extractive match | 32,768 |
| AIME24 | 30 | 5 | avg@5 | 38,912 |
| AIME25 | 30 | 5 | avg@5 | 38,912 |
| MMLU-Pro | 300 | 1 | extractive match | 32,768 |
| LCB v6 | 175 | 1 | checker pass@1 | 38,912 |

共同条件：seed=1234、temperature=1.0、top_p=0.95、top_k=-1、BF16 KV、eager、
上下文 40,960、max_num_seqs=16、max_num_batched_tokens=4096。数据来自已审计的固定
Arrow，独立复制并核对摘要。MiniMax 模板下最长 prompt 为 1,614 token，全部输入加上
对应生成上限均能完整容纳。LCB 实际每题生成一次，指标名中的 `:16` 不作为生成次数。

有限接续入口为 `python -m tools.minimax_m3_eval run --run <RUN>`。已启动的同一 RUN
会拒绝重复执行。流程等待已记录身份的 Ban 控制器 PID 2656176、matrix_norm 控制器
PID 2615910 及其子进程退出，等待下载完整，然后检查 GPU 2–7 空闲。
先进行整模型两题算术/连续 decode/有限 logprob 检查，再对五项任务各做两题短测；
短测仅把生成步数压到 256，保留正式上下文、并行、批容量、量化和 AIME 五回答语义。
全部通过后才顺序运行五项正式评测，任何错误停止后续阶段，不停止既有任务或自动占卡。

每个阶段使用独立模型目录视图，预测缓存留在该视图中，不与短测、其他任务或模型源目录
共用。结果核验题目 ID、实际 prompt/token、回答数量和无截断状态，保留原始
`GENERATIVE.parquet` 及正式评分；不将 final-only 诊断替换为原 avg@5。
运行状态和阶段日志见 RUN 的 `state.json`、`controller.log`；正式结果通过核验后写入
`accuracy.json`。当前已通过 63 项 CPU 回归（含真实父/子进程等待），正式准确率尚未产生。

2026-11-03 北京时间 09:12，原 RUN 的
[整模型生成短测](../../results/minimax_m3/continuation_20261102T084837Z/qualification/generation/summary.json)
通过；09:13 的 GSM8K 初始化因 `GenerationParameters.top_k=-1` 不符合 lighteval 的
非负参数 schema 退出，未开始正式评测。当前 vLLM 中 0 与 -1 都表示不限制 top-k，
因此只修正 MiniMax 独立入口为 `top_k=0`，不改变采样、任务和评分协议。新增针对性
CPU 检查及既有相关检查共 13 项通过，包含真实参数转换和复用证据拒绝条件。

新 RUN 保留旧失败产物，使用独立的模型视图/预测缓存。配置一致、生成计算代码及其他
原有绑定依赖未变后，复用已通过的整模型生成报告，从 GSM8K 两题短测接续；五项短测
全部通过后才进入正式阶段。来源、代码差异和校验见
[修复核验](../../results/minimax_m3/continuation_20261103T015711Z/repair_validation.json)、
[复用证据](../../results/minimax_m3/continuation_20261103T015711Z/generation_reuse.json) 和
[当前状态](../../results/minimax_m3/continuation_20261103T015711Z/state.json)。
修复后 GSM8K 两题的真实生成、评分、输入/回答数核验及六卡退出交接已通过，
接续进入 AIME24 五回答短测。该两题分数不作为正式准确率，正式阶段仍须等待其余
任务短测通过。启动标识、核验结果与交付快照见
[handoff.json](../../results/minimax_m3/continuation_20261103T015711Z/handoff.json)。

## 验证与证据

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  3rdparty/vllm/tests/models/test_minimax_m3_backport.py \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/ban/tests \
  Native_NVFP4_HiF4_Linear_Puncture/tests/test_checkpoint.py \
  Native_NVFP4_HiF4_Linear_Puncture/tests/e2e_diag_reconstruction/test_moe_vo_geometry.py \
  NVFP4/tests/test_vllm_emulation_cli_plumbing.py
```

本轮证据保存在
`results/model_compatibility/minimax_m3_1_flash_20261102T024140Z/`：

- `backport_config_check.log`：真实 checkpoint 的架构/配置检查。
- `backport_cpu_tests.log`：新量化、缓存契约及现有相关 CPU 回归。
- `real_weight_cpu_check.json`：第 3 层真实专家权重片段的反量化对照。
- `sm80_compile_check.json`：Triton 3.6.0 下的 SM80 离线编译，包含 TP8 的 prefill tile；没有 GPU 执行。
- `protected_source_sha256.json`：原入口、转换运行时与 Ban 等受保护源码的核验基线。
- `download_manifest.json`、`download_pid.json`、`download_https.log`：下载来源、revision 和运行标识。
- `gpu_kernel_tests.log`：真实 A800/SM80 上的 4 组数值对照，使用 64Q/4KV 和 8Q/1KV
  两种头数、非连续 combined-cache 视图、超过 top-16 的上下文及连续 decode。
  top-k 集合与 FP32 参考一致；注意力 NMSE 最大约 `1.19e-5`。
  8Q/1KV 仅检查 TP8 的每卡形状，不验证多卡通信。
- `gpu_layer_check.json`、`gpu_layer_check.log`、`validate_gpu_layer.py`：第 5 层
  1,562 个真实 checkpoint 张量、完整 128 专家/top-4，129 token prefill + 3 步 decode；
  输入是固定种子的随机 hidden states，不是整模型文本生成。
  MXFP8 投影对照误差为 0；两 token 的 NVFP4 MoE 对照 NMSE 为 `9.04e-6`。
  MoE 允许 BF16 计算/归约差异，要求总相对 RMSE 小于 1%，并按各专家和共享输出的
  绝对贡献幅度限制局部误差，避免相消为零处使用失真的相对误差。
  含数值参考的峰值分配显存约 5.25 GiB，保留显存约 6.37 GiB；不能外推整模型峰值。
- `gpu_followup_cpu_regression.log`：GPU 实跑修复后重跑，61 项 CPU 回归通过。
- `gpu_validation_report.json`：当前源码哈希、原有关键源码核验与验证范围。

复现 GPU 核测试（先确认 GPU 1 空闲；旧版 vLLM 要求使用数字 GPU 索引）：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python -m pytest -q -s \
  3rdparty/vllm/tests/kernels/attention/test_minimax_m3_backport.py

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 python -u \
  results/model_compatibility/minimax_m3_1_flash_20261102T024140Z/validate_gpu_layer.py
```

后续必须补齐完整模型加载、真实 prefill 和多步 decode、TP 通信、峰值显存，以及新入口的
lighteval 小样本衔接；单层测试不能代替这些验证。回移接口与测试边界的经验见
[vLLM 兼容性经验](../experience/lessons/vllm_build.md)。
