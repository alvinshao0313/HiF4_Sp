# 2026-10-23：matrix_norm HiF4 的 Ban 接入与短测

状态：实现、正式配置短测、512 条校准及三组 GSM8K 全量评测完成，2026-10-25 已复核原始
产物。固定 top-6 与动态 Ban 均达到本轮相对 top-8 保持率 ≥98% 的目标。下方早期“未启动”
和“运行中”描述保留为当时状态，GSM8K 最终结果见“2026-10-25：最终结果核验”。
追加 MMLU-Pro、LCB、AIME25@5 的首轮补测也已完成并复核；LCB/AIME 的三轮追加重复
已于 2026-10-27 全部完成，2026-10-28 复核通过，多轮结果见最后一节。
日期按北京时间。实现入口见 [Ban README](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/ban/README.md)，
执行边界见 [fake quant 经验](../lessons/fake_quant_boundaries.md)。

## 模型、来源与配置

目标为保留的 `results/aime25_repeats/matrix_norm_20261015T081958Z/export`：48 层、
128 专家、top-k=8、`fusable_r64` runtime ABI 3。权重仍为 BF16 导出，在线 R64 与
HiF4 激活 QDQ 保留；没有改变原权重或既有结果。

固定参考 commit 为 `3ac1169c9a48f6cda2477975a024aadba5d02f8b`，按该版本复现公式、
舍入和 KL 定义。环境实测为 `hif4`、Python 3.11.13、torch 2.10.0、editable vLLM
`0.19.2.dev0+gb1388b1fb.d20260430.cu128`，使用 GPU 2,3、A800 80GB、TP2、eager。
默认 `--expert_pruning_method none`，显式选择 `Ban` 时加载校准，lambda=0.7、k_min=3。

C4 train 明确读取本地两个 Arrow，使用 2 条×2048 token、seed 0。前 256 文档形成
116675-token 流，窗口起点 `[110680,50494]`，输入 SHA256 为
`d8c8f399d0e996be7ae89f95db2a34f680aaafef2e065f3f1e70cf6e10d46c2c`。
校准参考同一个未剪枝 HiF4 模型，逐层限制为 top-3，使用最后输入位置、参考词表
top-1000 支持上的 `KL(unpruned || pruned)`，不含 tail。

证据根目录为 `Native_NVFP4_HiF4_Linear_Puncture/results/ban/matrix_norm_20261023T093830Z/`。
实际命令、退出结果和最终源码哈希清单保留于该目录；未提交源码不冒充新 commit。

## 数值问题与修正

旧尝试已按用户要求撤回，其失败产物未复用。本轮确认了两个具体执行问题：

1. eager 路径的嵌套 MoE helper 会无显式路径查找 sidecar，不能只依赖模型构造期
   current-config。校准入口须像根 `main.py` 一样设置 `HIF4_RUNTIME_SPEC_PATH`。
   新 worker 核对隐式/显式 sidecar 为同一份；两张卡均实测 R64 已启用。
2. vLLM 的 top-3/4 专用求和核逐次 BF16 舍入，top-8 使用通用求和。同一输入
   `[1,2^-8,2^-8]` 实测分别为 `1.0`、`1.0078125`。
   compact 只执行保留专家的 GEMM，最终专家输出补零至原 top-8 再使用原求和核，
   保留累加边界。没有改动共享 vLLM 实现或其他实验代码。

logits 采集限制为每请求恰一次 `[1,vocab]`、关闭 prompt-logprobs、采样前复制。
native 重复完全一致。另有独立的路由算术现象：fused 与参考 PyTorch 路由在同一
局部输入上的专家集合均相同，权重最大差约 `1.2e-7`，单层输出最大相对 L2 约
`3.82e-5`；完整 48 层最后位置 logits 相对 L2 为 `0.066094`，KL 为 `0.021071`。
该差异随重算路由引入，不是 compact 的输出误差；离散量化传播放大是解释，不据此
推断准确率。验证同时保留原 fused 与参考 PyTorch 未剪枝对照。

## 实测与证据

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 固定上游源码 | 360 组 FP32/BF16 的 weights/IDs/counts 一致 | [源码对照](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/matrix_norm_20261023T093830Z/source_parity/result.json) |
| CPU 回归 | 25 passed，覆盖公式、采样、开关及 DP 拒绝 | 模块 `tests/`，命令为 `python -m pytest -q Native_NVFP4_HiF4_Linear_Puncture/experiments/ban/tests` |
| native 重复及原 fused no-op | logits 误差均为 0 | [诊断](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/matrix_norm_20261023T093830Z/qualification/diagnosis.json) |
| 参考 top-8 对 Ban k_min=8 | logits 最大差、相对 L2、KL 均为 0 | [短测](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/matrix_norm_20261023T093830Z/qualification/qualification.json) |
| masked 对 compact | logits 最大差、相对 L2、KL 均为 0，greedy token 一致 | 同上 |
| 48 层校准 | 2×2048，phase=diagnostic；完整校准约 28.87 秒，不含初始化 | [manifest](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/matrix_norm_20261023T093830Z/qualification/calibration/manifest.json) |
| prefill＋decode | 每层 2048＋31=2079 位置，TP 统计一致；平均专家 4.853114，实际分发 484302/原 798336 路由，减少 39.3361% | [路由](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/matrix_norm_20261023T093830Z/qualification/routing.json) |
| 现有 main.py 开关 | GSM8K 1 题×8 token，关闭/开启均退出 0；开启时平均专家 4.704262 | [命令](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/matrix_norm_20261023T093830Z/cli_commands.json) · [退出结果](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/matrix_norm_20261023T093830Z/cli_status.json) |

reference-top8、Ban-top8、masked、compact 的 rank-0 torch allocated 均为
67735546368 bytes，32-token 生成结束为 67734784512 bytes；仅说明本次重复调用的
观测，不证明任意长期运行均无增长。GPU 验证进程均正常退出，未停止他人任务。

本次 [c4_ban.pt](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/matrix_norm_20261023T093830Z/qualification/calibration/c4_ban.pt)
仅适用于诊断。GSM8K 输出限于 8 token，无法完成答案，分数没有准确率解释价值。
平均专家数包含 prefill 和 decode，不能写成纯 decode 平均；路由减少不等于吞吐提升。
正式 512 条校准、四项 reasoning 全量评测和性能测量均未启动。

## 追加：8→6 的精度保持验证准备

用户确认固定 top-6 与动态 Ban 两者都测，共用一份新跑的 top-8 基线。本轮只比较
GSM8K test 全部 1319 题，98% 定义为本轮剪枝分数除以 top-8 分数，不使用历史分数
代替新基线，不扩展到其他任务。

新增 `evaluate.py` 的有限工作流、固定本地数据的 `evaluation_main.py`、训练集专家预算
匹配 `budget.py` 和离线 `evaluation_report.py`。保留原 HiF4 权重、激活 QDQ、R64 与评分器。
各组使用新缓存，固定 top-6 仍调用原入口的 `--num_experts_per_tok 6`；全量结束后核对
逐题 prompt、gold、实际缓存 input token，并报告正确数、得失题、相对保留率与 Ban
实际 selected/dispatched 数。没有按测试准确率选 lambda。

正式推理条件为 TP2、eager、BF16 KV、batch/max_num_seqs=128、batched tokens=4096、
context=40960、max_new_tokens=32768、seed=1234、T=0.6、top_p=0.95、top_k=20、min_p=0。
`enable_thinking=None` 沿用现有模板默认行为；三组统一关闭 prefix cache，使输入位置
全部进入路由统计。正式配置短测只缩减为每组 2 题，检查实际生成正常结束，不判断精度。

Ban 使用全新 C4 512×2048、seed=0 校准；训练预算匹配固定 GSM8K train 前 8 题、
本轮显式 `k_min=5`，目标实际均值 `6±0.10`，最多在 [0,1] 搜索 lambda。默认推理
`k_min=3` 保留。测试集真实均值可能偏离训练预算，报告单独判断，不能自动把它写成 6。

CPU 回归为 38 passed；离线报告模块能从历史真实缓存还原 1254/1319 及实际输入 token
哈希，该回放只验证报告读取，不是新 top-8 评测结果。新预算数据准备的实际 8 条输入长度
为 `[50,43,72,65,38,77,69,118]`，不读取训练答案进行评分。

2026-10-23 14:31 UTC 的启动前检查没有空闲 TP2 卡对，未停止其他进程、未共享挤占 GPU、
未启动任何本轮 GPU 流程。准备记录、源码哈希、数据 SHA 和确切命令保留于
[preflight.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_preparation_20261023T142740Z/preflight.json)
和 [commands.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_preparation_20261023T142740Z/commands.json)。
有卡对后先执行其中的 `short_test`，通过后再后台启动 `formal_after_short_test_passes`；
正常推进后交付标识并停止主动监控。当前不能给出是否保住 98% 的结论。

经验增量：无新增 GPU 算法经验；沿用已有实际缓存核对与同轮基线约定。未删除任何旧产物，
没有安装依赖、暂存或提交。

## 2026-10-24：空闲卡恢复后的并行启动

用户授权按最大可用并行度开始测试。启动前 GPU 4–7 空闲，GPU 0–3 为其他用户活动任务；
只使用两对 TP2：4,5 运行校准→预算匹配→Ban，6,7 运行 top-8→固定 top-6。
仅修改 `evaluate.py` 的资源编排，使 Ban 短测及正式评测使用校准卡对，允许两条独立路径
同时运行；重叠卡对仍共用锁。算法、采样、批量和评分协议不变。与前次准备的源码摘要相比，
只有该编排文件变化，根 main.py、HiF4 runtime、Ban 核心文件均未变。

38 项 CPU 回归通过。正式配置短测（各组 2 题、32768 生成上限、TP2/batch128/context40960）
全部退出 0；三组实际 prompt/gold/input token 和推理配置一致、未截输入、所有请求均由
EOS 151645 正常结束。各组输出长度分别为 top-8 `[1182,1106]`、top-6 `[2687,980]`、
Ban `[1969,925]`。诊断 Ban 使用旧 2 条校准、lambda=0.7、k_min=5，实测平均专家数
6.054355，selected=dispatched=879383；TP ranks 一致而不重复相加。该诊断校准没有用于
正式 Ban。短测不判断 98% 精度保持率，也不作为吞吐比较。

短测证据为
[report.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_short_20261024T022233Z/report.json)。
正式启动时间为北京时间 2026-10-24 10:28:10，控制器 PID `2266729`，输出目录为
`Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_20261024T022233Z/`。
正式参数没有 `--max_samples`；Ban 重新校准 512×2048，再按 train 预算选 lambda。
后台使用独立 session、stdin=/dev/null、stdout/stderr 落盘，不建立自动重试或额外占卡任务。

- [确切命令](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_launch_20261024T022233Z/commands.json)
- [启动标识](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_launch_20261024T022233Z/launch.json)
- [运行状态](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_20261024T022233Z/status.json)
- [本轮模型、数据和源码标识](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_20261024T022233Z/manifest.json)

原始日志位于正式输出的 `logs/`，控制器日志位于 launch 目录的 `formal_controller.log`。
最终三组完整结果及实际预算须待 `report.json` 生成后重新核验；当前不能判断是否达到 98%。

启动检查已观察到校准 4/512 条完成全部 48 层，top-8 完成 25/1319 请求，两对 GPU 均在
实际计算；见 [启动推进证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_launch_20261024T022233Z/progress_verified.json)。
正常推进已确认，交付后停止主动监控，后台有限流程继续运行；不以该进度判断最终成功。

## 2026-10-25：最终结果核验

正式流程实际结束于北京时间 2026-10-24 13:30:39，五个阶段（top-8、512 条校准、top-6、
预算匹配、Ban）退出码均为 0，控制器状态为 complete，errors 为空。2026-10-25 从实际
results/details、GENERATIVE.parquet 和路由计数重新生成比较，与保存的最终报告一致。
三组各覆盖全部 1319 题，逐题 ID、prompt、gold、实际 input token 及采样配置一致；
lighteval 输入 token 哈希为 `d4fc9fa493e4a0b3`。

| 方案 | 保留专家数 | 正确题数 | 准确率 | 相对 top-8 保持率 |
| --- | --- | --- | --- | --- |
| 原 top-8 | 8 | 1254/1319 | 95.0720% | 100.0000% |
| 固定 top-6 | 6 | 1260/1319 | 95.5269% | 100.4785% |
| 动态 Ban | 实测均值 6.070009 | 1265/1319 | 95.9060% | 100.8772% |

相对保持率按本轮正确数之比计算，top-8 的 98% 整数门槛为至少 1229 题正确，两种方案
均达到。固定 top-6 相对 top-8 失去 12 题、得到 18 题；Ban 失去 10 题、得到 21 题。
这些是本模型和本轮 GSM8K 的观测，不证明其他任务、不同种子或统计下界达到 98%，也不
证明剪枝稳定提升精度。top-8 有 1 条生成到 32768 上限而按原协议结束；另外 1318 条和
top-6/Ban 的全部请求均正常 stop，没有为对照删除题目或重写评分。

正式校准完成 512×2048 输入的全部 48 层。本轮 `k_min=5`，train 前 8 题在首个
lambda=0.7 即测得平均 6.068739，无需继续搜参；未读取 test 准确率选参。test 实测均值
6.070009，在约定 `6±0.10` 内，包含 prefill 与 decode。保留与实际分发专家赋值总数
均为 854503012（140774592 个 token-layer），TP2 核对后只计一份。这个计数不等于
常驻权重压缩或吞吐收益。

预算日志在 `BAN_BUDGET_MATCHED` 和 `Shutdown complete` 之后，出现一条
`Engine core proc EngineCore died unexpectedly`。源码的进程监视线程会在收到退出
信号且 finalizer 尚存活时记录此消息；此处发生于退出收尾。阶段退出 0、预算文件完整，
后续独立 Ban 进程完成全部 1319 请求并通过原始产物复核，因此未发现本轮计算或结果受损；
保留日志，不把这条错误文本抹去或说成计算失败。

最终证据：
[report.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_20261024T022233Z/report.json)、
[正式校准](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_20261024T022233Z/calibration/manifest.json)、
[预算匹配](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_20261024T022233Z/budget/budget.json)、
[路由统计](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_20261024T022233Z/routing/Ban/routing_0000.json)。

经验增量：结果已回填本记录，无新增可推广的算法经验。未增加实验、依赖或 Git 操作，
保留所有模型、原始结果与核心日志；没有需要清理的一次性 GPU 进程或缓存。

## 2026-10-25：追加 Ban 三项 reasoning 补测

用户要求补测 Ban 下的 MMLU-Pro、LCB、AIME25@5。固定使用已完成的正式 C4 512 条
校准文件，不重新校准或按新任务分数调参；lambda=0.7、k_min=5、compact，保留原
HiF4 权重、R64、激活 QDQ 与根 main.py 的 lighteval/vLLM 路径。本轮仅运行 Ban，
没有额外启动同轮 top-8 或固定 top-6 基线，因此本轮三项结果不能直接写成同轮保持率。

MMLU-Pro 按项目默认 max_samples=300，LCB v6 为全部 175 题，AIME25 为全部 30 题×5
回答的 `avg@n:n=5`。MMLU-Pro 生成上限 32768，LCB/AIME25 上限 38912；其余沿用 TP2、
batch128、batched tokens=4096、context40960、BF16 KV、eager、seed1234、T=0.6、
top_p=0.95、top_k=20、min_p=0，enable_thinking=None，关闭 prefix cache。文档采样沿用
lighteval 的固定 shuffle seed42，实际题目 ID、prompt/gold/input token 摘要分别保存。

核对源码与 CPU 输入审计确认，当前 LCB 指标虽名为 `codegen_pass@1:16`，注册计算类
并非 SamplingMetric，实际 Doc.num_samples=1；本轮保持该已有评测路径，不改成 16 次。
AIME25 的实际 Doc.num_samples=5，保留正式原始回答 avg5 分数，不用 pass@5 或仅 final
文本重评分替代。三项分别绑定既有核验 Arrow/SHA；MMLU-Pro 使用修复后的固定 revision
`3373e0b32277875b8db2aa555a333b78a08477ea`。最大完整 prompt 长度分别为 768/1704/847，
均不需截输入。

新增 `supplement.py` 有限控制器、`additional_main.py` 数据绑定入口和
`additional_report.py` 产物验证；既有读取实际缓存 token 的 helper 仅新增任务名参数，
GSM8K 默认行为不变，38 项 CPU 回归通过。每项使用独立卡对，先运行 2/1/1 题短测，
保留正式长度、批量与 AIME 五回答语义；核对实际输入、生成覆盖、官方分数聚合和专家
分发后才自动启动该项正式评测。独立任务失败不取消其他任务，无自动重试或守护程序。

启动时间为北京时间 2026-10-25 23:09:26，控制器 PID `1192740`：GPU 0,1 跑 MMLU-Pro，
GPU 2,3 跑 LCB，GPU 4,5 跑 AIME25@5；GPU 6,7 未占用。输出目录为
`Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/`。
MMLU-Pro 两题短测已退出 0 并完成产物核验，2 条均 stop、平均专家数 6.062964，300 题
正式评测已启动；LCB 与 AIME25@5 的完整长度短测仍在运行，此时尚不能报告正式分数。

- [运行状态](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/status.json)
- [各阶段精确命令](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/commands.json)
- [模型、校准与源码标识](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/manifest.json)
- [数据绑定与 SHA](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/datasets.json)
- [启动信息](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_launch_20261025T150926Z/launch.json)

日志保留在输出 `logs/`，每项输出位于 `<task>/{short,formal}`，验证报告在各阶段的
`report.json`；全部任务结束后生成根目录的 `report.json`。本轮专家预算只观察并报告，
不因不同任务上的实际均值变化重新调整 lambda。

初始运行检查已观察到 MMLU-Pro 正式完成 43/300 题，LCB 与 AIME25@5 均已进入真实
连续生成。正常推进已确认，交付后停止主动监控，后续阶段由有限流程按各自验证结果继续。
本轮可复用增量为“指标名称与实际生成数须分别核对”，已补入
[评测协议经验](../lessons/evaluation_evidence.md)。未修改算法、依赖或既有实验，未删除旧产物，
没有执行 Git 暂存、提交或新增占卡任务。

## 2026-10-26：追加三项评测完成

全部流程实际结束于北京时间 2026-10-26 01:27:44，控制器状态 complete、errors 为空，
三项短测及三项正式评测的退出码均为 0。已从各自 results/details、实际 GENERATIVE
缓存和路由文件重新执行验证，结果与保存的主报告逐项一致。所有题目的实际输入 token、
prompt/gold 和生成覆盖均符合运行前审计；没有依赖进程退出或控制器状态单独判断成功。

| Ban 任务 | 评测覆盖 | 正式分数 | 实测平均专家数 |
| --- | --- | --- | --- |
| MMLU-Pro | 300 题，211/300 | 70.3333% | 6.069291 |
| LCB v6 | 175 题，91/175，每题 1 次生成 | 52.0000% | 6.071941 |
| AIME25@5 | 30 题×5 回答，正式评分等价 88/150 | 58.6667% avg@5 | 6.073645 |

MMLU-Pro 使用项目默认 300 题，并非完整 12032 题。AIME 使用现有正式评分器和原始五回答
avg5，未将只含 final 的重评分或人工最终答案复核结果混入该分数。MMLU-Pro 有 1 条生成
到达 32768 上限，AIME 有 2 条到达 38912 上限，均按既定协议保留；其余回答正常 stop，
LCB 全部 175 条正常 stop。没有删除达到长度上限的回答或重新抽取结果。

三项的 selected 与 dispatched 总数分别相等：313861203、1218012515、1221332359，
TP2 各只计一份。实际输入 token 哈希分别为 `faacb2474f7959cf`、`fe531aca695bb0f7`、
`56dd24fa7fe4c428`。专家预算没有针对新任务调参，仍为 lambda=0.7、k_min=5、compact。

本轮没有补跑对应的同轮 top-8 基线，因此只能报告上述 Ban 绝对分数，不能直接计算三项
98% 保持率，也不能拿 MMLU-Pro 历史全量成绩与这次 300 题相除。

证据为 [完整补测报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/report.json)，
以及分任务的
[MMLU-Pro](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/mmlu_pro/formal/report.json)、
[LCB](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/lcb/formal/report.json)、
[AIME25@5](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/supplement_20261025T150926Z/aime25/formal/report.json)。
原始回答、日志、数据绑定、源码摘要和模型校准文件均保留。无新增可推广的算法经验，
本节补全实际成绩与覆盖记录；没有新增运行、修改算法/依赖、删除既有产物或执行 Git 操作。

## 2026-10-26：LCB 与 AIME25@5 重复准备

用户要求这两项多测几轮。首次已询问各自追加轮数（推荐各 3 轮，加原轮共 4 轮），首次准备
完成时仍未收到轮数回复。当时全部 GPU 为其他用户活动任务占用，没有空闲 TP2 卡对；
没有启动新的 GPU 实验、等待资源的后台任务或占卡进程，既有第一轮成绩不变。

新增 `repeat.py`，仅复用已完成 supplement 的 LCB/AIME 正式命令，为新轮次替换输出和
缓存路径，保留 seed=1234、lambda=0.7、k_min=5、TP2、全部题目和原评分协议。启动前
核对来源短测/正式评测退出 0、源码摘要及模型/校准身份未变，并重新验证原始第一轮产物；
本次已确认这些来源条件不变，因此同条件重复可直接复用已有验证，不再重复长短测。

报告保留第一轮及每个新增轮次、均值、范围与轮间标准差，不取最高分替代第一轮；固定
seed 的多次运行不写成独立 seed。缓存隔离、拒绝短测命令当正式重复和全部轮次聚合的
CPU 检查通过，Ban 测试合计 40 passed，未安装依赖或改变算法。准备证据为
[preflight.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_preparation_20261026T042321Z/preflight.json)，
实际轮数和正式命令须在收到用户回复且有卡对可用后落盘。无新增算法经验，未删除旧产物、
修改全局记忆或执行 Git 操作。

用户随后要求“现在继续”，本次按前述方案固定为 LCB 和 AIME25@5 各追加 3 轮，分别记为
第 2/3/4 轮，连同保留的第一轮各共 4 轮。北京时间 2026-10-26 16:40 的实时检查只有
GPU 0 空闲，其余卡均有其他用户活动进程，仍无空闲 TP2 卡对；未占用 GPU 0、未停止
他人进程、未共享挤占显存，也未启动等待资源的后台程序。已询问可用卡对或运行窗口，
轮数不再重复询问。

再次确认来源源码未变、512 条校准身份和 SHA 一致，且 LCB/AIME 原始产物复核分别还原
52.00% 与 58.6667%，实际 input token 哈希与已存报告一致。已将 6 次正式运行的命令
（原参数、独立输出与缓存）保存为
[commands.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_preparation_20261026T084052Z/commands.json)，
状态与源文件摘要保存在
[preflight.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_preparation_20261026T084052Z/preflight.json)。
计划结果目录为 `results/ban/repeats_20261026T084052Z/`，目前尚未创建正式运行目录或进程。
交付前再次检查时 GPU 0 也已被新任务占用，全部卡均有其他用户活动进程，仍未启动；
此时唯一缺失条件为可用 TP2 卡对，追加轮数已确定，不再把轮数标为待确认。

## 2026-10-27：LCB 与 AIME25@5 各三轮重复已启动

用户再次要求继续后，启动前实测 GPU 0–3 仅有 1 MiB 系统上下文、无计算进程，GPU 4–7
仍由其他用户任务占用。因此使用两条独立 TP2 路径：0,1 依次执行 LCB 第 2/3/4 轮，2,3
依次执行 AIME25@5 第 2/3/4 轮。保留 supplement 第一轮，结束后每项共 4 轮。其他用户
进程保持原位，没有占用额外卡对或启动占卡程序。

实际启动时间为北京时间 2026-10-27 13:06:31，控制器 PID `1528415`；LCB 第 2 轮子进程
PID `1528772`，AIME25 第 2 轮子进程 PID `1528771`。控制器已完成来源短测和正式结果
核验、源码/校准摘要检查及原第一轮原始产物复核，没有重新校准或重复已适用的长短测。
这是首次正式启动追加重复，前两次 preparation 目录均为未启动的资源不足记录。

参数与第一轮相同：Ban lambda=0.7、k_min=5、compact、同一份 512 条校准、TP2、BF16 KV、
eager、seed1234、batch128、batched tokens4096、context40960、生成上限38912、T=0.6、
top_p=0.95、top_k20、min_p0。LCB 每轮完整 175 题、每题 1 次生成；AIME25 每轮完整
30 题×5 回答，使用正式原始回答 avg5。不同轮次的模型视图、缓存、日志和结果目录均独立，
第一轮文件不变；最终汇总包含全部 4 轮及均值/范围，不择优替换。

本轮路径：

- [运行状态](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_20261027T050631Z/status.json)
- [六次正式命令](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_20261027T050631Z/commands.json)
- [来源、预算和源码标识](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_20261027T050631Z/manifest.json)
- [控制器启动信息](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_launch_20261027T050631Z/launch.json)

输出目录为 `results/ban/repeats_20261027T050631Z/`，日志在 `logs/`，每轮结果在
`round_02|03|04/<task>/`；控制器日志在对应 launch 目录。当前两项第 2 轮运行中，无新增
正式分数，不能把已启动写成已完成。最终 `report.json` 将同时保留原第一轮与各新增轮次。
此轮只执行已准备的重复入口，没有改动模型/算法、依赖或运行中的正式源码，也无新的
Git 操作、产物删除或可推广算法经验。

北京时间 13:10:36 的有限启动检查确认两项均已完成模型初始化与全部输入渲染，进入
生成循环；所属四张卡持续计算，控制器 `errors=[]`，两份日志均无 traceback。此时尚无
完成的回答，不能据此判断最终精度。检查记录见
[progress_verified.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_launch_20261027T050631Z/progress_verified.json)。
按后台执行约定，此次交付后停止主动监控，保持有限的六次评测自行继续；后续再次唤起时
核对本次运行标识、退出状态和原始结果，不重复启动。

## 2026-10-28：六次追加重复完成与结果核验

本次重新核对运行 `repeats_20261027T050631Z`：控制器及六个阶段均为 `complete`，
六次评测全部退出 0，`errors=[]`。最后一项 AIME 第 4 轮结束于北京时间
2026-10-27 18:16:36；没有残留的本次评测进程，六份日志未发现 ERROR 或 traceback。
没有重复启动评测或改变原协议。

调用原有 `additional_report.summarize` 重新核验原首轮与六次追加的实际产物，全部与
各轮已存报告一致。每轮 LCB 覆盖完整 175 题、每题 1 次生成，AIME25 覆盖完整
30 题×5 回答；prompt、gold、实际输入 token 哈希及正式推理配置均与原首轮一致。
逐题指标重新聚合与正式分数一致，原始回答缓存和请求完成原因覆盖完整。

| 轮次 | LCB | AIME25 avg5 |
| --- | --- | --- |
| 第 1 轮（原补测） | 91/175，52.0000% | 88/150，58.6667% |
| 第 2 轮 | 88/175，50.2857% | 88/150，58.6667% |
| 第 3 轮 | 91/175，52.0000% | 87/150，58.0000% |
| 第 4 轮 | 88/175，50.2857% | 95/150，63.3333% |
| 四轮均值 | 51.1429%（合计 358/700） | 59.6667%（合计 358/600） |

LCB 四轮范围 50.2857%–52.0000%，轮间总体标准差 0.8571 个百分点；AIME 范围
58.0000%–63.3333%，总体标准差 2.1344 个百分点。均保留原首轮和全部追加轮次，
未取最高分替代正式汇总；这仍是同一题集、相同 seed=1234 和推理配置的重复运行，
不是新增独立题集或独立 seed 实验。AIME 使用原始回答的正式 avg5，未改为 pass@5
或仅 final 文本重评分。

各轮 48 层的 selected 与 dispatched 统计逐项相等，TP2 只计一份；跨四轮按 token-layer
计数加权，LCB 平均保留 6.072237 个专家，AIME 为 6.073711。达到生成长度上限的回答
均按原协议保留：LCB 四轮分别 0/1/2/1 条，AIME 分别 2/2/5/6 条，没有删样本或重抽。
本次没有这两项的同轮 top-8 对照，因此这些绝对分数仍不能单独证明 98% 精度保持率，
也不据此宣称吞吐提升。

证据为 [四轮汇总](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_20261027T050631Z/report.json)
及 [本次产物复核](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_20261027T050631Z/completion_verified.json)，
实际配置、源码摘要和命令沿用上一节链接。无新增可推广算法经验；本次只补全结果与
核验记录，未改模型/算法/依赖，未启动新 GPU 实验、删除既有产物或执行 Git 操作。

### 转换后开启与关闭 Ban 的现有结果比较

用户要求比较转换后模型是否开启 Ban 的精度。本次从历史关闭 Ban 的原始结果中提取
对照，没有补跑 GPU。历史 `matrix_norm` AIME 重测的资格记录确认恢复的 49 个权重
分片与原正式评测一致；当前 Ban 使用同一份恢复导出。MMLU-Pro、LCB 及 AIME 的
prompt、gold 和实际缓存 input token 均逐题与 Ban 输入审计匹配，推理配置核对时只
排除复制模型视图和 sidecar 的绝对路径差异。

MMLU-Pro 从历史全量 12032 题的逐题产物中提取本次完全相同的 300 题，得到关闭 Ban
216/300=72.0000%，开启为 211/300=70.3333%。因此无需拿全量 75.3324% 与 300 题分数
直接相减；这里是历史全量生成中的同题子集，生成批次构成仍与本次 300 题运行不同。

| 任务 | 关闭 Ban（top-8） | 开启 Ban（平均约 6.07 专家） | 变化，百分点 | 分数比值 |
| --- | --- | --- | --- | --- |
| MMLU-Pro，同一 300 题 | 72.0000% | 70.3333% | -1.6667 | 97.6852% |
| LCB，175 题 | 54.8571% | 51.1429% | -3.7143 | 93.2292% |
| GSM8K，1319 题 | 95.0720% | 95.9060% | +0.8340 | 100.8772% |
| AIME25@5，30 题×5 | 60.8333% | 59.6667% | -1.1667 | 98.0822% |

GSM8K 是已完成的同轮对照。其余均为历史关闭 Ban 与新 Ban 的跨运行参考，不将差值
全部归因为剪枝。LCB 为关闭 Ban 原正式一轮与 Ban 四轮均值；AIME 两侧均包含原正式
首轮及全部三轮追加，关闭 Ban 四轮原评分为 64.0000%、63.3333%、58.6667%、57.3333%，
合计 365/600，开启为 358/600。没有择优，也未混入按 final 答案人工修正的 AIME 分数。
两侧重复均为相同 seed 的独立进程和缓存，不称为独立 seed 实验。

现有证据支持 GSM8K 同轮保持率超过 98%；历史参考下 AIME 均值比值约 98.08%，
MMLU-Pro 与 LCB 分别约 97.69%、93.23%，不能据此声称所有任务达到 98%。该比较不
改变此前“LCB/AIME 未补同轮 top-8”这一边界。逐题/输入核验及来源保存在
[ban_vs_top8_comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_20261027T050631Z/ban_vs_top8_comparison.json)。
无新增算法实现或可推广算法经验，仅补全已有产物的比较口径。
