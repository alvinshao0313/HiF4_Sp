# 2026-10-28：自生成轨迹重新校准 Ban

用户批准复用已有原生生成轨迹、数学代码各半、匹配专家预算的三组对照，并明确要求
MMLU-Pro 使用完整12032题。实现位于
[独立实验入口](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/ban_trace_calibration/README.md)，
保留旧 Ban 算法、默认300题入口、模型权重及全部旧结果。日期按北京时间。

## 固定实验

模型沿用 `aime25_repeats/matrix_norm_20261015T081958Z/export`，权重为已有 BF16 QDQ 导出，
保留在线 R64、HiF4 激活量化、fusable_r64 ABI3；没有重新重构模型或训练 router。
三组为关闭 Ban 的top-8、已有正式512窗口C4 Ban、自生成轨迹重新校准Ban。
LCB 每组完整175题×3轮，MMLU-Pro 每组完整12032题×1轮，共12次正式评测。

推理为TP2、eager、BF16 KV、seed1234、batch128、batched tokens4096、context40960、
T=0.6、top_p0.95、top_k20、min_p0；LCB生成上限38912，MMLU上限32768。
模板及thinking沿用原协议，MMLU revision固定3373e0，实际Arrow和全部输入摘要落盘。
每次推理独立模型视图和预测缓存，不把历史分数当本轮top-8对照。

## 数据、预算与验证

复用 `reasoning_trace_calibration/continue_20261016T135141Z` 经排重的128条训练和32条
验证轨迹，数学代码各半。逐份核对原始生成JSON、校准tensor与隔离审计的摘要和token。
符合2048长度的训练轨迹为数学52条、代码63条，其余13条记录为长度排除。

seed0有放回抽取数学256、代码256个窗口，终点在原生成回答内、不跨题拼接、不重新
tokenize。实际窗口512×2048，来源114道题，token visits为1,048,576，唯一来源token
位置573,453，完全重复窗口1个；不是512道独立题或1,048,576个独立token。
保存的验证顺序中取数学8题、代码8题原始prompt，只用于lambda预算匹配，不按正确性筛选。

沿用原最后位置top1000支持KL、逐层top3校准和原在线Ban公式；k_min=5，两种Ban均
匹配开发集平均6.07±0.05个专家。先lambda0.7，必要时探测0/1及最多6步二分；失败不
进入该组正式评测，不根据测试成绩再调参。正式任务分别报告实际预算；两Ban相差超过
0.10时，不解释为相同预算下的精度收益。

CPU回归13项通过，覆盖抽样、原token、短轨迹、验证prompt、卡对互斥、缓存隔离、正式
全量MMLU命令与覆盖拒收、按题分簇bootstrap。真实输入审计通过，流式原始产物报告
回放还原旧LCB首轮91/175=52%，没有新GPU精度结论。旧Ban资格对应源码摘要未变，
复用其算法、native路由和紧凑执行核证据；本次不重做无关底层数值排查。

## 运行标识

本次结果根目录为
[run_20261028T090009Z](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/)。
输入准备已完成，以下文件为实际产物：

- [配置、模型身份、输入和源码摘要](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/manifest.json)
- [窗口与来源记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/windows.json)
- [LCB输入审计](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/lcb_input_audit.json)
- [全量MMLU-Pro输入审计](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/mmlu_pro_input_audit.json)
- [阶段、命令、PID与退出状态](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/status.json)

实时GPU检查后使用空闲0,1执行真实TP2预检；其他活动任务未动。预检先做两领域各一个
窗口的48层诊断校准及prefill/decode，再用正式上下文/批容量和16个验证prompt执行
64步生成。诊断不进入512窗口正式产物，不产生预算或精度结论。预检及后续实际状态以
上述文件为准；此处尚未宣称12次正式评测完成。

最终报告保留LCB全部三轮和全量MMLU结果，计算同轮top-8保持率、自生成Ban相对C4
变化及按题分簇10000次bootstrap区间（seed0，LCB同题三轮捆绑）。未达98%也完整交付，
不择优、不删除长度截断回答，不从路由减少推断吞吐提升。

参考经验为[评测证据](../lessons/evaluation_evidence.md)中的固定数据版本、实际token对齐、
校准测试隔离和预测缓存位置；本次保持冻结模型，只改变Ban校准和预算匹配，不沿用
此前重训权重的数据优劣结论。尚无新增算法经验，未安装依赖、删除旧产物或执行Git操作。

## 本轮交付状态

真实TP2预检两阶段均退出0：两窗口完成全部48层校准；2048输入加64个生成token的
每层计数为2111（prefill加后续63次decode），选择和实际分发一致；16条验证prompt的
正式引擎条件短测全部完成。新鲜模型文件视图也核对了全部49个权重分片和sidecar身份。
预检证据为
[preflight.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/preflight.json)，
CPU、真实数据和报告回放证据见
[implementation_validation.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/implementation_validation.json)。

北京时间2026-10-28 17:10:35，已用 `Popen(start_new_session=True, stdin=DEVNULL)`
启动有限后台控制器PID `4054485`，正式校准子进程PID `4054592`，使用GPU0,1；仅有一对
空闲TP2，GPU6单卡未占用，其他用户任务未改动。因此按trace→C4→top8依次执行各组，
每组先满足校准/预算/短测依赖，再运行LCB三轮和完整MMLU一轮。

截至17:15:06，正式校准已完成17/512窗口，每个完成窗口均覆盖48层，`errors=[]`，
不是只完成初始化。此时尚无新正式精度分数，12次下游评测尚未完成。
实际命令和日志见
[launch.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/launch.json)、
[controller.log](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/controller.log)，
交付快照见
[handoff.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/run_20261028T090009Z/handoff.json)。
按后台约定停止主动监控，保留任务运行和冻结依赖；后续再次唤起时核对本运行标识和原始
产物，不重复启动。未创建自动监控、资源等待或重试程序。

## 2026-10-29：首次终止核查与授权接续

10月29日核查确认，原运行于10月28日23:27:21退出，状态failed。此前512窗口校准全部
完成并退出0；自生成轨迹的开发预算匹配选择lambda=0.6125、平均6.082133个专家。
正式参数的LCB和MMLU两题短测均完成并通过原始产物核验。十二次正式评测均未实际启动，
原 `report.json` 为incomplete，不能拿短测分数当正式结果。

直接终止条件是下一阶段GPU0空闲检查未通过。原控制器仅等待主进程退出，未等待vLLM
worker及CUDA上下文清理；原报错未记录占用PID，因此不能断言历史时刻是外部占用还是
退出清理竞态。本次修复该阶段交接缺口：每个子命令使用独立session，只等待本session
worker退出和上下文清除，保留原空闲阈值并记录占用PID，不对其他活动进程等待或发信号。

用户明确要求重启后，先验证旧33份源码、181项输入及模型身份未变，并保存原控制器
快照；随后仅修改控制器进程管理，新增显式接续入口和生命周期验证。CPU合计15项通过，
其中实际子进程测试重现“主进程退出后子进程仍活着”，并验证等待不干预其他session。

原运行及其错误记录保持原位；新接续重新核验正式校准输入tensor、artifact摘要、开发
预算所有试探的输入/TP路由，以及两项短测原始回答，跳过已完成阶段，不重新选择数据或
lambda，不复用正式预测。旧控制器快照与重启前核验保存在
[restart_preparation_20261029T020036Z](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/restart_preparation_20261029T020036Z/)。

第一次连续启停验证的控制器在会话中断时退出，GPU已释放；因缺少退出状态，没有记为
验证通过。该目录 `continue_20261029T020036Z` 保留原状态快照并标为interrupted。
新的接续目录为
[continue_20261029T020854Z](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/continue_20261029T020854Z/)，
启停验证改用脱离会话方式启动。实际来源、复用阶段和控制器差异见
[continuation.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/continue_20261029T020854Z/continuation.json)，
当前阶段见
[status.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/continue_20261029T020854Z/status.json)。
GPU启停验证通过前不启动正式评测；当前记录不宣称已有正式精度结果。

### 接续启动确认

新目录的两次连续真实TP2验证均退出0，覆盖16个验证prompt、正式batch/context和BF16 KV，
仅生成步数缩至64；每次计算后确认GPU0,1均为0 MiB且无计算PID，再交给下一次引擎。
验证完成时间为北京时间10月29日10:11:44，证据见
[restart_verification.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/continue_20261029T020854Z/restart_verification.json)。
日志在完整输出和 `Shutdown complete` 后仍出现原vLLM客户端退出时的
`Engine core proc ... died unexpectedly` 提示；两次均退出0，实际16/16请求、路由和资源
释放验证通过。该现象与原Ban预算日志记录的退出提示一致，不冒充计算失败或精度结论。

北京时间10月29日10:12:39，正式接续控制器PID `1001928` 已脱离会话启动，GPU0,1；
从自生成轨迹Ban的LCB第1轮开始，子进程PID `1002043`。旧校准、开发预算和两项已完成
短测只读复用，所有正式模型视图和预测缓存均为新目录，原失败/中断目录保留。
后续仍是三组各LCB三轮及完整12032题MMLU一轮，共十二次正式评测，未降低协议。
实际命令与日志见
[launch.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/continue_20261029T020854Z/launch.json)、
[controller.log](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/continue_20261029T020854Z/controller.log)。

10:14:05交付检查确认控制器及正式LCB子进程存活，175条输入全部渲染并进入生成，GPU0,1
正在计算，`errors=[]`且无traceback；此时尚无完成的正式分数。检查快照为
[handoff.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/continue_20261029T020854Z/handoff.json)。
已停止主动监控，保持后台运行和冻结依赖；不等待全部实验结束或创建监控任务。

## 2026-11-02：6.07接续终止核查与6.4预算测试

核查 `continue_20261029T020854Z` 的状态和原始报告：自生成轨迹Ban的LCB前两轮均
完成175题、正确92题（52.5714%），实际平均6.081050个专家；第三轮未启动。第二轮
退出0且本session清理完成后，GPU0出现其他session的PID1338467，下一阶段记录
GPU0/1有PID1338467/1338033，资源检查失败。控制器于10月29日06:04:30 UTC退出，
状态failed。C4、top-8及三组全量MMLU均未运行，不能据两轮推断受控改善或98%保持率。

用户授权用空卡测试平均6.4个专家。保持三组、LCB175题三轮、MMLU-Pro12032题一轮
及原正式推理协议；只改变开发预算目标为6.40±0.05，k_min仍为5，各Ban独立重选lambda。
已有自生成轨迹512窗口校准和C4正式校准只读复用，不重做转换；全部短测和正式预测
重新生成，不复用6.07的lambda或回答。运行身份、来源及源码摘要见
[新运行目录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/)、
[manifest.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/manifest.json)、
[reuse.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/reuse.json)。

实现将目标预算从固定常量传到每次运行清单、搜索、失败提示和实际专家数报告；保留
原入口默认6.07，不修改Ban选择公式、执行核或模型。`rebudget`只验证实际校准依赖；
校准日志枚举但未导入的下游报告文件不作为校准算法依赖。原始输入、校准产物、模型
和sidecar身份均核验，当前38项源码和188项输入固定在清单；未安装依赖或执行Git操作。

CPU19项通过。新的真实TP2预检退出0，保留正式上下文40960、batch128、batched
tokens4096和BF16 KV，16个验证prompt各生成64token；TP路由一致，selected=dispatched，
资源清理后GPU0/1均0 MiB。该阶段使用诊断校准和lambda0.7，观测6.065922仅为短测
统计，不当作6.4预算匹配结果；两组正式预算仍须使用各自512窗口产物和完整生成搜索。
证据见
[implementation_validation.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/implementation_validation.json)、
[preflight.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/preflight.json)。

后台控制器PID2656176已用独立session和断开stdin启动；本次空闲卡分配为top8=0,1，
C4=2,3，trace=4,7，保留GPU5,6的既有任务。每组满足预算/短测条件后执行完整正式评测，
无自动重试、占卡或资源等待。命令、日志、阶段、PID和退出状态见
[launch.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/launch.json)、
[status.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/status.json)。
本次目标为下游精度对照；平均6.4只表示相对top-8减少20%的专家调用，不作吞吐或延迟结论。
尚无新增算法效果经验；参考既有固定输入、缓存隔离和session清理经验，正式分数待运行。

交付时三组引擎均已初始化并进入真实生成；top-8的LCB两题短测已返回1题，两种Ban正在
开发输入的第一个lambda探测，`errors=[]`。本次十二次正式评测尚未完成，暂无正式精度
分数。进度与进程核对快照见
[handoff.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/handoff.json)。
按后台约定停止主动监控，保留任务和源码依赖，不等待长实验完成。

## 2026-11-03：6.4预算评测完成与基线复用

两种Ban各三轮LCB及完整12032题MMLU-Pro均已完成，八次正式评测全部退出0，最后一次
结束于北京时间11月3日09:07。完整结果、每轮正确数、实际专家数与限制见
[REPORT.md](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/REPORT.md)，
机器可读对照见
[final_comparison.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/budget64_20261102T025500Z/final_comparison.json)。

LCB三轮均值：top-8 53.9048%，C4 Ban 53.3333%（保持98.9399%），自生成Ban
52.9524%（保持98.2332%）。实际平均专家分别6.3851和6.4009。两种Ban按均值达到
98%，不代表每一轮均达到，也不代表置信下界达到98%。自生成相对C4为−0.3810个百分点，
题目分簇配对95%bootstrap区间[−2.8571, 2.0952]，未显示LCB改善。

MMLU-Pro：C4 Ban 9031/12032=75.0582%，自生成Ban 9095/12032=75.5901%，实际
平均专家6.3645和6.3922。新top-8 MMLU因GPU占用未启动，未补跑；依用户明确要求，
复用历史同模型完整9064/12032=75.3324%基线，保持率分别99.6359%和100.3420%。
重新核验全部题目prompt/gold/实际input token、单回答数、推理协议一致，实际输入hash
均为8cae2f26fc9e7223；导出元数据一致、sidecar逐值一致，并复用已有49分片恢复逐位
一致的资格证据，当前模型身份与本轮清单一致。历史旧权重已清理，未重复做旧权重全文件
hash。自生成相对C4 +0.5319个百分点，配对95%区间[−0.0166, 1.0638]；该区间也跨0。

原控制器的failed及report.json的incomplete保留，准确记录新基线未启动；最终对照明确
标注complete_with_reused_baseline，不修改原始退出状态、生成或评分。仅做CPU汇总，
复现入口为运行目录的finalize_comparison.py；没有启动新GPU任务、重测基线或安装依赖。
两种Ban正式预算差均小于0.10；结果不支持自生成校准稳定优于C4，不从专家数下降推断
吞吐收益。经验边界已补入[评测证据](../lessons/evaluation_evidence.md)。原始产物均保留。

## 2026-11-04：固定6.4预算，等待GPU0/1补测其余任务

用户明确要求本对话只补测Ban6.4的其余下游指标，转换提速实验不在本对话继续，并等
GPU0、1空闲后运行。沿用原 `matrix_norm_20261015T081958Z/export`，两种512窗口
校准产物、k_min=5和已选lambda均不变：C4为0.8875，自生成轨迹为0.7375。没有重新
搜索预算、校准或转换。LCB与完整MMLU-Pro直接使用上节已完成结果。

本次新增独立补测入口，计划两组Ban各完成GSM8K1319题一次生成，以及AIME24@5、
AIME25@5各30题五次生成。同模型的完整GSM8K和AIME25 top-8已核验实际prompt、
gold、input token、推理协议和导出身份后复用；AIME24此前没有该模型基线，首次补测
一次，共7项正式GPU运行。全部新预测分组分任务建立独立模型文件视图与缓存。数据复用
已有固定Arrow快照，不继续其来源实验；revision、SHA及输入审计保存在新运行目录。

复用的GSM8K为1254/1319=95.0720%。AIME25保留原官方64.0%，另按既有最终提交
口径复核9条歧义回答后为99/150=66.0%，复核包含原回答摘要和逐条理由；不从thinking
补出未提交答案。新生成AIME同样保留两种分数，需人工复核时明确标记待复核，不伪造
最终保持率。模型49分片的恢复资格证据及当前模型/sidecar身份固定在清单。

CPU回归19项通过，核对了7项正式命令无样本截断、三个任务完整输入覆盖及1/5次生成、
TP2/context/batch/BF16 KV和采样配置、固定lambda与缓存隔离。新路径GPU验证尚未
运行；资源可用后先对top-8/C4/trace各做AIME24一题五次生成、64 token的真实TP2短测，
保留正式引擎形状，并核对完整路由统计、TP一致及selected=dispatched。原算法与执行
核未修改，复用已有适用数值验证；短测失败则不进入正式运行，不自动重试。

运行目录为
[remaining64_20261104T040720Z](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/)。
一次性脱离会话控制器PID1029071已启动，实际命令见
[launch.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/launch.json)，
进度见
[status.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/status.json)，
日志见
[controller.log](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/controller.log)。
交付核查时状态为waiting_for_gpu、phases为空、errors为空：GPU0、1分别由既有
PID1006265、1006536占用，各68161MiB，尚未启动GPU推理。本次用户授权的启动前
等待最多24小时，只读检查两卡同时空闲后再核验依赖并执行短测和正式任务；超时退出，
不占卡、不停止既有进程、不自动重试，阶段间资源不足仍立即失败。

交付快照与CPU证据为
[handoff.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/handoff.json)、
[implementation_validation.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/implementation_validation.json)。
按约定停止主动监控，保留一次性实验等待和冻结依赖；本次没有新增算法效果经验，不将
等待中、短测或目标专家数写成正式精度、实际任务预算或吞吐结果。

## 2026-11-05：其余三项完成与五任务汇总

核查确认，一次性等待控制器在11月4日13:07（北京时间）取得GPU0、1，三个真实TP2
短测均通过，随后7项正式评测全部完成、退出0，最后一项结束于11月4日22:53:42。
未剪枝模型与两组Ban均覆盖正式参数和真实请求；新正式运行共3388条回答，包含两次
完整1319题GSM8K和五次30题×5回答的AIME。当前控制器已退出，GPU0、1无该任务进程，
没有重试或启动转换提速实验。

AIME生成后状态曾为awaiting_aime_review，不能直接当作最终评分完成。本次用CPU
复核新生成的30条歧义回答，保存原回答SHA、二元判定及理由；缺少boxed的5条另从原始
缓存读取完整最终提交，避免被日志尾部截断或尾随空白误导。保留错误表达式和未完成回答
为0，不从thinking选取正确数字、不修复模型算错的中间结果。统一最终提交avg@5后：

| 本次补测 | top-8 | C4 Ban | 自生成Ban |
| --- | --- | --- | --- |
| GSM8K | 1254/1319=95.0720%（复用） | 1258/1319=95.3753% | 1261/1319=95.6027% |
| AIME24@5 | 113/150=75.3333% | 115/150=76.6667% | 114/150=76.0000% |
| AIME25@5 | 99/150=66.0000%（复用） | 93/150=62.0000% | 101/150=67.3333% |

C4/自生成在三项上的实际平均专家分别为6.3750/6.3967、6.3956/6.4103、6.3981/6.4144，
每项两组差均小于0.10；48层路由和TP ranks核对一致，selected=dispatched，只计一份。
C4的AIME25保持率93.9394%，未达98%；自生成为102.0202%，本轮五项观测保持率均
达到98%。这不是五项置信下界达标，也不能保证重复运行或其他校准/任务普遍成立。

原始自动评分独立保留：AIME24 top8/C4/自生成为74.0000/73.3333/70.6667%，
AIME25为64.0000/58.6667/60.6667%。最终提交口径在启动前已固定于manifest，所有
新组和复用AIME25基线同口径复核；不把原自动评分与最终提交评分混算保持率。按题目
整体保留五个回答、10000次seed0配对bootstrap，自生成减C4的AIME25为+5.3333
个百分点，95%区间[0.6667,10.6667]；其余补测GSM8K和AIME24的差值区间均跨0。
这是单轮30题的任务级证据，不是独立150题或多seed重复的稳定改善。

本次同时复用已完成的LCB三轮与完整12032题MMLU-Pro，汇成
[五任务报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/REPORT.md)及
[all_metrics.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/all_metrics.json)。
报告保留正确数、全部轮次来源、保持率、原分、截断数、实际专家、配对区间及复用证据。
CPU复现入口为
[finalize_results.py](../../../Native_NVFP4_HiF4_Linear_Puncture/results/ban_trace_calibration/remaining64_20261104T040720Z/finalize_results.py)。
原评分待复核状态快照、原始指标和回答保留；status.json现为complete，并另记finalized_at，
不改GPU原始结束时间或原6.4控制器的失败记录。经验边界补入
[评测证据](../lessons/evaluation_evidence.md)；未进行GPU重测、吞吐实验、安装依赖或暂存提交，
没有清理原始数据、模型和结果。
