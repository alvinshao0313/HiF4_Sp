# 格式转换章节补充实验计划：Qwen3-30B-A3B NVFP4 → HiF4

日期：2026-11-16。适用模型：固定原生 `nvidia/Qwen3-30B-A3B-NVFP4`。

状态：**2026-11-16 已进入授权实施阶段。** [独立实现、实际命令与当前证据](../../experiments/thesis_format_conversion_supplement/README.md)；运行状态以对应 RUN 的 manifest、短测、jobs 和日志为准，不能将实现完成写成正式结果完成。历史实验的配置、指标、checkpoint 和运行状态保持原位。

执行期追加要求（优先于下文原单轮及第9.2节条件式三轮）：**所有报告任务统一为5次完整评测的算术均值**，分别保存原始回答与缓存，保留各任务原协议与 seed；可复用的历史独立完整轮计入次数。同一原始生成的多个引用只计一次。不满5次仅报告进度，不报告正式均值/L5。无 AIME、六行主比较、两条同等优先强基线和 A/B 数据边界不变。

执行期校准追加授权（2026-11-17）：GPTQ 的原128条校准在第5层出现专家零命中，用户允许扩大校准集以覆盖所有专家。GPTQ 单独扩展为同版本S1K中通过原题簇隔离规则的850条完整序列（保留原128条；合计8,568,064 token），排除验证、开发、正式测试及已知官方test来源簇。A20、B20和正交消融的数据不变；此后GPTQ与A20不再是等校准量比较，必须列明样本数、token和全部转换成本。所有48层×128专家必须有真实有效路由，gate/up/down采集次数与路由计数一致；零命中仍停止，不能强制路由、补测试题或隐式RTN。扩大数据后的验证和正式状态见[续行记录](../../results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z/RUN.md)，本条不是全模型覆盖通过的声明。

## 1. 目标与范围

2026-11-18补测授权：用户新增GPU4、5，并明确S1K校准A20与自生成数据校准B20两版matrix_norm都补齐五任务、各五次完整评测均值。各复用已核验的原始首轮，补第2至5轮，使用独立模型视图/缓存及原协议；B20本次也执行完整五任务，不只补效率副表的LCB/MMLU-Pro。A20与GPTQ同S1K数据，B20保留自生成数据比较边界；GPU0 GPTQ及GPU6、7 scale保持运行，不改变其依赖。

2026-11-18 最新授权（替代下述850条S1K扩充和C4方案）：GPTQ改回与干净A20 `matrix_norm_s1k` **完全相同的128条S1K train完整序列**，1,272,746 token，最长18,945；同ID、顺序、token及有效mask，不使用32条val。参考[LLM Compressor Qwen3-MoE](https://github.com/vllm-project/llm-compressor/blob/0.9.0/src/llmcompressor/modeling/qwen3_moe.py)：校准时所有有效token经过全部128专家，gate/up及各专家真实SiLU(gate)×up的down输入均收集统计；最终层输出仍只聚合原top8选中的专家及原权重。自然路由零命中允许，校准输入零命中仍失败，两种计数分开记录。32专家Hessian驻留分组只分批收集，全部分组覆盖同一完整语料，不改变GPTQ量化顺序、G64、阻尼或在线推理。用户另明确授权停止上轮C4检查并保留产物。新转换从原生第0层开始；旧S1K/C4转换不接续。实现位于独立 `experiments/thesis_format_conversion_all_experts/`，保护活动scale评测依赖；五任务各五次及其他候选数据不变。

2026-11-18 后续校准授权：850条S1K仍在索引21层出现expert29零命中，用户明确选择“换成纯C4”，要求先检测全专家覆盖，并随后明确预算为**512段×512 token，共262,144 token**。候选数据改为固定revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`的C4 English train：本地首个原始train shard，seed42打乱文档，取通过隔离检查且足够长的512个不同文档，每文档以seed42抽一个512-token窗口；不跨文档拼接、不混S1K。这是用户对新C4采样定义的修改，不再沿用S1K完整长序列定义；停止并保留此前1024条完整文档的本轮临时检查。先扫描原生NVFP4全48层及真实gate/up/down输入，报告零命中与低命中；原生覆盖不能保证GPTQ学生前缀覆盖。通过必要验证后仍须从原生第0层转换，不能接用旧S1K的21层。其他方法的数据、五任务及五次均值不变；GPTQ需明确不同校准来源、token预算及全部成本。证据见[512×512覆盖记录](../../results/thesis_format_conversion_supplement/c4_512x512_20261118T025938Z/RUN.md)与[scale评测记录](../../results/thesis_format_conversion_supplement/c4_20261118T024647Z/RUN.md)。GPU资源追加为0、1、6、7；6、7优先运行已完成的scale模型评测。

本轮补齐格式转换章节的比较与解释，回答三个问题：

1. 在同一原生 NVFP4 checkpoint 上，我们的重构方法相对于直接转换、固定 R64、**R64＋scale 贪心搜索**、**R64＋HiF4-GPTQ**，分别恢复了多少下游精度？
2. 最终重构改变了哪些量化/输出误差？权重侧矩阵的非正交自由度是否带来收益？
3. 已完成的短程转换，在准确率与转换成本之间处于什么位置？关键 LCB 差异能否在有限重复中保持？

已确定的本轮口径：

- **不再运行或汇总 AIME24/AIME25**；AIME 不参与本轮正式指标、主表、方法选择或停止依据。历史 AIME 文件保留，旧报告不覆盖。
- 正式任务固定为 **MMLU-Pro、LiveCodeBench、GSM8K、ARC-E、ARC-C**。
- **R64＋scale 搜索与 R64＋HiF4-GPTQ 是同等地位的两条必补强基线**，均进入主表、五项完整评测与转换成本表；不根据局部误差或个别任务好坏取消其中一项。
- 更多模型上的验证另按迁移计划开展；先在 30B-A3B 上固定比较与方法定义。
- 本轮围绕已有成熟的 `matrix_norm` 研究线补证据。H-only 解析求解、MXFP4、剪枝等支线保持各自定义与状态，不视为已经替代主方法或默认并入本轮。

## 2. 主比较矩阵

以下六行是正式主表的固定结构。原生模型提供转换前参照，其余五行是目标 HiF4 对照。

| 标识 | 方法 | 权重处理 | 在线激活路径 | 本轮安排 |
| --- | --- | --- | --- | --- |
| native | 原生 NVFP4 | 原 checkpoint、ModelOpt 语义 | 原生路径 | 优先复用协议匹配的完整结果 |
| direct | 直接 HiF4 | 同源 effective weights 直接合法 HiF4 QDQ | 标准 HiF4，无新增 R64 | 优先复用匹配结果 |
| r64 | 固定 R64 | 按既有固定 R64 变换后，标准 HiF4 QDQ | 固定 R64＋标准 HiF4 | 优先复用匹配结果 |
| r64_scale | **R64＋HiF4 层级 scale 搜索** | 固定 R64 后，搜索合法 S0/e8/e4，以权重 MSE 选优 | 与 r64 相同 | **必补强基线：完整转换＋五任务** |
| r64_gptq | **R64＋本仓库 HiF4-GPTQ** | 固定 R64 后，Hessian 引导的逐列误差补偿与 HiF4 量化 | 与 r64 相同 | **必补强基线：MoE 适配＋完整转换＋五任务** |
| ours_a20 | 干净 S1K 的 matrix_norm，20 轮 | 普通 Linear 权重侧 G64 自由矩阵＋两处 norm 学习 | 固定 R64＋标准 HiF4，router 冻结 | 优先复用已完成的干净 A 模型与结果 |

`ours_a20` 对应现有 `matrix_norm_s1k`。它是已由独立开发集选定的 A，不以删除 AIME 后的测试排名重新选起点。自生成轨迹版本 `matrix_norm_native_traces` 记为 `b20`，用于已有数据对比和严格同数据的效率对照。该约定用于本轮比较，不宣称 A 在所有保留任务上都最好。

历史旧 S1K 上的 `matrix_norm`、DIAG、对称 rotation、router KL、两阶段方案保留为探索性证据。不能将它们与干净 A 混成只改变一个变量的消融。

## 3. 执行顺序与最小新增工作量

| 顺序 | 实验/工作 | 要解决的问题 | 可复用部分 | 最小新增量与交付 |
| --- | --- | --- | --- | --- |
| 先准备 | 无 AIME 汇总与产物绑定 | 主表是否同源、同输入、同评分；哪些结果真正缺失 | 既有报告、逐题记录、数据与模型 manifest | CPU 整理；生成六行主表框架、来源表、缺项清单 |
| 优先补强基线 | R64＋scale 搜索 | 合法量化参数搜索能否达到重构方法精度 | 现有层级 scale 搜索核心、R64 与导出路径 | **1 次整模型搜索转换＋5 个任务** |
| 同等优先补强基线 | R64＋HiF4-GPTQ | 通用二阶 PTQ 能否达到重构方法精度 | GPTQ 核心、主线逐层输入与导出 | **1 次整模型 GPTQ 转换＋5 个任务** |
| 有成品即可做 | 最终方法机制对照 | 改善发生在量化误差、层输出还是累计误差 | 真实 vLLM hook、既有机制工具、上述模型 | 固定输入、代表层诊断；**不新增整模型训练** |
| 补一个关键消融 | 单侧正交矩阵 vs 自由矩阵 | 非正交自由度是否有贡献 | 干净 A 自由矩阵模型；现有正交参数化思路 | 原则上 **1 条新的 48 层重构＋5 个任务**；自由对照匹配才复用 |
| 利用空档补齐 | 短程转换的精度/成本 | 20→5 轮及约半量 token 的实际取舍 | b20、full_e5、random50_e5、coverage50_e5 | **默认零重训**；先补 full/coverage 的 LCB、MMLU-Pro，共 4 个候选×任务执行 |
| 有剩余不确定性再做 | 有限 LCB 重复 | 关键方法差值是否受单轮生成波动主导 | 全部首轮 175 题记录 | 固定至多 3 轮/模型的预定协议，见第 9 节 |
| 仅在章节提出部署性能结论时 | 实际设备性能 | 是否有真实存储或运行收益 | 已有导出、运行边界证据 | 使用目标低比特执行路径另测；不以 A800 QDQ 吞吐代替 |

两条强基线**科研优先级相同**；若 scale 接入先完成，可以先运行，不等待 GPTQ 适配。已有 full/coverage 评测可在新算法准备期间填补空闲 TP2 时段。不得为了执行顺序而重复已适用的基线。

“最小新增量”不包含现有结果协议不匹配时确有必要的补评测，也不把已清理产物默认算作仍可直接运行。首批新整模型工作为 **两次强基线转换＋一次正交消融重构**；没有默认全矩阵多 seed 训练。

## 4. 固定模型、数据与评测协议

### 4.1 同源转换与目标格式

- 源模型固定 revision：`2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`。记录实际 checkpoint/config/tokenizer 与使用到的源码摘要。
- 以当前 30B-A3B **ModelOpt 原生 NVFP4 路径**解码得到的有效权重为转换输入，保留已确认的 scale 乘数与 BF16 舍入边界。不要换用 BF16 祖先模型。
- `NVFP4/dequantize.py` 和旧 `src/native_nvfp4.py` 的 compressed-tensors/H16 路径不能直接作为该 ModelOpt 转换器；实现时沿用主线 MoE 训练/导出的实际加载调用链。
- 所有目标方法固定 HiF4 格式、输入维 G64、目标投影清单、norm/router/lm_head 等保留策略；R64 相关行使用相同矩阵与实际 sidecar。
- 导出为 BF16 保存的 HiF4 QDQ 权重时明确写出；它是 W4A4 数值实验，不能称为已测得的 packed 4-bit 文件或原生 FP4 加速。

### 4.2 校准、验证与开发集

主强基线与正交消融使用已清洗的 **S1K 128 train / 32 val、seed 42、完整序列**，与 A20 对齐。已有干净数据和隔离 manifest 直接复用，不重新抽样。

- matrix_norm / GPTQ 需要校准数据时使用同一 train；搜索超参数及 checkpoint 选择只看对应验证/开发材料。
- **权重 scale 搜索本身不使用校准数据**，如实标为 data-free；不为了形式一致额外加入激活标定或教师目标。
- 独立开发集维持既有 **64 math＋64 code**。现有配置来源为固定版本 MATH 与 CodeContests，并非以 AIME benchmark 评分选择 A/B。实施准备仍绑定实际题目 ID 与隔离清单。
- 移除 AIME 正式评测不要求移除全部数学校准样本，不触发重新生成现有干净轨迹。
- 旧 S1K 已发现测试重合；相关旧模型不能冒充干净校准的主方法或严格单变量对照。清洗本身也不代表已量化出准确率偏差。

A20 重构配置参考：原生初始化，48 层，20 epochs/层，batch=4，AdamW，lr=3e-4，weight_decay=0，cosine，G64，block-delta NMSE，router 冻结且 router KL=0；按现有验证 NMSE 规则选每层训练 checkpoint，再传播学生输入。复用前核对实际 manifest，不用摘要替代原配置。

### 4.3 五项正式评测

| 任务 | 正式规模 | 入口与指标要求 |
| --- | --- | --- |
| MMLU-Pro | **12,032 题** | 根 `main.py` 路径的 lighteval＋vLLM，使用已核验的正式全量版本；不沿用默认 300 题 |
| LiveCodeBench | **v6，175 题** | 现有同一任务切片、代码评分器与单次生成 pass@1 |
| GSM8K | **1,319 题** | lighteval＋vLLM，保持原 prompt、thinking 与评分 |
| ARC-E | **2,376 题** | 现有 lm_eval 路径，保持原 metric 与 few-shot 等协议 |
| ARC-C | **1,172 题** | 同上 |

推理基准协议沿用已完成主线：TP2、BF16 KV、eager、context=40960、max_num_seqs=128、max_num_batched_tokens=4096；生成 seed=1234、temperature=0.6、top_p=0.95、top_k=20、min_p=0、原 thinking 设置与输出长度上限。各任务的 prompt、停止条件、few-shot、代码超时等从相应原始 protocol 继承，不统一覆盖成推测值。

LCB 指标名 `codegen_pass@1:16` 不等于实际 16 次生成；此处保持既有**每题一次**。后续有限重复单独记轮次，不能静默将主协议改成每题 16 次。

复用条件是权重、实际样本 ID / prompt token / gold、推理和评分协议一致，且完整记录可追溯；无需追求同一次进程或同一天运行。各历史 RUN 的 native 得分存在差异，不能挑选较有利的 native 或拼接多轮回答。需要建立一个明确的适用参照；不匹配的结果单列或只补该缺项。

### 4.4 汇总与统计

- 主表给出五项准确率及相对同协议 native 的有符号差值，单位为百分点。
- 同时固定报告五项等权平均正向损失：`L5 = mean_t(max(0, accuracy_native,t - accuracy_method,t))`，准确率按百分数计；增益不抵消另一任务的下降。
- MMLU-Pro/LCB/GSM8K 的三项摘要可作固定辅助视角，不根据结果改成更好看的主指标。
- 有完整逐题配对记录时，按题进行配对 bootstrap（10,000 次、seed 42），报告准确率差与损失差的 95% 区间。五项汇总区间须在每个任务内分别配对重采样再汇总，不能把所有题混成一个大池。
- 若某些旧 ARC 只有聚合结果，保留准确率，标记逐题配对统计缺失；不能用聚合分数伪造五项配对区间。是否补存逐题输出取决于它是否影响最终结论，不默认重跑所有 ARC。
- 删除 AIME 后必须从保留任务重新计算摘要和区间；旧四项/六项平均及其置信区间不沿用。
- 训练 seed 方差、生成重复方差、固定一轮回答下的题目重采样区间分别说明。一次转换不声称跨训练 seed 稳定。

## 5. 先做 CPU 准备：复用、缺项与独立输出

准备阶段只产生本计划的输入清单与新汇总，不改历史 RUN：

1. 绑定源 checkpoint、R64 sidecar、目标量化模块清单、干净 A 数据、B 数据、开发题及评测实际内容。
2. 为主表六行和效率四行登记已有结果：模型/数据/源码摘要、实际 protocol、逐题记录位置、可复用项与缺项。
3. 区分旧 S1K matrix_norm 与干净 A20；区分 b20 与旧 20.0406 小时参考，避免同名混用。
4. 检查已保留导出是否完整；完整则复用，只剩逐层选定参数则在新目录重新导出。参数也缺失时只记录该候选缺口，不能把补评测静默变成重训。
5. 生成新五任务报告，不调用旧含 AIME 的总控或必须等待 AIME 的 report。

2026-11-16 只读检查已经看到 efficiency RUN 的三个 `models/{full_e5,random50_e5,coverage50_e5}` 目录与模型分片；其中 full_e5 列出了 48 层分片、非层分片、权重索引和 HiF4 sidecar。**这仅确认文件目录证据，不等于已完成加载、摘要和实际推理验证。**

建议本次执行统一产物根目录：

`Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/run_<UTC时间>/`

首次实施时再创建，保存：

- `manifest.json`：本计划版本、模型/数据来源、任务列表、模型角色、实际资源、命令与代码版本/摘要。
- `reuse.json`：每个已复用结果的来源与适用依据；`protocol.json`：固定正式协议。
- `models/`、`training/`、`evaluation/`、`mechanism/`、`logs/`：只放本次新产物或明确引用，保留历史结果。
- `costs.json`、`summary.json`、`REPORT.md`：实际成本、无 AIME 指标、结论与限制。

这是一处运行记录，不另建通用调度器或多份计划。新评测使用独立模型视图和预测缓存；仅换结果目录不一定能隔离旧 lighteval 模型目录缓存。

## 6. 两条同等地位的强基线

### 6.1 R64＋scale 贪心搜索

正式名称采用 **R64＋HiF4 层级 scale 搜索**；“贪心搜索”作简称。复用 `HiFloat4/hif4_scale_threshold_optimization/src/weight_search.py::search_weight_groups()`。

固定定义：

- 对同源解码权重应用与 r64 完全相同的 R64，再沿输入维按 G64 搜索。
- 主结果固定现有 **full** 预算：标准 S0 的 E6M2 表示邻居偏移 `{-2,-1,0,+1,+2}`。
- 每个 8 元素块枚举合法的 `(e8,e4_left,e4_right)` 八种组合，对各 S0 候选汇总局部最小平方误差，再按组选择最低权重 MSE。
- 这是 **S0 邻域搜索＋组内局部枚举**，不声称搜索了所有可表示 S0 或全局最优。
- 保持 hardware S0 语义与合法 HiF4 格式；激活使用主线标准参数与固定 R64。此基线不学习 norm、router、权重侧自由矩阵，不加入激活 576 点阈值网格。
- 不默认再给 fast 单独跑五任务；full 是事先固定的强基线预算。

最小实现工作：

1. 接入当前 ModelOpt 的 effective weights 和全部 attention/expert 目标投影，不能沿用旧 Qwen3.5 脚本的模型加载与 dense-only 遍历假设。
2. 固定主线输入维量化轴，不使用旧脚本为凑 G64 整除而转置搜索的分支。
3. 保存搜索配置、选择的 scale/必要元数据与重建值；接入主线 R64 sidecar。
4. **导出不能再按标准 scale 量化一次而抹掉搜索结果。** 验证保存/重载后的实际权重与算子确实使用搜索结果。

首次有卡，选一个真实 attention Linear、一个已访问专家 gate/up、一个 down projection，检查完整解码→R64→搜索→导出/读取→实际算子链。标准候选在合法候选集合中，因此同一权重度量下搜索结果不应比所包含的标准候选更差；若不满足，应定位编码/重建或导出问题，而非直接解释为模型现象。

随后完成 48 层全部目标模块转换和五任务评测。记录逐模块权重 NMSE、S0 偏移分布、纯搜索与完整转换耗时、峰值显存。**权重 MSE 降低不预先等同于下游改善。**

### 6.2 R64＋本仓库 HiF4-GPTQ

可复用 `HiFloat4/hif4gptq/gptq/gptq_utils.py` 的 `GPTQ.add_batch()`、`fasterquant()` 与 HiF4 量化器。当前 `_get_quant_groups()` 是 dense 模型投影定义，**未支持本主线的 qwen3_moe**；不能直接拿旧 CLI 宣称整模型可跑。

固定定义：

- 同一 effective weights、R64、目标投影与合法 HiF4 格式；标准在线激活与 r64 相同。
- 复用干净 A 的 128 条 train 全序列，按2026-11-18授权对所有专家收集全部有效token的真实输入；保留有效 token mask，不为适配旧固定长度 driver 而截成新的校准集。原路由仅决定层输出的加权聚合，原路由计数与校准计数分别保存。
- 主基线按普通 GPTQ 的局部权重误差目标定义，Hessian 采集位置固定为 **R64 后、激活 QDQ 前**的实际投影输入；不追加教师输出拟合而仍称普通 GPTQ。正式 W4A4 的激活误差由最终下游统一评估。
- 起始阻尼采用核心默认 `percdamp=0.01`；量化分组、列顺序及其余核心配置在适配时照现有实现逐项记录并冻结。若需调整，仅在校准留出上进行事先有界的检查，不看正式任务调参。
- 不把第 6.1 节的 scale 搜索默认嵌入 GPTQ 形成第四条新方法；如当前 GPTQ 量化器自带 scale 优化，必须明确其实际策略，不能隐去差异。
- 论文名称先用“本仓库 HiF4-GPTQ 基线”；若称“HiGPTQ 复现”，需另核对上游原方法定义与本实现差异。

MoE 适配包括实际专家输入的 Hessian 收集、逐层传播、完整模块覆盖与主线导出。Hessian 缓存按模块/层流式处理，不默认同时驻留所有专家矩阵。

特别检查覆盖：现有 GPTQ 会对 Hessian 对角为零的列设置 dead-column 行为。先统计每个专家有效校准 token 数和零对角；**完全没有校准输入的专家不能被静默清零或隐式跳过**。所有专家gate/up/down均须收到完整train有效token数；自然top8零命中单独保留，不再等同于无校准输入。不得借验证/测试题补统计。该问题仅阻塞受影响的 GPTQ 转换，不阻塞 scale 或既有模型评测。

最小 GPU 验证覆盖真实 attention、专家 gate/up/down、一个完整 decoder layer 的逐层输入与保存读取，再安排整模型转换。完成全部五任务，同时记录 Hessian/量化/导出各阶段成本、专家覆盖与峰值；用实际代表层测量估算全层资源，不套用 matrix_norm 的训练时长。

## 7. 最终方法的机制证据

复用既有内部误差工具和真实 vLLM hook；不重新展开旧 token trajectory 全矩阵。

- 第一批比较 direct、r64、ours_a20；两条强基线导出后，加入**同一组**诊断，方便解释 scale 搜索、GPTQ 与重构分别改变了什么。
- 固定干净 S1K 的 32 条验证序列，固定输入与有效 token；它们未参与梯度拟合，但已用于 checkpoint 选择，报告中不称完全独立测试集。
- 代表层事先固定为 0、15、31、47，覆盖 early/middle/late；模块覆盖 attention 输出与 MoE gate/up、down/块输出。若需追加已有敏感层，注明来源，不按新结果挑最漂亮的层。
- 专家诊断只统计实际访问的输入，并报告命中分布；未访问专家不虚构激活误差。

两种观察分开：

| 观察 | 固定条件 | 指标与能回答的问题 |
| --- | --- | --- |
| 局部同输入 | 同一个教师/公共捕获输入，各方法独立执行真实量化算子 | 权重/激活零值比例、合法量化 level 占用、scale 分布、输出 NMSE、残差输出 NMSE；解释局部转换误差 |
| 学生前缀传播 | 同一 token 序列/forced history，各方法按自己的状态推进 | 层输出与累计 residual 误差、误差方向/交叉项；解释误差如何进入后续层 |

要讨论推理实际路径时，使用原生 vLLM TP2、incremental KV-cache 和已有 worker hook；普通离线层重放只作为局部证据。不可用旧 semantic replay 替代真实执行。

如进行误差分解，先把各项写在同一输出空间，并报告交叉项。对于 `e_total=e_fit+e_WQ+e_AQ` 的顺序反事实分解，平方误差包含交叉项，三项 NMSE 不能直接相加；分解顺序本身也应记录。非正交矩阵允许改变未量化映射，不能把拟合误差全部称为量化误差。

交付：一张代表层误差表、一张沿层的累计误差图或代表层对比图、必要的 scale/zero/level 统计，以及与五项任务结果对应的解释。观察到误差与精度共变时写相关证据；没有干预不宣称已证明单一因果机制。

## 8. 只补一个主设计消融：权重侧正交约束

问题：在**固定在线 R64**和其他学习范围相同的前提下，权重侧矩阵保持正交与允许自由矩阵有什么差别？

| 固定项 | 设计 |
| --- | --- |
| 数据与源模型 | 与 ours_a20 相同的干净 S1K 128/32、完整序列、原生初始化 |
| 激活侧 | 全程固定同一在线 R64，不跟随权重矩阵更新 |
| 权重侧 | 同一位置、G64 分组、共享粒度、普通 attention/expert 投影范围 |
| 初始映射 | 两组都从现有 R64 对应的相同映射开始；正交参数化不能为方便改成另一初值 |
| norm/router | 两组学习同样的 input/MoE norm；router 冻结、KL=0 |
| 重构 | 相同层数、轮数、batch、目标、验证选优、学生输入传播与导出规则 |
| 唯一设计变化 | 权重侧矩阵受正交约束，或自由学习 |

优先复用现有 Cayley 参数化思路，使正交组支持相同 R64 初值；只接入权重侧，不调用旧两侧绑定的 rotation 方案。对比结论是这种**约束与参数化**的实际效果，不把不同优化轨迹排除在外而声称纯几何证明。

默认沿用固定训练预算与学习率，不为某一组按测试成绩额外调参。若真实短测暴露参数化无法正常优化，先修复或用校准验证设置两组同等的有限调参预算，记录与既有 A 的匹配关系；不能让一组默认享有更多搜索。

自由组现存配置完全匹配时，新增正交组一条 48 层训练即可。否则将不匹配项明确列出，使用匹配自由对照，或收窄结论；不能拿历史不同数据模型冒充单变量对照。

新正交模型完成五任务，报告留出重构误差、任务差值、实际转换成本与矩阵正交性。旧 rotation 同时改激活与权重两侧，**不能直接充当本消融**。

本批不同时扫 group size、router KL、共享方式和三种 loss。只有论文把损失归一化本身作为主要创新时，再单独安排 output-MSE / output-NMSE / block-delta-NMSE；同一输入下 residual subtraction 不改变误差分子，不能把仅分母变化讲成不同的残差拟合目标。

## 9. 已有短程模型补评测与有限稳定性

### 9.1 精度/成本对照

效率四行固定：

| 候选 | 数据 | 轮数 | 回答的问题 |
| --- | --- | --- | --- |
| b20 | 完整干净 native traces 128/32 | 20 | 同一数据的完整预算参照 |
| full_e5 | 相同完整 native traces | 5 | 缩短独立训练 schedule 的收益与代价 |
| random50_e5 | 同来源，分层随机约 50% train tokens | 5 | 减少数据预算的收益与代价 |
| coverage50_e5 | 同来源，覆盖策略约 50% train tokens | 5 | 相近 token 预算下的选择策略 |

三条 e5 训练均已完成。random50 已有正式 MMLU-Pro、LCB、GSM8K，可在协议匹配后复用；旧报告中的两项 AIME 不参与本次。优先给 **full_e5 与 coverage50_e5 各补 LCB＋MMLU-Pro**，只做当前缺少的 4 个候选×任务执行。

随后：

- 若正文报告四候选完整五任务 Pareto 对比，补齐所有缺失的 GSM8K/ARC-E/ARC-C，再给 L5；不能给缺任务候选算五项均值。
- 若成本副实验仅保留 LCB/MMLU-Pro 双任务结果，明确其范围，四行都按相同两项呈现；不据其中优胜者替换主方法或省略不利结果。
- 主比较的六行仍须完整五任务，不受效率副表的简化影响。

20→5 比较使用 **b20**，不是旧 S1K matrix_norm 的 20.0406 小时。两者都是独立 cosine schedule，不是从 20 轮训练中截取第 5 轮。random/coverage 为约半量 **token**，不是 50 条样本；实际完整序列数、token 数、batch 数和 token visits 都记录。

已有 A20/B20 重构约 19.69/22.55 小时、random50_e5 约 3.82 小时，只是原设备与边界下的历史观测；不要将这些数当作新正交、GPTQ 或 scale 的承诺时长。旧 20.0406 小时仅作历史成本参考。

统一成本表区分：

- 校准数据准备/原生轨迹生成；
- 搜索、Hessian 量化或重构优化；
- 解码/旋转、导出、开发评测、正式评测；
- 单候选成本与全部探索成本；
- wall time、实际 GPU 数/型号、GPU-hours、峰值显存。

同设备和相同统计边界下才给加速比。短程配方同时改变数据量、轮数或 schedule 时，按整体配方比较，不越界归因。

### 9.2 LCB 的有限重复

先用完整首轮逐题记录评估 ours_a20 相对 r64 及两条强基线的差值与区间。若论文的关键领先结论仍依赖很小的单轮差值，执行以下预先固定安排：

- 对**仍用于支撑该关键领先结论的成对方法**，总共完成 3 轮，使用种子 `1234、1235、1236`；每轮都包含同一 175 题、每题一次回答。匹配的 1234 历史轮可复用。
- 重复对象在新增轮开始前写进 manifest。r64 是基本参照；若要宣称优于 scale/GPTQ，则相应强基线也须进入同一重复比较，不能只重复较弱对手。
- 只重复生成与评分，不重新训练。保留每轮分数、逐题差、配对汇总；先对同题跨轮结果汇总再按题重采样，不能把 175×3 当成互不相关的新题。
- **三轮结束即报告，无论结论正负或区间是否跨零。** 不采用“跑到显著为止”；少量轮次也不代表已充分刻画所有随机性。

如果首轮已经支持有限、明确的观察性结论，且论文不声称跨生成稳定领先，可以不追加重复，照实给单轮与区间限制。多训练 seed 验证仅在出现明显优化不稳定或论文提出相应结论时单独安排。

## 10. 有卡后的实际排队方式

| 当时已授权且可用的资源 | 优先安排 |
| --- | --- |
| 暂无 GPU | 输入/产物绑定、无 AIME 汇总、两条强基线与正交组的实现准备 |
| 1 张满足实际容量的 GPU | scale 流式转换、GPTQ/正交的真实层验证与逐层转换/训练；不为凑资源将既定 TP2 正式评测改成 TP1 |
| 2 张满足容量的 GPU | TP2 先补已有模型的缺项；新强基线就绪后优先完成其五任务。单卡训练/转换与 TP2 评测按实际资源错开 |
| 3 张满足容量的 GPU | 一组 TP2 评测＋一张卡逐层转换/训练，前提是共享 CPU/内存/I/O 无明显冲突 |
| 4 张及以上 | 根据待评模型数量选择两组 TP2，或一组 TP2 加独立转换/训练；不默认占满所有卡 |

GPU 型号、空闲显存、进程与调度限制以启动当时为准，不沿用历史 GPU 编号。GPTQ 全层资源先用代表层实测估计；大模型或长序列不能靠缩模型、截短正式校准或改变语义“通过”容量检查。

每个新增路径的正式启动前，做一次针对它的最小真实验证：训练覆盖前向/反向/更新/验证选优/保存与传播，转换覆盖实际量化与导出，TP2 覆盖正式引擎、关键批量形状、通信与短解码。短测只压缩时间，不将截断回答分数当正式精度。已适用证据直接复用。

长任务按仓库现有脱离会话方式运行，保存命令、配置、PID/会话或作业 ID、日志、产物目录与退出状态；确认实际计算推进后交付。无自动占卡、无限轮询、监控守护或自动重试任务。

## 11. 实现入口与明确的待补部分

以下是源码复用位置，不是已经实现的新命令：

| 工作 | 现有入口/模块 | 本轮必须处理的差异 |
| --- | --- | --- |
| scale 搜索 | `HiFloat4/hif4_scale_threshold_optimization/src/weight_search.py` | ModelOpt 解码、MoE 全投影、固定量化轴、搜索结果导出；旧脚本默认不同模型与任务 |
| GPTQ | `HiFloat4/hif4gptq/gptq/gptq_utils.py` | qwen3_moe driver、变长校准、真实专家输入/Hessian、覆盖与导出 |
| 主方法/正交消融 | `experiments/module_design_ablation/{parameters,student,train}.py` 与 `experiments/non_equivalent_reconstruction/` | 复用核心但不改写历史固定配置；新增单侧正交定义和干净数据绑定 |
| 干净 A/B 产物与协议 | `experiments/reasoning_trace_calibration/{config,evaluation,export}.py` | 复用单阶段核心；旧候选枚举、总控与 report 含 AIME |
| e5 单阶段入口 | `experiments/matrix_norm_efficiency/run.py` | 已核实 export/evaluate/report 等 argparse；旧 TASKS 无 ARC 且含 AIME24/25，不能直接当新五任务总队列 |
| 正式评测 | `experiments/downstream_accuracy_recovery/`、数据 pinned 入口、根 `main.py` | 显式固定本计划五任务，继承真实 protocol 和原始逐题记录 |
| 机制分析 | `experiments/internal_error_accumulation/` 与已有真实 vLLM hook | 给最终模型和新强基线接入相同捕获/干预接口，不恢复旧 semantic replay |

上表中未带顶层的 experiments 路径均相对 `Native_NVFP4_HiF4_Linear_Puncture/`。

独立入口已落实在 `experiments/thesis_format_conversion_supplement/`，核心算法复用各模块；实际短测/正式命令见[实现说明](../../experiments/thesis_format_conversion_supplement/README.md)和 RUN manifest。不要直接重启旧 `--stage experiment`，以免重复生成轨迹、A/B 训练、router 或 AIME。

全部 Python 执行使用 `hif4`，由调用者 shell 激活；实验脚本内不嵌 conda 激活。不改环境依赖，不停止其他任务，不提交或推送 Git。

## 12. 最终应交付的章节证据

| 产物 | 必须包含 |
| --- | --- |
| 主结果表 | native / direct / R64 / **R64＋scale** / **R64＋GPTQ** / ours 六行，五任务及适用的差值和统计 |
| 一项受控消融表 | 固定激活 R64 的单侧正交 vs 自由矩阵，数据/预算/选优一致 |
| 机制图表 | 相同输入的代表层误差、实际前缀传播差异，覆盖最终方法及两条强基线 |
| 成本与效率表 | 强基线与主方法转换成本；b20/full_e5/random50_e5/coverage50_e5 的同口径成本与精度 |
| 稳定性附表（若执行） | 预定 3 轮 LCB 原始成绩、配对统计、未显著或负结果也完整报告 |
| 复现记录 | 实际配置、模型/数据/代码摘要、命令、原始回答、指标、资源与适用限制 |

完成标准是问题有可追溯、可比较的证据，不是所有新实验必须证明主方法更好。若 scale 或 GPTQ 已经达到相近精度，应据实调整章节对复杂度与贡献的表述；若正交消融不支持非正交优势，也保留结果并收窄主张。

本计划只整理已有证据并提出待验假设，**无新增实测经验**；新的经验待运行完成后按已有主题更新，不把方案当作方法提升结论。

## 参考的仓库证据

- [评测协议、数据清洗、缓存与真实路径经验](../../../docs/experience/lessons/evaluation_evidence.md)：约束实际输入、clean 校准与可复用结果。
- [fake quant 与实际低比特执行边界](../../../docs/experience/lessons/fake_quant_boundaries.md)：约束存储/性能与实际算子解释。
- [既有模块消融报告](../../results/module_design_ablation/formal_20261008T074812Z/REPORT.md)：已有方案与主表历史证据，旧 S1K 对照适用范围需保留。
- [干净 A/B/C 运行报告](../../results/reasoning_trace_calibration/continue_20261016T135141Z/REPORT.md)及[逐项汇总](../../results/reasoning_trace_calibration/continue_20261016T135141Z/comparison.json)：干净模型、开发集选择、正式结果与成本。
- [校准数据配置](../../experiments/reasoning_trace_calibration/config.py)：数据来源、模型 revision 与训练/推理参数。
- [效率实验报告](../../results/matrix_norm_efficiency/run_20261029T023839Z/REPORT.md)及[配置](../../experiments/matrix_norm_efficiency/config.py)：三候选与已有 random50 结果；旧 AIME 套件不作为本轮五任务。
- [效率计划及 H-only 支线沿革](../../experiments/matrix_norm_efficiency/PLAN.md)：不同研究定义保持独立。
- [scale 搜索模块说明](../../../HiFloat4/hif4_scale_threshold_optimization/README.md)及[搜索核心](../../../HiFloat4/hif4_scale_threshold_optimization/src/weight_search.py)：合法 S0/e8/e4 搜索、fast/full 预算与旧模型范围。
- [HiFloat4 RTN/GPTQ 模块](../../../HiFloat4/README.md)及[GPTQ 核心](../../../HiFloat4/hif4gptq/gptq/gptq_utils.py)：可复用低层算法与 dense driver 的限制。
- [既有内部误差计划](2026-09-09-qwen3-30b-hif4-internal-error-accumulation-mechanism-plan-cn.md)：基础工具与历史机制问题；不因本计划重复全部旧实验。
