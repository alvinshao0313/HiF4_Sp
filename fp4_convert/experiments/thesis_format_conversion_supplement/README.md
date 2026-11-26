# 格式转换章节补充实验

实施[已确认计划](../../../Native_NVFP4_HiF4_Linear_Puncture/plans/qwen3_30b_a3b/2026-11-16-thesis-format-conversion-supplement-plan-cn.md)，不修改历史训练、导出和评测入口。初始 RUN 为
[run_20261116T103300Z](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/run_20261116T103300Z)。

2026-11-16 执行期间用户追加：**测试结果统一成 5 次均值**。本入口对每个报告任务执行五次完整评测，分别使用独立模型视图和缓存；每次保留任务原协议与 seed。匹配历史结果可以充当一次，指向同一原始生成的重复引用不能计为两次。五次均值不是 pass@5、投票或五个训练 seed。未满五次不输出正式均值/L5。该要求替代原计划条件式 LCB 三轮；不运行 AIME。

2026-11-18追加授权：GPU4、5用于**S1K A20和自生成数据B20两版matrix_norm**的五任务五次均值补测；两者均已有完整首轮，分别补第2至5轮。为保护GPU0 GPTQ及GPU6、7 scale的冻结依赖，新卡组入口放在`../thesis_matrix_norm_repeats/`，复用原评测核心、协议和汇总。[运行配置、顺序及证据](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/matrix_norm_repeats_20261118T034732Z/RUN.md)。A20与新GPTQ同S1K数据，B20保留自生成数据边界，不混为同数据消融。

**2026-11-18 最新GPTQ方案**：用户改回与干净A20完全相同的S1K train（128条完整序列，1,272,746 token），采用LLM Compressor式全token、全专家校准，层输出仍只使用原top8路由。独立模块位于`../thesis_format_conversion_all_experts/`，复用原GPTQ核心、ModelOpt解码、R64、学生传播及直接导出，不修改活动scale评测的冻结依赖。[当前运行记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z/RUN.md)。用户已明确授权停止上轮C4扫描；其部分证据及旧850条S1K结果保留，但不用于新转换续层。下面扩充/C4段落描述历史过程。

2026-11-19核实该GPTQ已完成48层转换及导出，全部专家有校准输入，退出0；用户要求直接开展五任务五次评测。使用GPU0、1，先完成真实导出模型TP2短测，再由RUN内`evaluation_pipeline.py`复用本模块入口顺序评测并汇总；实际PID、日志和完成状态见上述运行记录。

2026-11-17 用户允许扩充 GPTQ 校准。新增[独立续行目录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z/RUN.md)准备850条完整干净S1K（原128条全部保留），五次评测和其他训练数据不变。`gptq_calibration.py`复用原1000条S1K的完整隔离图，排除test/known-test/val/dev整簇；`gptq_coverage.py`重现旧失败层的真实路由并核对输入采集。正式导出可显式传入`--gptq-calibration RUN/gptq_calibration`，不传时保留原128条语义。GPTQ新增逐层全专家路由记录，并检查所有专家投影采集计数一致；不以多取样本本身宣称覆盖通过。

2026-11-18 实测：850条S1K的GPTQ完成21层后，索引21层的expert29仍零命中而退出；正交A20的48层训练与导出均完成。用户随后选定**纯C4**，明确缩减为**512段×512 token**，要求先检查全专家覆盖，并分配GPU0、1、6、7。[当前覆盖记录](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/c4_512x512_20261118T025938Z/RUN.md)使用固定版本C4 English train（seed42；262,144 token），每个独立文档抽一个512-token窗口，不混S1K。先前1024条完整文档的本轮临时检查已停止，保留其数据与部分记录。数据准备及原生NVFP4逐层覆盖扫描放在独立的`../thesis_format_conversion_c4/`，避免改动[活动scale评测](../../../Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/c4_20261118T024647Z/RUN.md)的依赖。覆盖扫描按窗口分两份，各自传播48层并核对真实gate/up/down输入计数，结束后合并6144个专家的记录。原生覆盖不能保证转换后的覆盖；新的GPTQ正式转换尚未启动。

## 固定实现

- `config.py`：六行主表、A20/B20 身份、五任务与五次重复；ARC 保持原 lm_eval、0-shot、acc、seed42、context4096，reasoning 保持原 lighteval、seed1234、TP2、context40960。不将 reasoning 生成参数覆盖 ARC。
- `prepare.py`：验证完整 calibration、固定 Arrow、每题 prompt/token/gold、权重分片 SHA-256 与来源；只写新 RUN。开发/隔离与效率数据另见 RUN 的 `isolation_and_development.json`、`efficiency_data.json`。
- `quantization.py`、`export.py`：ModelOpt 主线逐层解码、固定 R64、输入维 G64。scale 采用 full 的五个 S0 邻居与八种 e8/e4 枚举，data-free。直接保存选定 BF16 QDQ 值并验证重载，不调用会再次标准量化的 `_state_to_tensors()`。
- GPTQ 复用 `GPTQ.add_batch/fasterquant`、`WeightHiFxQuantizer(hifx4)`，自然列顺序、block128、group64、damping0.01，量化器使用标准缓存层级 scale，无额外搜索。旧适配按自然路由；最新独立适配在 R64 后、AQ 前采集每个专家的全部有效token，down使用该专家真实的SiLU(gate)×up输入。32专家分组限制Hessian驻留；各组遍历完整语料，当前组dense计算、其他专家维持稀疏计算，最终输出只使用原路由及原权重。自然路由零命中单独记录；任一投影校准输入数不等于完整train token数则失败。转换完成后恢复稀疏传播，导出不包含强制全专家推理。
- `training.py`：Cayley 只接权重侧，初值严格等于自由组的 R64。在线激活固定 R64，学习同样两个 norm，router 冻结。使用干净 A 的 128/32 完整序列、48 层×20 epochs、batch4、AdamW 3e-4、无 weight decay、独立 cosine、验证 NMSE 选优与学生输入传播。
- `evaluation.py`：新模型视图/缓存，调用真实根 `main.py` 或既有 lm_eval。`report.py` 读取原始记录，逐题核对与配对 bootstrap；无逐题 ARC 时不伪造五任务配对区间。效率副表固定 LCB/MMLU-Pro 两任务，各五次，不据其排名重选主方法。
- `workflow.py`：有限命令列表，每步检查指定 GPU，遇到占用或失败停止；无等待他人释放、重试、占卡或守护。正式启动前必须已有对应短测证据。

## 实际命令

在仓库根目录、调用者已激活 `hif4` 后运行。下面的正式操作不重复执行已有同名 `jobs/`；先读其 `status.json`、日志与进程。

最新S1K全专家入口（短测通过后才可执行convert；正式从原生第0层使用全部128条）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -u -m fp4_convert.experiments.thesis_format_conversion_all_experts.workflow qualify --run Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python -u -m fp4_convert.experiments.thesis_format_conversion_all_experts.workflow convert --run Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/s1k_all_experts_20261118T032650Z
```

以下为旧续行入口参考；850条GPTQ命令已被上面的方案替代，不重新启动：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 python -u -m fp4_convert.experiments.thesis_format_conversion_supplement.workflow convert --candidate r64_gptq --run Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 python -u -m fp4_convert.experiments.thesis_format_conversion_supplement.workflow orthogonal --run Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 python -u -m fp4_convert.experiments.thesis_format_conversion_supplement.workflow efficiency --run Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=8 python -u -m fp4_convert.experiments.thesis_format_conversion_supplement.workflow evaluate --candidate r64_scale --run Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z
python -m fp4_convert.experiments.thesis_format_conversion_supplement.report --run Native_NVFP4_HiF4_Linear_Puncture/results/thesis_format_conversion_supplement/gptq_expanded_20261117T062000Z
```

这些是不同资源时段的单独命令，不能同时占用同一 GPU。正式执行由已激活环境的调用者以 `Popen(start_new_session=True, stdin=DEVNULL)` 或 nohup 脱离会话，stdout/stderr 落盘；当前实际后台标识见对应 RUN 的 `launch.json` 与 `jobs/*/status.json`，不能仅凭本页命令认定已运行。

准备和短测使用 `prepare --run NEW_RUN`、`qualify --run RUN`、`export --candidate r64_gptq --qualification --run RUN`、`training train --qualification --run RUN`。本次追加授权后，评测允许TP2对`0,1`或`6,7`；仍须核实实时占用，旧运行继续保留原源码摘要。C4运行入口为`python -m fp4_convert.experiments.thesis_format_conversion_c4.calibration --run RUN`和`python -m fp4_convert.experiments.thesis_format_conversion_c4.coverage --run RUN`；后者仅使用GPU0、1，两份扫描完成后汇总并退出，不启动GPTQ、不重试、不占卡。

## 当前证据与后续依赖

scale 的真实第0层全部388投影、最长完整4条输入、attention/专家 gate/up/down 的实际运行时算子与导出读取已验证。正交组两个完整 decoder 层、最长 train/val batch、两轮更新/验证、选优保存读取与传播已验证。GPTQ 的整层校准/实际专家覆盖/导出证据以 RUN 的 `qualification/gptq.json` 为准；CPU 核心检查不替代它。

历史 A20/B20 和三条 e5 的训练与模型分片可复用；direct/R64 的历史权重分片已清理，需在独立目录按原固定变换重新导出再补评测，不能重训。机制部分仍需接入同一组模型和代表层 0/15/31/47 的32条验证序列；离线局部证据和真实 vLLM TP2 incremental KV 路径分别报告，不能拿本次层短测冒充机制结论。此阶段无新增方法精度结论。

来源核验与工程经验见[实施记录](../../../docs/experience/records/thesis_format_conversion_20261116.md)。
