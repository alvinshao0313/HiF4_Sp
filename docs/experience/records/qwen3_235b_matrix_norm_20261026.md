# 2026-10-26：235B matrix_norm 实现与资源阻塞

状态（北京时间 2026-10-26 12:19 实测）：独立实现、CPU 回归和固定数据准备已完成；
GPU 0–7 均被其他用户进程占用，TP8 入口记录 `blocked_resources`，没有启动任何 GPU
阶段。**尚未完成真实 GPU 路径验证，尚无 235B 自生成轨迹、转换模型或前后指标。**

## 目标与实现

用户确认使用原生 235B 自生成轨迹、完整四任务前后对照，允许空闲时使用 GPU 0–7。
方法、固定超参数、入口职责、执行顺序及 AIME 复核方法维护在
[实验 README](../../../Native_NVFP4_HiF4_Linear_Puncture/experiments/qwen3_235b_matrix_norm/README.md)。
本次没有复用旧 E4/Group 权重、30B 回答或历史原生分数，也没有修改旧实验源码或 Git 暂存状态。

- 模型：`nvidia/Qwen3-235B-A22B-NVFP4`，revision
  `21cfa2c9e152032eb60647ee7b46a2bbcd8d76d2`；真实配置为 94 层、hidden=4096、
  Q/KV heads=64/4，128 experts，top-8。
- 校准协议：原生 TP8，数学/代码各 64 条训练、16 条验证，完整 token IDs、无正确性筛选。
  题簇划分 seed=42；生成/评测 seed=1234；生成参数及预算保持已确认方案。
- 转换协议：固定激活 R64，训练 G64 自由矩阵和两处 norm；router 权重冻结，
  block-delta NMSE，94 层×20 epochs、全局 batch=4、LR=`3e-4`。
- 导出格式：BF16 HiF4 QDQ 权重与 W4A4 运行时 sidecar；无打包压缩或推理加速实测。

## 当前证据

运行目录：[run_20261026T025055Z](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/)。

1. `hif4` Python 3.11.13 下的 13 项 CPU 回归通过，覆盖真实 235B 几何、完整题数和
   TP8、AIME 五回答、部分/诊断训练禁止正式导出、四项正向损失、真实 Adam 标量状态
   摘要，以及旧 TP2 结果不能冒充 TP8。
2. 27 个原生分片存在，配置、tokenizer、索引及分片身份已记录；当前运行绑定 179 个
   源码文件。未提交，版本证据为
   [source_hashes.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/source_hashes.json)。
   准备阶段源码另存，GPU 阶段启动前的编排状态修正及摘要变更记录于
   [pre_execution_revision.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/pre_execution_revision.json)。
3. 固定来源共 26,110 条记录（包括来源官方 test/valid），四项正式测试 13,556 条；
   合并 39,666 条后得到 36,485 个题簇，排除 17,817 个测试关联题簇及 21 个 prompt
   超预算题簇。数学候选 train/val=5,207/1,302，代码=9,608/2,402；现有划分函数
   为每个领域另留的 64 条开发池本轮不生成、不评测、不参与选优。
4. 实际固定 Arrow 与 tokenizer 的全量输入审计通过：GSM8K 1,319×1、MMLU-Pro
   12,032×1、LCB v6 175×1、AIME25 30×5。58 个准备文件摘要通过复核。
   证据见 [inputs.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/inputs.json)、
   [排重审计](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/isolation/audit.json)和
   [prompt/token/gold 审计](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/prompt_audit.json)。
5. 正式编排入口于北京时间 12:19 执行资源检查，PID=2530103、exit=1，状态为资源阻塞，
   `stages={}`。没有评测子进程、训练进程或占卡等待程序；没有停止他人任务。
   见 [launch.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/launch.json)、
   [status.json](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/status.json)和
   [runner.log](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/runner.log)。

CPU 数据准备由 PID=2368971 完成，原始日志与启动记录保留在结果父目录；
该非子进程的退出码未回溯，完成依据为完整产物及输入/源码摘要核验，不伪造 exit=0。

## 接续与限制

8 张卡空闲后，在已激活的 `hif4` shell 中，从仓库根目录运行：

```bash
nohup python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_235b_matrix_norm.run \
  --run Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z \
  --gpus 0,1,2,3,4,5,6,7 --stage all \
  >> Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_235b_matrix_norm/run_20261026T025055Z/runner.log 2>&1 < /dev/null &
```

入口重查 GPU 占用与准备文件，不会等待资源或自动重试。首个尚未启动阶段是原生 TP8
生成短测；后续仍须完成实际最长完整 batch 的两层四卡训练短测、94 层导出及真实
TP8 HiF4 短测，不能以当前 CPU 结果替代。后台正常推进后交付运行信息并停止主动监控。

本轮无新增算法经验；固定数据与执行路径沿用
[模型迁移经验](../lessons/model_transfer_execution.md)及
[评测版本、缓存、AIME 评分边界](../lessons/evaluation_evidence.md)。
未删除既有结果、权重或其他任务产物。
