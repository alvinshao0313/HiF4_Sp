# 两类校准数据对比与追加 Router 对齐

状态：独立实现与验证；用户已于 2026-10-16 批准清洗补足 S1K 并重训 A。运行状态以独立 RUN 的 manifest/日志为准。

## 三个结果、两阶段

| 候选 | 数据 | 初始化 | 训练目标与范围 |
| --- | --- | --- | --- |
| matrix_norm_s1k | 清洗补足的 S1K 128 train/32 val | 原生 NVFP4 | block-delta NMSE；仅普通 Linear 的 per-G64 块矩阵和 input/MoE norm；router 冻结 |
| matrix_norm_native_traces | 独立 math/code 原生自生成轨迹，128/32 | 同一原生 NVFP4 | 与上一组相同，只改变数据 |
| winner_router_aligned | 胜出组的原数据 | 胜出 matrix_norm 完整 checkpoint | 仅 top-mass KL，直接训练实际 router gate 权重；其他参数冻结 |

第一阶段 48 层、20 epochs/层、batch=4、seed=42、lr=3e-4、AdamW/weight_decay=0、cosine；按各自验证 NMSE 选 epoch。清洗已改变 S1K 数据契约，A 必须从原生 NVFP4 重训，不复用旧 checkpoint。

第二阶段 48 层、5 epochs/层；第一层 lr=1e-5/3e-5/1e-4 各短测 3 epochs，验证 KL 选优后重新初始化正式训练。实际 router 权重用 FP32 主参数、BF16 前向/导出；128 专家完整 softmax，teacher top-8 加外部概率质量，T=1，在有效 token 上平均。不加入 NMSE，不训练 router 块变换。每层允许采用初始 epoch 0。缓存同一层 detached router 输入和原生 teacher logits；每层选优后通过完整 student 传播下一层。

## 数据与隔离

原生模型固定 Qwen3-30B-A3B-NVFP4 snapshot 2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3。自生成题目仅来自以下 train split：

- DigitalLearningGmbH/MATH-lighteval，revision 0530c78699ea5e8eb5530600900e1f328b48acad。
- deepmind/code_contests，revision 802411c3010cb00d1b05bad57ca77365a3c699d6。

生成前对当前六项正式测试、来源数据官方测试划分做来源 ID/URL、规范化题干、字符 5-gram Jaccard>=0.8 排重，数字掩码后重复检查；近重复按连通簇整体剔除。同簇不得跨 train/val/dev。首次只读审计发现原 S1K 15 个测试重合簇，随后用户批准清洗补足并重训 A。新 RUN 将完整 S1K 1000 题加入同一次分簇，保留原划分内 145 个干净样本及其位置，按 seed=42 的原未选顺序补入 15 题（train 10、val 5）；剔除已知官方 test 来源簇，禁止跨划分同簇。S1K 固定 revision=f375c33cb9f540d89b5fa39bb18f6903e2b97e33，保存原始/新索引、替换证据、token 验证与摘要；旧数据、旧模型不覆盖。

自生成 train 为 math/code 各64，val 各16；另留共同开发集 math/code 各64，与两类校准和正式测试隔离。题目在生成前固定划分和顺序。仅题目及公开样例进入 prompt；参考答案、隐藏测试只存在 judging 侧，不进入生成或校准。每题一次生成，正常结束且有最终输出才接受；不按答对筛选，记录判断结果。保留原始 prompt、generated token IDs、完整 thinking/final；不截断完整轨迹，不把校准轨迹当作正式测试。两组条数固定，token 数/耗时可能不同，必须报告。

原生 packed NVFP4、真实 vLLM TP2、thinking 开启；temperature=.6、top_p=.95、top_k=20、min_p=0；context=40960、max_new_tokens=38912。训练样本不足、输入超限、来源版本不符直接失败，不自动降协议。

## 选择、评测与交付

A/B 在共同开发集各生成一次，以 math accuracy 与 code pass@1 等权平均选 router 起点，平手选 S1K。之后锁定 checkpoint 和 router 配置，再评测 A/B/C，不用正式测试改起点。

完整评测 MMLU-Pro 12032、LCB 175、GSM8K 1319、AIME25 30x5；ARC-E/C 作辅助。固定实际数据内容和 prompt/token/gold；MMLU-Pro 使用已核验的 3373e0b32277875b8db2aa555a333b78a08477ea。AIME 只对最终提交答案评分，保留逐回答记录，含糊情况先复核再汇总。

四项 reasoning 各自 max(0,native-candidate) 的 pp 均值为主指标。旧原生基线固定 MMLU-Pro=75.3740%、LCB=54.2857%、GSM8K=95.4511%、纠正后 AIME=106/150。报告 B-A、router 后-前以及逐题配对区间；AIME 以题为抽样单位，不把150个回答当作独立题。旧基线是跨运行参照，不复用原生同轮配对区间。单轮收益不冒充跨seed稳定性。

验证覆盖数据隔离/覆盖/token、最长真实 batch 前后向与保存选优/换层、冻结状态、router 零步与缓存梯度等价、导出仅改变 router、真实 TP2 推理及既有 AIME 误判回归。独立目录和日志；不修改活动实验依赖、不覆盖旧产物、不自动删除权重、不自动暂存提交。正式后台正常推进后交付并停止主动监控。

2026-10-16 续跑：160 条原生轨迹已完整生成，后续评分因代理连接失败中止。用户要求继续后，在独立 RUN 复用这些固定轨迹，预先缓存评分测试材料并保留失败记录。优化生成请求补入调度，独立测试 CUDA Graph；既有轨迹不重新生成，正式采样和优化目标不变。资源允许时 A/B 分别单卡并行训练，组内逐层依赖、C 的选优依赖不变。

参考经验：[数据版本、实际评分输入与误判](../../../docs/experience/lessons/evaluation_evidence.md)。当前未知：自生成校准及 router-only 对齐的真实下游收益。
