# Qwen3 路由专家：UOS、R16扩展R32与G512组合

2026-11-13。用户要求测试三种方法的两两组合及三者组合，沿用上一轮GPQA超参数和prompt。
用户已明确确认按本计划的定义和资源执行；当前正在实施。运行状态以实际产物为准。

## 四个候选

| 实验标识 | R16扩展R32 | FP32外层G512 | group32内层scale |
| --- | --- | --- | --- |
| `r16x2_uos_c7p25` | 有 | 无 | UOS c7.25 |
| `dualscale_g512_uos_c7p25` | 无 | 有 | UOS c7.25 |
| `r16x2_dualscale_g512` | 有 | 有 | 原floor规则 |
| `r16x2_dualscale_g512_uos_c7p25` | 有 | 有 | UOS c7.25 |

四个候选均量化专家权重和激活。UOS是替换内层指数选择，不额外串联一次MXFP4量化；
G512外层分母始终为6，不改成7.25。无PNQ、无Attention量化变更、无额外旋转矩阵搜索。

## 运算定义与边界

权重从原始NVFP4 snapshot `2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3`
独立转换。沿用已验证的Qwen解码顺序：E4M3组scale乘weight_scale_2，再乘E2M1，
明确舍入BF16。不得从已量化的R32/UOS/G512权重再量化。

对专家输入激活或上述BF16权重，按以下顺序处理：

1. 有旋转时沿K维连续32元素块做FP32右乘R；否则仅转为FP32。
   R严格复用现有`qwen3_mxfp4_r16x2.transform`的
   `(H2/sqrt(2)) ⊗ R4 ⊗ R4`及分阶段运算次序，记录矩阵和源码摘要。
   权重布局为[out,in]，两侧均右乘同一正交R；不跨SwiGLU移动旋转。
2. 有G512时对旋转后的每行、沿全局K每512元素求
   `t = FP32(absmax/6)`，零组取1；FP32计算`z=xR/t`。
   无G512时`z=xR`。旋转与归一化之间均不先舍入BF16。
3. 对z按K32分组：UOS取`e=clamp(ceil(log2(amax/7.25)),-127,127)`；
   无UOS取`e=clamp(floor(log2(amax))-2,-127,127)`。
   零组均取e=0、s=1。沿用frexp/FP32位比较实现精确边界，E8M0保存scale。
4. `q=E2M1_RNE(z/s)`，最近偶数舍入并饱和到±6。GEMM操作数为BF16(s*q)。
   有G512时在FP32累加器中对每个全局G512部分和乘`t_activation*t_weight`后求和，
   最终按原边界输出BF16；不把外层scale提前乘入BF16操作数。

生产GEMM使用BF16操作数、FP32累加，禁止FP32/TF32 GEMM。
只转换48层×128路由专家×gate/up/down＝18,432个矩阵。
gate/up共用一次输入变换，down在原生SwiGLU的BF16输出后独立变换。
Router接收原始hidden states；Attention、Router、Embedding、Norm、LM Head的张量、
方法及原top8、路由加权、moe_sum、输出SUM通信保持不变。

TP2下gate/up输入K2048，down局部K384，全局K768。
R32与内层group32均不跨rank。G512仍按全局[0,512)、[512,768)分组：
先在各rank对旋转后局部值求对应全局组的amax，再在真实TP组上MAX all-reduce，
最后归一化和内层QDQ。不能把rank1的K起点重新视为0；尾组仅处理有效256元素。
外层权重scale离线按全局K计算，不增加在线权重通信。

## 实现和必要验证

新增隔离组合模块，不修改旧候选、旧checkpoint和评分产物；复用已验证的packed BF16
GEMM、原生SwiGLU与路由实现。转换与在线量化显式记录rotation、outer_group_size、
inner_scale_rule及格式版本。每个候选独立导出，所有非专家张量及tokenizer等逐字节核对；
保存重载与完整覆盖通过后才发布index。

1. CPU验证四种组合的零组、指数阈值邻域、E2M1中点、打包、尾组，以及首末层真实权重。
   核验R方向和未量化乘积恒等式；旋转结果先与独立矩阵参考比较，再将量化误差与旋转
   FP32运算误差分开。保留CPU参考的正确舍入FP32除法边界。
2. 真实TP2验证tokens=1/256/8192、K2048/K384、top8和全局G512跨rank边界。
   检查scale、编码、重建、两次GEMM、原生SwiGLU、MoE及MAX/SUM通信；
   对同一量化后的BF16操作数参考，沿用NMSE<1e-4与贡献幅度归一化误差检查。
   确认实际BF16 MMA，不把实现错误归入量化损失。
3. 对受影响的融合QDQ、MAX通信与完整MoE做CUDA events计时并记录峰值显存。
   复用旧证据，不重新搜索KV预算，也不以必须加速作为准确率评测条件。
4. 每候选一次正式TP2/batch256配置的26题×10、64-token执行检查，核对两个rank、
   全部48层方法和权重/scale切片、非专家路径、真实prompt/token与缓存。短测分数不作精度。
5. 以上通过后启动四组正式GPQA×10。有限顺序流程，失败即退出，不自动重试或更改协议；
   正式生成确认推进后交付运行标识、日志、命令与结果路径，停止主动监控。

## 正式协议与已授权资源

- 只测GPQA Diamond，198题×10，主指标1980个回答的平均准确率。
- 复用AISBench commit `aff1554354fbf19dccbdf2199cc3c22119f815a2`的固定prompt、
  上轮实际输入token、选项顺序、chat template、数据内容和最终答案复核规则。
  执行框架继续为lighteval＋vLLM。
- thinking开启；seed1234；temperature0.6、top_p0.95、top_k20、min_p0。
- max_new_tokens32768、上下文40960、batch256、max_num_batched_tokens8192、eager。
- 四个新候选统一TP2、每卡56GiB BF16 KV，沿用最新UOS/G512的预算。
- 本轮已授权仅使用GPU0–3，两路TP2：GPU0–1顺序运行R32+UOS、R32+G512；
  GPU2–3顺序运行UOS+G512、三者组合。四个候选独立，不依据前序分数改后序定义。
  GPU4–7不安排本任务。启动前重新核对资源；当前只读检查0–3空闲。

已有原生、直接及三种单方法结果直接复用，注明单R32为TP4/32GiB，
UOS/G512为TP2/56GiB；不把这点执行差异隐藏为完全相同配置。
来源为[上一轮七组最终复核报告](../../../Native_NVFP4_HiF4_Linear_Puncture/results/qwen3_mxfp4/aisbench_gpqa_20261113T023500Z/answer_review_20261113T144526Z/REPORT.md)。
不重测这些已完成基线，不恢复取消的G256。

## 最终结果

新增四行：准确率、10个采样序号准确率、样本方差（除9）、总体方差（除10）、标准差、
相对单方法/直接/原生的百分点差、截断率与解析失败率。保留逐条原始回答、独立预测缓存、
原自动分数、盲审决定、输入/代码/矩阵/checkpoint摘要、实际运行配置及性能记录。
组合收益未知；不由局部量化误差或旧单方法排名推定最终准确率。

参考[低比特算子边界](../../../docs/experience/lessons/fake_quant_boundaries.md)中的
NVFP4解码、两次SwiGLU舍入、UOS精确阈值及FP32除法经验，
以及[评测证据](../../../docs/experience/lessons/evaluation_evidence.md)中的
固定输入、短测范围、KV预算、worker释放与独立评分要求。
本轮新增待验项是旋转后G512全局scale与UOS内层的组合路径；旧单方法通过不能代替验证。
