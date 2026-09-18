# 非等价变换重建：独立实验

目标模型：`nvidia/Qwen3-30B-A3B-NVFP4`。默认使用 Phase-A E4 的 adopted DIAG 初始化。

代码、配置、训练入口、artifact 和结果由本目录独立管理；通过 import 使用现有 checkpoint、teacher、量化、校准数据及评测底层工具。

## 变换

先融合四组 DIAG：norm 乘输入缩放，普通权重变为 `D_out W D_in^-1`，router 权重做对应逆补偿。冻结这些基础权重。

训练两个 RMSNorm 参数，以及普通投影和 router 的输入侧 64×64 矩阵：

- `--matrix_sharing linear`：每个 Linear 的全部 G64 输入组共享矩阵。
- `--matrix_sharing group`：每个 G64 输入组独立矩阵。
- 不同投影和专家不共享。普通投影初始化为 R64，router 初始化为 I64。
- 学习矩阵不受正交、逆关系或旧 DIAG 截断约束。在线 R64 固定。

导出时矩阵全部融合进权重。普通权重保存为 HiF4 QDQ 后的 BF16，使用现有 vLLM R64→HiF4 激活算子及 MoE Triton 路径；router 保持 BF16。这里沿用项目的 HiF4 数值模拟 checkpoint 格式。

## Router 目标

`--router_loss top_partial|top_mass` 均先对全部专家归一化，再取 teacher top-k。设 teacher/student 概率为 q/p：

- partial：`T² Σ_top q_i log(q_i/p_i)`，可以为负。
- mass：再加 `T² q_out log(q_out/p_out)`，是 K+1 类完整 KL。

teacher 使用同一层输入的原生 NVFP4 层实际路由。辅助分支在 MoE norm **之前** detach，只更新 MoE norm 和 router 矩阵。主重建损失正常训练当前层全部参数。

默认 k=8、T=1、λ=1，可用 `--router_top_k`、`--router_temperature`、`--router_loss_weight` 修改。有效 token 求平均，另外记录 top-k 集合一致率、重合率、外部概率及各专家 token 数。指标中的 top-k 使用模型实际路由 k=8。

训练与 teacher 均使用因果 attention；右侧 padding 不影响有效 token。

## 训练和恢复

在仓库根目录执行：

```bash
MODULE=Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n hif4 python -m "$MODULE.train" \
  --output_dir Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/linear_top_mass \
  --matrix_sharing linear --router_loss top_mass
```

默认 48 层、20 epochs/层、batch size=4；AdamW、lr=1e-4、weight decay=0、cosine；复用 `s1k_original` 128 train / 32 val / seed=42。S1K 使用完整文本，`calib_seqlen` 不截断 S1K。

每层使用该实验自己的 progressive student 输入。总目标为 `block_delta_nmse + λ router_loss`。只在训练后的 epoch 中选验证目标最优者，初始 E4 仅作对照，不执行 identity 或 router 回滚。

给相同训练命令加 `--resume` 可从最后完成的 epoch 继续。当前未提交 epoch 会重新执行。配置必须一致。

每个 run 保存：

- `manifest.json`、`initialization.pt`：来源、配置、初始 DIAG 和已完成层。
- `resume.pt`：当前层输入、参数、优化器、调度器、下一 epoch 和最佳状态；原子替换。
- `layers/NNN/selected.pt`：该层选中的学习参数。
- `layers/NNN/initial_metrics.json`、`epochs/*.json`、`metrics.json`：初始对照及训练验证记录。
- `summary.json`：全部 48 层完成后的汇总。

矩阵按层保存，训练只加载当前层。`resume.pt` 是进度提交点；恢复时从它重建 manifest 的完成层列表。初始化对照与训练后结果使用同一层输入，但不同实验的 progressive 输入可能不同。

## 导出与评测

```bash
conda run --no-capture-output -n hif4 python -m "$MODULE.materialize" \
  --run_dir /absolute/path/to/run --output_dir /absolute/path/to/model --device cpu

CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n hif4 python -m "$MODULE.evaluate" \
  --model_dir /absolute/path/to/model --output_dir /absolute/path/to/run --task arc

CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n hif4 python -m "$MODULE.evaluate" \
  --model_dir /absolute/path/to/model --output_dir /absolute/path/to/run --task mmlu_pro
```

导出要求完整 48 层且输出目录为空。`--initialization_only` 导出同一起点的 E4 基线。评测直接消费导出的模型：ARC 走 lm_eval；MMLU-Pro 300 走仓库根 `main.py` + vLLM/lighteval，启用 thinking，生成上限 32768。

ARC 入口显式持有 vLLM 实例，在评测结束后关闭引擎并等待输出线程退出，再保存指标；关闭失败会直接报错。检查任务完成时还需确认进程返回码为 0，不能只依据 GPU 空闲或指标文件存在。

## 四组完整实验

```bash
RUN_ROOT=/absolute/path/to/new_run_root GPU_A=0 GPU_B=1 \
  bash Native_NVFP4_HiF4_Linear_Puncture/experiments/non_equivalent_reconstruction/scripts/run_matrix.sh
```

使用项目 GPU 池 0–3 中明确指定的两张空闲卡；脚本按两波执行 `linear/group × top_partial/top_mass`。全部训练成功后，顺序导出并评测四组及 E4，生成 `comparison.json`。不要用不同定义下的总目标比较四组效果。

恢复训练阶段时设 `RESUME=1`，并使用原 `RUN_ROOT`。已经开始的 run 恢复，尚未开始的 run 正常创建。导出和评测可通过上述独立命令继续。

## 验证

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 conda run --no-capture-output -n hif4 python -m pytest \
  Native_NVFP4_HiF4_Linear_Puncture/experiments/non_equivalent_reconstruction/tests -q

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n hif4 python -m "$MODULE.verify" \
  --output Native_NVFP4_HiF4_Linear_Puncture/results/non_equivalent_reconstruction/verification/layer0.json
```

单测检查公式、梯度、初始化、因果性、保存恢复、导出参考前向和实际 vLLM 激活算子。`verify` 使用真实 checkpoint 检查 E4 初始化、完整反向和一次参数更新后的实际 vLLM MoE 算子输出；它是算子检查，不代表整模型下游精度已验证。
