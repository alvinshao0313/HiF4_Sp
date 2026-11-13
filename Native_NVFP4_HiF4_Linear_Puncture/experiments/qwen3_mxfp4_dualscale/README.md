# Qwen3 路由专家二级 scale MXFP4

算法和精度边界见 [计划](PLAN.md)。两个候选为 `dualscale_g256`、`dualscale_g512`：FP32 外层 absmax/6，内层沿 K 的 group32 原 floor MXFP4，无旋转。生产 GEMM 使用 BF16 操作数、FP32 累加及大组部分和缩放，输出 BF16。

所有命令由已激活 `hif4` 的仓库根目录 shell 执行。独立入口注册 `qwen3_routed_mxfp4_dualscale`，不改旧候选和活动 UOS 代码。

## 转换与验证

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_dualscale.convert --source /path/to/original_nvfp4 --output /path/to/new_checkpoint --variant dualscale_g256 --threads 8
CUDA_VISIBLE_DEVICES=4,5,6,7 OMP_NUM_THREADS=2 python -m torch.distributed.run --standalone --nproc_per_node=4 -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_dualscale.validate_gpu --model /path/to/new_checkpoint --direct-model /path/to/old_direct_checkpoint --output /path/to/gpu_validation.json
```

G512 使用相应 variant 和独立输出路径。真实 TP4 验证覆盖 1/256/8192 tokens、两个 GEMM、原生 SwiGLU、跨卡 MAX 和输出 SUM，核对 BF16 MMA 并记录新旧路径计时。参考量化在 CPU 执行，保持 FP32 除以 6 的既定舍入；生产 Triton 用正确舍入除法。

## 有限评测流程

```bash
python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_dualscale.workflow prepare --root /path/to/run --baseline /path/to/native_run --prior-table /path/to/five_model_summary.json --g256-validation /path/to/g256_validation.json --g512-validation /path/to/g512_validation.json
python -u -m Native_NVFP4_HiF4_Linear_Puncture.experiments.qwen3_mxfp4_dualscale.workflow run --root /path/to/run
```

`prepare` 复用 `root/checkpoints/{dualscale_g256,dualscale_g512}`，检查转换及数值验证，冻结源码、旧结果、输入和协议；不启动推理。`run` 在 GPU4–7 顺序执行两个候选的 26题×10、64-token 短测，然后 G256、G512 的全部 198题×10 正式评测。任一步失败即退出，不自动重试、缩小正式配置或停止其他进程。

正式设置为 TP4、batch256、8192 batched tokens、32768 输出上限、40960 上下文、每卡32GiB BF16 KV、eager、thinking 开启、seed1234 和原采样设置。启动前核实 GPU4–7 无其他计算任务；GPU0–3 UOS 不受影响。

完成推理后保留严格 JSON 原分数，生成 `answer_review/pending_blind_review_queue.json`；只在显式完成最终答案复核后使用 `report --root ... --decisions ...` 输出七组 JSON／CSV。不会自动猜测人工复核结论。

本次运行、失败诊断和原始性能报告保存在 [dualscale_gpu47_20261110T085025Z](../../results/qwen3_mxfp4/dualscale_gpu47_20261110T085025Z/)。当前状态以 `workflow_status.json`、阶段退出记录和缓存为准；计划、PID、局部数值通过均不代表最终准确率。
