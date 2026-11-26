# 手动 GPU 评测配置

这些是历史评测 preset，执行后会加载真实模型并占用 GPU，不是普通单元测试。它们不代表当前 NVFP4→HiF4 / MXFP4 正式协议，也不构成 GPU 资源授权。

| 文件 | 原路径 | 实际用途 |
| --- | --- | --- |
| `qwen3_5_aime25_hif4_activation.sh` | 根目录 `test.sh` | Qwen3.5 AIME25、HiF4 激活、无 KV fake quant，重复配置保留原样 |
| `qwen3_5_hif4_rtn_kv_mxfp8_q.sh` | `scripts/test.sh` | HiF4 RTN 权重、HiF4-1 激活/KV、MXFP8 Q 的 AIME25 配置 |

仅改存放位置与文件名，脚本内容逐字节保留；两个原路径为相对符号链接，已有调用仍有效。执行前从仓库根目录激活 `hif4`，核对脚本中的模型路径、输出与 GPU 资源。当前转换实验使用[专用入口](../../fp4_convert/experiments/README.md)，通用 CLI 见[命令参考](../../docs/guides/gpu_commands.md)。
