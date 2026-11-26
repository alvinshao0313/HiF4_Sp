# HiF4_Sp：NVFP4 格式转换与 GPU 验证

本仓库用于把已有 **NVFP4 模型分别转换为 HiF4 和 MXFP4**，实现转换方法、GPU 推理适配、数值验证和下游评测，并保留论文结论所需的实验来源。

两条转换路线分别报告转换范围、精度和成本。GPU 上的 fake quant / QDQ 与原生低比特 GEMM 是不同执行路径，具体能力以实现和实测记录为准。

## 从哪里开始

| 需要做的事 | 入口 |
| --- | --- |
| 查找 HiF4 / MXFP4 方法与实验 | [转换实验导航](fp4_convert/experiments/README.md) |
| 了解目录职责、本地与远程收录边界 | [目录与文件归属](docs/guides/repository_layout.md) |
| 安装 GPU 环境 | [安装指南](docs/guides/install.md)，统一使用 `hif4` |
| 使用评测 CLI 或已有 RTN / GPTQ | [GPU 命令参考](docs/guides/gpu_commands.md) |
| 查阅实验协议、证据和历史 | [文档入口](docs/README.md) · [经验库](docs/experience/README.md) |
| 修改代码或执行实验 | [工作约定](AGENTS.md) |

## 当前代码布局

```text
HiF4_Sp/
├── fp4_convert/                      # NVFP4→HiF4 / MXFP4 的维护主包
│   ├── src/                          # 格式、源 checkpoint、旋转及共享工具
│   ├── experiments/                  # 转换、重构、GPU runtime 与实验入口
│   ├── tests/                        # 数值语义与功能回归
│   └── configs/                      # 可复现配置
├── Native_NVFP4_HiF4_Linear_Puncture/  # 历史路径保留区
│   ├── plans/                        # 论文引用的协议；继续收录 Git
│   ├── results/                      # 本地原始结果、checkpoint、缓存
│   └── ...                           # 本地冻结源码，供旧实验恢复；不再分发
├── NVFP4/                            # 源格式解码、QDQ、GPU emulation 验证
├── HiFloat4/                         # HiF4 CUDA、量化器及 GPTQ 共享实现
├── 3rdparty/{vllm,lighteval}/          # 含本项目适配的第三方源码
├── tasks/                            # 下游任务与评分定义
├── tools/                            # 文档工具、模型适配及独立评测工具
├── docs/                             # 指南、经验、实验与维护记录
├── archive/{legacy,reference}/       # 历史实验与非 GPU 主线参考
├── scripts/eval/                     # 按用途命名的旧手动评测配置
├── main.py                           # 通用 vLLM + lighteval 评测入口
├── install.sh                        # 安装入口
├── results/                          # 本地通用评测及工具产物
└── LaTeX/                            # 独立论文工作区，本次整理不改动
```

后续开发、导入及 `python -m` 使用 `fp4_convert`。旧源码留在本机原路径，供已挂起的两组实验按原版本恢复；历史结果与论文引用的计划保持原位。迁移不改变算法、CLI 参数、校准数据或默认产物路径，详见[目录说明](docs/guides/repository_layout.md)。

## 运行与复现

在已安装环境中：

```bash
source /home/shaoyuantian/anaconda3/etc/profile.d/conda.sh
conda activate hif4
```

按实验 README 选择转换、验证或评测入口。正式实验记录模型和数据版本、转换范围、采样与评分协议、代码哈希、配置、日志及原始指标。结果可复用与否依赖这些条件，不由目录名决定。

`3rdparty/vllm` 和 `3rdparty/lighteval` 是当前运行依赖，包含项目适配，不能直接换成未核验的上游安装。原 `NVFP4/llm-compressor` 的本地示例已归入 `archive/reference/llm_compressor_examples/`，无效子模块登记已退出跟踪；它不属于运行依赖。

## Git 收录范围

- 应收录：源码、CUDA/Triton 算子、测试、配置、实验方案、文档、第三方适配及许可证。
- 本地保留：模型权重、原始数据、checkpoint、评测缓存、日志、构建产物和客户端状态；它们不因被 Git 忽略就变得可删除。
- `LaTeX/` 按现有约定留在本地，内容和路径不变。
- `.gitignore` 按产物目录分类；不能按 `data/`、`models/`、JSON 或 YAML 名称一概屏蔽源码和配置。

本地结果链接需要对应产物，单独 clone 代码不能取得全部原始证据。本次整理已按授权调整暂存与跟踪清单；尚未提交或推送，因此远程分支尚未更新。
