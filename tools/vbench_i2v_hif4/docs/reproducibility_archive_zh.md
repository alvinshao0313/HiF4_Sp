# 复现材料说明

本工具包依赖 VBench-I2V pip 包及其外部模型、图像资产。由于该环境与 HiF4 / Wan 生成环境常发生版本冲突，提交包额外保留了 `reproducibility_archive/`，用于排查兼容性问题。

## 为什么不把复现材料放进工具目录

`tools/vbench_i2v_hif4/` 应保持为可维护代码目录，只包含源码、配置模板、README 和 smoke test。环境快照、资产表、历史 diff 和实验结果表属于复现证据，体积和时效性都不同，因此放在提交包根目录的 `reproducibility_archive/`。

## 内容含义

```text
reproducibility_archive/env_snapshots/      已验证评测环境的 pip/conda 快照和导入报告
reproducibility_archive/asset_manifests/    VBench 图像/模型资产清单
reproducibility_archive/git_diffs/          当时相关仓库的状态记录，用于定位本地补丁
reproducibility_archive/result_tables/      已完成实验的结果表，仅作参考
```

其中绝对路径已脱敏为 `${PROJECT_ROOT}`、`${CONDA_PREFIX}`、`${USER_HOME}` 等占位符。它们用于对照，不要求在新机器上逐字还原。

## 使用建议

1. 新环境优先安装 `requirements-vbench.txt`。
2. 如果导入失败，再对照 `requirements-vbench-key-pinned.txt` 和 `env_snapshots/*/requirements.freeze.txt`。
3. 如果 VBench 运行到一半找不到图片或模型，对照 `asset_manifests/` 检查文件数量和相对路径。
4. 如果 VBench API 或依赖版本不同，先运行 `python -m hif4_vbench_i2v.preflight` 生成新的 JSON report。
