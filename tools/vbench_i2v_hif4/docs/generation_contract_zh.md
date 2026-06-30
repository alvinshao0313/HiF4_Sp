# VBench-I2V 生成输入契约

本工具包不负责视频生成，只负责把已经生成好的 mp4 整理成 VBench-I2V case input。生成阶段必须满足下面契约。

## 1. 每个 prompt 真实采样 5 次

对 full_info 中每个 image-prompt pair，需要真实调用生成模型 5 次，保存为：

```text
<prompt>-0.mp4
<prompt>-1.mp4
<prompt>-2.mp4
<prompt>-3.mp4
<prompt>-4.mp4
```

这 5 个文件应来自 5 次采样。可以使用固定基准 seed 加 index 偏移，也可以每个 repeat 记录独立随机 seed，但不能把一个 mp4 复制成 5 个文件。

## 2. 推荐记录 seed manifest

建议生成阶段同时写出一个 TSV/CSV：

```text
filename	prompt	image_name	dimension	repeat_index	seed	variant	checkpoint
```

这样 reviewer 可以确认：

- repeat index 与文件名一致；
- seed 不是事后挑选；
- 不同 variant 的生成参数可追溯；
- 评测失败时能定位到具体 prompt/repeat。

## 3. 构建器的行为

`build_eval_inputs.py` 只做 exact filename match：

```text
template/.../base-3.mp4  <-  generated_dir/base-3.mp4
```

如果 `generated_dir` 只有 `base-0.mp4`，命令会失败。这个失败是正确的，因为缺失的 repeat 应该回到生成阶段补齐。

## 4. 验收器的行为

`validate_case_input.py` 默认检查：

1. mp4 数量是否符合预期；
2. 每个 base 是否拥有 `0..4` 完整 repeat；
3. 同一 base 的 5 个 repeat 是否 SHA256 完全相同。

第 3 项用于发现“复制一个视频凑 5 份”的非标准输入。极少数特殊实验确实需要放行时，才显式添加：

```bash
--allow-identical-repeat-files
```
