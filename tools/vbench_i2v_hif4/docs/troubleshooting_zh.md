# 故障排查：HiF4 VBench-I2V 兼容层

## 1. `/tmp` 空间不足

错误：

```text
mkdir: cannot create directory '/tmp/...': No space left on device
```

原因：节点本地 `/tmp` 可能被其他作业占满。

修复：

```bash
export SCRATCH_PARENT=$PWD/.scratch_vbench_i2v
```

或在命令中使用：

```bash
--scratch-dir .scratch_vbench_i2v
```

---

## 2. 缺失 repeat 视频

错误：

```text
RuntimeError: Error reading ... videos_quant_sb/...-4.mp4
No such file or directory
```

原因：VBench-I2V 的 subject/background/camera 输入需要 exact repeats，某些 `-4.mp4` 没有被真实生成。注意：不能把 `-0.mp4` 复制成 `-1..-4.mp4` 来凑数；官方语义是每个 image-prompt pair 采样 5 个视频。

修复：先回到生成阶段补齐真实的 `prompt-0..prompt-4.mp4`，再重建输入目录：

```bash
python -m hif4_vbench_i2v.repair_exact_repeats \
  --template-case evaluation_inputs/empty_seed42 \
  --case-input evaluation_inputs/hif4_seed42 \
  --generated-dir outputs/hif4_seed42_exact_repeats
```

---

## 3. identical repeat 文件被拒绝

错误：

```text
IDENTICAL_REPEAT_ERROR ... all_5_repeat_files_have_same_sha256=...
```

原因：同一 prompt 的 5 个 repeat 文件字节完全相同，通常意味着把一个 mp4 复制成 5 份。请使用不同 seed/index 重新生成。确有特殊原因要放行时，显式添加：

```bash
--allow-identical-repeat-files
```

---

## 4. symlink 解析失败

默认不要用 symlink。建议：

```bash
--copy-mode physical
```

---

## 5. 外部模型下载失败

VBench 的 AMT / RAFT / CoTracker / DreamSim 等模型建议提前下载到本地，然后在配置中指定路径。

---

## 6. 不能只看作业状态

一个 VBench job 可能中途失败，但前几个维度已经成功写出 JSON。最终完成度必须用：

```bash
python -m hif4_vbench_i2v.scan_missing ...
```

而不是只看 Slurm `COMPLETED/FAILED`。
