"""Run and persist one real CPU Linear comparison without loading a full model."""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import time
import traceback

import torch

from ..matrix_norm_efficiency.artifacts import sources as previous_sources
from ..matrix_norm_efficiency.config import REPO
from ..non_equivalent_reconstruction.artifact import atomic_save, load, sha256, write_json
from ..reasoning_trace_calibration.artifacts import state_digest
from .data import load_cpu_projection
from .solver import evaluate_projection, fit_projection


def source_hashes():
    paths = list(Path(__file__).parent.rglob("*.py"))
    paths += [REPO / "Native_NVFP4_HiF4_Linear_Puncture/src/formats.py",
              REPO / "Native_NVFP4_HiF4_Linear_Puncture/experiments/diag_gradient/r64_transform.py",
              REPO / "NVFP4/torch_fake.py",
              REPO / "3rdparty/vllm/vllm/model_executor/layers/quantization/utils/nvfp4_emulation_utils.py"]
    paths += list((REPO / "HiFloat4/hif4_scale_threshold_optimization/src").glob("*.py"))
    return {**previous_sources(), **{str(p.relative_to(REPO)): sha256(p) for p in paths}}


def run(root):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("run this CPU diagnostic with CUDA_VISIBLE_DEVICES='' explicitly")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    command = f"CUDA_VISIBLE_DEVICES='' python -m Native_NVFP4_HiF4_Linear_Puncture.experiments.matrix_only_analytic.run --output {root}"
    status = {"status": "preparing", "pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat(),
              "device": "cpu", "command": command, "exit_code": None}
    write_json(status, root / "status.json")
    try:
        sources = source_hashes()
        print("Loading native layer-0 q_proj and 256 train / 256 validation tokens", flush=True)
        weight, training, validation, frozen, manifest = load_cpu_projection()
        frozen_digest = state_digest(frozen)
        manifest.update(seed=42, cpu_threads=2, cpu_interop_threads=1, python=platform.python_version(),
                        torch_version=torch.__version__, cpu=platform.processor(), sources=sources,
                        weight_shape=list(weight.shape), h_shape=[weight.shape[1] // 64, 64, 64],
                        frozen_state_digest=frozen_digest, command=command)
        write_json(manifest, root / "manifest.json")
        preparation_seconds = time.monotonic() - started
        status.update(status="running", preparation_seconds=preparation_seconds)
        write_json(status, root / "status.json")
        reports, selections = {}, {}
        with (root / "progress.jsonl").open("w", encoding="utf-8") as progress:
            import json

            for method in ("adam", "analytic"):
                def record(entry):
                    progress.write(json.dumps({"method": method, **entry}, allow_nan=False) + "\n")
                    progress.flush()
                    print(f"{method} {entry['iteration']}: train NMSE={entry['train']['nmse']:.7g}, "
                          f"validation NMSE={entry['validation']['nmse']:.7g}, "
                          f"{entry['elapsed_seconds']:.2f}s", flush=True)

                result = fit_projection(weight, training, validation, method, callback=record)
                reports[method] = result.report
                path = root / f"{method}.pt"
                atomic_save({"selected_h": result.selected_h, "final_h": result.final_h,
                             "h_convention": "H=M.T; folded weight W@M; [out,in] storage"}, path)
                saved = load(path)
                if not all(torch.equal(saved[key], getattr(result, key)) for key in ("selected_h", "final_h")):
                    raise ArithmeticError("H save/reload changed tensor bits")
                if state_digest(frozen) != frozen_digest:
                    raise ArithmeticError("frozen model parameters or inputs changed")
                selected = evaluate_projection(validation, weight, saved["selected_h"])
                expected = result.report["curve"][result.report["selected_iteration"]]["validation"]
                if selected != expected:
                    raise ArithmeticError("selected-H reloaded validation metric changed")
                selections[method] = {"validation": selected, "checkpoint_sha256": sha256(path),
                                      "frozen_state_unchanged": True, "checkpoint_exact_reload": True}
                write_json(result.report, root / f"{method}.json")

        if torch.cuda.is_initialized():
            raise RuntimeError("CPU diagnostic unexpectedly initialized CUDA")
        if source_hashes() != sources:
            raise RuntimeError("source changed during diagnostic")
        report = {"status": "complete", "device": "cpu", "cuda_initialized": False,
                  "preparation_seconds": preparation_seconds, "total_seconds": time.monotonic() - started,
                  "methods": reports, "selected": selections,
                  "scope": manifest["scope"], "native_snapshot": manifest["native_snapshot"]}
        write_json(report, root / "report.json")
        lines = ["# H-only CPU 单 Linear 结果", "", "原生 Qwen3-30B-A3B-NVFP4 第0层 q_proj，权重4096×2048。",
                 "固定原生 norm、基础权重与激活 R64；两线程 CPU，seed=42。4条train与4条validation各取64个token，共256+256。",
                 "层0 embedding + RMSNorm在attention之前逐token独立；本取样方式不适用于更深层或完整block。", "",
                 "| 方法 | 验证选优步/扫块 | 初始验证NMSE | 选优验证NMSE | 最后验证NMSE | 含测量耗时/s | 扣除测量耗时/s |",
                 "| --- | --- | --- | --- | --- | --- | --- |"]
        for method, result in reports.items():
            lines.append(f"| {method} | {result['selected_iteration']} | {result['curve'][0]['validation']['nmse']:.9g} | "
                         f"{selections[method]['validation']['nmse']:.9g} | {result['curve'][-1]['validation']['nmse']:.9g} | "
                         f"{result['fit_seconds_including_validation']:.3f} | {result['solver_seconds_excluding_measurement']:.3f} |")
        lines += ["", "Adam 20步，lr=3e-4；解析2次顺序扫块，阻尼0.01，真实QDQ回溯步长1/0.5/0.25/0.125/0.0625。",
                  "初始R64参与验证选优。NMSE是平方误差和/教师输出平方和，不是准确率。",
                  "解析更新保留完整跨块Gram；闭式求解只针对冻结量化残差的二次代理，原含量化问题仍非凸。",
                  "本次为BF16操作数/输出的HiF4 QDQ数值验证，不是packed HiF4计算核或整模型速度测试。",
                  "验证集用于选优，只支持局部诊断；没有独立测试集、整层误差或下游精度结果。", "",
                  "配置/输入与代码哈希见 [manifest.json](manifest.json)，完整曲线及检查见 [report.json](report.json)，",
                  "进度见 [progress.jsonl](progress.jsonl)，退出状态见 [status.json](status.json)。"]
        (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        status.update(status="complete", exit_code=0, total_seconds=report["total_seconds"])
        write_json(status, root / "status.json")
        print(f"Complete: {root / 'REPORT.md'}", flush=True)
        return report
    except Exception:
        status.update(status="failed", exit_code=1, error=traceback.format_exc())
        write_json(status, root / "status.json")
        raise
