"""Keep canonical layer orchestration and distribute only E4 optimizer updates.

Rank zero owns calibration, teacher targets, validation, adoption, propagation,
and artifacts. Other ranks retain their layer replica and optimizer between
epochs, and release them as soon as the last training epoch finishes.
"""
from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import E2ETrainConfig
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import (
    load_qwen3_moe_layer_state,
    release_qwen3_moe_layer_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_semantic_hif4 import (
    MoEFusableDiagState,
    StudentQwen3MoELayerRuntime,
    build_moe_diag_state,
)
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import build_length_bucket_batches
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training import moe_trainer as canonical
from .parallel_execution import broadcast_parameters, e4_parallel_step


def _parameter_layout(module: torch.nn.Module) -> list[tuple]:
    return [
        (name, tuple(parameter.shape), str(parameter.dtype), parameter.requires_grad)
        for name, parameter in sorted(module.named_parameters())
    ]


class _ParallelDriver:
    def __init__(self, cfg: E2ETrainConfig, device: torch.device, control_group) -> None:
        self.cfg = cfg
        self.device = device
        self.control_group = control_group
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.runtime = None
        self.optimizer = None
        self.scheduler = None
        self.snapshot = None
        self.total_steps = 0

    def command(self, payload: dict | None = None) -> dict:
        items = [payload]
        dist.broadcast_object_list(items, src=0, group=self.control_group)
        if not isinstance(items[0], dict):
            raise RuntimeError("parallel training command must be a dictionary")
        return items[0]

    def _initialize_worker(self, command: dict) -> None:
        if self.runtime is not None:
            raise RuntimeError("previous parallel layer was not released")
        self.snapshot = Path(command["snapshot"])
        state = load_qwen3_moe_layer_state(self.snapshot, command["layer_idx"], self.device)
        diag = build_moe_diag_state(state.spec, self.cfg.diag_mode).to(self.device)
        if not isinstance(diag, MoEFusableDiagState):
            raise TypeError("parallel E4 requires fusable DIAG parameters")
        diag.configure_fusable_components(self.cfg.fusable_diag_components)
        if _parameter_layout(diag) != command["parameter_layout"]:
            raise RuntimeError("parallel E4 replica parameter layout differs from rank zero")
        self.runtime = StudentQwen3MoELayerRuntime(
            state, diag, use_r64=self.cfg.use_r64, rot_order=self.cfg.rot_order,
        ).to(self.device)
        broadcast_parameters(diag)
        params = [p for p in diag.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=self.cfg.diag_lr, weight_decay=float(self.cfg.weight_decay),
        )
        scheduler_state = command["scheduler_state"]
        self.scheduler = None
        if scheduler_state is not None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=scheduler_state["T_max"], eta_min=scheduler_state["eta_min"],
            )
            self.scheduler.load_state_dict(scheduler_state)
        self.optimizer.load_state_dict(command["optimizer_state"])
        print(json.dumps({"event": "parallel_worker_loaded", "rank": self.rank,
                          "layer": state.layer_idx}), flush=True)

    def _scatter_batch(self, command: dict, packed_tensors: list | None = None) -> list:
        local = []
        for index, field in enumerate(command["fields"]):
            shape = tuple(field["shape"])
            dtype = getattr(torch, field["dtype"])
            received = torch.empty((1, *shape[1:]), dtype=dtype, device=self.device)
            # Only one global input tensor is staged on rank zero's GPU at a
            # time. Every rank receives the original batch's full padded length.
            staged = None
            chunks = None
            if self.rank == 0:
                staged = packed_tensors[index].to(self.device)
                chunks = list(staged.split(1, dim=0))
            dist.scatter(received, scatter_list=chunks, src=0)
            local.append(received)
            del staged, chunks
        return local

    @torch.no_grad()
    def _check_replicas(self) -> None:
        """Verify replicated update state, not full-batch numerical equivalence."""
        mismatch = torch.zeros((), dtype=torch.int32, device=self.device)
        metadata = {"scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
                    "state_keys": {}, "grad_none": []}
        tensors = []
        for name, parameter in sorted(self.runtime.diag_state.named_parameters()):
            tensors.append(parameter.detach())
            metadata["grad_none"].append((name, parameter.grad is None))
            if parameter.grad is not None:
                tensors.append(parameter.grad)
            state = self.optimizer.state.get(parameter, {})
            metadata["state_keys"][name] = sorted(state)
            for key in sorted(state):
                value = state[key]
                if isinstance(value, torch.Tensor):
                    tensors.append(value.to(self.device))
                else:
                    metadata[f"{name}.{key}"] = value
        replicas = [None] * self.world_size
        dist.all_gather_object(replicas, metadata, group=self.control_group)
        if any(item != replicas[0] for item in replicas):
            raise RuntimeError("parallel E4 gradient presence/optimizer/scheduler metadata diverged")
        for tensor in tensors:
            reference = tensor.detach().clone()
            dist.broadcast(reference, src=0)
            mismatch.add_(int(not torch.equal(tensor, reference)))
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        if int(mismatch):
            raise RuntimeError("parallel E4 parameters, gradients, or Adam state diverged across ranks")

    def _step(self, local: list) -> dict:
        result = e4_parallel_step(
            self.runtime, *local, self.snapshot, self.cfg, self.optimizer, self.scheduler,
        )
        self.total_steps += 1
        return result

    def _release(self) -> None:
        if self.runtime is None:
            raise RuntimeError("parallel E4 release without an active replica")
        layer_idx = self.runtime.state.layer_idx
        # Rank zero's runtime still belongs to canonical adoption/propagation.
        if self.rank != 0:
            state = self.runtime.state
            self.runtime = None
            self.optimizer = None
            self.scheduler = None
            release_qwen3_moe_layer_state(state)
            del state
        self.runtime = self.optimizer = self.scheduler = self.snapshot = None
        if self.rank != 0:
            print(json.dumps({"event": "parallel_worker_released", "rank": self.rank,
                              "layer": layer_idx}), flush=True)

    def train_epoch(
        self, runtime, snapshot, train_samples, collator, x_cache, targets, cfg,
        device, epoch, optimizer, scheduler,
    ) -> int:
        if self.rank != 0:
            raise RuntimeError("only rank zero executes canonical layer orchestration")
        batches = build_length_bucket_batches(train_samples, cfg.diag_batch_size, cfg.calib_seed + epoch)
        if not batches or any(len(batch) != self.world_size for batch in batches):
            raise ValueError("parallel E4 requires complete global batches of four samples")
        if self.runtime is None:
            if epoch != 0 or optimizer.state:
                raise RuntimeError("parallel E4 only initializes at a fresh layer boundary")
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                raise TypeError("parallel E4 supports only the canonical cosine scheduler")
            self.runtime, self.optimizer, self.scheduler = runtime, optimizer, scheduler
            self.snapshot = Path(snapshot)
            self.command({
                "op": "load", "snapshot": str(snapshot), "layer_idx": runtime.state.layer_idx,
                "parameter_layout": _parameter_layout(runtime.diag_state),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            })
            broadcast_parameters(runtime.diag_state)
        elif self.runtime is not runtime or self.optimizer is not optimizer or self.scheduler is not scheduler:
            raise RuntimeError("parallel E4 canonical runtime/optimizer changed within a layer")

        epoch_started = time.monotonic()
        for batch_index, batch in enumerate(batches):
            step_started = time.monotonic()
            packed = collator(batch)
            sample_ids = [sample.sample_id for sample in batch]
            hidden, _ = x_cache.assemble(sample_ids, torch.device("cpu"))
            target = canonical._assemble_targets(targets, sample_ids, torch.device("cpu"))
            tensors = [hidden, target, packed["loss_mask"], packed["attention_mask"]]
            fields = [{"shape": tuple(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
                      for t in tensors]
            command = self.command({"op": "step", "fields": fields})
            local = self._scatter_batch(command, tensors)
            result = self._step(local)
            del local, tensors, hidden, target, packed
            if epoch == 0 and batch_index == 0:
                self.command({"op": "check"})
                self._check_replicas()
                print(json.dumps({"event": "parallel_first_step_replicas_equal",
                                  "layer": runtime.state.layer_idx, "world_size": self.world_size}), flush=True)
            print(json.dumps({
                "event": "parallel_optimizer_step", "layer": runtime.state.layer_idx,
                "epoch": epoch, "batch": batch_index, "global_optimizer_steps": self.total_steps,
                "batch_sample_ids": sample_ids, "global_padded_length": fields[0]["shape"][1],
                "elapsed_seconds": round(time.monotonic() - step_started, 4),
                "loss": result["loss"], "lr": optimizer.param_groups[0]["lr"],
            }), flush=True)
        self.command({"op": "check"})
        self._check_replicas()
        print(json.dumps({"event": "parallel_epoch_complete", "layer": runtime.state.layer_idx,
                          "epoch": epoch, "steps": len(batches), "replicas_equal": True,
                          "elapsed_seconds": round(time.monotonic() - epoch_started, 4)}), flush=True)
        if epoch == cfg.diag_epochs - 1:
            self.command({"op": "release"})
            self._release()
        return len(batches)

    def worker_loop(self) -> None:
        while True:
            command = self.command()
            operation = command["op"]
            if operation == "load":
                self._initialize_worker(command)
            elif operation == "step":
                if self.runtime is None:
                    raise RuntimeError("parallel E4 step before layer initialization")
                local = self._scatter_batch(command)
                self._step(local)
                del local
            elif operation == "check":
                self._check_replicas()
            elif operation == "release":
                self._release()
            elif operation == "stop":
                if self.runtime is not None:
                    raise RuntimeError("parallel E4 stopped before releasing a layer")
                return
            else:
                raise ValueError(f"unknown parallel E4 command {operation!r}")


def run_parallel_training(cfg: E2ETrainConfig, device: torch.device, *, resume: bool = False) -> None:
    """Run on all four torchrun ranks after rank zero prepares the output tree.

    The caller sets its CUDA device and seeds. No existing file or process is
    modified. A resumed run must already contain the verified adopted prefix and
    calibration/config files required by ``moe_trainer_resume``.
    """
    if cfg.diag_batch_size != 4 or cfg.diag_mode != "fusable" or cfg.recon_loss != "block_delta_nmse":
        raise ValueError("parallel E4 requires global batch 4, fusable DIAG, block_delta_nmse")
    if cfg.optimizer != "AdamW" or float(cfg.router_align_loss_weight) != 0.0:
        raise ValueError("parallel E4 requires AdamW and no additive router training loss")
    if cfg.diag_epochs < 1 or cfg.diag_scheduler not in ("cosine", "constant"):
        raise ValueError("parallel E4 requires positive epochs and the canonical scheduler")
    if cfg.start_layer > 0 and not resume:
        raise ValueError("nonzero start_layer requires adopted-prefix replay with resume=True")
    created_group = not dist.is_initialized()
    timeout = timedelta(hours=2)  # Workers also wait through prefix replay/teacher/validation.
    if created_group:
        dist.init_process_group("nccl", timeout=timeout, device_id=device)
    if dist.get_world_size() != 4 or dist.get_backend() != "nccl":
        raise ValueError("parallel E4 requires exactly four ranks using NCCL")
    control_group = dist.new_group(backend="gloo", timeout=timeout)
    configs = [None] * dist.get_world_size()
    dist.all_gather_object(configs, cfg.to_dict(), group=control_group)
    if any(config != configs[0] for config in configs):
        raise ValueError("parallel E4 configurations differ across ranks")
    driver = _ParallelDriver(cfg, device, control_group)
    if dist.get_rank() == 0:
        original_train_layer = canonical._train_layer
        canonical._train_layer = driver.train_epoch
        try:
            if resume:
                from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.training.moe_trainer_resume import train_qwen3_moe_lazy
                train_qwen3_moe_lazy(cfg, device)
            else:
                canonical.train_qwen3_moe_lazy(cfg, device)
            driver.command({"op": "stop"})
        finally:
            canonical._train_layer = original_train_layer
    else:
        driver.worker_loop()
    # An exception intentionally escapes before these normal-completion
    # collectives, allowing torchrun to terminate the remaining ranks promptly.
    dist.barrier(group=control_group)
    dist.destroy_process_group(control_group)
    if created_group:
        dist.destroy_process_group()
