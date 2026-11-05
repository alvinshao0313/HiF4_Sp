"""Distribute Group training updates while retaining canonical orchestration."""
from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.artifact import select_layer_diag
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.modelopt_moe_checkpoint import load_qwen3_moe_layer_state, release_qwen3_moe_layer_state
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.batching import build_length_bucket_batches
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction import train as canonical
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import load
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.student import Student
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.transforms import fold_initial_diag
from .group_parallel_step import ROUTER_METRICS, group_parallel_step


def _layout(module):
    return [(name, tuple(parameter.shape), str(parameter.dtype), parameter.requires_grad)
            for name, parameter in sorted(module.named_parameters())]


class _GroupDriver:
    def __init__(self, cfg, device, control_group, original_epoch):
        self.cfg, self.device, self.control_group = cfg, device, control_group
        self.original_epoch = original_epoch
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.student = self.optimizer = self.scheduler = self.snapshot = None
        self.total_steps = 0
        self.initial = None

    def command(self, payload=None):
        items = [payload]
        dist.broadcast_object_list(items, src=0, group=self.control_group)
        if not isinstance(items[0], dict):
            raise RuntimeError("Group command must be a dictionary")
        return items[0]

    @torch.no_grad()
    def _broadcast_parameters(self):
        for _, parameter in sorted(self.student.named_parameters()):
            dist.broadcast(parameter, src=0)

    def _load_worker(self, command):
        if self.student is not None:
            raise RuntimeError("previous Group worker layer was not released")
        if self.initial is None:
            self.initial = load(self.cfg.init_artifact)
            if self.initial["source_model"] != self.cfg.model_path or self.initial["diag_mode"] != "fusable":
                raise ValueError("Group worker E4 initialization differs from configured source")
        self.snapshot = Path(command["snapshot"])
        native_state = load_qwen3_moe_layer_state(self.snapshot, command["layer"], self.device)
        base = fold_initial_diag(native_state, select_layer_diag(self.initial["layers"][str(command["layer"])], "adopted"))
        # The folded state shares immutable metadata with native_state. Drop the
        # native reference instead of clearing metadata through the release helper.
        del native_state
        self.student = Student(base, self.cfg.matrix_sharing).to(self.device)
        if _layout(self.student) != command["layout"]:
            raise RuntimeError("Group worker parameter layout differs from rank zero")
        self._broadcast_parameters()
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.cfg.lr, weight_decay=0.0)
        scheduler_state = command["scheduler"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=scheduler_state["T_max"], eta_min=scheduler_state["eta_min"],
        )
        self.scheduler.load_state_dict(scheduler_state)
        self.optimizer.load_state_dict(command["optimizer"])
        print(json.dumps({"event": "group_parallel_worker_loaded", "rank": self.rank,
                          "layer": command["layer"]}), flush=True)

    def _scatter(self, command, tensors=None):
        local = []
        for index, field in enumerate(command["fields"]):
            received = torch.empty((1, *field["shape"][1:]), dtype=getattr(torch, field["dtype"]),
                                   device=self.device)
            staged = chunks = None
            if self.rank == 0:
                staged = tensors[index].to(self.device)
                chunks = list(staged.split(1, dim=0))
            dist.scatter(received, scatter_list=chunks, src=0)
            local.append(received)
            del staged, chunks
        return local

    @torch.no_grad()
    def _check_replicas(self):
        metadata = {"scheduler": self.scheduler.state_dict(), "state": {}, "grad_none": []}
        tensors = defaultdict(list)
        for name, parameter in sorted(self.student.named_parameters()):
            tensors[(parameter.device, parameter.dtype)].append(parameter.detach())
            metadata["grad_none"].append((name, parameter.grad is None))
            if parameter.grad is not None:
                tensors[(parameter.grad.device, parameter.grad.dtype)].append(parameter.grad)
            state = self.optimizer.state.get(parameter, {})
            metadata["state"][name] = sorted(state)
            for key, value in sorted(state.items()):
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    tensors[(value.device, value.dtype)].append(value)
                else:
                    metadata[f"{name}.{key}"] = float(value) if isinstance(value, torch.Tensor) else value
        replicas = [None] * self.world_size
        dist.all_gather_object(replicas, metadata, group=self.control_group)
        if any(item != replicas[0] for item in replicas):
            raise RuntimeError("Group gradient-presence/Adam/scheduler metadata differs across ranks")
        mismatch = torch.zeros((), dtype=torch.int32, device=self.device)

        def check_bucket(bucket):
            flat = torch.cat([tensor.reshape(-1) for tensor in bucket])
            reference = flat.clone()
            dist.broadcast(reference, src=0)
            mismatch.add_(int(not torch.equal(flat, reference)))

        for values in tensors.values():
            bucket, size = [], 0
            for value in values:
                nbytes = value.numel() * value.element_size()
                if bucket and size + nbytes > 64 * 1024 * 1024:
                    check_bucket(bucket)
                    bucket, size = [], 0
                bucket.append(value)
                size += nbytes
            if bucket:
                check_bucket(bucket)
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        if int(mismatch):
            raise RuntimeError("Group parameters, gradients, or Adam moments differ across ranks")

    def _step(self, local):
        result = group_parallel_step(self.student, *local, self.snapshot, self.cfg,
                                     self.optimizer, self.scheduler)
        self.total_steps += 1
        return result

    def _release(self):
        if self.student is None:
            raise RuntimeError("Group release without an active layer")
        layer = self.student.base.layer_idx
        if self.rank != 0:
            base = self.student.base
            self.student = self.optimizer = self.scheduler = None
            release_qwen3_moe_layer_state(base)
            del base
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps({"event": "group_parallel_worker_released", "rank": self.rank,
                              "layer": layer}), flush=True)
        # Rank zero's student remains owned by canonical selection/propagation.
        self.student = self.optimizer = self.scheduler = self.snapshot = None

    def run_epoch(self, student, cfg, snapshot, samples, collator, hidden, targets, router_targets,
                  device, *, epoch, optimizer=None, scheduler=None):
        if optimizer is None:
            return self.original_epoch(student, cfg, snapshot, samples, collator, hidden, targets,
                                       router_targets, device, epoch=epoch)
        if self.rank != 0:
            raise RuntimeError("only rank zero owns canonical Group orchestration")
        batches = build_length_bucket_batches(samples, cfg.batch_size, cfg.calib_seed + epoch)
        if not batches or any(len(batch) != self.world_size for batch in batches):
            raise ValueError("Group parallel training requires complete global batches of four")
        if self.student is None:
            if epoch != 0 or optimizer.state:
                raise ValueError("this Group parallel driver starts at fresh layer boundaries")
            if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                raise TypeError("Group parallel training requires the canonical cosine scheduler")
            self.student, self.optimizer, self.scheduler, self.snapshot = student, optimizer, scheduler, Path(snapshot)
            self.command({"op": "load", "layer": student.base.layer_idx, "snapshot": str(snapshot),
                          "layout": _layout(student), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict()})
            self._broadcast_parameters()
        elif self.student is not student or self.optimizer is not optimizer or self.scheduler is not scheduler:
            raise RuntimeError("canonical Group runtime or optimizer changed within a layer")

        total_num = total_den = router_sum = 0.0
        tokens = 0
        metric_sums = {name: 0.0 for name in ROUTER_METRICS}
        counts = torch.zeros(student.base.spec.num_experts, dtype=torch.long)
        started = time.monotonic()
        for batch_index, batch in enumerate(batches):
            step_started = time.monotonic()
            packed = collator(batch)
            tensors = [canonical.assemble(values, batch, "cpu") for values in (hidden, targets, router_targets)]
            tensors.extend((packed["loss_mask"], packed["attention_mask"]))
            fields = [{"shape": tuple(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
                      for tensor in tensors]
            command = self.command({"op": "step", "fields": fields})
            local = self._scatter(command, tensors)
            result = self._step(local)
            del local, tensors, packed
            if epoch == 0 and batch_index == 0:
                self.command({"op": "check"})
                self._check_replicas()
                print(json.dumps({"event": "group_parallel_first_step_replicas_equal",
                                  "layer": student.base.layer_idx, "world_size": self.world_size}), flush=True)
            total_num += result["num"]
            total_den += result["den"]
            router_sum += result["router_sum"]
            tokens += result["tokens"]
            for name in ROUTER_METRICS:
                metric_sums[name] += result["router_metric_sums"][name]
            counts += torch.tensor(result["expert_token_counts"], dtype=torch.long)
            print(json.dumps({"event": "batch", "layer": student.base.layer_idx, "epoch": epoch,
                              "train": True, "batch": batch_index + 1, "batches": len(batches),
                              "tokens": result["tokens"], "objective": result["objective"],
                              "elapsed_seconds": round(time.monotonic() - started, 2),
                              "step_seconds": round(time.monotonic() - step_started, 4),
                              "parallel_world_size": self.world_size,
                              "global_padded_length": fields[0]["shape"][1],
                              "sample_ids": [sample.sample_id for sample in batch]}), flush=True)
        self.command({"op": "check"})
        self._check_replicas()
        print(json.dumps({"event": "group_parallel_epoch_complete", "layer": student.base.layer_idx,
                          "epoch": epoch, "steps": len(batches), "replicas_equal": True,
                          "elapsed_seconds": round(time.monotonic() - started, 4)}), flush=True)
        if epoch == cfg.epochs - 1:
            self.command({"op": "release"})
            self._release()
        if tokens == 0 or total_den <= 0:
            raise RuntimeError("empty Group epoch or zero reconstruction target energy")
        reconstruction, routing = total_num / total_den, router_sum / tokens
        return {"reconstruction_nmse": reconstruction, "router_loss": routing,
                "objective": reconstruction + cfg.router_loss_weight * routing, "tokens": tokens,
                **{name: value / tokens for name, value in metric_sums.items()},
                "expert_token_counts": counts.tolist()}

    def worker_loop(self):
        while True:
            command = self.command()
            operation = command["op"]
            if operation == "load":
                self._load_worker(command)
            elif operation == "step":
                if self.student is None:
                    raise RuntimeError("Group step before worker initialization")
                local = self._scatter(command)
                self._step(local)
                del local
            elif operation == "check":
                self._check_replicas()
            elif operation == "release":
                self._release()
            elif operation == "stop":
                if self.student is not None:
                    raise RuntimeError("Group driver stopped before releasing its layer")
                return
            else:
                raise ValueError(f"unknown Group parallel command {operation!r}")


def run_parallel_group(cfg, device, *, runner=None):
    """Run on all four ranks; optional rank-zero runner is a no-argument callable.

    Only the training branch is replaced. Validation, teacher targets, checkpoint
    writes, selection, and propagation retain their existing canonical paths.
    This fresh-run driver does not implement loading an in-progress Group epoch.
    """
    cfg.validate()
    if cfg.batch_size != 4 or cfg.matrix_sharing != "group" or cfg.router_loss != "top_mass":
        raise ValueError("Group sample parallelism requires batch 4, group matrices, top_mass router loss")
    created_group = not dist.is_initialized()
    timeout = timedelta(hours=2)
    if created_group:
        dist.init_process_group("nccl", timeout=timeout, device_id=device)
    if dist.get_world_size() != 4 or dist.get_backend() != "nccl":
        raise ValueError("Group parallel training requires exactly four NCCL ranks")
    control = dist.new_group(backend="gloo", timeout=timeout)
    configs = [None] * dist.get_world_size()
    dist.all_gather_object(configs, cfg.to_dict(), group=control)
    if any(config != configs[0] for config in configs):
        raise ValueError("Group configurations differ across ranks")
    original_epoch = canonical.run_epoch
    driver = _GroupDriver(cfg, device, control, original_epoch)
    if dist.get_rank() == 0:
        from . import smoke_group
        original_smoke_epoch = smoke_group.run_epoch
        canonical.run_epoch = smoke_group.run_epoch = driver.run_epoch
        try:
            if runner is None:
                canonical.train(cfg)
            else:
                runner()
            driver.command({"op": "stop"})
        finally:
            canonical.run_epoch = original_epoch
            smoke_group.run_epoch = original_smoke_epoch
    else:
        driver.worker_loop()
    # On failure torchrun terminates the peers; do not enter a completion barrier.
    dist.barrier(group=control)
    dist.destroy_process_group(control)
    if created_group:
        dist.destroy_process_group()
