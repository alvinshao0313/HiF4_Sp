"""Rank-local full-prefix intervention on existing real-vLLM worker hooks.

No Attention, decoder, normalization or expert computation is reproduced here.
The only mutation is replacement of an existing production boundary state pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BOUNDARIES = ("layer_in", "post_attn_norm", "layer_out")
SESSION_ATTR = "_hif4_format_intervention_session"
HANDLES_ATTR = "_hif4_format_intervention_handles"


def canonical_scope(sample: dict, layer: int, boundary: str, target_decode_index: int) -> dict:
    if boundary not in BOUNDARIES or not 0 <= layer < 48:
        raise ValueError("unsupported intervention layer/boundary")
    prompt = [int(x) for x in sample["input_ids"]]
    target = int(target_decode_index)
    output = [int(x) for x in sample["output_ids"]]
    if not prompt or not 0 <= target < len(output):
        raise ValueError("target predictor must exist in the canonical E0 trajectory")
    digest = lambda ids: hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    return {
        "sample_key": str(sample["prompt_key"]),
        "prompt_len": len(prompt),
        "prompt_ids_sha256": digest(prompt),
        "canonical_output_sha256": digest(output[:target + 1]),
        "target_decode_index": target,
        "target_abs_position": len(prompt) - 1 + target,
        "layer": int(layer),
        "boundary": boundary,
    }


def interpolate_pair(pair0: tuple, pair1: tuple, alpha: float) -> tuple:
    """Keep exact endpoint tensors; interpolate both roles in FP32 otherwise."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    result = []
    for a, b in zip(pair0, pair1, strict=True):
        if a is None or b is None:
            if a is not None or b is not None:
                raise RuntimeError("canonical residual presence differs")
            result.append(None)
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError("canonical state shape/dtype differs")
        if alpha == 0:
            result.append(a)
        elif alpha == 1:
            result.append(b)
        else:
            result.append((a.float() + float(alpha) * (b.float() - a.float())).to(a.dtype))
    return tuple(result)


class StateInterventionSession:
    """One sample, rank, layer and boundary; all prefix positions are mandatory."""

    def __init__(self, scope: dict, rank: int, variant: str, mode: str,
                 reference: dict | None = None, deviation: dict | None = None,
                 alpha: float = 0.0, gate: dict | None = None):
        if variant not in {"E0", "E1", "E2", "E3"} or mode not in {"capture", "noop", "reset", "inject"}:
            raise ValueError("invalid state intervention mode/variant")
        if rank not in {0, 1} or scope["boundary"] not in BOUNDARIES:
            raise ValueError("state intervention requires TP2 and a canonical pair boundary")
        if mode in {"noop", "inject"} and variant != "E0":
            raise ValueError("no-op and injection must run the E0 tail")
        if mode == "reset" or (mode == "inject" and alpha != 0):
            required = "alpha0" if mode == "inject" else "noop"
            if not gate or gate.get("status") != "PASS" or gate.get("kind") != required or gate.get("scope") != scope:
                raise RuntimeError(f"STATE_RESET_BLOCKED: missing matching {required} identity gate")
            if set(gate.get("ranks", [])) != {0, 1}:
                raise RuntimeError("STATE_RESET_BLOCKED: identity gate must cover both TP ranks")
        self.scope = dict(scope)
        self.rank, self.variant, self.mode = rank, variant, mode
        self.alpha = float(alpha)
        self.reference, self.deviation = reference, deviation
        self.positions: list[int] = []
        self.records: dict[int, tuple] = {}
        self.seen: set[int] = set()
        self.closed = False
        for data, expected_variant in ((reference, "E0"), (deviation, "E1")):
            if data is None:
                continue
            if data.get("scope") != scope or data.get("rank") != rank or data.get("variant") != expected_variant:
                raise RuntimeError("reference sample/history/layer/boundary/rank/variant mismatch")
            if data.get("world_size") != 2 or not data.get("full_prefix_complete"):
                raise RuntimeError("reference is not a complete real TP2 prefix")
            expected = set(range(scope["target_abs_position"] + 1))
            if set(data["pairs"]) != expected:
                raise RuntimeError("reference is missing prefix positions")
        if mode != "capture" and reference is None:
            raise ValueError("state patch requires an E0 reference")
        if mode == "inject" and deviation is None:
            raise ValueError("injection requires fixed-E0-history E1 deviation states")

    def set_positions(self, sample_key: str | None, positions: torch.Tensor) -> None:
        if self.closed or sample_key != self.scope["sample_key"]:
            raise RuntimeError("intervention sample isolation violation")
        self.positions = [int(x) for x in positions.detach().cpu().reshape(-1).tolist()]
        if any(p < 0 for p in self.positions) or len(self.positions) != len(set(self.positions)):
            raise RuntimeError("single-request positions must be unique and nonnegative")

    def apply_pair(self, sample_key: str | None, layer: int, boundary: str,
                   branch: torch.Tensor, residual: torch.Tensor | None) -> tuple:
        if self.closed or sample_key != self.scope["sample_key"]:
            raise RuntimeError("intervention sample isolation violation")
        if layer != self.scope["layer"] or boundary != self.scope["boundary"]:
            return branch, residual
        if branch.ndim < 2 or branch.shape[0] != len(self.positions):
            raise RuntimeError("boundary row/position mapping mismatch")
        if residual is None and not (layer == 0 and boundary == "layer_in"):
            raise RuntimeError("canonical branch/residual pair is incomplete")
        if residual is not None and (residual.shape != branch.shape or residual.dtype != branch.dtype):
            raise RuntimeError("branch/residual shape/dtype mismatch")
        selected = [(row, pos) for row, pos in enumerate(self.positions) if pos <= self.scope["target_abs_position"]]
        if not selected:
            return branch, residual
        patched_branch = branch.clone() if self.mode != "capture" else branch
        patched_residual = residual.clone() if self.mode != "capture" and residual is not None else residual
        for row, pos in selected:
            if pos in self.seen:
                raise RuntimeError(f"duplicate full-prefix state at position {pos}")
            self.seen.add(pos)
            pair = (branch[row], None if residual is None else residual[row])
            if self.mode == "capture":
                self.records[pos] = tuple(None if x is None else x.detach().cpu().clone() for x in pair)
                continue
            desired = self.reference["pairs"][pos]
            if self.mode == "inject":
                desired = interpolate_pair(desired, self.deviation["pairs"][pos], self.alpha)
            for actual, source in zip(pair, desired, strict=True):
                if (actual is None) != (source is None):
                    raise RuntimeError("reference residual presence differs")
                if actual is not None and (actual.shape != source.shape or actual.dtype != source.dtype):
                    raise RuntimeError("reference state shape/dtype mismatch")
            patched_branch[row].copy_(desired[0])
            if patched_residual is not None:
                patched_residual[row].copy_(desired[1])
        return patched_branch, patched_residual

    def finish(self) -> dict:
        try:
            expected = set(range(self.scope["target_abs_position"] + 1))
            if self.seen != expected:
                raise RuntimeError(f"STATE_RESET_BLOCKED: incomplete full prefix; missing={sorted(expected - self.seen)[:16]}")
            return {
                "schema_version": 1, "scope": self.scope, "rank": self.rank, "world_size": 2,
                "variant": self.variant, "mode": self.mode, "alpha": self.alpha,
                "full_prefix_complete": True, "num_positions": len(self.seen),
                "pairs": dict(self.records),
            }
        finally:
            self.closed = True
            self.positions.clear()
            self.seen.clear()
            self.records.clear()
            self.reference = self.deviation = None


class InstallStateInterventionOp:
    """Attach temporary hooks after existing BeginSampleOp, before generate()."""

    def __init__(self, scope: dict, mode: str, reference_root: str | None = None,
                 deviation_root: str | None = None, alpha: float = 0.0, gate: dict | None = None):
        self.scope, self.mode = scope, mode
        self.reference_root, self.deviation_root = reference_root, deviation_root
        self.alpha, self.gate = alpha, gate

    def __call__(self, model) -> dict:
        from ..worker_hooks import _state, _get_positions, _parse_layer_inputs
        state = _state(model)
        if state.sample_key != self.scope["sample_key"] or state.world_size != 2:
            raise RuntimeError("begin the matching TP2 hook sample before intervention installation")
        if hasattr(model, SESSION_ATTR):
            raise RuntimeError("previous intervention was not flushed")
        def load(root):
            return None if root is None else torch.load(Path(root) / f"rank{state.rank}.pt", map_location="cpu", weights_only=False)
        session = StateInterventionSession(self.scope, state.rank, state.variant, self.mode,
                                           load(self.reference_root), load(self.deviation_root), self.alpha, self.gate)
        layer_idx, boundary = self.scope["layer"], self.scope["boundary"]
        layer = model.model.layers[layer_idx]
        def positions_hook(_module, args, kwargs):
            session.set_positions(state.sample_key, _get_positions(args, kwargs))
        def output_hook(_module, _args, _kwargs, output):
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError("production boundary must expose the canonical state pair")
            return session.apply_pair(state.sample_key, layer_idx, boundary, *output)
        def input_hook(_module, args, kwargs):
            pair = session.apply_pair(state.sample_key, layer_idx, boundary, *_parse_layer_inputs(args, kwargs))
            new_args, new_kwargs = list(args), dict(kwargs)
            for index, name, tensor in ((1, "hidden_states", pair[0]), (2, "residual", pair[1])):
                if len(new_args) > index:
                    new_args[index] = tensor
                else:
                    new_kwargs[name] = tensor
            return tuple(new_args), new_kwargs
        handles = [model.register_forward_pre_hook(positions_hook, with_kwargs=True, prepend=True)]
        try:
            if boundary == "layer_in":
                handles.append(layer.register_forward_pre_hook(input_hook, with_kwargs=True, prepend=True))
            else:
                module = layer if boundary == "layer_out" else layer.post_attention_layernorm
                handles.append(module.register_forward_hook(output_hook, with_kwargs=True, prepend=True))
        except BaseException:
            for handle in handles:
                handle.remove()
            raise
        setattr(model, SESSION_ATTR, session)
        setattr(model, HANDLES_ATTR, handles)
        return {"rank": state.rank, "scope": self.scope, "mode": self.mode, "num_handles": len(handles)}


class FlushStateInterventionOp:
    """Always remove temporary handles and clear references before ordinary flush."""

    def __init__(self, output_root: str):
        self.output_root = output_root

    def __call__(self, model) -> dict:
        session = getattr(model, SESSION_ATTR)
        try:
            payload = session.finish()
            path = Path(self.output_root) / f"rank{session.rank}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)
            return {k: v for k, v in payload.items() if k != "pairs"} | {"path": str(path)}
        finally:
            for handle in getattr(model, HANDLES_ATTR):
                handle.remove()
            delattr(model, HANDLES_ATTR)
            delattr(model, SESSION_ATTR)


def decision_metrics(logits: torch.Tensor, reference: torch.Tensor, target: int, competitor: int) -> dict:
    x, y = logits.double().reshape(-1), reference.double().reshape(-1)
    if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("raw finite full-vocabulary logits of matching shape are required")
    lp, lq = torch.log_softmax(y, 0), torch.log_softmax(x, 0)
    return {"top1": int(x.argmax()), "target_rank": int((x > x[target]).sum()) + 1,
            "target_competing_margin": float(x[target] - x[competitor]),
            "kl_to_e0": float((lp.exp() * (lp - lq)).sum()),
            "max_abs_delta": float((x - y).abs().max()), "l2_delta": float((x - y).norm())}


def make_identity_gate(scope: dict, kind: str, rank_rows: list[dict]) -> dict:
    """Noise floor is measured, never relaxed; caller supplies both rank tensors."""
    if kind not in {"noop", "alpha0"} or {row["rank"] for row in rank_rows} != {0, 1} or len(rank_rows) != 2:
        raise ValueError("identity gate requires noop/alpha0 and exactly two TP ranks")
    evidence = []
    for row in rank_rows:
        base, repeat, patched = (row[name].double().reshape(-1) for name in ("baseline", "repeat", "patched"))
        target = int(row["target"])
        b = decision_metrics(base, base, target, target)
        r = decision_metrics(repeat, base, target, target)
        p = decision_metrics(patched, base, target, target)
        passed = bool(row["forced_exact"] and row["full_prefix_complete"] and
                      r["top1"] == b["top1"] == p["top1"] and
                      r["target_rank"] == b["target_rank"] == p["target_rank"] and
                      p["max_abs_delta"] <= r["max_abs_delta"] and p["l2_delta"] <= r["l2_delta"])
        evidence.append({"rank": row["rank"], "repeat_noise_floor": r,
                         "patched_metrics": p, "passed": passed})
    return {"schema_version": 1, "scope": scope, "kind": kind, "ranks": [0, 1],
            "status": "PASS" if all(row["passed"] for row in evidence) else "STATE_RESET_BLOCKED",
            "evidence": evidence}


def main() -> None:
    from ..build_llm import build_real_vllm
    from ..forced_trajectory import make_forced_sampling_params
    from ..worker_hooks import BeginSampleOp, InstallHooksOp, flush_sample, remove_hooks
    from ...config import DEFAULT_MODEL_PATH, DEFAULT_PHASEA_ROOT
    from vllm.inputs import TokensPrompt
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["E0", "E1", "E2", "E3"], required=True)
    p.add_argument("--mode", choices=["capture", "noop", "reset", "inject"], required=True)
    p.add_argument("--sample_json", required=True)
    p.add_argument("--sample_key")
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--boundary", choices=BOUNDARIES, required=True)
    p.add_argument("--target_decode_index", type=int, required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--reference_root")
    p.add_argument("--deviation_root")
    p.add_argument("--identity_gate")
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--phasea_root", default=str(DEFAULT_PHASEA_ROOT))
    args = p.parse_args()
    payload = json.loads(Path(args.sample_json).read_text())
    if "samples" in payload:
        matches = [s for s in payload["samples"] if s["prompt_key"] == args.sample_key]
        if len(matches) != 1:
            raise ValueError("sample_key must select exactly one probe-plan sample")
        sample = matches[0]
    else:
        sample = payload
    scope = canonical_scope(sample, args.layer, args.boundary, args.target_decode_index)
    gate = None if args.identity_gate is None else json.loads(Path(args.identity_gate).read_text())
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    llm, runtime = build_real_vllm(args.variant, model_path=args.model_path, phasea_root=Path(args.phasea_root))
    llm.apply_model(InstallHooksOp(args.variant, str(root / "hooks")))
    llm.apply_model(BeginSampleOp(scope["sample_key"], scope["prompt_len"],
                                  {scope["target_abs_position"]: args.target_decode_index}))
    llm.apply_model(InstallStateInterventionOp(scope, args.mode, args.reference_root,
                                             args.deviation_root, args.alpha, gate))
    ids = [int(x) for x in sample["output_ids"][:args.target_decode_index + 1]]
    params = make_forced_sampling_params(ids, sample_key=scope["sample_key"], variant=args.variant,
                                         probe_decode_indices=[args.target_decode_index], logits_root=str(root / "raw_logits"))
    try:
        outputs = llm.generate([TokensPrompt(prompt_token_ids=sample["input_ids"])], [params], use_tqdm=False)
        generated = list(outputs[0].outputs[0].token_ids)
        if generated != ids:
            raise RuntimeError("STATE_RESET_BLOCKED: forced trajectory mismatch")
    finally:
        state_receipts = llm.apply_model(FlushStateInterventionOp(str(root / "states")))
    if {r["rank"] for r in state_receipts} != {0, 1}:
        raise RuntimeError("state capture must contain both TP ranks")
    hook_receipts = llm.apply_model(flush_sample)
    llm.apply_model(remove_hooks)
    manifest = {"schema_version": 1, "scope": scope, "variant": args.variant, "mode": args.mode,
                "alpha": args.alpha, "runtime": runtime, "forced_exact": True,
                "generated_ids": generated, "states": state_receipts, "hooks": hook_receipts,
                "raw_logits_root": str(root / "raw_logits"), "identity_gate": args.identity_gate}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    if not __package__:
        __package__ = "Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion"
    main()
