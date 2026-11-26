"""Run the agreed two-sample, all-layer TP2 Ban qualification."""
from __future__ import annotations

import copy
from pathlib import Path

from .artifact import write_json
from .calibrate import calibrate, parser
from .data import sample_c4
from .diagnose import diagnose, generate
from .engine import configure, create_engine, snapshot


def qualify(args):
    if (args.phase, args.nsamples, args.seqlen, args.seed, args.tensor_parallel_size) != ("diagnostic", 2, 2048, 0, 2):
        raise ValueError("qualification uses diagnostic C4 2x2048, seed 0, TP2")
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    samples, data = sample_c4(args.model_path, args.c4_arrow, args.nsamples, args.seqlen, args.seed)
    engine = create_engine(args.model_path, args.hif4_runtime_spec, args.tensor_parallel_size)
    diagnosis = diagnose(engine, samples[0], root)
    calibration_args = copy.copy(args)
    calibration_args.output = root / "calibration"
    engine, artifact, samples = calibrate(calibration_args, engine=engine, samples=samples, data=data)
    engine.collective_rpc("ban_load_artifact", args=(str(root / "calibration/c4_ban.pt"),))
    ids = samples[0]
    checks, observations = {}, {}
    configure(engine, mode="full", capture="reference", execution="masked")
    full = generate(engine, ids)
    observations["reference_top8"] = snapshot(engine)
    configure(engine, mode="ban", k_min=8, capture="compare", execution="compact")
    unpruned = generate(engine, ids)
    observations["ban_top8"] = snapshot(engine)
    checks["source_router_unpruned"] = observations["ban_top8"]["capture"]
    checks["unpruned_token_equal"] = full[0].outputs[0].token_ids == unpruned[0].outputs[0].token_ids
    configure(engine, mode="ban", k_min=3, capture="reference", execution="masked")
    masked = generate(engine, ids)
    observations["masked"] = snapshot(engine)
    configure(engine, mode="ban", k_min=3, capture="compare", execution="compact")
    compact = generate(engine, ids)
    observations["compact"] = snapshot(engine)
    checks["compact_vs_masked"] = observations["compact"]["capture"]
    checks["compact_token_equal"] = masked[0].outputs[0].token_ids == compact[0].outputs[0].token_ids
    configure(engine, mode="ban", k_min=3, capture=None, execution="compact")
    generation = generate(engine, ids, max_tokens=32)
    observations["decode"] = snapshot(engine)
    checks["generated_token_ids"] = generation[0].outputs[0].token_ids
    checks["average_experts"] = observations["decode"]["average_experts"]
    checks["physical_dispatch_matches_selection"] = observations["decode"]["selected"] == observations["decode"]["dispatched"]
    checks["prefill_decode_tokens_per_layer"] = observations["decode"]["tokens"]
    checks["saved_expert_routes_fraction"] = 1 - sum(observations["decode"]["dispatched"]) / (8 * sum(observations["decode"]["tokens"]))
    checks["native_original_identity"] = diagnosis["native_route_identity"]["capture"]
    checks["source_vs_native_router"] = diagnosis["torch_top8"]["capture"]
    checks["passed"] = (checks["unpruned_token_equal"] and checks["compact_token_equal"]
                         and checks["source_router_unpruned"]["relative_l2"] == 0
                         and checks["native_original_identity"]["relative_l2"] == 0
                         and checks["compact_vs_masked"]["relative_l2"] <= 0.002
                         and checks["compact_vs_masked"]["kl"] <= 1e-4
                         and checks["physical_dispatch_matches_selection"]
                         and checks["prefill_decode_tokens_per_layer"] == [2079] * 48
                         and 3 <= checks["average_experts"] < 8)
    write_json(root / "qualification.json", checks)
    write_json(root / "routing.json", observations)
    if not checks["passed"]:
        raise RuntimeError(f"Ban qualification failed: {checks}")
    print(f"BAN_QUALIFIED average_experts={checks['average_experts']:.4f} saved_routes={checks['saved_expert_routes_fraction']:.4%}", flush=True)


if __name__ == "__main__":
    qualify(parser().parse_args())
