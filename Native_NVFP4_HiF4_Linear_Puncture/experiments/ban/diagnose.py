"""Reproduce and isolate the earlier no-pruning discrepancy before calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

from .artifact import write_json
from .engine import configure, create_engine, snapshot


def generate(engine, ids, max_tokens=1):
    from vllm import SamplingParams
    return engine.generate([{"prompt_token_ids": ids.tolist()}],
                           SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=True,
                                          prompt_logprobs=None), use_tqdm=False)


def diagnose(engine, ids, output):
    result = {"sidecar": engine.collective_rpc("ban_context")}
    configure(engine, mode="native", capture="reference", probe=True)
    generate(engine, ids)
    result["native_reference"] = snapshot(engine)
    configure(engine, mode="native", capture="compare")
    generate(engine, ids)
    result["native_repeat"] = snapshot(engine)
    configure(engine, mode="identity", capture="compare", execution="compact")
    generate(engine, ids)
    result["native_route_identity"] = snapshot(engine)
    configure(engine, mode="full", capture="compare", execution="masked")
    generate(engine, ids)
    result["torch_top8"] = snapshot(engine)
    write_json(Path(output) / "diagnosis.json", result)
    repeat = result["native_repeat"]["capture"]
    print("BAN_DIAGNOSIS", {"native_repeat": repeat,
          "torch_top8": result["torch_top8"]["capture"],
          "sidecar": result["sidecar"]}, flush=True)
    identity = result["native_route_identity"]["capture"]
    if repeat is None or repeat["relative_l2"] != 0 or identity is None or identity["relative_l2"] != 0:
        raise RuntimeError("native repeated forward is not identical; diagnose before calibration")
    return result


def main():
    from .data import sample_c4
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--hif4_runtime_spec", required=True)
    parser.add_argument("--c4_arrow", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    ids, data = sample_c4(args.model_path, args.c4_arrow, 2, 2048, 0)
    write_json(output / "data.json", data)
    engine = create_engine(args.model_path, args.hif4_runtime_spec)
    diagnose(engine, ids[0], output)


if __name__ == "__main__":
    main()
