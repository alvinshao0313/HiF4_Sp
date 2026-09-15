"""Build unbiased WikiText2 + s1k_original internal-error state cohort."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch

from .config import (
    COHORT_SEED,
    DEFAULT_MODEL_PATH,
    DISCOVERY_PER_SOURCE,
    HOLDOUT_PER_SOURCE,
    MIN_CALIBRATION_LENGTH,
    PREFIX_LENGTHS_J,
    S1K_SHARED_CALIB,
    WIKITEXT2_SHARED_CALIB,
)
from .run_state import atomic_write_json, write_jsonl


def _sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def _load_shared_split(root: Path, split: str) -> list[Any]:
    path = root / "calibration" / f"{split}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def ensure_wikitext2_shared_calibration(*, model_path: str = DEFAULT_MODEL_PATH) -> Path:
    """Build shared WikiText2 cache if missing (same recipe as e2e shared_calib)."""
    root = WIKITEXT2_SHARED_CALIB
    train_pt = root / "calibration" / "train.pt"
    if train_pt.is_file():
        return root
    from transformers import AutoTokenizer

    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.config import (
        E2ETrainConfig,
    )
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.data.calibration import (
        build_or_load_calibration,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cfg = E2ETrainConfig.for_test(
        calib_source="wikitext2",
        calib_nsamples=128,
        calib_val_nsamples=32,
        calib_seed=42,
        calib_seqlen=1024,
    )
    # for_test may override; force the shared recipe fields.
    cfg.calib_source = "wikitext2"
    cfg.calib_nsamples = 128
    cfg.calib_val_nsamples = 32
    cfg.calib_seed = 42
    cfg.calib_seqlen = 1024
    root.mkdir(parents=True, exist_ok=True)
    build_or_load_calibration(cfg, tokenizer, native_model=None, out_dir=root)
    if not train_pt.is_file():
        raise RuntimeError(f"failed to materialize WikiText2 shared calibration at {train_pt}")
    return root


def _eligible(samples: list[Any], source: str) -> list[dict[str, Any]]:
    rows = []
    for s in samples:
        ids = [int(x) for x in s.input_ids.tolist()]
        if len(ids) < MIN_CALIBRATION_LENGTH:
            continue
        rows.append(
            {
                "calibration_sample_id": str(s.sample_id),
                "source": source,
                "source_index": s.source_index,
                "input_ids": ids,
                "length": len(ids),
                "input_ids_sha256": _sha256_ids(ids),
            }
        )
    if len(rows) < DISCOVERY_PER_SOURCE + HOLDOUT_PER_SOURCE:
        raise RuntimeError(
            f"source={source} has only {len(rows)} samples with length>={MIN_CALIBRATION_LENGTH}"
        )
    return rows


def select_cohort_samples(*, seed: int = COHORT_SEED, model_path: str = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    ensure_wikitext2_shared_calibration(model_path=model_path)
    s1k = _eligible(_load_shared_split(S1K_SHARED_CALIB, "train"), "s1k_original")
    wiki = _eligible(_load_shared_split(WIKITEXT2_SHARED_CALIB, "train"), "wikitext2")
    rng = random.Random(int(seed))
    wiki_pick = rng.sample(wiki, DISCOVERY_PER_SOURCE + HOLDOUT_PER_SOURCE)
    s1k_pick = rng.sample(s1k, DISCOVERY_PER_SOURCE + HOLDOUT_PER_SOURCE)
    discovery = wiki_pick[:DISCOVERY_PER_SOURCE] + s1k_pick[:DISCOVERY_PER_SOURCE]
    holdout = wiki_pick[DISCOVERY_PER_SOURCE:] + s1k_pick[DISCOVERY_PER_SOURCE:]
    return {
        "seed": int(seed),
        "discovery": discovery,
        "holdout": holdout,
        "prefix_lengths_j": list(PREFIX_LENGTHS_J),
        "min_calibration_length": MIN_CALIBRATION_LENGTH,
        "sources": {
            "wikitext2": str(WIKITEXT2_SHARED_CALIB),
            "s1k_original": str(S1K_SHARED_CALIB),
        },
    }


def expand_states(selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("discovery", "holdout"):
        for sample in selection[split_name]:
            ids = sample["input_ids"]
            for j in selection["prefix_lengths_j"]:
                j = int(j)
                if len(ids) <= j:
                    raise RuntimeError(f"sample {sample['calibration_sample_id']} too short for j={j}")
                sample_key = f"{sample['calibration_sample_id']}__j{j}"
                rows.append(
                    {
                        "sample_key": sample_key,
                        "calibration_sample_id": sample["calibration_sample_id"],
                        "source": sample["source"],
                        "source_index": sample["source_index"],
                        "calibration_input_ids_sha256": sample["input_ids_sha256"],
                        "calibration_length": sample["length"],
                        "fixed_history_source": "calibration_tokens",
                        "prefix_length_j": j,
                        "target_token_id": int(ids[j]),
                        "abs_position": j - 1,
                        "split": split_name,
                        "prompt_token_ids": ids[:1],
                        "forced_token_ids": ids[1 : j + 1],
                        "decode_index": j - 1,
                        "full_calibration_token_ids": ids,
                    }
                )
    return rows


def build_and_write_cohort(protocol_dir: Path, *, seed: int = COHORT_SEED, model_path: str = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    protocol_dir = Path(protocol_dir)
    protocol_dir.mkdir(parents=True, exist_ok=True)
    selection = select_cohort_samples(seed=seed, model_path=model_path)
    # Persist sample IDs without dumping full token lists into meta; states jsonl keeps tokens.
    meta = {
        "seed": selection["seed"],
        "prefix_lengths_j": selection["prefix_lengths_j"],
        "min_calibration_length": selection["min_calibration_length"],
        "sources": selection["sources"],
        "discovery_sample_ids": [s["calibration_sample_id"] for s in selection["discovery"]],
        "holdout_sample_ids": [s["calibration_sample_id"] for s in selection["holdout"]],
        "discovery_sources": [s["source"] for s in selection["discovery"]],
        "holdout_sources": [s["source"] for s in selection["holdout"]],
        "n_discovery_samples": len(selection["discovery"]),
        "n_holdout_samples": len(selection["holdout"]),
        "n_states": (len(selection["discovery"]) + len(selection["holdout"])) * len(selection["prefix_lengths_j"]),
        "forbidden": [
            "first_divergence",
            "trajectory_survival",
            "mmlu_pro",
            "livecodebench",
            "e0_free_continuation",
        ],
    }
    states = expand_states(selection)
    if len(states) != meta["n_states"]:
        raise RuntimeError("state expansion size mismatch")
    # Strip bulky full ids from on-disk optional? Plan requires calibration tokens as history —
    # keep them; they are the immutable forced history.
    write_jsonl(protocol_dir / "internal_error_state_cohort.jsonl", states)
    atomic_write_json(protocol_dir / "internal_error_state_cohort.meta.json", meta)
    return {"meta": meta, "n_states": len(states), "path": str(protocol_dir / "internal_error_state_cohort.jsonl")}
