from __future__ import annotations

CORE_BOUNDARIES = (
    "layer_in",
    "input_norm",
    "attention_core",
    "o_proj",
    "post_attn_norm",
    "router_logits",
    "moe_out",
    "layer_out",
    "final_norm",
    "raw_logits",
)

# Drill-down levels reuse core captures on selected frontiers only
# (see run_drilldown.py). No full-model Q/K/V or per-expert tensors.
ATTENTION_DRILLDOWN_BOUNDARIES = (
    "attention_core",
    "o_proj",
)

MOE_DRILLDOWN_BOUNDARIES = (
    "post_attn_norm",  # fused MoE input (mlp input after post-attn norm)
    "router_logits",
    "moe_out",  # fused MoE output
)


def predictor_abs_position(prompt_len: int, decode_index: int) -> int:
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    if decode_index < 0:
        raise ValueError(f"decode_index must be non-negative, got {decode_index}")
    return int(prompt_len) + int(decode_index) - 1


def build_probe_map(prompt_len: int, positions: list[dict]) -> dict[int, int]:
    result: dict[int, int] = {}
    for row in positions:
        decode_index = int(row["decode_index"])
        abs_pos = predictor_abs_position(prompt_len, decode_index)
        if abs_pos in result:
            raise ValueError(f"duplicate abs position {abs_pos}")
        result[abs_pos] = decode_index
    return result
