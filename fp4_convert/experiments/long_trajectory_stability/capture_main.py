#!/usr/bin/env python3
"""Run the repository lighteval/vLLM entrypoint while retaining exact token trajectories.

This wrapper intentionally does not modify ``main.py``.  It only replaces the
result tracker in this process so detail JSON contains raw text plus exact input
and generated token ids.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as root_main  # noqa: E402


def coerce_unrestricted_topk(top_k: int | None) -> int | None:
    """Map vLLM's unrestricted sentinel ``top_k=-1`` to lighteval's ``None``.

    ``lighteval.GenerationParameters.top_k`` is ``NonNegativeInt | None``.
    Passing ``-1`` through ``main.py`` therefore fails pydantic validation.
    ``None`` omits the field; vLLM then uses its unrestricted default.
    With greedy ``temperature=0``, vLLM stores ``top_k=0`` regardless.
    """
    if top_k is not None and int(top_k) < 0:
        return None
    return top_k


def install_lighteval_topk_adapter() -> None:
    from lighteval.models.model_input import GenerationParameters

    if getattr(GenerationParameters, "_hif4_topk_adapter", False):
        return
    orig_init = GenerationParameters.__init__

    def _init(self, *args, **kwargs):
        if "top_k" in kwargs:
            kwargs["top_k"] = coerce_unrestricted_topk(kwargs["top_k"])
        return orig_init(self, *args, **kwargs)

    GenerationParameters.__init__ = _init
    GenerationParameters._hif4_topk_adapter = True


class RawTrajectoryEvaluationTracker(root_main.CustomEvaluationTracker):
    def _filter_detail_record(self, record: dict) -> dict:
        out = super()._filter_detail_record(record)
        model_response = record.get("model_response") or {}
        target = out.setdefault("model_response", {})
        target.update(
            {
                "text": model_response.get("text"),
                "input_tokens": model_response.get("input_tokens"),
                "output_tokens": model_response.get("output_tokens"),
                "reasonings": model_response.get("reasonings"),
            }
        )
        return out


if __name__ == "__main__":
    install_lighteval_topk_adapter()
    root_main.CustomEvaluationTracker = RawTrajectoryEvaluationTracker
    root_main.main()
