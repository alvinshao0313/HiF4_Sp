"""Metadata contracts using the real pinned source; no synthetic checkpoint."""
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import Qwen3MoeModelSpec
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.moe_materialize import _copy_tokenizer_assets
from Native_NVFP4_HiF4_Linear_Puncture.experiments.non_equivalent_reconstruction.artifact import KIND, SCHEMA_VERSION, load_manifest, write_json
from .config import diag_config, group_config, source_snapshot
from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.core.moe_model_spec import QWEN3_30B_A3B_NVFP4
from Native_NVFP4_HiF4_Linear_Puncture.src.checkpoint import resolve_local_snapshot


def test_real_235b_geometry():
    spec = Qwen3MoeModelSpec.from_snapshot(source_snapshot())
    assert (spec.num_layers, spec.hidden_size, spec.moe_intermediate_size) == (94, 4096, 1536)
    assert spec.target_linear_count == 36472
    assert spec.source_model == str(source_snapshot())


def test_48_layer_artifact_is_not_a_complete_235b_model(tmp_path):
    write_json({"kind": KIND, "schema_version": SCHEMA_VERSION,
                "source_snapshot": str(source_snapshot()), "completed_layers": list(range(48))},
               tmp_path / "manifest.json")
    with pytest.raises(ValueError, match="94 completed layers"):
        load_manifest(tmp_path, require_complete=True)


def test_recipe_preserves_original_e4_conditions(tmp_path):
    original = Path(__file__).resolve().parents[2] / "results/e2e_diag_reconstruction/phaseA_refactor_20260825T035730Z/E4_fusable_r64/config.json"
    old = json.loads(original.read_text())
    new = diag_config(tmp_path).to_dict()
    changed_for_model = {"model_path", "output_dir", "num_layers", "end_layer", "calib_cache_dir"}
    assert {k: v for k, v in new.items() if k not in changed_for_model} == {
        k: old[k] for k in new if k not in changed_for_model}
    group = group_config(tmp_path)
    assert (group.matrix_sharing, group.router_loss, group.batch_size, group.epochs) == ("group", "top_mass", 4, 20)
    smoke = diag_config(tmp_path, smoke=True)
    assert smoke.diag_batch_size == 4 and smoke.num_layers == 94 and smoke.end_layer == 0


@pytest.mark.parametrize("model", ["235b", "30b"])
def test_export_preserves_source_template_and_prompt_tokens(tmp_path, model):
    source = source_snapshot() if model == "235b" else Path(resolve_local_snapshot(QWEN3_30B_A3B_NVFP4))
    _copy_tokenizer_assets(source, tmp_path)
    original = AutoTokenizer.from_pretrained(source, local_files_only=True)
    exported = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert exported.chat_template == original.chat_template
    messages = [{"role": "user", "content": "Compute 7 + 9."}]
    for thinking in (True, False):
        kwargs = {"add_generation_prompt": True, "enable_thinking": thinking}
        assert exported.apply_chat_template(messages, **kwargs) == original.apply_chat_template(messages, **kwargs)


def test_tp8_preserves_full_protocol_and_reaches_both_evaluation_engines():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.protocol import protocol_for
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.evaluate import reasoning_command
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.e2e_diag_reconstruction.evaluation.lm_eval_vllm import build_lm_eval_vllm_kwargs
    protocol = protocol_for("mmlu_pro", tensor_parallel_size=8)
    assert protocol["count"] == 12032 and protocol["max_samples"] is None
    assert {**protocol, "tensor_parallel_size": 2} == protocol_for("mmlu_pro")
    cmd = reasoning_command(source_snapshot(), Path("unused"), protocol, "native_nvfp4", None)
    assert cmd[cmd.index("--tensor_parallel_size") + 1] == "8" and "--max_samples" not in cmd
    kwargs = build_lm_eval_vllm_kwargs(model_path=str(source_snapshot()), native_nvfp4=True, tensor_parallel_size=8)
    assert kwargs["tensor_parallel_size"] == 8 and kwargs["kv_cache_dtype"] == "bfloat16"


def test_235b_embedded_template_hash_matches_export(tmp_path):
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.reporting import tokenizer_asset_hashes
    _copy_tokenizer_assets(source_snapshot(), tmp_path)
    source = AutoTokenizer.from_pretrained(source_snapshot(), local_files_only=True)
    exported = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert tokenizer_asset_hashes(source_snapshot(), source) == tokenizer_asset_hashes(tmp_path, exported)


def test_existing_tp2_result_cannot_be_relabelled_as_tp8():
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.downstream_accuracy_recovery.reporting import validate_metrics
    root = Path(__file__).resolve().parents[2] / "results/downstream_accuracy_recovery/paired_full_20260930T131000Z"
    result = json.loads((root / "formal/native_nvfp4/eval/arc/metrics.json").read_text())
    assert set(validate_metrics(result, "arc", smoke=False, model_kind="native_nvfp4")) == {"ARC-E", "ARC-C"}
    with pytest.raises(ValueError, match="wrong evaluation protocol"):
        validate_metrics(result, "arc", smoke=False, model_kind="native_nvfp4", tensor_parallel_size=8)


def test_layer_source_preflight_requires_complete_real_shards_but_not_unread_layers(tmp_path):
    source = resolve_local_snapshot(QWEN3_30B_A3B_NVFP4)
    index_file = source / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())["weight_map"]
    prefixes = ("model.embed_tokens.", "model.layers.0.")
    required = {shard for key, shard in index.items() if key.startswith(prefixes)}
    assert required and required < set(index.values())
    (tmp_path / index_file.name).symlink_to(index_file)
    for shard in required:
        (tmp_path / shard).symlink_to(source / shard)
    assert resolve_local_snapshot(str(tmp_path), tensor_prefixes=prefixes) == tmp_path
    with pytest.raises(SystemExit):
        resolve_local_snapshot(str(tmp_path))
    with pytest.raises(ValueError, match="absent from checkpoint index"):
        resolve_local_snapshot(str(tmp_path), tensor_prefixes=("model.layers.999.",))
    with pytest.raises(ValueError, match="nonempty prefixes"):
        resolve_local_snapshot(str(tmp_path), tensor_prefixes=())
    # Removing a link in the isolated test view leaves the real checkpoint intact.
    (tmp_path / sorted(required)[0]).unlink()
    with pytest.raises(SystemExit):
        resolve_local_snapshot(str(tmp_path), tensor_prefixes=prefixes)
