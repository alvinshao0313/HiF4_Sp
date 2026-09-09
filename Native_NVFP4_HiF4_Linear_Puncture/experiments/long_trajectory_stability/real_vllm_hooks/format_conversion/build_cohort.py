"""Read the formal sampled benchmark and select an auditable matched cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.trajectory_io import prompt_key, write_jsonl

VARIANT_DIRS = {"E0": "E0_native_nvfp4", "E1": "E1_direct_hif4", "E2": "E2_r64_only", "E3": "E3_fusable"}
FORMAL_PROTOCOL = {"dataset": "lcb:codegeneration_v6|0", "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                   "min_p": 0.0, "max_new_tokens": 38912, "tensor_parallel_size": 2,
                   "kv_cache_dtype": "bfloat16", "enforce_eager": True, "max_model_length": 40960}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_artifacts(phasea_root: Path) -> tuple[dict, dict]:
    """Reject ambiguous result files or mismatched formal generation protocols."""
    details, manifests = {}, {}
    for variant, directory in VARIANT_DIRS.items():
        root = Path(phasea_root) / directory / "eval/livecodebench"
        paths = sorted(root.glob("**/details_lcb:codegeneration_v6|0_*.json"))
        results = sorted(root.glob("**/results_*.json"))
        if len(paths) != 1 or len(results) != 1:
            raise ValueError(f"ambiguous or absent formal artifacts for {variant}: {paths}, {results}")
        data = json.loads(results[0].read_text())
        config = data["config_general"]["model_config"]
        protocol = {key: config["generation_parameters"][key] if key in config["generation_parameters"] else config[key]
                    for key in FORMAL_PROTOCOL if key != "dataset"}
        tasks = [key for key in data["results"] if key == FORMAL_PROTOCOL["dataset"]]
        protocol["dataset"] = tasks[0] if len(tasks) == 1 else None
        if protocol != FORMAL_PROTOCOL:
            raise ValueError(f"FORMAL_PROTOCOL_MISMATCH {variant}: {protocol}")
        details[variant] = json.loads(paths[0].read_text())
        manifests[variant] = {"details": str(paths[0].resolve()), "details_sha256": sha256_file(paths[0]),
                              "results": str(results[0].resolve()), "results_sha256": sha256_file(results[0]),
                              "protocol": protocol, "enable_thinking_config": config.get("enable_thinking"),
                              "max_num_seqs": config.get("max_num_seqs"),
                              "max_num_batched_tokens": config.get("max_num_batched_tokens")}
    return details, manifests


def align_formal_details(details_by_variant: dict) -> dict[str, dict[str, dict]]:
    if set(details_by_variant) != set(VARIANT_DIRS):
        raise ValueError("formal matrix requires exactly E0/E1/E2/E3")
    aligned = {}
    for variant, rows in details_by_variant.items():
        index = {str(row["doc"]["id"]): row for row in rows}
        if len(index) != len(rows):
            raise ValueError(f"duplicate formal doc IDs: {variant}")
        aligned[variant] = index
    ids = set(aligned["E0"])
    if any(set(index) != ids for index in aligned.values()):
        raise ValueError("formal doc ID sets differ across variants")
    for doc_id in ids:
        e0 = aligned["E0"][doc_id]
        for variant in VARIANT_DIRS:
            other = aligned[variant][doc_id]
            if other["doc"]["specific"] != e0["doc"]["specific"]:
                raise ValueError(f"formal judge payload mismatch: {variant}/{doc_id}")
    return aligned


def enrich_formal_from_raw_caches(details_by_variant: dict, cache_paths: list[Path]) -> dict:
    """Accept cached raw output only after all doc IDs, prompts and judged texts match."""
    import pyarrow.parquet as pq
    from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.judge_isolated_lcb import remove_reasoning_tags
    aligned = align_formal_details(details_by_variant)
    matches = {variant: [] for variant in VARIANT_DIRS}
    for path in cache_paths:
        if "E4" in str(path):
            raise ValueError("E4 is excluded from this experiment")
        table = pq.read_table(path).to_pylist()
        cached = {str(row["sample_id"]): json.loads(row["sample"]) if isinstance(row["sample"], str) else row["sample"] for row in table}
        if len(cached) != len(table):
            raise ValueError(f"duplicate cache sample IDs: {path}")
        for variant, index in aligned.items():
            if set(cached) != set(index):
                continue
            exact = all(cached[doc_id]["input"] == item["model_response"]["input"] and
                        [remove_reasoning_tags(text, [("<think>", "</think>")]) for text in cached[doc_id]["text"]] == item["model_response"]["text_post_processed"]
                        for doc_id, item in index.items())
            if exact:
                matches[variant].append({"path": str(path.resolve()), "sha256": sha256_file(path), "matched_docs": len(index)})
                for doc_id, item in index.items():
                    raw = cached[doc_id]
                    item["model_response"].update({field: raw[field] for field in ("text", "input_tokens", "output_tokens")})
    if any(len(value) > 1 for value in matches.values()):
        raise ValueError(f"multiple exact formal raw cache lineages: {matches}")
    return matches


def _one_generation(value):
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("expected exactly one formal generation")
        return value[0]
    return value


def formal_generation_metadata(response: dict, tokenizer=None) -> dict:
    """Never infer raw length or thinking completion from postprocessed text."""
    ids = response.get("output_tokens")
    if ids is not None and ids and isinstance(ids[0], list):
        ids = _one_generation(ids)
    raw = _one_generation(response.get("text"))
    if ids:
        if tokenizer is None:
            raise ValueError("tokenizer is required to inspect raw formal output IDs")
        end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        from Native_NVFP4_HiF4_Linear_Puncture.experiments.long_trajectory_stability.real_vllm_hooks.format_conversion.judge_isolated_lcb import contains_subsequence
        finished = contains_subsequence(ids, end_ids)
        return {"generation_len": len(ids), "generation_len_source": "raw_output_token_ids",
                "finished_thinking": finished, "finished_thinking_source": "raw_output_token_ids"}
    if raw is not None:
        return {"generation_len": len(tokenizer.encode(raw, add_special_tokens=False)) if tokenizer else None,
                "generation_len_source": "raw_text_retokens" if tokenizer else "unknown",
                "finished_thinking": "</think>" in raw, "finished_thinking_source": "raw_text"}
    return {"generation_len": None, "generation_len_source": "unknown_raw_generation_not_saved",
            "finished_thinking": "unknown", "finished_thinking_source": "unknown_raw_generation_not_saved"}


def build_formal_matrix(details_by_variant: dict, tokenizer=None) -> tuple[list[dict], dict]:
    aligned = align_formal_details(details_by_variant)
    rows = []
    for doc_id in sorted(aligned["E0"], key=int):
        row = {"doc_id": doc_id, "label_protocol": "formal_sampled_temperature_0.6"}
        for variant in VARIANT_DIRS:
            item = aligned[variant][doc_id]
            row[f"{variant}_prompt_equal_E0"] = item["model_response"]["input"] == aligned["E0"][doc_id]["model_response"]["input"]
            metric = float(item["metric"]["codegen_pass@1:16"])
            if metric not in (0.0, 1.0):
                raise ValueError(f"nonbinary pass@1 for {variant}/{doc_id}: {metric}")
            row[f"{variant}_pass"] = metric == 1.0
            row.update({f"{variant}_{key}": value for key, value in formal_generation_metadata(item["model_response"], tokenizer).items()})
        row["formal_group"] = ("formal_regression" if not row["E1_pass"] else "formal_robust") if row["E0_pass"] else ("formal_e1_only" if row["E1_pass"] else "formal_both_fail")
        for variant in ("E2", "E3"):
            row[f"{variant}_formal_observed_recovery"] = row["formal_group"] == "formal_regression" and row[f"{variant}_pass"]
        rows.append(row)
    summary = {"schema_version": 1, "n_tasks": len(rows), "protocol": FORMAL_PROTOCOL,
               "group_counts": dict(Counter(row["formal_group"] for row in rows)),
               "pass_counts": {variant: sum(row[f"{variant}_pass"] for row in rows) for variant in VARIANT_DIRS},
               "observed_recovery_doc_ids": {variant: [row["doc_id"] for row in rows if row[f"{variant}_formal_observed_recovery"]] for variant in ("E2", "E3")},
               "raw_length_available_counts": {variant: sum(row[f"{variant}_generation_len"] is not None for row in rows) for variant in VARIANT_DIRS},
               "prompt_equal_E0_counts": {variant: sum(row[f"{variant}_prompt_equal_E0"] for row in rows) for variant in VARIANT_DIRS},
               "FORMAL_PROMPT_PROTOCOL_ALIGNED": all(row[f"{variant}_prompt_equal_E0"] for row in rows for variant in VARIANT_DIRS),
               "observed_recovery_interpretation": "single_sample_observation_not_causal_rescue"}
    return rows, summary


def _require_lengths(rows: list[dict]) -> None:
    missing = [str(row["doc_id"]) for row in rows if row.get("E0_generation_len") is None]
    if missing:
        raise ValueError(f"FORMAL_E0_GENERATION_LENGTH_UNAVAILABLE: {missing}; postprocessed answer length cannot replace raw trajectory length")


def select_regressions(matrix: list[dict], count: int = 16, excluded_ids=()) -> list[dict]:
    candidates = [row for row in matrix if row["formal_group"] == "formal_regression" and str(row["doc_id"]) not in set(map(str, excluded_ids))]
    _require_lengths(candidates)
    ordered = sorted(candidates, key=lambda row: (row["E0_generation_len"], int(row["doc_id"])))
    selected = [row for row in ordered if str(row["doc_id"]) == "58"][:count]
    ordered = [row for row in ordered if row not in selected]
    n = min(max(count - len(selected), 0), len(ordered))
    if n == 1:
        selected.append(ordered[(len(ordered) - 1) // 2])
    elif n > 1:
        # Integer nearest quantile, with explicit half-up ties: deterministic on every platform.
        selected.extend(ordered[(2 * i * (len(ordered) - 1) + n - 1) // (2 * (n - 1))] for i in range(n))
    return selected


def match_robust(regressions: list[dict], robust_candidates: list[dict], prompt_lengths: dict[str, int]) -> list[tuple[dict, dict]]:
    _require_lengths(regressions + robust_candidates)
    available = {str(row["doc_id"]): row for row in robust_candidates}
    pairs = []
    for regression in regressions:
        if not available:
            break
        reg_id = str(regression["doc_id"])
        robust_id = min(available, key=lambda doc_id: (abs(available[doc_id]["E0_generation_len"] - regression["E0_generation_len"]),
                                                     abs(prompt_lengths[doc_id] - prompt_lengths[reg_id]), int(doc_id)))
        pairs.append((regression, available.pop(robust_id)))
    return pairs


def _tokenize_e0_prompt_rows(e0_details: list[dict], tokenizer, existing_rows: list[dict] | None = None) -> dict[str, dict]:
    by_id = {str(item["doc"]["id"]): item for item in e0_details}
    saved = {str(row["doc_id"]): row for row in existing_rows or []}
    prompt_rows = {}
    for doc_id, item in by_id.items():
        text = item["model_response"]["input"]
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if doc_id in saved:
            if saved[doc_id]["prompt_text_sha256"] != text_hash:
                raise ValueError(f"saved prompt changed: {doc_id}")
            ids = saved[doc_id]["input_ids"]
        else:
            ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        if not ids:
            raise ValueError(f"empty prompt: {doc_id}")
        prompt_rows[doc_id] = {"prompt_key": prompt_key(ids), "doc_id": doc_id, "input_ids": ids,
                               "prompt_text_sha256": text_hash, "prompt_source": "phaseA_formal_e0_text_retokens_once",
                               "output_ids": [], "output_len": 0, "raw_text": "", "gold": item.get("gold"),
                               "specific": {**item["doc"]["specific"], "task": "livecodebench"}}
    return prompt_rows


def _pair_cohort(selected: list[dict], robust_candidates: list[dict], prompt_rows: dict[str, dict],
                 *, formal_prompt_protocol_aligned: bool, role: str) -> tuple[list[dict], dict]:
    lengths = {doc_id: len(row["input_ids"]) for doc_id, row in prompt_rows.items()}
    pairs = match_robust(selected, robust_candidates, lengths)
    cohort = []
    for pair_index, (regression, robust) in enumerate(pairs):
        for own, other in ((regression, robust), (robust, regression)):
            cohort.append({**prompt_rows[str(own["doc_id"])], "formal_group": own["formal_group"],
                           "matched_doc_id": str(other["doc_id"]), "pair_index": pair_index,
                           "E0_generation_len": own["E0_generation_len"],
                           "cohort_role": role,
                           "formal_format_only_regression": bool(formal_prompt_protocol_aligned and
                                                                own["formal_group"] == "formal_regression")})
    if len({row["prompt_key"] for row in cohort}) != len(cohort):
        raise ValueError("duplicate mechanism prompt keys")
    manifest = {"schema_version": 1, "n_samples": len(cohort), "n_pairs": len(pairs),
                "selection": "doc58_plus_uniform_E0_raw_generation_length_quantiles",
                "matching": ["E0_raw_generation_length_distance", "prompt_token_length_distance", "doc_id"],
                "doc_ids": [row["doc_id"] for row in cohort], "tokenized_prompt_rows": list(prompt_rows.values()),
                "formal_prompt_ids_exact": False,
                "FORMAL_PROMPT_PROTOCOL_ALIGNED": formal_prompt_protocol_aligned,
                "cohort_role": role,
                "mechanism_prompt_policy": "all_variants_reuse_identical_E0_chat_wrapped_input_ids"}
    return cohort, manifest


def build_benchmark_cohort(matrix: list[dict], e0_details: list[dict], tokenizer, count: int = 16,
                           existing_rows: list[dict] | None = None) -> tuple[list[dict], dict]:
    """Tokenize every formal E0 prompt once; expansion reuses saved prompt token IDs."""
    if any(not row[f"{variant}_prompt_equal_E0"] for row in matrix for variant in VARIANT_DIRS):
        raise ValueError("FORMAL_PROMPT_PROTOCOL_MISMATCH: formal regression cannot be treated as a format-only cohort")
    prompt_rows = _tokenize_e0_prompt_rows(e0_details, tokenizer, existing_rows)
    selected = select_regressions(matrix, count)
    return _pair_cohort(selected, [row for row in matrix if row["formal_group"] == "formal_robust"], prompt_rows,
                        formal_prompt_protocol_aligned=True, role="formal_aligned_benchmark_cohort")


def verify_e0_wraps_other_questions(details_by_variant: dict) -> dict:
    """Confirm E0 chat wrapper is exactly the same question body used by E1/E2/E3."""
    aligned = align_formal_details(details_by_variant)
    prefix, suffix = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"
    counts = {}
    for variant in ("E1", "E2", "E3"):
        exact = 0
        for doc_id, e0 in aligned["E0"].items():
            other = aligned[variant][doc_id]["model_response"]["input"]
            if e0["model_response"]["input"] == prefix + other + suffix:
                exact += 1
        counts[variant] = exact
        if exact != len(aligned["E0"]):
            raise ValueError(f"E0 does not exactly wrap {variant} questions: {exact}/{len(aligned['E0'])}")
    return {"status": "PASS", "exact_wrap_counts": counts, "n_docs": len(aligned["E0"])}


def build_interest_cohort_under_prompt_mismatch(matrix: list[dict], details_by_variant: dict, tokenizer,
                                                count: int = 16, existing_rows: list[dict] | None = None,
                                                excluded_ids=()) -> tuple[list[dict], dict]:
    """Interest seed only: formal E0-pass/E1-fail under mismatched prompts.

    All mechanism variants reuse identical E0 chat-wrapped input_ids. Formal
    observed differences are NOT format-only causal labels.
    """
    if all(row[f"{variant}_prompt_equal_E0"] for row in matrix for variant in VARIANT_DIRS):
        raise ValueError("prompt protocols already align; use build_benchmark_cohort")
    wrap = verify_e0_wraps_other_questions(details_by_variant)
    prompt_rows = _tokenize_e0_prompt_rows(details_by_variant["E0"], tokenizer, existing_rows)
    selected = select_regressions(matrix, count, excluded_ids=excluded_ids)
    cohort, manifest = _pair_cohort(selected, [row for row in matrix if row["formal_group"] == "formal_robust"],
                                    prompt_rows, formal_prompt_protocol_aligned=False,
                                    role="interest_seed_under_formal_prompt_mismatch")
    manifest["e0_wraps_other_questions"] = wrap
    manifest["causal_warning"] = (
        "formal_observed_difference_is_confounded_by_chat_wrapper; "
        "only matched-prompt isolated greedy + official judge defines mechanism labels"
    )
    return cohort, manifest


def audit_formal_protocol(phasea_root: Path, run_root: Path, model_path: str | None = None,
                          raw_cache_paths: list[Path] | None = None) -> dict:
    """Reproducible read-only source audit; always emits the factual matrix before blocking."""
    from transformers import AutoTokenizer
    details, manifests = load_formal_artifacts(Path(phasea_root))
    configs = {variant: json.loads(Path(manifest["results"]).read_text())["config_general"]["model_config"]
               for variant, manifest in manifests.items()}
    if raw_cache_paths is None:
        raw_cache_paths = sorted({path for config in configs.values()
                                 for path in Path(config["model_name"]).glob("*/lcb:codegeneration_v6|0/*/GENERATIVE.parquet")})
    raw_sources = enrich_formal_from_raw_caches(details, raw_cache_paths)
    tokenizer = AutoTokenizer.from_pretrained(model_path or configs["E0"]["model_name"], trust_remote_code=True, local_files_only=True)
    matrix, summary = build_formal_matrix(details, tokenizer)
    summary.update({"source_manifests": manifests, "raw_cache_sources": raw_sources})
    output = Path(run_root) / "01_formal_matrix"
    write_jsonl(output / "formal_task_matrix.jsonl", matrix)
    (output / "formal_task_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    def digest(value):
        serialized = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(serialized.encode()).hexdigest()

    indexed = {variant: {str(item["doc"]["id"]): item for item in items} for variant, items in details.items()}
    audit_rows = []
    for formal in matrix:
        doc_id = formal["doc_id"]
        e0 = indexed["E0"][doc_id]["model_response"]
        row = {"doc_id": doc_id, "variants": {}}
        for variant, index in indexed.items():
            response = index[doc_id]["model_response"]
            text, ids, outputs, raw = response["input"], response.get("input_tokens"), response.get("output_tokens"), response.get("text")
            row["variants"][variant] = {
                "formal_input_sha256": digest(text), "formal_postprocessed_text_sha256": digest(response["text_post_processed"][0]),
                "formal_label_pass": formal[f"{variant}_pass"], "formal_prompt_equal_E0": text == e0["input"],
                "prompt_form": "qwen_chat_wrapper" if text.startswith("<|im_start|>user\n") else "plain_question",
                "raw_cache_exact_lineage_validated": bool(raw_sources[variant]),
                "raw_input_ids_sha256": digest(ids) if ids else None,
                "raw_output_ids_sha256": digest(outputs[0]) if outputs else None,
                "raw_text_sha256": digest(raw[0]) if raw else None,
                "raw_input_len": len(ids) if ids else None, "raw_output_len": len(outputs[0]) if outputs else None,
                "raw_input_ids_head": ids[:8] if ids else None, "raw_input_ids_tail": ids[-12:] if ids else None,
                "raw_think_start_present": "<think>" in raw[0] if raw else "unknown",
                "raw_think_end_present": "</think>" in raw[0] if raw else "unknown",
            }
            if variant != "E0":
                row["variants"][variant]["E0_exactly_wraps_same_question"] = e0["input"] == "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"
        audit_rows.append(row)
    config_sources = {}
    for variant, config in configs.items():
        model = Path(config["model_name"])
        tokenizer_config, template = model / "tokenizer_config.json", model / "chat_template.jinja"
        config_sources[variant] = {
            "formal_results": manifests[variant]["results"], "formal_results_sha256": manifests[variant]["results_sha256"],
            "model_path": str(model), "tokenizer_override": config.get("tokenizer"),
            "override_chat_template": config.get("override_chat_template"), "enable_thinking": config.get("enable_thinking"),
            "tokenizer_config_exists_now": tokenizer_config.exists(), "tokenizer_config_is_symlink_now": tokenizer_config.is_symlink(),
            "chat_template_jinja_exists_now": template.exists(), "chat_template_jinja_is_symlink_now": template.is_symlink(),
        }
        if tokenizer_config.exists():
            config_sources[variant].update({"tokenizer_config_sha256_now": sha256_file(tokenizer_config),
                                           "tokenizer_config_chat_template_nonnull_now": json.loads(tokenizer_config.read_text()).get("chat_template") is not None})
        if template.exists():
            config_sources[variant]["chat_template_jinja_sha256_now"] = sha256_file(template)
    aligned = summary["FORMAL_PROMPT_PROTOCOL_ALIGNED"]
    audit = {
        "schema_version": 1, "status": "PASS" if aligned else "FORMAL_PROMPT_PROTOCOL_MISMATCH",
        "FORMAL_LCB_E0_E3_REUSABLE_AS_FORMAT_ONLY_COMPARISON": aligned,
        "label_counts_reusable_as_observed_facts": True, "formal_rows": len(matrix),
        "summary": {variant: {
            "prompt_equal_E0_count": sum(row["variants"][variant]["formal_prompt_equal_E0"] for row in audit_rows),
            "chat_wrapper_count": sum(row["variants"][variant]["prompt_form"] == "qwen_chat_wrapper" for row in audit_rows),
            "raw_lineage_docs": sum(row["variants"][variant]["raw_cache_exact_lineage_validated"] for row in audit_rows),
            "raw_think_start_count": sum(row["variants"][variant]["raw_think_start_present"] is True for row in audit_rows) if raw_sources[variant] else None,
            "raw_think_end_count": sum(row["variants"][variant]["raw_think_end_present"] is True for row in audit_rows) if raw_sources[variant] else None,
        } for variant in VARIANT_DIRS},
        "raw_cache_sources": raw_sources,
        "searched_raw_cache_candidates": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in raw_cache_paths],
        "unmatched_raw_cache_candidates": [str(path.resolve()) for path in raw_cache_paths if not any(str(path.resolve()) == match["path"] for values in raw_sources.values() for match in values)],
        "formal_protocol_sources": config_sources,
        "official_judge_identity": "04_greedy_judge/formal_judge_identity_summary.json",
        "interpretation": {
            "facts": "Formal labels remain observed run outcomes; per-doc prompt and raw-cache lineage are recorded below.",
            "association": "When prompts differ, the accuracy difference is confounded by prompt formatting.",
            "causal": "Different formal prompts do not identify the effect of format conversion alone.",
        },
        "rows": audit_rows,
    }
    audit_dir = Path(run_root) / "00_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / "FORMAL_PROTOCOL_MISMATCH.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 正式 LCB 输入协议审计", "", f"状态：`{audit['status']}`。", "",
             "正式标签作为运行事实保留；输入不一致时，不能将准确率差归因于格式转换。", "",
             "| 方案 | 与 E0 相同输入 | chat wrapper | 原始 cache 同源题数 | 原始 `<think>` | 原始 `</think>` |",
             "|---|---:|---:|---:|---:|---:|"]
    for variant, counts in audit["summary"].items():
        values = [counts[key] if counts[key] is not None else "unknown" for key in ("prompt_equal_E0_count", "chat_wrapper_count", "raw_lineage_docs", "raw_think_start_count", "raw_think_end_count")]
        lines.append("| " + " | ".join(map(str, [variant, *values])) + " |")
    lines.extend(["", "缓存必须逐题满足 doc ID、完整 input、官方 remove_reasoning_tags 后完整输出全部一致才接受。未匹配缓存不进入正式矩阵；未找到 raw cache 的长度和 thinking 标记记 unknown。",
                  "", "175 题逐输入/输出 SHA256、原始边界 token、缓存路径与文件 SHA256、正式配置来源见同名 JSON。`enable_thinking=null` 不能视为明确 true。",
                  "", "若原始输出连 `<think>` 都没有，不能只因缺少 `</think>` 就认定 thinking 未结束；首先需要核实生成输入协议。",
                  "", "当前 lighteval VLLMModel 调用 uses_chat_template，以加载 tokenizer 的 chat_template 是否存在决定包装。当前文件状态是排错线索，历史原始输入 token 才是此次协议错位的直接证据。",
                  "", "事实：四次运行标签可保留。统计关联：输入协议不同会混入准确率差。因果证据：这组正式结果无法单独识别格式转换的准确率效应。"])
    audit_path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    return {"status": audit["status"], "FORMAL_PROMPT_PROTOCOL_ALIGNED": aligned,
            "audit_path": str(audit_path), "matrix_path": str(output / "formal_task_matrix.jsonl"),
            "summary_path": str(output / "formal_task_summary.json"), "raw_cache_sources": raw_sources,
            "source_manifests": manifests}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phasea_root", type=Path, required=True)
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--model_path")
    parser.add_argument("--matrix_only", action="store_true")
    parser.add_argument("--raw_cache", type=Path, nargs="*", default=[])
    parser.add_argument("--audit_protocol", action="store_true")
    args = parser.parse_args()
    if args.audit_protocol:
        result = audit_formal_protocol(args.phasea_root, args.run_root, args.model_path, args.raw_cache or None)
        print(json.dumps(result, ensure_ascii=False))
        return
    details, manifests = load_formal_artifacts(args.phasea_root)
    raw_sources = enrich_formal_from_raw_caches(details, args.raw_cache) if args.raw_cache else {}
    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    matrix, summary = build_formal_matrix(details, tokenizer)
    summary["source_manifests"] = manifests
    summary["raw_cache_sources"] = raw_sources
    output = args.run_root / "01_formal_matrix"
    write_jsonl(output / "formal_task_matrix.jsonl", matrix)
    (output / "formal_task_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if not args.matrix_only:
        if tokenizer is None:
            raise ValueError("--model_path is required for cohort tokenization")
        rows, manifest = build_benchmark_cohort(matrix, details["E0"], tokenizer)
        output = args.run_root / "02_cohort"
        write_jsonl(output / "benchmark_cohort.jsonl", rows)
        (output / "benchmark_cohort_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "source_manifests"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
