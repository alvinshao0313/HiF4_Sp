"""Freeze inputs and existing baseline evidence for the 6.4-budget supplement."""
from pathlib import Path

from ..ban.additional_main import install_datasets
from ..ban.artifact import load_artifact, model_identity
from ..ban.evaluate import REPO, file_sha256
from ..ban.evaluation_report import _sha256
from .config import INFERENCE, now, read, save, source_identity
from .remaining_report import TASKS, summarize


def prepare(source, output, data_source):
    from transformers import AutoTokenizer
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.prompt_manager import PromptManager

    source, root, data_source = map(lambda p: Path(p).resolve(), (source, output, data_source))
    if root.exists():
        raise FileExistsError(f"supplement output must be new: {root}")
    previous = read(source / "manifest.json")
    if previous["budget"]["target"] != 6.4:
        raise ValueError("source must be the matched 6.4-budget experiment")
    identity = model_identity(previous["model"], previous["spec"])
    assert identity == previous["model_identity"]
    for name, digest in previous["code"]["source_sha256"].items():
        assert file_sha256(REPO / name) == digest, f"qualified source changed: {name}"
    arms, inputs = {}, {}
    for arm in ("c4", "trace"):
        path = source / f"budget/{arm}/budget.json"
        budget = read(path)
        assert budget["status"] == "complete" and budget["budget_matched"] and budget["target"] == 6.4
        assert budget["k_min"] == 5 and not budget["test_set_tuning"]
        artifact = Path(budget["artifact"]).resolve()
        parsed = load_artifact(artifact, identity)
        assert parsed["phase"] == "formal" and parsed["num_samples"] == 512
        arms[arm] = {"artifact": str(artifact), "lambda": budget["chosen_lambda"], "k_min": 5}
        for p in (path, artifact):
            inputs[str(p)] = file_sha256(p)
    routes = {}
    data = read(data_source / "evaluation/data/manifest.json")
    for name in TASKS:
        item = data["tasks"][name]
        route = {"revision": item["registry_revision"], "source_revision": item["task_revision"], "splits": {}}
        for split, entry in item["splits"].items():
            path = (data_source / entry["path"]).resolve(strict=True)
            assert file_sha256(path) == entry["sha256"]
            route["splits"][split] = {"path": str(path), "sha256": entry["sha256"]}
            inputs[str(path)] = entry["sha256"]
        routes[item["repo"]] = route
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    save(root / "datasets.json", routes)
    install_datasets(root / "datasets.json")
    tokenizer = AutoTokenizer.from_pretrained(previous["model"], local_files_only=True)
    manager = PromptManager(use_chat_template=True, tokenizer=tokenizer, enable_thinking=None)
    registry = Registry(tasks=",".join(t["task"] for t in TASKS.values()),
                        custom_tasks=str(REPO / "tasks/custom_tasks.py")).load_tasks()
    for name, definition in TASKS.items():
        task = registry[definition["task"]]
        docs = task.get_docs(max_samples=definition["count"])
        assert len(docs) == definition["count"] and all(d.num_samples == definition["generations"] for d in docs)
        documents = []
        for doc in docs:
            prompt = manager.prepare_prompt(doc)
            ids = tokenizer.encode(prompt, add_special_tokens=True)
            # All supplemental prompts fit the formal input budget. This also
            # makes the 64-token qualification use exactly the same prompt IDs.
            assert len(ids) <= 40960 - definition["max_new_tokens"]
            documents.append({"id": str(doc.id), "prompt_sha256": _sha256(prompt),
                              "gold_sha256": _sha256(doc.get_golds()), "input_tokens_sha256": _sha256(ids)})
        assert len({d["id"] for d in documents}) == definition["count"]
        save(root / f"{name}_input_audit.json", {"documents": documents, "dataset": task.dataset_path,
             "revision": routes[task.dataset_path]["source_revision"], **definition})
    historical = read(REPO / "Native_NVFP4_HiF4_Linear_Puncture/results/ban/repeats_20261027T050631Z/ban_vs_top8_comparison.json")
    gsm = read(REPO / "Native_NVFP4_HiF4_Linear_Puncture/results/ban/gsm8k_8to6_20261024T022233Z/report.json")["evaluations"]["top8"]
    aime = historical["historical_sources"]["aime25"][0]
    metrics = read(aime["metrics_file"])
    assert metrics["exit_code"] == 0
    baseline_sources = {"gsm8k": {"results_file": gsm["results_file"], "prediction_cache": gsm["prediction_cache"]},
                        "aime25": {"results_file": metrics["results_file"], "prediction_cache": aime["prediction_cache"]}}
    # The restored export has already been proved identical to these historical
    # model weights. Check metadata and bind that qualification again.
    qualification = Path(previous["model"]).parent / "qualification.json"
    proof = read(qualification)
    assert proof["passed"] and len(proof["verified_weight_shards"]) == 49
    inputs[str(qualification)] = file_sha256(qualification)
    for name, evidence in baseline_sources.items():
        old_model = Path(read(evidence["results_file"])["config_general"]["model_config"]["model_name"])
        for asset in ("config.json", "model.safetensors.index.json", "ablation_export.json", "chat_template.jinja"):
            assert file_sha256(old_model / asset) == file_sha256(Path(previous["model"]) / asset)
        target = root / "top8" / name
        target.mkdir(parents=True)
        summarize(target, name, read(root / f"{name}_input_audit.json"), arm="top8", historical=evidence)
        for p in map(Path, evidence.values()):
            inputs[str(p)] = file_sha256(p)
    for name in ("datasets.json", *(f"{t}_input_audit.json" for t in TASKS)):
        inputs[str(root / name)] = file_sha256(root / name)
    code = source_identity()
    for path in (REPO / "tasks/custom_tasks.py", REPO / "Native_NVFP4_HiF4_Linear_Puncture/experiments/reasoning_trace_calibration/scoring.py"):
        code["source_sha256"][str(path.relative_to(REPO))] = file_sha256(path)
    save(root / "manifest.json", {"created_at": now(), "source_run": str(source),
         "model": previous["model"], "spec": previous["spec"], "model_identity": identity,
         "budget_target": 6.4, "arms": arms, "tasks": TASKS, "inference": INFERENCE,
         "code": code, "input_sha256": inputs, "authorized_gpu_pair": "0,1", "wait_timeout_hours": 24,
         "baseline_sources": baseline_sources, "new_baseline_tasks": ["aime24"],
         "scoring": "official score retained; AIME final_submission_avg5 additionally reviewed for all arms"})
    save(root / "status.json", {"status": "prepared", "phases": {}, "errors": [], "created_at": now()})
    print(f"BAN_REMAINING_PREPARED {root}", flush=True)
    return root


def verify(root):
    manifest = read(Path(root) / "manifest.json")
    for name, digest in manifest["code"]["source_sha256"].items():
        assert file_sha256(REPO / name) == digest, f"frozen source changed: {name}"
    for name, digest in manifest["input_sha256"].items():
        assert file_sha256(name) == digest, f"frozen input changed: {name}"
    assert model_identity(manifest["model"], manifest["spec"]) == manifest["model_identity"]
    return manifest
