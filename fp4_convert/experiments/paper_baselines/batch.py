"""Execute the fixed, qualified single-method conversion/QA plan once."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .results import render, validate_qa, validate_reasoning


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(".writing")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run(spec_path):
    spec_path = spec_path.resolve()
    spec = json.loads(spec_path.read_text())
    root = spec_path.parent
    (root / "verified").mkdir(exist_ok=True)
    (root / "verified/reasoning").mkdir(exist_ok=True)
    (root / "jobs").mkdir(exist_ok=True)
    configs = {row["id"]: row for row in spec["configs"]}
    state = {"pid": os.getpid(), "started_at": timestamp(), "status": "running",
             "requested_configurations": len(configs), "requested_task_seed_runs": 19 * len(configs),
             "reasoning_status": spec["reasoning_status"], "completed_jobs": [],
             "validated_task_seed_runs": 4 * len(list((root / "verified").glob("*.json")))}
    status_path = root / "status.json"
    if status_path.exists():
        raise FileExistsError("This batch has already been started; inspect its status before creating a new execution plan")
    code = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")}
    write_json(root / "code_sha256.json", code)
    write_json(status_path, state)

    def job(item):
        directory = root / "jobs" / item["id"]
        directory.mkdir()
        record = {**item, "started_at": timestamp(), "status": "running"}
        with (directory / "output.log").open("xb") as log:
            process = subprocess.Popen(item["argv"], cwd=spec["cwd"], stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       env={**os.environ, **item["environment"]})
            record["pid"] = process.pid
            write_json(directory / "status.json", record)
            record["exit_code"] = process.wait()
        try:
            if record["exit_code"]:
                raise RuntimeError(f"{item['id']} exited {record['exit_code']}")
            if item["kind"] == "calibration":
                recipe = json.loads((Path(item["output"]) / "recipe.json").read_text())
                if (recipe["purpose"] != "paper" or not recipe["scope_complete"]
                        or len(recipe["sample_ids"]) != 128):
                    raise ValueError("Formal calibration must cover the full model and all 128 complete S1K sequences")
                tokens = sum(recipe["tokens"])
                if any(group["tokens"] != tokens or group["sequences"] != 128 for group in recipe["groups"]):
                    raise ValueError("Calibration groups did not observe all complete sequences")
                record["tokens"] = tokens
            elif item["kind"] == "export":
                info = json.loads((Path(item["output"]) / "conversion.json").read_text())
                if info.get("purpose") != "paper":
                    raise ValueError("Refuse qualification-calibrated weights in formal evaluation")
            elif item["kind"] == "qa":
                report = validate_qa(item["output"])
                report["config"] = configs[item["config"]]
                report["exit_code"] = 0
                report["verified_at"] = timestamp()
                write_json(root / "verified" / (item["config"] + ".json"), report)
            elif item["kind"] == "reasoning":
                candidate = configs[item["config"]]
                settings = spec["reasoning"][candidate["model"]][item["task"]]
                report = validate_reasoning(item["output"], settings, item["seed"])
                report.update(config=item["config"], verified_at=timestamp(), exit_code=0)
                name = f"{item['config']}--{item['task']}--{item['seed']}.json"
                write_json(root / "verified/reasoning" / name, report)
            else:
                raise ValueError(f"Unknown stage {item['kind']}")
            record["status"] = "completed"
        except Exception as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            record["finished_at"] = timestamp()
            write_json(directory / "status.json", record)
        return item["id"]

    try:
        render(spec, root)
        for group in spec["stages"]:
            state["active_jobs"] = [item["id"] for item in group]
            write_json(status_path, state)
            # Each group specifies either one TP2 evaluation, or two independent
            # calibrations on distinct GPUs. No retries or resource polling.
            with ThreadPoolExecutor(max_workers=len(group)) as pool:
                results = list(pool.map(job, group))
            state["completed_jobs"].extend(results)
            state["validated_task_seed_runs"] = (4 * len(list((root / "verified").glob("*.json")))
                                                  + len(list((root / "verified/reasoning").glob("*.json"))))
            render(spec, root)
            write_json(status_path, state)
        if state["validated_task_seed_runs"] != state["requested_task_seed_runs"]:
            raise ValueError("The fixed batch ended without completing all requested task/seed runs")
        state["status"] = "completed"
        state["active_jobs"] = []
    except Exception as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        state["updated_at"] = timestamp()
        write_json(status_path, state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    run(parser.parse_args().spec)
