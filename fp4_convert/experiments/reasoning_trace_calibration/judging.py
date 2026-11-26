"""Record calibration correctness after generation; never select by correctness."""
import argparse
from pathlib import Path

from .artifacts import read_json, sha256, write_json
from .scoring import code_score, math_score
from .sources import reference


def judge(root):
    root = Path(root)
    manifest = read_json(root / "generation/manifest.json")
    if not manifest["complete"]:
        raise ValueError("generation must complete before judging")
    pools = read_json(root / "isolation/pools.json")
    questions = {r["id"]: r for d in pools.values() for split in ("train", "val") for r in d[split]}
    selected = []
    for row in manifest["attempts"]:
        if not row.get("accepted"):
            continue
        path = root / "generation" / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError("generated response changed")
        response = read_json(path)
        answer = reference(root, questions[row["id"]])
        value = math_score(response["text"], answer["solution"]) if row["domain"] == "math" else code_score(response["text"], answer)
        selected.append({"id": row["id"], "domain": row["domain"], "split": row["split"], "score": value})
    if len(selected) != 160:
        raise ValueError("judging must cover the fixed 160 selected traces")
    write_json({"complete": True, "correctness_used_for_selection": False, "per_question": selected}, root / "judging/calibration_correctness.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    judge(parser.parse_args().run)
