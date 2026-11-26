"""One-shot independent conversions of G256 and G512 from the original NVFP4."""
import argparse
import os
from pathlib import Path
from .artifacts import save
from .convert import convert, SOURCE
from .format import VARIANTS
from ..qwen3_mxfp4.workflow import now


def run(root):
    root=Path(root).absolute()
    state={"status":"running","pid":os.getpid(),"started_at":now(),"completed":[]}
    try:
        for variant in VARIANTS:
            state["variant"]=variant
            save(root/"weights_status.json",state)
            convert(SOURCE,root/"checkpoints"/variant,variant,threads=8)
            state["completed"].append(variant)
        state.update(status="completed",returncode=0,finished_at=now())
    except Exception as exc:
        state.update(status="failed",returncode=1,error=f"{type(exc).__name__}: {exc}",finished_at=now())
        raise
    finally:
        save(root/"weights_status.json",state)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    run(parser.parse_args().root)
