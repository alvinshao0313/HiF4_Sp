"""Finite stage execution; no daemon, retry, reservation, or wait for foreign GPUs."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from ..ban_trace_calibration.processes import require_free, join_session
from .artifacts import *
from .config import *


def execute(root, action, candidate=None):
    root=run_root(root);verify_sources(root)
    key=action+("_"+candidate if candidate else "")
    directory=root/"jobs"/key
    directory.mkdir(parents=True,exist_ok=False)
    pair=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if pair not in ("0","1","6","7","0,1","6,7"):raise ValueError("only GPUs 0,1,6,7 are allocated")
    require_free(pair)
    if action=="convert":
        qualification=root/"qualification"/("scale.json" if candidate=="r64_scale" else "gptq.json")
        if not read_json(qualification).get("complete"):raise ValueError("conversion qualification incomplete")
        commands=[[sys.executable,"-u","-m",MODULE+".export","--run",str(root),"--candidate",candidate]]
        if candidate == "r64_gptq" and (root / "gptq_calibration/manifest.json").exists():
            coverage = read_json(root / "qualification/expanded_gptq.json")
            if not coverage["complete"]:
                raise ValueError("expanded GPTQ qualification incomplete")
            commands[0] += ["--gptq-calibration", str(root / "gptq_calibration")]
    elif action=="orthogonal":
        qualification=read_json(root/"qualification/training/orthogonal_a20/manifest.json")
        if not qualification["complete"] or qualification["completed_layers"]!=[0,1]:raise ValueError("orthogonal qualification incomplete")
        commands=[[sys.executable,"-u","-m",MODULE+".training",stage,"--run",str(root)] for stage in ("train","export")]
    else:
        if pair not in ("0,1","6,7"):raise ValueError("evaluation requires an allocated TP2 pair (0,1 or 6,7)")
        candidates=(candidate,) if candidate else ("full_e5","coverage50_e5")
        tasks=("lcb","mmlu_pro") if action=="efficiency" else REASONING+("arc",)
        existing=read_json(root/"summary.json")["rows"]
        commands=[]
        for repetition in range(1,6):
            for role in candidates:
                for task in tasks:
                    keys=("arc_easy","arc_challenge") if task=="arc" else (task,)
                    if all(str(repetition) in existing[role]["round_scores"][k] for k in keys):continue
                    commands.append([sys.executable,"-u","-m",MODULE+".evaluation","--run",str(root),
                                     "--candidate",role,"--task",task,"--repetition",str(repetition)])
    state={**process_record(sys.argv),"status":"running","commands":commands,"completed":[]}
    write_json(state,directory/"status.json")
    with (root/f"gpu_{pair.replace(',','_')}.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for index,command in enumerate(commands):
            require_free(pair);verify_sources(root)
            log=directory/f"{index:03d}.log";started=time.monotonic()
            with log.open("w") as stream:
                child=subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL,stdout=stream,
                                       stderr=subprocess.STDOUT,start_new_session=True)
                state["current"]={"index":index,"pid":child.pid,"session_id":child.pid,"command":command,"log":str(log)}
                write_json(state,directory/"status.json")
                code=child.wait()
            children=join_session(child.pid)
            state["completed"].append({**state["current"],"exit_code":code,"seconds":time.monotonic()-started,"children":children})
            if code:
                state["status"]="failed";write_json(state,directory/"status.json")
                raise RuntimeError(f"stage exit={code}; see {log}")
            write_json(state,directory/"status.json")
        state["status"]="complete";state["exit_code"]=0
        write_json(state,directory/"status.json")


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__);p.add_argument("action",choices=("convert","orthogonal","efficiency","evaluate"))
    p.add_argument("--run",type=Path,required=True);p.add_argument("--candidate",choices=MAIN+EFFICIENCY_ROLES+("orthogonal_a20",))
    a=vars(p.parse_args());a["root"]=a.pop("run");execute(**a)
