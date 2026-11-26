"""Run one subprocess at a time and preserve its own exit code and artifacts."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from .plan import REPO, digest, json_bytes, read


def now():
    return datetime.now(timezone.utc).isoformat()


def unique():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid4().hex[:8]


def write(path, obj):
    data = json_bytes(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    temp.write_bytes(data)
    os.replace(temp, path)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another live lane owns {path}') from None
        yield handle.fileno()


def free_gpu(gpu):
    uuid = subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid',
                                    '--format=csv,noheader']).decode().strip()
    holders = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                       '--format=csv,noheader']).decode().splitlines()
    occupied = [line for line in holders if line.split(',')[0].strip() == uuid]
    if occupied:
        raise RuntimeError(f'GPU {gpu} already has compute processes: {occupied}')
    return uuid


def code_digest():
    base = REPO/'fp4_convert/experiments'
    paths = [REPO/'main.py', *Path(__file__).parent.glob('*.py'), *(base/'paper_baselines').glob('*.py')]
    for folder in ('qwen3_mxfp4','qwen3_mxfp4_uos','qwen3_mxfp4_dualscale'):
        paths.extend((base/folder).glob('*.py'))
    paths.extend((REPO/'HiFloat4/hif4_scale_threshold_optimization/src').glob('*.py'))
    return {str(p.relative_to(REPO)):digest(p) for p in sorted(set(paths))}


def call(root, config, phase, arguments, gpu, locks, metadata):
    job = root/'attempts'/config['id']/phase/unique()
    job.mkdir(parents=True)
    output = job/'output'
    argv = [sys.executable, '-u', '-m', *map(str, arguments(output))]
    env = dict(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=os.pathsep.join(filter(None,(str(REPO),os.environ.get('PYTHONPATH')))))
    status = dict(status='launching', config=config['id'], phase=phase, argv=argv,
                  environment=env, gpu=gpu, gpu_uuid=free_gpu(gpu), controller=os.getpid(), cwd=str(REPO),
                  job_status=str(job/'status.json'), output=str(output), log=str(job/'output.log'),
                  plan_sha256=digest(root/'plan.json'), code_sha256=code_digest(), started=now(), **metadata)
    write(job/'status.json', status)
    with (job/'output.log').open('ab',buffering=0) as log:
        try:
            child = subprocess.Popen(argv, cwd=REPO, env={**os.environ,**env}, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, pass_fds=tuple(locks))
            status.update(status='running', pid=child.pid)
            write(job/'status.json', status)
            print(f'{now()} PID {child.pid} GPU {gpu} {phase} {config["id"]} {job}', flush=True)
            rc = child.wait()
            status.update(status='completed' if rc == 0 else 'failed', exit_code=rc, finished=now())
        except BaseException as error:
            status.update(status='interrupted', error=repr(error), updated=now())
            raise
        finally:
            write(job/'status.json', status)
    if rc != 0:
        raise RuntimeError(f'{phase} exited {rc}: {job}/output.log')
    return status
