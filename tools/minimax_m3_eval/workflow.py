"""One authorized continuation: wait, qualify all tasks, evaluate, then exit."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.processes import (
    gpu_snapshot, join_session, require_free,
)
from .data import sha256
from .download import save
from .report import summarize


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_bindings(root):
    for filename, digest in json.loads((root / 'bindings.json').read_text()).items():
        if sha256(filename) != digest:
            raise ValueError(f"Queued input/source changed: {filename}")


def wait_predecessor(record):
    """Use process identities, including children, without signalling any PID."""
    try:
        process = psutil.Process(record['pid'])
        if process.create_time() != record['create_time']:
            return {'already_exited': True}
        if process.cmdline() != record['command'] or process.uids().real != os.getuid():
            raise ValueError(f"Predecessor identity changed: {record['pid']}")
    except psutil.NoSuchProcess:
        return {'already_exited': True}
    children = {}
    while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
        try:
            for child in process.children(recursive=True):
                children[(child.pid, child.create_time())] = child
            process.wait(timeout=30)
            break
        except psutil.TimeoutExpired:
            continue
        except psutil.NoSuchProcess:
            break
    for child in children.values():
        child.wait()
    return {'finished_at': now(), 'children_joined': len(children)}


def join_gpus(session, devices):
    owned = join_session(session)
    deadline = time.monotonic() + 30
    while True:
        state = gpu_snapshot(devices)
        if any(pid not in owned['owned_pids'] for d in state.values() for pid in d['pids']):
            raise RuntimeError(f"GPUs were occupied by another process after our stage: {state}")
        if all(not d['pids'] and d['memory_mib'] <= 1024 for d in state.values()):
            require_free(devices)
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU contexts did not clear after our stage: {state}")
        time.sleep(.2)


def model_view(root, phase, name):
    config = json.loads((root / 'config.json').read_text())
    manifest = json.loads((root / 'download_manifest.json').read_text())
    view = root / phase / name / 'model'
    view.mkdir(parents=True, exist_ok=False)
    # Link files individually. Prediction caches stay in this stage's real
    # directory; linking a model directory would alias historical caches.
    for entry in manifest['files']:
        target = view / entry['name']
        source = Path(config['model']) / entry['name']
        if not source.is_file() or source.stat().st_size != entry['size']:
            raise FileNotFoundError(f"Checkpoint incomplete: {entry['name']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
    return view


def command(root, config, phase, name, view):
    task = config['tasks'].get(name)
    result = [sys.executable, '-u', 'tools/minimax_m3.py', 'qualify' if name == 'generation' else 'evaluate',
        '--model_path', str(view), '--dataset_manifest', str(root / 'datasets.json'),
        '--tensor_parallel_size', str(config['tp']),
        '--pipeline_parallel_size', str(config['pp']), '--max_model_len', str(config['max_model_len']),
        '--max_new_tokens', str(256 if phase == 'qualification' else task['max_new_tokens']),
        '--batch_size', str(config['batch_size']), '--max_num_batched_tokens', str(config['max_num_batched_tokens']),
        '--gpu_memory_utilization', str(config['gpu_memory_utilization']),
        '--temperature', str(config['temperature']), '--top_p', str(config['top_p']),
        '--output_dir', str(root / phase / name / 'output')]
    if task is not None:
        result += ['--datasets', task['registry']]
    if task is not None and (phase == 'qualification' or name == 'mmlu_pro'):
        result += ['--max_samples', str(2 if phase == 'qualification' else task['count'])]
    return result


def reuse_generation(root):
    config = json.loads((root / 'config.json').read_text())
    source = config.get('generation_qualification_source')
    if source is None:
        return None
    source = Path(source)
    previous = json.loads((source / 'config.json').read_text())
    for settings in (config, previous):
        settings.pop('created_at', None)
        settings.pop('generation_qualification_source', None)
    if config != previous:
        raise ValueError('Generation qualification configuration differs from its source')
    record = json.loads((source / 'state.json').read_text())['phases']['qualification/generation']
    summary = json.loads((source / 'qualification/generation/summary.json').read_text())
    output = json.loads((source / 'qualification/generation/output/qualification.json').read_text())
    if (record['status'] != 'complete' or record['exit_code'] != 0
            or summary['status'] != 'passed' or record['summary'] != summary or output != summary):
        raise ValueError('Generation qualification source is not a completed, verified result')
    return {'status': 'reused', 'source_run': str(source), 'summary': summary,
            'original_record': record}


def run(root):
    root = Path(root).resolve()
    config = json.loads((root / 'config.json').read_text())
    if (root / 'state.json').exists():
        raise FileExistsError('This finite continuation was already launched')
    verify_bindings(root)
    generation = reuse_generation(root)
    state = {'status': 'waiting_predecessors', 'pid': os.getpid(), 'started_at': now(),
             'gpu_devices': config['gpus'], 'formal_started': False, 'phases': {}}
    if generation is not None:
        state['phases']['qualification/generation'] = generation
    save(root / 'state.json', state)
    try:
        for record in config['predecessors']:
            state['waiting_pid'] = record['pid']
            save(root / 'state.json', state)
            state.setdefault('predecessor_handoffs', {})[str(record['pid'])] = wait_predecessor(record)
        state['status'] = 'waiting_download'
        save(root / 'state.json', state)
        launch = json.loads((root / 'download_launch.json').read_text())
        wait_predecessor(launch)
        download = json.loads((root / 'download_status.json').read_text())
        if download['status'] != 'complete':
            raise RuntimeError(f"Download did not complete: {download}")
        verify_bindings(root)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': config['gpus'], 'HF_HUB_OFFLINE': '1',
               'HF_DATASETS_OFFLINE': '1', 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
        results = {}
        for phase in ('qualification', 'formal'):
            state.update(status=phase, formal_started=phase == 'formal')
            save(root / 'state.json', state)
            names = (['generation', *config['tasks']]
                     if phase == 'qualification' and generation is None else config['tasks'])
            for name in names:
                verify_bindings(root)
                require_free(config['gpus'])
                view = model_view(root, phase, name)
                argv = command(root, config, phase, name, view)
                log_path = root / phase / name / 'run.log'
                with log_path.open('w') as log:
                    child = subprocess.Popen(argv, cwd=Path(__file__).resolve().parents[2], env=env,
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    label = phase + '/' + name
                    state['current_phase'] = label
                    record = {'status': 'running', 'pid': child.pid, 'command': argv,
                              'started_at': now(), 'log': str(log_path)}
                    state['phases'][label] = record
                    save(root / 'state.json', state)
                    code = child.wait()
                record.update(status='validating' if code == 0 else 'failed',
                              exit_code=code, finished_at=now())
                record['gpu_handoff'] = join_gpus(child.pid, config['gpus'])
                if code:
                    raise RuntimeError(f"{label} failed with exit code {code}; see {log_path}")
                if name == 'generation':
                    result = json.loads((root / phase / name / 'output/qualification.json').read_text())
                    if result['status'] != 'passed':
                        raise ValueError('Whole-model qualification failed')
                else:
                    result = summarize(root, phase, name)
                save(root / phase / name / 'summary.json', result)
                record.update(status='complete', summary=result)
                if phase == 'formal':
                    results[name] = result
                    save(root / 'accuracy.json', results)
                save(root / 'state.json', state)
        state.update(status='complete', finished_at=now())
    except Exception as error:
        current = state['phases'].get(state.get('current_phase'))
        if current is not None and current['status'] in ('running', 'validating'):
            current.update(status='failed', error=repr(error))
        state.update(status='failed', error=repr(error), finished_at=now())
        save(root / 'state.json', state)
        raise
    save(root / 'state.json', state)
