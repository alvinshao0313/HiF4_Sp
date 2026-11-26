import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/long_pipeline.py'
spec = importlib.util.spec_from_file_location('kld_long_pipeline_test', SCRIPT)
driver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = driver
spec.loader.exec_module(driver)


def queue(tmp_path, monkeypatch, tasks, failure=None):
    root, control = tmp_path/'run', tmp_path/'control'
    root.mkdir()
    control.mkdir()
    q = driver.Queue(root, control, train_gpus=('6','7'), eval_gpus=('8','9'))
    monkeypatch.setattr(q, 'check_source', lambda: None)
    monkeypatch.setattr(driver, 'idle_devices', lambda ids: [{'index': x} for x in ids.split(',')])
    original = subprocess.Popen
    children = []
    def start(argv, **kwargs):
        command = argv[4]
        output = next(t.output for t in tasks if t.command == command and list(t.args) == argv[7:])
        if command == failure:
            code = 'raise SystemExit(3)'
        else:
            code = ("import pathlib,json,time,os; time.sleep(.2); "
                    f"p=pathlib.Path({output!r}); p.parent.mkdir(parents=True,exist_ok=True); "
                    "p.write_text(json.dumps(dict(status='COMPLETE',steps=1,gpus=os.environ['CUDA_VISIBLE_DEVICES']))); "
                    "(p.parent/'final.pt').write_bytes(b'checkpoint')")
        p = original([sys.executable,'-c',code], **kwargs)
        children.append(p)
        return p
    monkeypatch.setattr(driver.subprocess, 'Popen', start)
    return q, children


def test_two_one_gpu_jobs_and_cpu_export_overlap_but_tp2_is_exclusive(tmp_path, monkeypatch):
    tasks = [driver.Task('a','train',('--recipe','a'),1,str(tmp_path/'a/manifest.json')),
             driver.Task('b','train',('--recipe','b'),1,str(tmp_path/'b/manifest.json')),
             driver.Task('cpu','export',('--output','cpu'),0,str(tmp_path/'cpu/export.json')),
             driver.Task('tp','capture',('--variant','candidate'),2,str(tmp_path/'tp/manifest.json'),('a','b'))]
    q, _ = queue(tmp_path,monkeypatch,tasks)
    q.run_phase('test',tasks)
    rows = [json.loads(line) for line in (q.root/'stages.jsonl').read_text().splitlines()]
    by_key = {r['task']:r for r in rows}
    assert {by_key['a']['devices'],by_key['b']['devices']} == {'6','7'}
    assert by_key['tp']['devices'] == '8,9' and by_key['cpu']['devices'] is None
    assert by_key['b']['started_at'] < by_key['a']['finished_at']
    assert by_key['cpu']['started_at'] < min(by_key[k]['finished_at'] for k in ('a','b'))
    assert by_key['tp']['started_at'] >= max(by_key[k]['finished_at'] for k in ('a','b'))
    q.resume = True
    q.run_phase('test',tasks)
    assert len((q.root/'stages.jsonl').read_text().splitlines()) == 4
    Path(tasks[0].output).write_text('{"status":"COMPLETE","steps":2}')
    with pytest.raises(RuntimeError,match='changed'):
        q.run_phase('test',tasks)


def test_failure_does_not_dispatch_followup_tasks(tmp_path, monkeypatch):
    tasks = [driver.Task('bad','export',(),0,str(tmp_path/'bad/export.json')),
             driver.Task('dependent','capture',(),2,str(tmp_path/'later/manifest.json'),('bad',))]
    q, children = queue(tmp_path,monkeypatch,tasks,failure='export')
    with pytest.raises(RuntimeError,match='bad failed'):
        q.run_phase('test',tasks)
    q.stop_owned_jobs()
    assert len(children)==1 and children[0].returncode==3
    assert not Path(tasks[1].output).exists()


def test_training_plan_keeps_fifteen_independent_single_gpu_configs():
    tasks=driver.training_tasks(Path('/run'))
    assert len(tasks)==15 and len({t.key for t in tasks})==15
    assert all(t.slots==1 for t in tasks)
    assert [t.key for t in tasks[:3]]==['L08_mse','L24_mse','L40_mse']


def test_failure_stops_owned_running_job_and_records_interruption(tmp_path, monkeypatch):
    tasks = [driver.Task('bad','export',(),0,str(tmp_path/'bad/export.json')),
             driver.Task('training','train',('--recipe','running'),1,str(tmp_path/'training/manifest.json'))]
    q, children = queue(tmp_path,monkeypatch,tasks,failure='export')
    original = subprocess.Popen
    def slow_training(argv, **kwargs):
        if argv[4] == 'train':
            p = real_popen([sys.executable,'-c','import time; time.sleep(60)'],**kwargs)
            children.append(p)
            return p
        return original(argv,**kwargs)
    # The real constructor was saved before installing the test interception.
    real_popen = subprocess.Popen.__closure__[1].cell_contents if False else None
    # Retrieve the existing test interceptor's closed-over original constructor.
    real_popen = dict(zip(original.__code__.co_freevars, (c.cell_contents for c in original.__closure__)))['original']
    monkeypatch.setattr(driver.subprocess,'Popen',slow_training)
    with pytest.raises(RuntimeError,match='bad failed'):
        q.run_phase('test',tasks)
    q.stop_owned_jobs()
    assert len(children)==2 and all(p.poll() is not None for p in children)
    state=json.loads((q.control/'jobs/training/status.json').read_text())
    assert state['status']=='INTERRUPTED' and state['returncode']!=0


def test_four_batches_then_two_fixed_tp2_pairs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    tasks = driver.training_tasks(tmp_path / 'run')
    q, _ = queue(tmp_path, monkeypatch, tasks)
    q.train_gpus = q.eval_gpus = ('0', '1', '2', '3')
    phases, reserves = [], []
    def phase(name, jobs, **kwargs):
        keys = {t.key for t in jobs}
        assert all(set(t.dependencies) <= keys for t in jobs)
        phases.append((name, jobs, kwargs))
    monkeypatch.setattr(q, 'run_phase', phase)
    monkeypatch.setattr(q, 'start_reservation', reserves.append)
    monkeypatch.setattr(q, 'stop_reservations', lambda: None)
    monkeypatch.setattr(driver, 'common_steps', lambda ds: {8: 2, 24: 4, 40: 8})
    q.run_train_evaluate(SimpleNamespace(root=q.root, ids=lambda split: [split]))
    assert [len(jobs) for _, jobs, _ in phases[:4]] == [4, 4, 4, 3]
    assert reserves == ['3']
    assert phases[3][2]['hold_finished']
    assert phases[4][0] == 'evaluate' and len(phases[4][1]) == 120
    assert phases[5][0] == 'common_steps' and len(phases[5][1]) == 30
    task = driver.Task('eval', 'capture', (), 2, 'unused', pool='eval')
    assert q._assign(task) == ['0', '1']
    q.running = {'engine0': {'pool': 'eval', 'gpus': ['0', '1']}}
    assert q._assign(task) == ['2', '3']
    q.running['engine1'] = {'pool': 'eval', 'gpus': ['2', '3']}
    assert q._assign(task) is None
    del q.running['engine0']
    assert q._assign(task) == ['0', '1']


def test_cross_phase_dependencies_fail_before_dispatch(tmp_path, monkeypatch):
    tasks = [driver.Task('bad', 'export', (), 0, '/unused', ('old_training',))]
    q, children = queue(tmp_path, monkeypatch, tasks)
    with pytest.raises(ValueError, match='outside the phase'):
        q.run_phase('bad', tasks)
    assert not children


def test_reservation_ready_release_and_reuse(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    q, _ = queue(tmp_path, monkeypatch, [])
    def reserve(argv, **kwargs):
        ready = argv[argv.index('--ready-file') + 1]
        code = ("import os,signal,pathlib,json,threading; done=threading.Event(); "
                "signal.signal(signal.SIGTERM,lambda *args:done.set()); "
                f"pathlib.Path({ready!r}).write_text(json.dumps(dict(pid=os.getpid()))); done.wait()")
        return real_popen([sys.executable, '-c', code], **kwargs)
    monkeypatch.setattr(driver.subprocess, 'Popen', reserve)
    q.start_reservation('3')
    assert q.reservations['3']['process'].poll() is None
    q.stop_reservations(['3'])
    q.start_reservation('3')
    q.stop_reservations()
    rows = [json.loads(line) for line in (q.root/'stages.jsonl').read_text().splitlines()]
    assert len(rows) == 2
    assert all(r['returncode'] == 0 and r['status'] == 'RELEASED' for r in rows)
    assert len(list((q.control/'reservations').glob('*.log'))) == 2
