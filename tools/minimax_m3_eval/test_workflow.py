"""Real process handoff and immutable-input checks; no GPU/engine substitutes."""
import json
import subprocess
import sys

import psutil
import pytest

from .data import sha256
from .workflow import reuse_generation, verify_bindings, wait_predecessor


def test_generation_reuse_preserves_provenance_and_rejects_changed_conditions(tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'target'
    (source / 'qualification/generation/output').mkdir(parents=True)
    target.mkdir()
    config = {'tp': 2, 'pp': 3, 'temperature': 1.0}
    summary = {'status': 'passed', 'cases': [{'text': '4'}, {'text': '56'}]}
    record = {'status': 'complete', 'exit_code': 0, 'summary': summary}
    (source / 'config.json').write_text(json.dumps(config))
    (source / 'state.json').write_text(json.dumps({'phases': {'qualification/generation': record}}))
    (source / 'qualification/generation/summary.json').write_text(json.dumps(summary))
    output = source / 'qualification/generation/output/qualification.json'
    output.write_text(json.dumps(summary))
    config['generation_qualification_source'] = str(source)
    (target / 'config.json').write_text(json.dumps(config))
    reused = reuse_generation(target)
    assert reused['source_run'] == str(source) and reused['original_record'] == record
    config['pp'] = 2
    (target / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='configuration differs'):
        reuse_generation(target)
    config['pp'] = 3
    (target / 'config.json').write_text(json.dumps(config))
    output.write_text(json.dumps({'status': 'failed'}))
    with pytest.raises(ValueError, match='not a completed, verified result'):
        reuse_generation(target)


def test_changed_queued_source_is_rejected(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('initial source\n')
    (tmp_path / 'bindings.json').write_text(json.dumps({str(source): sha256(source)}))
    verify_bindings(tmp_path)
    source.write_text('changed source\n')
    with pytest.raises(ValueError, match='Queued input/source changed'):
        verify_bindings(tmp_path)


def test_wait_joins_real_parent_and_its_longer_lived_child():
    code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.4)']); print(p.pid,flush=True); time.sleep(.1)"
    parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline())
    process = psutil.Process(parent.pid)
    result = wait_predecessor({'pid': parent.pid, 'create_time': process.create_time(),
                               'command': process.cmdline()})
    assert result['children_joined'] == 1
    assert parent.wait() == 0
    assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
