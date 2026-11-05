"""Real process-lifecycle checks, with no replacement process or GPU tools."""
from __future__ import annotations

import subprocess
import sys
import time

import psutil

from .processes import join_session


def test_join_waits_for_descendant_after_leader_exit():
    code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(0.6)']); print(p.pid,flush=True)"
    leader = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                              stdout=subprocess.PIPE, text=True)
    child = int(leader.stdout.readline().strip())
    leader.wait()
    result = join_session(leader.pid, timeout=5)
    assert child in result["owned_pids"]
    assert not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE


def test_join_does_not_wait_for_or_signal_foreign_session():
    foreign = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"], start_new_session=True)
    own = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    own.wait()
    started = time.monotonic()
    result = join_session(own.pid, timeout=5)
    assert foreign.pid not in result["owned_pids"]
    assert time.monotonic() - started < 1
    assert foreign.poll() is None
    foreign.wait()
