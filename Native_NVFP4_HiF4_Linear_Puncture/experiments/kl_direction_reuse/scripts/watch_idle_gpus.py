"""Wait for two unoccupied GPUs, release our identified holder, run one queue."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def query(scope, fields):
    text = subprocess.check_output(['nvidia-smi', f'--query-{scope}={fields}', '--format=csv,noheader,nounits'], text=True)
    return [[x.strip() for x in row] for row in csv.reader(text.splitlines())]


def select_pair(inventory, apps, holder_pid):
    busy = {row[0] for row in apps if int(row[1]) != holder_pid}
    free = [row[0] for row in inventory if row[1] not in busy and
            ((holder_pid is not None and any(a[0] == row[1] and int(a[1]) == holder_pid for a in apps))
             or (int(row[2]) < 1024 and int(row[3]) == 0))]
    free.sort(key=lambda gpu: (gpu not in ('2', '3'), gpu != '2', int(gpu)))
    return free[:2] if len(free) >= 2 else []


def identity(pid):
    path = Path(f'/proc/{pid}')
    try:
        return (path.stat().st_uid, (path/'stat').read_text().rsplit(')', 1)[1].split()[19],
                (path/'cmdline').read_bytes())
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--control', type=Path, required=True)
    p.add_argument('--holder-pid', type=int, required=True)
    p.add_argument('--gpus', required=True)
    args = p.parse_args()
    if Path(sys.prefix).name != 'hif4':
        raise RuntimeError('hif4 environment required')
    args.control.mkdir(parents=True, exist_ok=True)
    with (args.control/'watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(state, **kw):
            data = dict(status=state, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kw)
            tmp = args.control/'status.partial'
            tmp.write_text(json.dumps(data, indent=2)+'\n')
            tmp.replace(args.control/'status.json')
            print(json.dumps(data), flush=True)
        holder = identity(args.holder_pid)
        if holder is not None and (holder[0] != os.getuid() or b'/reserve_gpus.py\0' not in holder[2]):
            raise RuntimeError('holder process identity mismatch')
        try:
            while True:
                current = identity(args.holder_pid)
                if current is not None and current != holder:
                    raise RuntimeError('holder PID was reused')
                inv = query('gpu', 'index,uuid,memory.used,utilization.gpu')
                inv = [row for row in inv if row[0] in args.gpus.split(',')]
                apps = query('compute-apps', 'gpu_uuid,pid')
                pair = select_pair(inv, apps, args.holder_pid if current else None)
                if not pair:
                    status('WAITING_FOR_TWO_GPUS', inventory=inv)
                    time.sleep(30)
                    continue
                status('HANDOFF', gpus=pair)
                if current:
                    # Kill only the same owned reservation process observed at startup.
                    if identity(args.holder_pid) != holder:
                        raise RuntimeError('holder changed before handoff')
                    os.kill(args.holder_pid, signal.SIGTERM)
                    deadline = time.monotonic()+30
                    while identity(args.holder_pid) is not None:
                        if time.monotonic() > deadline:
                            raise RuntimeError('holder did not exit')
                        time.sleep(0.5)
                # Fail visibly if a competing job won the handoff race. No retries.
                check = query('compute-apps', 'gpu_uuid,pid')
                uuids = {row[1] for row in inv if row[0] in pair}
                if any(row[0] in uuids for row in check):
                    raise RuntimeError('selected GPUs acquired by another process during handoff')
                queue = args.control/'queue'
                status('RUNNING', gpus=pair, queue_control=str(queue))
                script = Path(__file__).with_name('long_pipeline.py')
                with (args.control/'pipeline.log').open('x') as log:
                    code = subprocess.call([sys.executable, '-u', str(script), '--root', str(args.root),
                                            '--control', str(queue), '--gpus', ','.join(pair)], stdout=log, stderr=subprocess.STDOUT)
                status('COMPLETE' if code == 0 else 'FAILED', gpus=pair, returncode=code, queue_control=str(queue))
                return code
        except BaseException as e:
            status('FAILED', error=repr(e))
            raise


if __name__ == '__main__':
    sys.exit(main())
