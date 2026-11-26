"""Prepare Table 3 or execute a disjoint list of assigned configurations on one GPU."""
import argparse
import json
import os
from pathlib import Path
import sys

from . import weights
from .jobs import lock, now, unique, write
from .plan import ROOT, digest, make_plan, prior_inventory, read, tasks


def prepare(root):
    with lock(root/'locks/prepare'):
        plan, routes = make_plan(root)
        for route in routes.values():
            for split in route['splits'].values():
                if digest(split['path']) != split['sha256']:
                    raise ValueError(f"Pinned data changed: {split['path']}")
        for model in plan['reference_inputs'].values():
            for ref in model.values():
                if digest(ref.get('samples') or ref['file']) != ref['sha256']:
                    raise ValueError(f'Reference inputs changed: {ref}')
        inventory = prior_inventory(plan, root)
        for rel, data in [('data/manifest.json', routes),('plan.json',plan)]:
            path = root/rel
            if path.exists():
                if read(path) != json.loads(json.dumps(data)):
                    raise ValueError(f'Existing immutable plan differs: {path}')
            else:
                write(path, data)
        write(root/'inventory.json', inventory)
    return inventory


def lane(root, gpu, ids, phase):
    plan = read(root/'plan.json')
    configs = {row['id']:row for row in plan['configs']}
    if not ids or len(set(ids))!=len(ids) or set(ids)-set(configs):
        raise ValueError(f'Expected distinct Table 3 IDs drawn from {list(configs)}')
    if os.environ.get('CONDA_DEFAULT_ENV') != 'hif4':
        raise ValueError('Activate the project hif4 environment')
    record = root/'lanes'/f'gpu{gpu}-{unique()}.json'
    status = dict(gpu=gpu, pid=os.getpid(), argv=sys.argv, configs=ids, phase=phase,
                  started=now(), status='running', plan_sha256=digest(root/'plan.json'))
    with lock(root/'locks'/f'gpu-{gpu}') as card:
        write(record, status)
        try:
            for key in ids:
                with lock(root/'locks'/f'config-{key}') as config_lock:
                    locks = [card, config_lock]
                    fingerprint = weights.ensure(root, configs[key], gpu, locks)
                    if phase != 'export':
                        from .evaluation import evaluate
                        evaluate(root, plan, configs[key], fingerprint, gpu, locks, phase)
        except BaseException as exc:
            status.update(status='failed', error=repr(exc), exit_code=1)
            raise
        else:
            status.update(status='completed',exit_code=0)
        finally:
            status.update(finished=now())
            write(record,status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    sub.add_parser('inventory')
    run = sub.add_parser('lane')
    run.add_argument('--gpu',type=int,required=True)
    run.add_argument('--configs',nargs='+',required=True)
    run.add_argument('--phase',choices=['export','qualify','all'],default='all')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command=='prepare':
        inv = prepare(root)
        print(f'{root}: {inv["configuration_count"]} configurations, {inv["maximum_new_rounds"]} rounds')
    elif args.command=='inventory':
        from .evaluation import accepted
        p = read(root/'plan.json')
        for row in p['configs']:
            completed = [(t,s) for t,s in tasks() if accepted(root,row,t,s)]
            print(row['id'], f'{len(completed)}/{len(tasks())}', 'missing', [(t,s) for t,s in tasks() if (t,s) not in completed])
    else:
        lane(root, args.gpu, args.configs, args.phase)


if __name__=='__main__':
    main()
