"""Isolate fresh adjoint, router, and combined gradients on unchanged parameters."""
import argparse
import json
from pathlib import Path

import torch

from ..artifacts import write_json, source_fingerprint
from ..capture import CaptureStore
from ..config import PROTOCOL, idle_devices, require_environment, require_gpus
from ..data import Dataset
from ..losses import router_term, signed_term
from ..optimization import collect_direction
from ..runtime import Runtime


def stats(a, b):
    a, b = a.detach().cpu().double().reshape(-1), b.detach().cpu().double().reshape(-1)
    if a.shape != b.shape or not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise RuntimeError('nonfinite tensor or shape mismatch')
    d = a-b
    na, nb = a.norm(), b.norm()
    return dict(exact=bool(torch.equal(a,b)), max_abs=float(d.abs().max()),
                l2_relative=float(d.norm()/na) if na else None,
                cosine=float(a.dot(b)/(na*nb)) if na and nb else None,
                norm_left=float(na), norm_right=float(nb),
                close=bool(torch.allclose(a,b,atol=1e-7,rtol=1e-5)))


def grad_stats(left, right):
    if set(left) != set(right):
        raise RuntimeError('gradient parameter coverage mismatch')
    return {k: stats(left[k], right[k]) for k in left}


def parameter_gradients(runtime, x, teacher, router_target, denominator, objective, record):
    runtime.student.zero_grad(set_to_none=True)
    out = runtime.local(x)
    main = None
    if objective.startswith('direct'):
        main = runtime.kl(out.output, teacher, denominator)
    elif objective.startswith('cached'):
        main = signed_term(out.output, record['anchor'].to(x.device), record['gradient'].to(x.device))
    aux = router_term(out.router_logits, router_target, denominator,
                     top_k=runtime.student.base.spec.top_k)
    loss = aux if objective == 'router' else main + aux if objective.endswith('total') else main
    incoming = {}
    def capture(grad):
        incoming['boundary'] = grad.detach().cpu().clone()
    if main is not None:
        out.output.register_hook(capture)
    loss.backward()
    grads = {name: p.grad.detach().cpu().clone() for name,p in runtime.student.learned.named_parameters()
             if p.grad is not None}
    return grads, incoming, float(loss.detach())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--layer',type=int,choices=PROTOCOL.layers,default=24)
    parser.add_argument('--gpu',required=True)
    args=parser.parse_args()
    require_environment()
    idle_devices(args.gpu)
    require_gpus(1)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(PROTOCOL.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ds=Dataset(args.root)
    native=CaptureStore(ds.root/'captures/native',ds)
    baseline=CaptureStore(ds.root/'captures/baseline',ds)
    ids=ds.protocol['batches'][0]
    denominator=sum(len(ds.samples[s]['input_ids']) for s in ids)
    report=dict(status='RUNNING',source=source_fingerprint(),layer=args.layer,gpu=args.gpu,
                samples=ids,denominator=denominator,comparisons={})
    write_json(report,args.output)
    try:
        runtime=Runtime(ds.snapshot,ds.root/'baseline',args.layer)
        for sid in ids:
            row={}
            report['comparisons'][sid]=row
            x=baseline.layer(sid,args.layer)['input'].unsqueeze(0).to(runtime.device)
            target=native.layer(sid,args.layer)['router_logits'].unsqueeze(0).to(runtime.device)
            teacher=native.logits(sid)
            print(f'{sid}: collect fresh direction',flush=True)
            record=collect_direction(runtime,x,teacher,denominator)
            out=runtime.local(x)
            row['no_grad_vs_grad_forward']=stats(record['anchor'],out.output)
            del out
            gradients={}
            incoming={}
            for objective in ('direct_main','cached_main','router','direct_total','cached_total','direct_total_repeat'):
                mode='direct_total' if objective=='direct_total_repeat' else objective
                print(f'{sid}: {objective}',flush=True)
                grads,boundary,loss=parameter_gradients(runtime,x,teacher,target,denominator,mode,record)
                gradients[objective]=grads
                incoming[objective]=boundary
                row[objective+'_loss']=loss
                write_json(report,args.output)
            row['boundary_gradient']=stats(incoming['direct_main']['boundary'],record['gradient'])
            for label,a,b in [('kl_parameter_gradient','direct_main','cached_main'),
                              ('total_parameter_gradient','direct_total','cached_total'),
                              ('direct_repeat','direct_total','direct_total_repeat')]:
                row[label]=grad_stats(gradients[a],gradients[b])
                failed=[k for k,v in row[label].items() if not v['close']]
                print(f'{sid}: {label} failing={len(failed)} max_abs={max(v["max_abs"] for v in row[label].values())}',flush=True)
            print(f'{sid}: boundary {row["boundary_gradient"]}',flush=True)
            write_json(report,args.output)
        report['status']='COMPLETE'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error))
        raise
    finally:
        write_json(report,args.output)
    print('diagnosis COMPLETE',flush=True)


if __name__=='__main__':
    main()
