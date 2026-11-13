"""Real four-rank NCCL MAX and fused QDQ, compared with full-K quantization."""
import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from .artifacts import save, code_hashes
from .format import quantize, unpack_codes, mx_dequantize
from .kernels import activation_qdq
from ..qwen3_mxfp4.validate_gpu import measure, device_state


def setup():
    rank=int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl",timeout=timedelta(minutes=5),device_id=torch.device("cuda",rank))
    if dist.get_world_size()!=4:
        raise ValueError("Validation requires TP4")
    return rank


def check(x, group, rank, sharded):
    full=x.cpu()
    packed,scales,outer=quantize(full,group)
    start,end=(rank*192,(rank+1)*192) if sharded else (0,x.shape[1])
    args={"global_k":768,"offset":start,"process_group":dist.group.WORLD} if sharded else {}
    got,t,c,s=activation_qdq(x[:,start:end].contiguous(),group,diagnostics=True,**args)
    torch.testing.assert_close(t.cpu(),outer,rtol=0,atol=0)
    torch.testing.assert_close(c.cpu(),unpack_codes(packed)[:,start:end],rtol=0,atol=0)
    torch.testing.assert_close(s.cpu(),scales[:,start//32:end//32],rtol=0,atol=0)
    torch.testing.assert_close(got.cpu(),mx_dequantize(packed,scales)[:,start:end],rtol=0,atol=0)
    return {"rows":x.shape[0],"global_k":x.shape[1],"outer_group":group,"sharded":sharded,"exact":True}


def validate(output):
    rank=setup()
    result={"status":"running","rank":rank,"source_sha256":code_hashes(),"tests":[],"performance":[]}
    if rank==0: result["gpu_state_before"]=device_state()
    try:
        for group in (256,512):
            for count in (1,256,8192):
                torch.manual_seed(1234)
                for label,rows,k,sharded in (("gate_up",count,2048,False),("down",count*8,768,True)):
                    x=torch.randn(rows,k,device="cuda",dtype=torch.bfloat16)
                    if rows>1:x[0].zero_()
                    result["tests"].append(check(x,group,rank,sharded))
                    local=x[:,rank*192:(rank+1)*192].contiguous() if sharded else x
                    args={"global_k":768,"offset":rank*192,"process_group":dist.group.WORLD} if sharded else {}
                    dist.barrier()
                    stats=measure(lambda:activation_qdq(local,group,**args),rows)
                    result["performance"].append({"boundary":label,"tokens":count,"group":group,**stats})
                if rank==0: print(f"QDQ/NCCL exact: group={group}, tokens={count}",flush=True)
        result["status"]="passed"
    except Exception as exc:
        result.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save(Path(output).with_name(Path(output).stem+f"_rank{rank}.json"),result)
    gathered=[None]*4 if rank==0 else None
    dist.gather_object(result,gathered,dst=0)
    if rank==0: save(output,{"status":"passed","ranks":gathered,"gpu_state_after":device_state()})
    dist.destroy_process_group()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    validate(parser.parse_args().output)
