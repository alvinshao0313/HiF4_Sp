"""GPU numerical validation using real converted experts and formal TP2 slices."""
import argparse
import json
from pathlib import Path

from safetensors import safe_open
import torch

from tools.minimax_m3_eval.download import save
from vllm.model_executor.models.minimax_m3.quantization import swiglu_oai
from .format import quantize, dequantize, decode_scales
from .runtime import activation_qdq, routed_experts


def validate(model, output):
    torch.set_num_threads(4)
    torch.manual_seed(2134)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Path(model)
    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    tensors = {}
    wanted = []
    for eid in range(4):
        for part in ('w1','w2','w3'):
            stem = f'language_model.model.layers.5.block_sparse_moe.experts.{eid}.{part}'
            wanted.extend((stem + '.weight_packed', stem + '.weight_scale'))
    for shard in {index[name] for name in wanted}:
        with safe_open(model / shard, framework='pt', device='cpu') as handle:
            for name in wanted:
                if index[name] == shard:
                    tensors[name] = handle.get_tensor(name)
    first, second, sfirst, ssecond = [], [], [], []
    for eid in range(4):
        def get(part, suffix):
            return tensors[f'language_model.model.layers.5.block_sparse_moe.experts.{eid}.{part}.{suffix}']
        # Rank 0 of the formal TP2 partition: FC1 splits rows; FC2 splits K.
        first.append(torch.cat([get(part,'weight_packed')[:1536] for part in ('w1','w3')]))
        sfirst.append(torch.cat([get(part,'weight_scale')[:1536] for part in ('w1','w3')]))
        second.append(get('w2','weight_packed')[:,:768])
        ssecond.append(get('w2','weight_scale')[:,:48])
    w13, w2 = torch.stack(first).cuda(), torch.stack(second).contiguous().cuda()
    raw13, raw2 = torch.stack(sfirst).cuda(), torch.stack(ssecond).contiguous().cuda()
    s13, s2 = decode_scales(raw13), decode_scales(raw2)
    ref13, ref2 = dequantize(w13,raw13), dequantize(w2,raw2)
    ones = torch.ones(4,device='cuda')
    records = []
    for count in (2,129):
        x = torch.randn(count,6144,device='cuda',dtype=torch.bfloat16)
        x[0].zero_()
        if count>2: x[1] *= .01; x[-1] *= 100
        cpu_qdq = dequantize(*quantize(x.cpu())).cuda()
        torch.testing.assert_close(activation_qdq(x),cpu_qdq,rtol=0,atol=0)
        selected = torch.arange(4,device='cuda',dtype=torch.int32).expand(count,-1).contiguous()
        weights = torch.rand(count,4,device='cuda'); weights /= weights.sum(-1,keepdim=True)
        actual = routed_experts(x,w13,w2,s13,s2,ones,weights,selected)
        contributions=[]
        for eid in range(4):
            first_output = torch.nn.functional.linear(cpu_qdq,ref13[eid])
            activated = swiglu_oai(first_output)
            # Independent CPU quantization prevents a matching error in the GPU
            # activation kernel from being hidden by the GEMM comparison.
            second_input = dequantize(*quantize(activated.cpu())).cuda()
            result = torch.nn.functional.linear(second_input,ref2[eid])
            contributions.append((result.float()*weights[:,eid,None]).bfloat16())
        expected = torch.stack(contributions,1).float().sum(1).bfloat16()
        amplitude = torch.stack(contributions,1).float().abs().sum(1)
        delta=actual.float()-expected.float()
        nmse=delta.square().sum()/expected.float().square().sum().clamp_min(1e-30)
        assert torch.isfinite(actual).all()
        assert torch.all(delta.abs() <= 4*torch.finfo(torch.bfloat16).eps*amplitude+1e-5)
        assert nmse.item()<1e-4
        records.append({'tokens':count,'nmse':nmse.item(),'max_abs':delta.abs().max().item()})
    save(Path(output),{'status':'passed','source':str(model),'layer':5,'experts':[0,1,2,3],
        'partition':'actual TP2 rank-0 weight slices','activation_cpu_gpu_exact':True,
        'cases':records,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    validate(args.model,args.output)
