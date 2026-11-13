"""Independent scale, encoding, real-weight and TP partition contracts."""
import json
import math
import unittest
import torch
from safetensors import safe_open
from .convert import SOURCE
from .format import quantize, convert_weight, unpack_codes, group_size, VARIANTS, nvfp4_to_bf16
from ..qwen3_mxfp4.test_format import scalar_reference


def reference(x, group):
    values=x.float()
    chunks=[];scales=[]
    for start in range(0,values.shape[-1],group):
        block=values[...,start:start+group]
        maximum=block.abs().amax(-1)
        scale=maximum/torch.tensor(6.,dtype=torch.float32)
        scale=torch.where(maximum==0,1.,scale)
        chunks.append(block/scale[...,None]);scales.append(scale)
    normalized=torch.cat(chunks,-1)
    codes,inner=scalar_reference(normalized)
    return codes,inner,torch.stack(scales,-1)


class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(8)

    def check(self,x,g):
        p,s,t=quantize(x,g)
        codes,scales,outer=reference(x,g)
        self.assertTrue(torch.equal(unpack_codes(p),codes))
        self.assertTrue(torch.equal(s,scales))
        self.assertTrue(torch.equal(t,outer))

    def test_zero_ties_and_tail(self):
        x=torch.zeros(3,768)
        x[1,:32]=torch.tensor([6.,.25,.75,1.25,1.75,2.5,3.5,5.]*4)
        x[2]=-x[1]
        x[2,512:]=-0.
        for g in (256,512):
            self.check(x,g)
            p,s,t=quantize(x,g)
            self.assertTrue(torch.all(t[0]==1))
            self.assertTrue(torch.all(s[0]==127))
            self.assertTrue(torch.all(unpack_codes(p)[2,512:]==8))

    def test_fp32_scale_and_normalization_boundary(self):
        x=torch.randn(4,768,generator=torch.Generator().manual_seed(1234)).bfloat16()
        for g in (256,512):
            self.check(x,g)
            _,_,t=quantize(x,g)
            self.assertEqual(t.dtype,torch.float32)
            self.assertFalse(torch.equal(t,t.bfloat16().float()))
        for exponent in (-130,-80,0,80,120):
            self.check((x.float()*2.**exponent).bfloat16(),256)

    def test_tp4_global_groups_match_full_quantization(self):
        x=torch.randn(16,768,generator=torch.Generator().manual_seed(17)).bfloat16().float()
        x[0].zero_()
        for g in (256,512):
            groups=math.ceil(768/g)
            local=[]
            for rank in range(4):
                maximum=torch.zeros(x.shape[0],groups)
                for j in range(groups):
                    lo=max(rank*192,j*g);hi=min((rank+1)*192,(j+1)*g,768)
                    if lo<hi: maximum[:,j]=x[:,lo:hi].abs().amax(-1)
                local.append(maximum)
            maximum=torch.stack(local).amax(0)
            outer=torch.where(maximum==0,1.,maximum/6.)
            p,s,t=quantize(x,g)
            self.assertTrue(torch.equal(t,outer))
            from ..qwen3_mxfp4.format import quantize as inner
            pieces=[];scale_pieces=[]
            for rank in range(4):
                ids=torch.arange(rank*192,(rank+1)*192)
                a,b=inner(x[:,ids]/outer[:,ids//g])
                pieces.append(a);scale_pieces.append(b)
            self.assertTrue(torch.equal(p,torch.cat(pieces,-1)))
            self.assertTrue(torch.equal(s,torch.cat(scale_pieces,-1)))

    def test_real_first_and_last_experts(self):
        mapping=json.loads((SOURCE/'model.safetensors.index.json').read_text())["weight_map"]
        for layer,expert in ((0,0),(47,127)):
            for projection in ('gate_proj','up_proj','down_proj'):
                prefix=f'model.layers.{layer}.mlp.experts.{expert}.{projection}'
                tensors=[]
                for suffix in ('weight','weight_scale','weight_scale_2'):
                    name=prefix+'.'+suffix
                    with safe_open(str(SOURCE/mapping[name]),framework='pt') as reader:
                        tensor=reader.get_tensor(name)
                        tensors.append(tensor[:4].contiguous() if tensor.ndim==2 else tensor)
                b=nvfp4_to_bf16(*tensors)
                for variant in VARIANTS:
                    p,s,t=convert_weight(*tensors,variant)
                    codes,inner,outer=reference(b,group_size(variant))
                    self.assertTrue(torch.equal(unpack_codes(p),codes))
                    self.assertTrue(torch.equal(s,inner))
                    self.assertTrue(torch.equal(t,outer))


if __name__=='__main__':
    unittest.main()
