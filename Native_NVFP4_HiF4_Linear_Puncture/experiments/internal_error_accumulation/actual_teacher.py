"""Read actual production teacher tensors with sample/layer provenance."""
from pathlib import Path
import torch
from .mechanism_phase import VERSION
from .candidate_runtime import sha256
from .run_state import read_json

FIELDS=('input','output','attn','post_attn','moe','moe_input')


def _manifest(root: Path) -> dict:
    m=read_json(root/'manifest.json')
    if m.get('status')!='COMPLETE' or m.get('execution_path')!='actual_vllm_tp2_teacher_forced_prefill' or m.get('training_path_version')!=VERSION:
        raise RuntimeError('invalid actual teacher provenance')
    return m


def load_actual_layer_cache(root: Path, *, layer: int, sample_ids: list[str]) -> dict:
    manifest=_manifest(root)
    if set(sample_ids)!=set(manifest['sample_token_sha256']):
        raise RuntimeError('actual teacher sample coverage differs')
    result={f'e0_{f}':{} for f in FIELDS}
    result.update({f'e1_{f}':{} for f in ('input','post_attn','moe_input')})
    for sid in sample_ids:
        for variant,fields in [('E0',FIELDS),('E1',('input','post_attn','moe_input'))]:
            entry=manifest['samples'][sid][variant]['layers'][str(layer)]
            path=Path(entry['path'])
            if sha256(path)!=entry['sha256']:raise RuntimeError('actual teacher layer hash changed')
            payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            for field in fields:
                x=payload[field]
                if x.ndim!=2 or x.shape[0]!=manifest['lengths'][sid] or x.dtype!=torch.bfloat16 or not torch.isfinite(x).all():
                    raise RuntimeError('invalid actual cache shape, dtype, length or finite state')
                result[f'{variant.lower()}_{field}'][sid]=x
    return result


class ChunkedLogits:
    """Only the requested token block is resident; no whole-corpus logits load."""
    def __init__(self, entries, length):
        if not entries or length <= 0:
            raise RuntimeError('teacher logits require nonempty token coverage')
        self.entries=entries
        self.shape=(length,int(entries[0]['vocab']))
        self.current=None
        self.current_tensor=None
        self.verified={}
        cursor=0
        for entry in entries:
            if entry['start']!=cursor or entry['stop']<=cursor or entry['vocab']!=self.shape[1]:
                raise RuntimeError('invalid teacher logit chunk coverage')
            cursor=entry['stop']
        if cursor!=length:raise RuntimeError('teacher logits incomplete')

    def __getitem__(self,item):
        if not isinstance(item,slice) or item.step not in (None,1):
            raise ValueError('teacher logits require contiguous token slices')
        start,stop,_=item.indices(self.shape[0]);chunks=[]
        for entry in self.entries:
            lo,hi=max(start,entry['start']),min(stop,entry['stop'])
            if lo>=hi:continue
            path=Path(entry['path'])
            stat=path.stat()
            signature=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
            if path in self.verified and self.verified[path]!=signature:
                raise RuntimeError('actual logit chunk changed after validation')
            if self.current!=path:
                if path not in self.verified:
                    if sha256(path)!=entry['sha256']:raise RuntimeError('actual logit chunk hash changed')
                    self.verified[path]=signature
                self.current_tensor=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                if self.current_tensor.shape!=(entry['stop']-entry['start'],self.shape[1]) or not torch.isfinite(self.current_tensor).all():
                    raise RuntimeError('invalid teacher logit chunk tensor')
                self.current=path
            chunks.append(self.current_tensor[lo-entry['start']:hi-entry['start']])
        if not chunks:
            return torch.empty((0,self.shape[1]),dtype=torch.float32)
        return chunks[0] if len(chunks)==1 else torch.cat(chunks)


def load_actual_final_logits(root: Path, sample_ids: list[str]) -> dict:
    manifest=_manifest(root)
    return {sid:ChunkedLogits(manifest['samples'][sid]['E0']['logits'],manifest['lengths'][sid])
            for sid in sample_ids}
