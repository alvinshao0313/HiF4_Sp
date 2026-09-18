"""Detached preparation of the mechanism study, without automatic S5."""
from pathlib import Path
import os
import subprocess
import sys
import torch
from .candidate_runtime import sha256
from .mechanism_phase import prepare_protocol, initial_gates, phase_root, VERSION
from .corrected_phase import write_frozen
from .run_state import read_json, atomic_write_json


def assemble_teacher(root):
    out=phase_root(root)/'actual_teacher'
    protocol=read_json(phase_root(root)/'protocol.json')
    by_variant={}
    for variant in ('E0','E1'):
        complete=read_json(out/f'{variant}_complete.json')
        if complete['status']!='COMPLETE':raise RuntimeError('incomplete actual teacher')
        rows={r['sample_id']:r for r in complete['samples']}
        if set(rows)!=set(protocol['sample_token_sha256']):raise RuntimeError('actual teacher sample coverage differs')
        by_variant[variant]=rows
    samples={};lengths={}
    for sid,digest in protocol['sample_token_sha256'].items():
        samples[sid]={}
        for variant in ('E0','E1'):
            row=by_variant[variant][sid]
            if row['token_sha256']!=digest:raise RuntimeError('teacher token hash differs')
            if sid in lengths and lengths[sid]!=row['length']:raise RuntimeError('teacher lengths differ')
            lengths[sid]=row['length'];samples[sid][variant]=row['ranks'][0]
    router=out/'router';router.mkdir(exist_ok=True)
    router_files={}
    for layer in (19,31):
        for split in ('train','val'):
            payload={}
            for sid in protocol[split+'_ids']:
                entry=samples[sid]['E0']['layers'][str(layer)]
                if sha256(Path(entry['path']))!=entry['sha256']:raise RuntimeError('teacher layer changed before router extraction')
                data=torch.load(entry['path'],map_location='cpu',weights_only=False,mmap=True)
                payload[sid]={'length':lengths[sid], 'sample_id':sid,
                    **{k:data[k].clone() for k in ('router_logits','topk_ids','topk_weights')}}
            path=router/f'L{layer:02d}_{split}.pt'
            if path.exists():raise RuntimeError('refusing router teacher overwrite')
            torch.save(payload,path);router_files[path.name]=sha256(path)
    atomic_write_json(router/'manifest.json', {'status':'COMPLETE','training_path_version':VERSION,
        'execution_path':protocol['teacher_execution_path'],'layers':[19,31], 'files_sha256':router_files,
        'train_ids':protocol['train_ids'],'val_ids':protocol['val_ids']})
    atomic_write_json(out/'manifest.json', {'status':'COMPLETE','training_path_version':VERSION,
        'execution_path':protocol['teacher_execution_path'],'sample_token_sha256':protocol['sample_token_sha256'],
        'lengths':lengths,'samples':samples,'variant_manifest_sha256':{
            v:sha256(out/f'{v}_complete.json') for v in ('E0','E1')},
        'router_manifest_sha256':sha256(router/'manifest.json')})


def run_prepare(*,run_root,model_path,phasea_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('authorized devices must be 2,3')
    root=Path(run_root).resolve()
    prepare_protocol(root);initial_gates(root)
    out=phase_root(root);teacher=out/'actual_teacher';teacher.mkdir(exist_ok=True)
    code=Path(__file__).parent
    write_frozen(out/'preparation_execution.json', {'protocol_sha256':sha256(out/'protocol.json'),
        'source_sha256':{name:sha256(code/name) for name in ('mechanism_prepare.py','actual_teacher_capture.py',
             'actual_teacher.py','mechanism_phase.py','mechanism_objectives.py','selected_layer_objective_trainer.py')},
        'moe_source_sha256':sha256(code.parent/'e2e_diag_reconstruction/core/moe_semantic_hif4.py'),
        'teacher_execution':'actual TP2 causal prefill with original full token sequences',
        'training_started':False,'no_automatic_s5':True})
    for variant in ('E0','E1'):
        if (teacher/f'{variant}_complete.json').exists():
            raise RuntimeError('completed teacher reuse requires explicit content validation; refusing overwrite')
        with (teacher/f'{variant}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m',
                'Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.actual_teacher_capture',
                '--root',str(root),'--variant',variant,'--model-path',model_path,'--phasea-root',str(phasea_root)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    assemble_teacher(root)
    atomic_write_json(out/'preparation_complete.json', {'status':'COMPLETE',
        'teacher_manifest_sha256':sha256(teacher/'manifest.json'),
        'next':'objective correctness and longest training checks; then pilot/mechanism experiments',
        'training_started':False})
