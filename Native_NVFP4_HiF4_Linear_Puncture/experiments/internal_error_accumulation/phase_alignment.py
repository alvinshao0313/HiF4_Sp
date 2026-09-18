"""Detached v3 alignment stage. Never starts training on an unresolved path."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys
from .candidate_runtime import sha256
from .corrected_phase import prepare_protocol,phase_root,write_frozen
from .path_audit import selected_states
from .run_state import atomic_write_json,read_json,read_jsonl,write_jsonl


def finish_alignment(root:Path) -> dict:
    out=phase_root(root);a=out/'alignment';reasons=[];summary={}
    for variant in ('E0','E1'):
        production=read_json(a/f'production_{variant}_complete.json')
        if production['status']!='COMPLETE':raise RuntimeError('production probes incomplete')
        rows=read_jsonl(a/f'local_{variant}/local_boundaries.jsonl')
        summary[variant]={}
        for metric in ('same_input_router','same_input_moe','same_input_o_proj'):
            summary[variant][metric]={'exact':sum(r[metric]['equal'] for r in rows),'total':len(rows),
                'mean_nmse':sum(r[metric]['nmse'] for r in rows)/len(rows),'max_abs':max(r[metric]['max_abs'] for r in rows)}
            if not all(r[metric]['equal'] for r in rows):reasons.append(f'{variant}: {metric} training diagnostic differs from actual TP2; no tolerance waiver')
        for reply in production['replies']:
            if not reply['replay_pass']:reasons.append(f'{variant} TP{reply["rank"]}: production replay identity failed')
            if not reply['lm_head_pass']:reasons.append(f'{variant} TP{reply["rank"]}: actual LM Head identity failed')
    for rank in (0,1):
        e0=read_json(a/f'production_E0_rank{rank}_summary.json');e1=read_json(a/f'production_E1_rank{rank}_summary.json')
        if e0['lm_head_weight_sha256']!=e1['lm_head_weight_sha256']:reasons.append(f'TP{rank} LM Head weight differs between E0/E1')
    for kind in ('native','student'):
        smoke=read_json(out/f'smoke_{kind}.json')
        if smoke['status']!='PASS':reasons.append(f'{kind} smoke failed')
    # These are additional required gates, not implied by baseline local checks.
    for name in ('complete_attention_history_alignment','nonzero_diag_deployment_alignment'):
        gate=a/f'{name}.json'
        if not gate.exists() or read_json(gate).get('status')!='PASS':reasons.append(f'{name}: not established')
    payload={'status':'BLOCKED' if reasons else 'PASS','phase':'corrected_objectives_v3',
             'protocol_sha256':sha256(out/'protocol.json'),'reasons':reasons,'local_summary':summary,
             'training_started':False,'old_artifacts_unchanged':True,
             'evidence_sha256':{str(p.relative_to(out)):sha256(p) for p in sorted(a.glob('*.json*'))}}
    atomic_write_json(out/'alignment_gate.json',payload)
    lines=['# 修正阶段路径门控','',f'状态：{payload["status"]}。尚未启动重训。','',
        '已修正 O1_M 同输入、O1_A 分支提取与原生因果 SDPA；实际 teacher 接口禁止语义缓存替代。',
        '已冻结 8+15 个候选协议和 64 个独立确认样本。','',
        '| 路径 | 算子 | 同输入逐位一致 | 平均 NMSE |','|---|---|---|---|']
    for variant,metrics in summary.items():
        for key,value in metrics.items():lines.append(f'| {variant} | {key} | {value["exact"]}/{value["total"]} | {value["mean_nmse"]:.8g} |')
    lines += ['','## 尚未通过的条件','']+['- '+r for r in reasons]
    lines += ['','真实生产算子的 replay、MoE 四状态分解、BF16 能量闭合和生产 LM Head identity 见 alignment 下逐状态原始记录。',
              '本阶段只定位训练/部署偏差，不用新容差放行，不把未完成的后续实验标记为完成。']
    (out/'ALIGNMENT_REPORT.md').write_text('\n'.join(lines)+'\n')
    return payload


def run_alignment(*,run_root:Path,model_path:str,phasea_root:Path):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('authorized GPUs are exactly 2,3')
    root=Path(run_root).resolve();out=phase_root(root)
    prepare_protocol(root)
    a=out/'alignment';a.mkdir(parents=True,exist_ok=True)
    for kind in ('native','student'):
        if read_json(out/f'smoke_{kind}.json').get('status')!='PASS':raise RuntimeError('real-GPU smoke must pass before formal alignment')
    cohort=selected_states(root)
    path=a/'cohort.jsonl'
    if path.exists() and read_jsonl(path)!=cohort:raise RuntimeError('alignment cohort changed')
    write_jsonl(path,cohort)
    code=Path(__file__).parent
    write_frozen(a/'execution_manifest.json',{'protocol_sha256':sha256(out/'protocol.json'),
        'cohort_sha256':sha256(path),'source_sha256':{name:sha256(code/name) for name in (
        'phase_alignment.py','production_alignment.py','operator_alignment.py','selected_layer_objective_trainer.py','actual_teacher.py')},
        'models':['E0','E1'],'gpus':[2,3],'tp':2})
    for variant in ('E0','E1'):
        if (a/f'production_{variant}_complete.json').exists():raise RuntimeError('explicit completed stage reuse is required; refusing overwrite')
        with (a/f'production_{variant}.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m',
                'Native_NVFP4_HiF4_Linear_Puncture.experiments.internal_error_accumulation.production_alignment',
                '--root',str(root),'--variant',variant,'--model-path',model_path,'--phasea-root',str(phasea_root)],
                stdout=log,stderr=subprocess.STDOUT,check=True)
    return finish_alignment(root)
