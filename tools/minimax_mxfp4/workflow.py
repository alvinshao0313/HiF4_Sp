"""Finite MXFP4 continuation: join existing work, validate, then score five tasks."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.minimax_m3_eval.data import sha256
from tools.minimax_m3_eval.download import save
from tools.minimax_m3_eval.workflow import (
    command, join_gpus, model_view, verify_bindings, wait_predecessor,
)
from tools.minimax_m3_eval.report import summarize
from Native_NVFP4_HiF4_Linear_Puncture.experiments.ban_trace_calibration.processes import require_free


def now():
    return datetime.now(timezone.utc).isoformat()


def run(root):
    root=Path(root).resolve()
    if (root/'state.json').exists():
        raise FileExistsError('This MXFP4 continuation was already launched')
    config=json.loads((root/'config.json').read_text())
    state={'status':'waiting_conversion','pid':os.getpid(),'started_at':now(),
           'gpu_devices':config['gpus'],'formal_started':False,'phases':{}}
    save(root/'state.json',state)
    try:
        verify_bindings(root)
        wait_predecessor(json.loads((root/'conversion_launch.json').read_text()))
        model=Path(config['model'])
        conversion=json.loads((model/'conversion_status.json').read_text())
        if conversion['status']!='complete' or conversion['exit_code']!=0:
            raise RuntimeError(f'Conversion did not complete: {conversion}')
        manifest=model/'conversion_manifest.json'
        if sha256(manifest)!=conversion['manifest_sha256']:
            raise ValueError('Conversion manifest changed')
        save(root/'download_manifest.json',json.loads(manifest.read_text()))
        verify_bindings(root)
        bindings=json.loads((root/'bindings.json').read_text())
        for path in [root/'download_manifest.json',model/'conversion_manifest.json',
                     model/'conversion_status.json',model/'config.json',model/'model.safetensors.index.json']:
            bindings[str(path)]=sha256(path)
        save(root/'bindings.json',bindings)
        state.update(status='waiting_baseline',conversion_nmse=conversion['nmse'])
        save(root/'state.json',state)
        for predecessor in config['predecessors']:
            state['waiting_pid']=predecessor['pid'];save(root/'state.json',state)
            wait_predecessor(predecessor)
        verify_bindings(root)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':config['gpus'],'HF_HUB_OFFLINE':'1',
             'HF_DATASETS_OFFLINE':'1','OMP_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}

        def execute(label,argv):
            verify_bindings(root);require_free(config['gpus'])
            log=root/label/'run.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x') as output:
                child=subprocess.Popen(argv,cwd=Path(__file__).resolve().parents[2],env=env,
                    stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                record={'status':'running','pid':child.pid,'command':argv,'log':str(log),'started_at':now()}
                state['phases'][label]=record;state['current_phase']=label;save(root/'state.json',state)
                code=child.wait()
            record.update(exit_code=code,finished_at=now(),status='validating' if code==0 else 'failed')
            record['gpu_handoff']=join_gpus(child.pid,config['gpus'])
            if code:
                raise RuntimeError(f'{label} failed; see {log}')
            return record

        state['status']='qualification';save(root/'state.json',state)
        record=execute('qualification/kernels',[sys.executable,'-u','-m','tools.minimax_mxfp4.validate',
            '--model',config['model'],'--output',str(root/'qualification/kernels/report.json')])
        if json.loads((root/'qualification/kernels/report.json').read_text())['status']!='passed':
            raise ValueError('MXFP4 GPU numerical checks failed')
        record['status']='complete';save(root/'state.json',state)
        results={}
        for phase in ('qualification','formal'):
            state.update(status=phase,formal_started=phase=='formal');save(root/'state.json',state)
            names=['generation',*config['tasks']] if phase=='qualification' else config['tasks']
            for name in names:
                view=model_view(root,phase,name)
                argv=command(root,config,phase,name,view);argv[2]=config['entrypoint']
                record=execute(phase+'/'+name,argv)
                if name=='generation':
                    summary=json.loads((root/phase/name/'output/qualification.json').read_text())
                    if summary['status']!='passed':raise ValueError('MXFP4 generation checks failed')
                else:
                    summary=summarize(root,phase,name)
                save(root/phase/name/'summary.json',summary)
                record.update(status='complete',summary=summary)
                if phase=='formal':
                    results[name]=summary;save(root/'accuracy.json',results)
                    baseline_path=Path(config['baseline_run'])/'accuracy.json'
                    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
                    comparison={}
                    for task,row in results.items():
                        previous=baseline.get(task)
                        comparison[task]={'status':'complete' if previous else 'baseline_result_missing',
                            'baseline_pct':previous['accuracy_pct'] if previous else None,
                            'mxfp4_pct':row['accuracy_pct'],
                            'delta_pp':row['accuracy_pct']-previous['accuracy_pct'] if previous else None,
                            'baseline_source':previous['result_file'] if previous else None,
                            'mxfp4_source':row['result_file']}
                    save(root/'comparison.json',comparison)
                save(root/'state.json',state)
        state.update(status='complete',finished_at=now());save(root/'state.json',state)
    except Exception as error:
        record=state['phases'].get(state.get('current_phase'))
        if record and record['status'] in ('running','validating'):record['status']='failed'
        state.update(status='failed',error=repr(error),finished_at=now());save(root/'state.json',state)
        raise
