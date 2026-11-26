"""Evaluate only missing rounds and publish proof after the real scorer exits."""
import json
from pathlib import Path

from fp4_convert.experiments.paper_baselines.results import validate_qa, validate_reasoning

from .jobs import call, now, write
from .plan import METRICS, PREVIOUS, QA, REASONING, SEEDS, digest, read


def reference_rows(path, reasoning):
    if reasoning:
        return {row['id']: row for row in read(path)}
    fields = ('doc_hash','prompt_hash','target_hash','arguments')
    with Path(path).open() as stream:
        rows = (json.loads(line) for line in stream)
        return {row['doc_id']:[row[k] for k in fields] for row in rows}


def check_inputs(candidate, reference, task, count):
    raw = reference.get('file') or reference['samples']
    if digest(raw) != reference['sha256']:
        raise ValueError(f'Changed reference {raw}')
    a = reference_rows(candidate, task in REASONING)
    b = reference_rows(raw, task in REASONING)
    if len(a) != count or any(key not in b or b[key] != val for key,val in a.items()):
        raise ValueError(f'Task inputs or option order differ for {task}: {candidate}')


def check_raw(plan, config, status):
    output = Path(status['output'])
    seed = status['seed']
    counts = status['counts']
    references = plan['reference_inputs'][config['model']]
    results = {}
    if set(counts) <= set(QA):
        result = validate_qa(output, counts)
        for task, count in counts.items():
            samples = result['evidence'][task]
            check_inputs(samples['samples'], references[task], task, count)
            metric = METRICS[task]
            results[task] = dict(results=result['metrics'], score=result['scores'][task][metric],
                                 metric=metric, count=count, samples=samples['samples'],
                                 samples_sha256=samples['sha256'])
    else:
        task, = counts
        count = counts[task]
        engine = read(output/'runtime_config.json')
        expected = dict(tensor_parallel_size=1, kv_cache_dtype='bfloat16', max_model_length=40960,
                        max_num_seqs=128,max_num_batched_tokens=8192,seed=seed)
        if any(engine.get(k) != v for k,v in expected.items()):
            raise ValueError(f'Wrong effective reasoning configuration: {output}')
        result = validate_reasoning(output, plan['reasoning'][config['model']][task], seed, count)
        if result['metric'] != METRICS[task]:
            raise ValueError(f'Wrong reasoning metric {result["metric"]} for {task}')
        prompt = output/'prompt_audit.json'
        check_inputs(prompt, references[task], task, count)
        results[task] = dict(**result,prompt_audit=str(prompt),prompt_sha256=digest(prompt),
                             runtime_config=str(output/'runtime_config.json'))
    for task, row in results.items():
        row.update(seed=seed, task=task, config=config['id'], tp=1, exit_code=0, gpu=status['gpu'],
                   job_status=status['job_status'],results_sha256=digest(row['results']),
                   plan_sha256=status['plan_sha256'], checkpoint_receipt_sha256=status['checkpoint_receipt_sha256'],
                   reference_sha256=references[task]['sha256'], verified_at=now())
    return results


def accepted(root, config, task, seed, qualification=False):
    folder = 'qualification' if qualification else 'verified'
    path = root/folder/f'{config["id"]}--{task}--{seed}.json'
    if not path.is_file():
        return False
    row = read(path)
    status = read(row['job_status'])
    expected = (40 if task in QA else 8) if qualification else (QA|REASONING)[task]
    if (row['config'] != config['id'] or row['task'] != task or row['seed'] != seed
            or row['plan_sha256'] != digest(root/'plan.json') or row['tp'] != 1
            or row['metric'] != METRICS[task] or row['count'] != expected
            or row['exit_code'] != 0 or status['status'] != 'completed' or status['exit_code'] != 0
            or status['config'] != config['id'] or status['qualification'] != qualification
            or row['checkpoint_receipt_sha256'] != digest(root/'exports'/f'{config["id"]}.json')
            or digest(row['results']) != row['results_sha256']):
        raise ValueError(f'Invalid prior Table 3 proof: {path}')
    raw_key = 'samples' if task in QA else 'details'
    if digest(row[raw_key]) != row[raw_key+'_sha256']:
        raise ValueError(f'Changed raw evidence: {path}')
    return True


def commands(root, config, counts, seed, output, qualification):
    if set(counts) <= set(QA):
        model_args = (f"pretrained={config['checkpoint']},dtype=auto,trust_remote_code=True,"
                      "tensor_parallel_size=1,max_model_len=4096,kv_cache_dtype=bfloat16,enforce_eager=False,"
                      "gpu_memory_utilization=0.90,max_num_seqs=128,max_num_batched_tokens=8192,seed=1234")
        cmd = ['fp4_convert.experiments.paper_baselines.entry','lm_eval','--model','vllm',
               '--model_args',model_args,'--tasks',','.join(counts),'--num_fewshot','0','--batch_size','128',
               '--seed','1234','--log_samples','--output_path',output]
        return cmd + (['--limit','40'] if qualification else [])
    task, = counts
    cmd = ['fp4_convert.experiments.paper_baselines.reasoning',root/'plan.json',config['id'],
           task,str(seed),output,'--tp','1']
    return cmd + (['--max-samples',str(counts[task])] if qualification else [])


def publish(root, plan, config, status):
    results = check_raw(plan, config, status)
    folder = 'qualification' if status['qualification'] else 'verified'
    for task, result in results.items():
        path = root/folder/f'{config["id"]}--{task}--{status["seed"]}.json'
        if path.exists():
            if not accepted(root, config, task, status['seed'], status['qualification']):
                raise ValueError(f'Stale proof: {path}')
        else:
            write(path,result)


def batch(root, plan, config, checkpoint, counts, seed, gpu, locks, qualification):
    missing = {task:n for task,n in counts.items()
               if not accepted(root,config,task,seed,qualification)}
    if not missing:
        return
    stage = ('short-' if qualification else '') + ('qa' if set(counts) <= set(QA)
                                                    else f'{next(iter(counts))}-{seed}')
    previous = root/'attempts'/config['id']/stage
    for file in sorted(previous.glob('*/status.json')):
        old = read(file)
        if (old['status']=='completed' and old['exit_code']==0 and old['seed']==seed
                and old['qualification']==qualification and set(old['counts']) & set(missing)
                and old['plan_sha256']==digest(root/'plan.json') and old['checkpoint_receipt_sha256']==checkpoint):
            publish(root, plan, config, old)
            missing = {t:n for t,n in missing.items() if t not in old['counts']}
    if not missing:
        return
    metadata = dict(seed=seed, counts=missing, qualification=qualification, checkpoint_receipt_sha256=checkpoint)
    status = call(root, config, stage, lambda output:commands(root,config,missing,seed,output,qualification),
                  gpu,locks,metadata)
    publish(root,plan,config,status)


def evaluate(root, plan, config, checkpoint, gpu, locks, phase):
    # The prior TP1 capacity audit and full real task runs cover 128 concurrent
    # sequences. Exercise the new model and its worker with real, full-length
    # decoding, as well as the 128-request QA scorer, before full task execution.
    batch(root,plan,config,checkpoint,{t:40 for t in QA},1234,gpu,locks,True)
    batch(root,plan,config,checkpoint,{'gsm8k':8},1234,gpu,locks,True)
    if phase=='qualify':
        return
    batch(root,plan,config,checkpoint,QA,1234,gpu,locks,False)
    for seed in SEEDS:
        for task,n in REASONING.items():
            batch(root,plan,config,checkpoint,{task:n},seed,gpu,locks,False)
