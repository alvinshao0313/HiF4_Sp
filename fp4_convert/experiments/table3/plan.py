"""Fixed inputs and reusable evidence for the two-model, six-task scale study."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / 'Native_NVFP4_HiF4_Linear_Puncture/results/paper_baselines'
PREVIOUS = RESULTS / 'single_methods_20261120T050800Z/all_single_methods'
ROOT = RESULTS / 'table3_six_tasks_20261126'
METHODS = {'H06': ('hif4', 'scale_search'), 'H18': ('hif4', 'matrix_c'),
           'M06': ('mxfp4', 'uos'), 'M07': ('mxfp4', 'g256'), 'M08': ('mxfp4', 'g512')}
QA = {'arc_challenge': 1172, 'openbookqa': 500, 'boolq': 3270}
REASONING = {'gsm8k': 1319, 'gpqa': 198, 'lcb': 175}
METRICS = {'arc_challenge': 'acc_norm', 'openbookqa': 'acc_norm', 'boolq': 'acc',
           'gsm8k': 'extractive_match', 'gpqa': 'gpqa_pass@k:k=1', 'lcb': 'codegen_pass@1:16'}
SEEDS = (1234, 1235, 1236)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    with Path(path).open('rb') as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    h = sha256()
    for block in iter(lambda: stream.read(4 << 20), b''):
        h.update(block)
    return h.hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode()


def tasks():
    return [(task, 1234) for task in QA] + [(task, seed) for seed in SEEDS for task in REASONING]


def make_plan(root):
    prior = read(PREVIOUS/'plan.json')
    data = read(PREVIOUS/'data/manifest.json')
    routes = {k:data[k] for k in ('Idavidrein/gpqa','openai/gsm8k','lighteval/code_generation_lite')}
    configs, inputs = [], {}
    for model, source in prior['sources'].items():
        for suffix, (target, method) in METHODS.items():
            key = f'{model.upper()}-{suffix}'
            configs.append(dict(id=key,model=model,target=target,method=method,source=source,
                                checkpoint=str(root/'models'/key)))
        qa_reports = {p.resolve() for p in PREVIOUS.rglob(f'verified/{model}-gptq-hif4.json')}
        if len(qa_reports) != 1:
            raise ValueError(f'Expected one prior accepted QA reference for {model}: {qa_reports}')
        ref = read(qa_reports.pop())
        if ref['exit_code'] != 0 or ref['config']['source'] != source:
            raise ValueError('Prior QA input reference has a different source or failed')
        inputs[model] = {task: ref['evidence'][task] for task in QA}
        for task in REASONING:
            path = PREVIOUS/'reasoning'/f'{model}-gptq-hif4--{task}--1234'/'prompt_audit.json'
            inputs[model][task] = {'file':str(path), 'sha256':digest(path)}
    return dict(configs=configs, reasoning={m:{k:prior['reasoning'][m][k] for k in REASONING}
                                            for m in prior['sources']},
                qa={'counts':QA, 'tp':1, 'seed':1234, 'batch':128, 'num_fewshot':0,
                    'apply_chat_template':False, 'max_model_len':4096},
                seeds=SEEDS,tp=1,reference_inputs=inputs,
                data_manifest_sha256=sha256(json_bytes(routes)).hexdigest(),
                source_plan=str(PREVIOUS/'plan.json'),source_plan_sha256=digest(PREVIOUS/'plan.json')), routes


def prior_inventory(plan, root):
    """Baseline results are references, never runnable assignments."""
    source = REPO/'LaTeX/tables/manual_scores.tex'
    rows = re.findall(r'^\\setpaperresult\{([^}]+)\}\{([^}]+)\}\{([0-9.]+)\}',
                      source.read_text(), re.M)
    references = {f'{m}-{kind}': {} for m in ('N9','Q4Q') for kind in ('N0','H01','M01')}
    for key, task, score in rows:
        if key in references and task in METRICS:
            references[key][task] = score
    eligible = {}
    known = {(c['source'],c['target'],c['method']):c['id'] for c in plan['configs']}
    for record in RESULTS.rglob('conversion.json'):
        if record.is_symlink() or root in record.parents:
            continue
        info = read(record)
        key = known.get((info.get('source'),info.get('target'),info.get('method')))
        if key is not None:
            if key in eligible and eligible[key] != str(record.parent):
                raise ValueError(f'Multiple candidate weights for {key}: {record.parent}, {eligible[key]}')
            eligible[key] = str(record.parent)
    # These are maintained proofs, not labels inferred from filenames of diagnostic runs.
    reports = REPO/'LaTeX/tables/score_sources.json'
    previous_ids = Counter()
    for filename in read(reports)['source_sha256']:
        report = read(REPO/filename)
        config = report['config']
        key = config['id'] if isinstance(config,dict) else config
        previous_ids[key] += 1
    if set(previous_ids) & {c['id'] for c in plan['configs']}:
        raise ValueError('Existing Table 3 rounds already in the single-method archive: reconcile before launching')
    return dict(reference_source=str(source), references_percent=references,
                reference_missing={c:sorted(set(METRICS)-set(v)) for c,v in references.items() if len(v)<6},
                prior_single_method_verified=str(reports), prior_configuration_report_counts=previous_ids,
                reusable_export_candidates=eligible,
                configuration_count=len(plan['configs']), maximum_new_rounds=len(plan['configs'])*len(tasks()))
