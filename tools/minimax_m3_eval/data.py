"""Pin the existing audited task data and check actual MiniMax chat inputs."""
import hashlib
import json
from pathlib import Path
import shutil

from .download import save


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def install(manifest):
    from datasets import Dataset, DatasetDict
    from lighteval.tasks.lighteval_task import LightevalTask
    routes = json.loads(Path(manifest).read_text())
    for route in routes.values():
        for entry in route['splits'].values():
            if sha256(entry['path']) != entry['sha256']:
                raise ValueError(f"Pinned task data changed: {entry['path']}")

    def load(task):
        route = routes[task.dataset_path]
        if task.dataset_revision != route['registry_revision'] or task.dataset_config_name != route['subset']:
            raise ValueError(f"Task registry no longer matches pinned data: {task.dataset_path}")
        return DatasetDict({key: Dataset.from_file(value['path'])
                            for key, value in route['splits'].items()})

    LightevalTask.download_dataset_worker = staticmethod(load)


def prepare(root):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from lighteval.tasks.lighteval_task import LightevalTask
    from lighteval.tasks.prompt_manager import PromptManager
    from lighteval.tasks.registry import Registry
    from vllm.model_executor.models.minimax_m3.integration import register
    register()
    root = Path(root).resolve()
    config = json.loads((root / 'config.json').read_text())
    source_root = Path(config['source_dataset_root'])
    evidence = json.loads((source_root / 'evaluation/data/manifest.json').read_text())
    routes = {}
    for name in config['tasks']:
        source = evidence['tasks'][name]
        route = {k: source[k] for k in ('repo', 'subset', 'registry_revision', 'source_revision')}
        route['splits'] = {}
        for split, item in source['splits'].items():
            original = source_root / item['path']
            if sha256(original) != item['sha256']:
                raise ValueError(f"Source dataset changed: {original}")
            dest = root / 'data' / name / (split + '.arrow')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(dest)
            shutil.copyfile(original, dest)
            route['splits'][split] = {'path': str(dest), 'sha256': sha256(dest),
                'count': len(Dataset.from_file(str(dest)))}
        routes[source['repo']] = route
    save(root / 'datasets.json', routes)
    install(root / 'datasets.json')
    tokenizer = AutoTokenizer.from_pretrained(config['model'], trust_remote_code=False, local_files_only=True)
    prompts = PromptManager(use_chat_template=True, tokenizer=tokenizer)
    registry = Registry(tasks=','.join(t['registry'] for t in config['tasks'].values()),
                        custom_tasks=str(Path.cwd() / 'tasks/custom_tasks.py'))
    tasks = registry.load_tasks()
    LightevalTask.load_datasets(tasks, 1)
    audit = {}
    for name, definition in config['tasks'].items():
        task = next(task for task in tasks.values() if task.full_name == definition['registry'])
        docs = task.get_docs(definition['count'])
        if len(docs) != definition['count'] or any(doc.num_samples != definition['samples_per_question'] for doc in docs):
            raise ValueError(f"Task coverage/sample count mismatch: {name}")
        rows = []
        for doc in docs:
            prompt = prompts.prepare_prompt(doc)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if len(ids) + definition['max_new_tokens'] > config['max_model_len']:
                raise ValueError(f"Prompt would be truncated: {name}, {doc.id}, {len(ids)} tokens")
            rows.append({'id': doc.id, 'prompt': prompt, 'input_ids': ids, 'gold': doc.get_golds(),
                         'num_samples': doc.num_samples})
        save(root / 'data' / name / 'prompts.json', rows)
        audit[name] = {'count': len(rows), 'samples_per_question': definition['samples_per_question'],
            'max_prompt_tokens': max(len(row['input_ids']) for row in rows),
            'prompts_sha256': sha256(root / 'data' / name / 'prompts.json'),
            'metric_names': [metric.metric_name for metric in task.metrics]}
    save(root / 'preparation.json', {'complete': True, 'tasks': audit,
        'datasets_sha256': sha256(root / 'datasets.json'),
        'tokenizer_sha256': {p.name: sha256(p) for p in Path(config['model']).iterdir()
            if p.is_file() and (p.name.startswith('tokenizer') or p.name == 'chat_template.jinja')}})
    print(json.dumps(audit, indent=2))
