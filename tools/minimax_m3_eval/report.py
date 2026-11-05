"""Validate real saved requests/responses before publishing task accuracy."""
import json
from pathlib import Path

import pyarrow.parquet as pq

from .data import sha256


def summarize(root, phase, name):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    definition = config['tasks'][name]
    target = root / phase / name
    files = list((target / 'output').rglob('results_*.json'))
    caches = list((target / 'model').rglob('GENERATIVE.parquet'))
    if len(files) != 1 or len(caches) != 1:
        raise ValueError(f"Missing/ambiguous results or raw cache: {phase}/{name}")
    result = json.loads(files[0].read_text())['results'][definition['registry']]
    metrics = {k: v for k, v in result.items() if not k.endswith('_stderr')}
    if len(metrics) != 1:
        raise ValueError(f"Unexpected metrics: {metrics}")
    rows = pq.read_table(caches[0]).to_pylist()
    wanted = json.loads((root / 'data' / name / 'prompts.json').read_text())
    if phase == 'qualification':
        wanted = wanted[:2]
    expected = {str(row['id']): row for row in wanted}
    if len(rows) != len(expected) or {str(row['sample_id']) for row in rows} != set(expected):
        raise ValueError(f"Incomplete question coverage: {phase}/{name}")
    completions = 0
    maximum = 0
    for row in rows:
        sample, prepared = row['sample'], expected[str(row['sample_id'])]
        if sample['input'] != prepared['prompt'] or sample['input_tokens'] != prepared['input_ids']:
            raise ValueError(f"Actual prompt/tokens differ from preparation: {name}/{row['sample_id']}")
        if sample['truncated_tokens_count']:
            raise ValueError(f"Truncated prompt: {name}/{row['sample_id']}")
        if len(sample['text']) != definition['samples_per_question'] or len(sample['output_tokens']) != definition['samples_per_question']:
            raise ValueError(f"Wrong number of responses: {name}/{row['sample_id']}")
        if any(not tokens or not text.strip() for tokens, text in zip(sample['output_tokens'], sample['text'])):
            raise ValueError(f"Empty response: {name}/{row['sample_id']}")
        completions += len(sample['text'])
        maximum = max(maximum, max(map(len, sample['output_tokens'])))
    metric, value = next(iter(metrics.items()))
    if not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"Invalid accuracy: {metric}={value}")
    return {'status': 'complete', 'task': name, 'questions': len(rows),
        'responses': completions, 'metric': metric, 'accuracy_pct': 100 * value,
        'maximum_output_tokens': maximum, 'formal': phase == 'formal',
        'raw_cache': str(caches[0]), 'raw_cache_sha256': sha256(caches[0]),
        'result_file': str(files[0]), 'result_sha256': sha256(files[0])}
