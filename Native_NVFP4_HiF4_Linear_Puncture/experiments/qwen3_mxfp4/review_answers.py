"""Offline final-submission review; never overwrites the original JSON scores."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

from ..nvfp4_gpqa.task import extract_final_answer
from .artifacts import file_hash, save


RULES = {
    "scope": "Only the completed final response after the last </think>; no reasoning answers or gold-dependent extraction.",
    "explicit": "Accept case-insensitive JSON/Markdown/LaTeX answer fields and single-letter boxed answers; require agreement of all explicit submissions.",
    "prose": "Accept an unambiguous explicit final/correct answer statement when no structured submission exists.",
    "bare": "Accept a final response consisting solely of one choice letter, allowing quotes/code fences.",
    "review": "Review conflicts, missing final submissions, and disagreements with prose without displaying model identity or gold.",
    "incomplete": "Unfinished thinking and unresolved/absent final submissions score zero; all 1980 answers remain in each denominator.",
    "preservation": "Save as a separate final-submission review, preserving original responses, original parser, and original scores.",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def plain(text):
    text = re.sub(r'\\(?:text|mathrm|mathbf|textbf)\s*\{\s*(answer|[A-D])\s*\}', r'\1', text, flags=re.I)
    return text.translate(str.maketrans('', '', '*`"\'_$')).replace('“', '').replace('”', '').replace('’', '')


FIELD = re.compile(r'\b(?:final\s+)?answer\s*[:=]\s*(?:(?:option|choice)\s+)?\(?\s*([A-D])\b', re.I)
PROSE = re.compile(r'\b(?:(?:the\s+)?(?:correct|final|best)\s+)?(?:answer|choice|option)\s+(?:is|would\s+be|should\s+be)\s*[:,]?\s*(?:(?:option|choice)\s+)?\(?\s*((?-i:[A-D]))\b', re.I)


def boxed_letters(text):
    matches = []
    for match in re.finditer(r'\\boxed\s*\{', text):
        start = match.end()
        depth, end = 1, start
        while depth and end < len(text):
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        if depth:
            continue
        content = text[start:end-1]
        content = re.sub(r'\\(?:text|mathrm|mathbf|textbf|operatorname)\b', '', content)
        content = plain(content).translate(str.maketrans('', '', '{}')).strip()
        letter = re.fullmatch(r'(?:(?:Option|Choice)\s+)?([A-D])[.)]?', content, re.I)
        if letter:
            matches.append({'kind': 'boxed', 'answer': letter[1].upper(), 'text': text[match.start():end]})
    return matches


def parse_final(raw):
    if raw.rfind('<think>') > raw.rfind('</think>'):
        return {'answer': None, 'status': 'unfinished_thinking', 'candidates': [], 'final': ''}
    final = raw.rsplit('</think>', 1)[-1].strip()
    normalized = plain(final)
    explicit = [{'kind': 'answer_field', 'answer': m[1].upper(), 'text': m[0]} for m in FIELD.finditer(normalized)]
    explicit += boxed_letters(final)
    prose = [{'kind': 'answer_statement', 'answer': m[1], 'text': m[0]} for m in PROSE.finditer(normalized)]
    values = {m['answer'] for m in explicit}
    prose_values = {m['answer'] for m in prose}
    candidates = explicit + prose
    if len(values) > 1 or (values and prose_values - values):
        answer, status = None, 'review_conflict'
    elif values:
        answer, status = values.pop(), 'explicit_final'
    elif len(prose_values) == 1:
        answer, status = prose_values.pop(), 'prose_final'
    elif len(prose_values) > 1:
        answer, status = None, 'review_conflict'
    else:
        bare = re.fullmatch(r'\s*([A-D])[.)]?\s*', normalized)
        answer, status = (bare[1], 'bare_final') if bare else (None, 'review_missing')
    return {'answer': answer, 'status': status, 'candidates': candidates, 'final': final}


def extract(root, output):
    root, output = Path(root).absolute(), Path(output).absolute()
    output.mkdir(exist_ok=False)
    comparison = json.loads((root/'comparison.json').read_text())
    manifest = json.loads((root/'workflow_manifest.json').read_text())
    protocol_path = Path(manifest['baseline'])/'protocol_inputs.json'
    records, queue, source_hashes = [], [], {}
    source_hashes[str(root/'comparison.json')] = file_hash(root/'comparison.json')
    source_hashes[str(protocol_path)] = file_hash(protocol_path)
    for model, summary in comparison['results'].items():
        seen = set()
        counts = Counter()
        for cache in summary['raw_caches']:
            source_hashes[cache] = file_hash(cache)
            for row in pq.read_table(cache).to_pylist():
                qid, sample = str(row['sample_id']), row['sample']
                if qid in seen or len(sample['text']) != 10:
                    raise ValueError('Duplicate question or incomplete sampling')
                seen.add(qid)
                for index, raw in enumerate(sample['text']):
                    parsed = parse_final(raw)
                    original, original_status = extract_final_answer(raw)
                    record = {'id': digest(f'{model}:{qid}:{index}')[:16], 'model': model, 'question_id': qid,
                              'sampling_index': index, 'raw_sha256': digest(raw), 'input_sha256': digest(sample['input']),
                              'input_tokens_sha256': digest(json.dumps(sample['input_tokens'], separators=(',', ':'))),
                              'output_tokens': len(sample['output_tokens'][index]), 'original_answer': original,
                              'original_status': original_status, **parsed}
                    records.append(record)
                    counts[parsed['status']] += 1
                    if parsed['status'].startswith('review_'):
                        queue.append({'id': record['id'], 'status': parsed['status'], 'candidates': parsed['candidates'],
                                      'output_tokens': record['output_tokens'], 'final': parsed['final'], 'prompt': sample['input']})
        if len(seen) != 198:
            raise ValueError('Expected all 198 questions')
        print(model, dict(counts), flush=True)
    with (output/'extractions.jsonl').open('x') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    save(output/'blind_review_queue.json', sorted(queue, key=lambda x:x['id']))
    save(output/'manifest.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'source_root': str(root),
        'protocol': str(protocol_path), 'rules': RULES, 'parser_sha256': file_hash(__file__),
        'original_parser_sha256': file_hash(Path(__file__).parent.parent/'nvfp4_gpqa/task.py'),
        'source_sha256': source_hashes, 'extractions_sha256': file_hash(output/'extractions.jsonl'),
        'review_queue_sha256': file_hash(output/'blind_review_queue.json'), 'review_count': len(queue)})
    print('review_count',len(queue),'output',output,flush=True)


def score(output, decisions_path):
    output = Path(output).absolute()
    if (output/'reviewed_summary.json').exists():
        raise FileExistsError('Do not overwrite a completed review')
    manifest = json.loads((output/'manifest.json').read_text())
    if file_hash(__file__) != manifest['parser_sha256']:
        raise ValueError('Parser changed after extraction')
    for name, expected in manifest['source_sha256'].items():
        if file_hash(name) != expected:
            raise ValueError(f'Source changed: {name}')
    for name, key in [('extractions.jsonl','extractions_sha256'),('blind_review_queue.json','review_queue_sha256')]:
        if file_hash(output/name) != manifest[key]:
            raise ValueError(f'Review artifact changed: {name}')
    records = [json.loads(line) for line in (output/'extractions.jsonl').read_text().splitlines()]
    queue = json.loads((output/'blind_review_queue.json').read_text())
    decisions = json.loads(Path(decisions_path).read_text())
    if set(decisions) != {row['id'] for row in queue}:
        raise ValueError('Every queued final submission must have one explicit review decision')
    protocol = {str(row['id']): row for row in json.loads(Path(manifest['protocol']).read_text())}
    original = json.loads((Path(manifest['source_root'])/'comparison.json').read_text())
    groups = {}
    for record in records:
        reference = protocol[record['question_id']]
        if record['input_sha256'] != reference['prompt_sha256'] or record['input_tokens_sha256'] != reference['input_tokens_sha256']:
            raise ValueError('Actual prompt/token/options changed')
        if record['id'] in decisions:
            decision = decisions[record['id']]
            if decision.get('answer') not in (None,'A','B','C','D') or not decision.get('reason'):
                raise ValueError('Each manual decision needs an answer or null and a reason')
            record.update(answer=decision['answer'], review_reason=decision['reason'], status='manual_final' if decision['answer'] else 'no_final_submission')
        record['gold'] = reference['gold']
        record['score'] = int(record['answer'] == reference['gold'])
        record['original_score'] = int(record['original_answer'] == reference['gold'])
        groups.setdefault(record['model'],[]).append(record)
    summaries = {}
    for model, items in groups.items():
        if len(items) != 1980 or sum(r['original_score'] for r in items) != original['results'][model]['correct']:
            raise ValueError('Original scores do not reproduce')
        correct = sum(r['score'] for r in items)
        by_index = [sum(r['score'] for r in items if r['sampling_index']==i) for i in range(10)]
        summaries[model] = {'correct':correct, 'generations':1980, 'accuracy':correct/1980,
            'correct_by_sampling_index':by_index, 'accuracy_by_sampling_index':[n/198 for n in by_index],
            'original_correct': original['results'][model]['correct'], 'original_accuracy':original['results'][model]['accuracy'],
            'final_status':dict(Counter(r['status'] for r in items)),
            'unresolved_count':sum(r['answer'] is None for r in items),
            'changed_answers':sum(r['answer']!=r['original_answer'] for r in items),
            'changed_binary_scores':sum(r['score']!=r['original_score'] for r in items)}
    with (output/'reviewed_scores.jsonl').open('x') as stream:
        for record in records:
            stream.write(json.dumps({k:v for k,v in record.items() if k!='final'},ensure_ascii=False)+'\n')
    save(output/'reviewed_summary.json', {'status':'completed', 'rules':RULES, 'results':summaries,
        'decisions_sha256':file_hash(decisions_path), 'scores_sha256':file_hash(output/'reviewed_scores.jsonl'),
        'original_metrics_preserved':True, 'input_tokens_and_original_scores_verified':True,
        'r64_minus_direct_percentage_points':100*(summaries['r64']['accuracy']-summaries['direct']['accuracy']),
        'direct_minus_native_percentage_points':100*(summaries['direct']['accuracy']-summaries['native']['accuracy']),
        'r64_minus_native_percentage_points':100*(summaries['r64']['accuracy']-summaries['native']['accuracy'])})
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    command = sub.add_parser('extract')
    command.add_argument('--root', type=Path, required=True)
    command.add_argument('--output', type=Path, required=True)
    command = sub.add_parser('score')
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--decisions', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'extract':
        extract(args.root, args.output)
    else:
        score(args.output, args.decisions)
