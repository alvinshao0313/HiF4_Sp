"""Whole-model communication/decode checks before scoring real tasks."""
import math
from pathlib import Path
import re

from vllm import LLM, SamplingParams
from vllm.model_executor.models.minimax_m3.evaluation import engine_kwargs
from .download import save


def qualify(args):
    options = engine_kwargs(args.model_path, args.tensor_parallel_size, args.max_model_len,
        args.gpu_memory_utilization, args.batch_size, args.max_num_batched_tokens,
        args.pipeline_parallel_size)
    options['seed'] = 1234
    model = LLM(**options)
    questions = [('What is 2 + 2? Reply with only the final integer.', '4'),
                 ('What is 8 times 7? Reply with only the final integer.', '56')]
    outputs = model.chat([[{'role': 'user', 'content': question}] for question, _ in questions],
        SamplingParams(temperature=args.temperature, top_p=args.top_p,
                       max_tokens=args.max_new_tokens, logprobs=1))
    records = []
    for (_, expected), request in zip(questions, outputs, strict=True):
        output = request.outputs[0]
        final = output.text.rsplit('</mm:think>', 1)[-1]
        if re.findall(r'(?<!\d)-?\d+(?!\d)', final) != [expected]:
            raise ValueError(f"Whole-model arithmetic check failed: {output.text!r}")
        if len(output.token_ids) < 2 or not output.logprobs:
            raise ValueError('Whole-model check did not exercise repeated decoding/logprobs')
        if not all(math.isfinite(value.logprob) for row in output.logprobs for value in row.values()):
            raise ValueError('Nonfinite generated logprob')
        records.append({'text': output.text, 'output_tokens': list(output.token_ids),
                        'finish_reason': output.finish_reason, 'expected': expected})
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    save(root / 'qualification.json', {'status': 'passed', 'tp': args.tensor_parallel_size,
        'pp': args.pipeline_parallel_size, 'max_model_len': args.max_model_len,
        'max_num_seqs': args.batch_size, 'cases': records,
        'coverage': 'Real full checkpoint, two requests, repeated decode, finite logprobs and elementary arithmetic. Not a downstream accuracy score.'})
