import logging
import time

import torch


TASK_METRIC_MAP = {
    "mmlu_abstract_algebra": "acc,none",
    "mmlu_business_ethics": "acc,none",
    "mmlu_college_computer_science": "acc,none",
    "mmlu_college_mathematics": "acc,none",
    "mmlu_conceptual_physics": "acc,none",
    "mmlu_formal_logic": "acc,none",
    "mmlu_machine_learning": "acc,none",
    "mmlu_miscellaneous": "acc,none",
    "mmlu_philosophy": "acc,none",
    "mmlu_global_facts": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
    "rte": "acc,none",
    "openbookqa": "acc_norm,none",
    "lambada": "acc,none",
    "lambada_openai": "acc,none",
    "lambada_standard": "acc,none",
    "aime": "acc,none",
}


def _input_device(model):
    if hasattr(model, "_hif4_input_device"):
        return model._hif4_input_device

    if hasattr(model, "hf_device_map"):
        for _, dev in model.hf_device_map.items():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)

    return next(model.parameters()).device


def map_tensors(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(map_tensors(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: map_tensors(v, device) for k, v in obj.items()}
    return obj


def sync_gpus() -> None:
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device=i)


@torch.no_grad()
def evaluate_ppl(model, pad_token_id, testloader) -> float:
    start_time = time.time()
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss(
        reduction="none",
        ignore_index=pad_token_id if pad_token_id is not None else -100,
    )

    device = _input_device(model)
    nlls = []

    logging.info("Evaluating perplexity...")
    for batch in testloader:
        batch = map_tensors(batch, device)
        logits = model(**batch).logits

        logits = logits[:, :-1, :]
        shift_labels = batch["input_ids"][:, 1:]

        nll = loss_fn(logits.permute(0, 2, 1), shift_labels).float()
        mask = shift_labels != loss_fn.ignore_index
        nll_means = (nll * mask).sum(dim=1) / mask.sum(dim=1)
        nlls.append(nll_means)

    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())

    sync_gpus()
    elapsed = time.time() - start_time
    logging.info(
        "Time spent on evaluation: %s",
        time.strftime("%H:%M:%S.{}".format(str(elapsed % 1)[2:])[:13], time.gmtime(elapsed)),
    )

    return ppl.item()


def calculate_avg_accuracy(task_names, results) -> float:
    import lm_eval

    n_tasks = len(task_names)
    acc_cumul = sum(results[task].get(TASK_METRIC_MAP[task], 0.0) for task in results if "mmlu" not in task)

    mmlu_questions = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if "mmlu" in task_name
    }
    if not mmlu_questions:
        return acc_cumul / max(n_tasks, 1)

    acc_mmlu = sum(
        results[task].get(TASK_METRIC_MAP[task], 0.0) * mmlu_questions[task]
        for task in results
        if "mmlu" in task and task in mmlu_questions
    )
    acc_mmlu_avg = acc_mmlu / sum(mmlu_questions.values())
    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(mmlu_questions) + 1)


def eval_zero_shot_task(model, tokenizer, tasks, logger):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=8)
    task_manager = TaskManager()
    task_names = task_manager.match_tasks(tasks)
    logger.info("Selected tasks: %s", task_names)

    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=0, batch_size=2)["results"]

    metric_vals = {}
    for task, result in results.items():
        metric_name = TASK_METRIC_MAP.get(task)
        if metric_name is None:
            continue
        metric_vals[task] = round(result.get(metric_name, 0.0), 4)

    metric_vals["average"] = round(calculate_avg_accuracy(task_names, results), 4)
    logger.info("Zero-shot metrics: %s", metric_vals)
