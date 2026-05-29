import random

import datasets
import torch
import transformers
from datasets import load_dataset


def _get_tokenizer(model_name, hf_token=None):
    if hf_token is None:
        return transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)
    return transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=hf_token)


def get_wikitext2(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    tokenizer = _get_tokenizer(model, hf_token)
    split = "test" if eval_mode else "train"
    data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(data["text"])
    encoded = tokenizer(text, return_tensors="pt")

    if eval_mode:
        return encoded

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = encoded.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_ptb(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    tokenizer = _get_tokenizer(model, hf_token)
    split = "test" if eval_mode else "train"
    data = datasets.load_dataset("ptb_text_only", "penn_treebank", split=split)
    text = " ".join(data["sentence"])
    encoded = tokenizer(text, return_tensors="pt")

    if eval_mode:
        return encoded

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = encoded.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_c4(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    tokenizer = _get_tokenizer(model, hf_token)

    if eval_mode:
        val_data = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        encoded = tokenizer(" ".join(val_data[:1100]["text"]), return_tensors="pt")
        encoded.input_ids = encoded.input_ids[:, : 256 * seqlen]
        return encoded

    train_data = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(train_data) - 1)
            encoded = tokenizer(train_data[idx]["text"], return_tensors="pt")
            if encoded.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, encoded.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = encoded.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_wikitext2_test(seed, seqlen, model):
    del seed
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    testenc = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")
    return testenc


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model="", hf_token=None, eval_mode=False):
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, model, hf_token, eval_mode)
    raise NotImplementedError(f"Unsupported calibration dataset: {name}")
