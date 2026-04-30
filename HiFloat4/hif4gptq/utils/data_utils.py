import logging

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


def get_dataset(name: str) -> datasets.DatasetDict:
    logging.info("Loading dataset: %s", name)

    ds_properties = {
        "wikitext2": {"path": "wikitext", "config_name": "wikitext-2-raw-v1"},
        "ptb": {"path": "ptb_text_only", "config_name": "penn_treebank"},
        "c4": {
            "path": "allenai/c4",
            "config_name": "allenai--c4",
            "data_files": {
                "train": "en/c4-train.00000-of-01024.json.gz",
                "validation": "en/c4-validation.00000-of-00008.json.gz",
            },
            "cols_to_remove": ["url", "timestamp"],
        },
    }

    if name not in ds_properties:
        raise NotImplementedError(f"Unsupported dataset: {name}")

    props = ds_properties[name]
    ds = datasets.load_dataset(props["path"], name=props.get("config_name"), data_files=props.get("data_files"))

    if "cols_to_remove" in props:
        ds = ds.remove_columns(props["cols_to_remove"])

    logging.info("Loading dataset done")
    return ds


def prepare_test_dataloader(
    dataset: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    seqlen: int = 2048,
    batch_size: int = 1,
) -> DataLoader:
    class TestDataset(Dataset):
        def __init__(self, ds, tok, seq_len=2048):
            tokenized = tok("\n\n".join(ds["text"]), return_tensors="pt")
            nsamples = tokenized.input_ids.numel() // seq_len

            input_ids = tokenized.input_ids[0, : nsamples * seq_len].reshape(nsamples, seq_len)
            attn_mask = tokenized.attention_mask[0, : nsamples * seq_len].reshape(nsamples, seq_len)

            self.input_ids = input_ids
            self.attn_mask = attn_mask

        def __getitem__(self, idx):
            return {"input_ids": self.input_ids[idx], "attention_mask": self.attn_mask[idx]}

        def __len__(self):
            return len(self.input_ids)

    test_ds = TestDataset(dataset, tokenizer, seqlen)
    return DataLoader(test_ds, batch_size=batch_size)
