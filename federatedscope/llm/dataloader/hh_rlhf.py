# Define Option 2 Alpaca-style template for multi-turn conversations
HH_RLHF_PROMPT_DICT = {
    "multi_turn": (
        "Below is an instruction that describes a task. "
        "The following conversation history provides context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Conversation History:\n{history}\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:"
    )
}

import os
import json
import copy
import pickle
import datasets
import random

from federatedscope.core.splitters.generic.lda_splitter import LDASplitter
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonls, load_jsonl
from federatedscope.llm.dataset.llm_dataset import LLMComparisonDataset, \
    LLMDataset


def _download_hh_rlhf(data_root):
    """Download HH-RLHF dataset via Hugging Face and save splits to disk"""
    os.makedirs(data_root, exist_ok=True)

    # Define paths for saved splits
    raw_split_paths = {
        'train': os.path.join(data_root, 'train_raw.json'),
        'val': os.path.join(data_root, 'val_raw.json'),
        'test': os.path.join(data_root, 'test_raw.json')
    }

    # Load from saved files if they exist
    if all(os.path.exists(p) for p in raw_split_paths.values()):
        train_data = datasets.load_dataset('json', data_files=raw_split_paths['train'])['train']
        val_data = datasets.load_dataset('json', data_files=raw_split_paths['val'])['train']
        test_data = datasets.load_dataset('json', data_files=raw_split_paths['test'])['train']
    else:
        # Download fresh dataset from Hugging Face
        dataset = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base")

        # Split into train/val/test
        train_test = dataset["train"].train_test_split(test_size=0.2, shuffle=True)
        train_data = train_test["train"]
        test_val = train_test["test"].train_test_split(test_size=0.5)
        val_data, test_data = test_val["train"], test_val["test"]

        # Save raw splits to disk
        train_data.to_json(raw_split_paths['train'])
        val_data.to_json(raw_split_paths['val'])
        test_data.to_json(raw_split_paths['test'])

    # Add simulated worker IDs (1024 workers)
    def add_category(example):
        if 'category' not in example:
            example['category'] = str(random.randint(0, 1023))
        return example

    train_data = train_data.map(add_category)
    val_data = val_data.map(lambda x: {'category': str(random.randint(0, 1023))})
    test_data = test_data.map(lambda x: {'category': str(random.randint(0, 1023))})

    return train_data, val_data, test_data


def load_comparison_dataset(data_root, tokenizer, config, max_num_test=-1):
    token_name = os.path.basename(tokenizer.name_or_path)
    cache_paths = [
        os.path.join(data_root, f'hh_{token_name}_{split}.pickle')
        for split in ['train', 'val', 'test']
    ]

    # Load from cache if exists
    if all(os.path.exists(p) for p in cache_paths):
        with open(cache_paths[0], 'rb') as f:
            train_dataset = pickle.load(f)
        with open(cache_paths[1], 'rb') as f:
            val_dataset = pickle.load(f)
        with open(cache_paths[2], 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        # Download and process dataset via Hugging Face
        train_data, val_data, test_data = _download_hh_rlhf(data_root)

        # Convert to LLMComparisonDataset format (adapt fields as needed)
        def format_example(example):
            return {
                "history": example["chosen"].split("Assistant:")[0],
                "instruction": example["chosen"].split("Human:")[-1].split("Assistant:")[0],
                "output_A": example["chosen"].split("Assistant:")[-1],
                "output_B": example["rejected"].split("Assistant:")[-1],
                "choice": 0,  # 0 = chosen (A), 1 = rejected (B)
                "category": example["category"]
            }

        train_data = [format_example(e) for e in train_data]
        val_data = [format_example(e) for e in val_data]
        test_data = [format_example(e) for e in test_data]

        # Create datasets
        train_dataset = LLMComparisonDataset(
            train_data,
            tokenizer,
            prompt_input=HH_RLHF_PROMPT_DICT["multi_turn"],
            prompt_no_input=HH_RLHF_PROMPT_DICT["multi_turn"],
            output_A="output_A",
            output_B="output_B",
            choice="choice"
        )
        val_dataset = LLMComparisonDataset(val_data, tokenizer, ...)  # Same args as above
        test_dataset = LLMComparisonDataset(test_data, tokenizer, ...)

        # Cache datasets
        with open(cache_paths[0], 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(cache_paths[1], 'wb') as f:
            pickle.dump(val_dataset, f)
        with open(cache_paths[2], 'wb') as f:
            pickle.dump(test_dataset, f)

    # Truncate test datasets if needed
    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.win_dataset.input_ids = \
            val_dataset.win_dataset.input_ids[:max_num_test]
        val_dataset.lose_dataset.input_ids = \
            val_dataset.lose_dataset.input_ids[:max_num_test]
        test_dataset.win_dataset.input_ids = \
            test_dataset.win_dataset.input_ids[:max_num_test]
        test_dataset.lose_dataset.input_ids = \
            test_dataset.lose_dataset.input_ids[:max_num_test]

    return train_dataset, val_dataset, test_dataset