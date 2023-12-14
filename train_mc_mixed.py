from datasets import Dataset
from datasets.formatting import format_table, get_formatter, query_table
from collections.abc import Sequence
import math
import os
from argparse import ArgumentParser, Namespace
from functools import partial

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForMultipleChoice,
                          AutoTokenizer, default_data_collator, get_scheduler)

import re
import wandb
from lib.lib_mc.constants import MC_DATA_FILE_WITH_DATABASE
from lib.lib_mc.trainer import MCTrainer
from lib.lib_mc.preprocess import flatten_list, unflatten_list, MC_MAX_SEQ_LEN, MC_ENDING_LEN
from lib.optimizer import get_optimizer
from lib.utils.train_utils import set_random_seeds

from transformers.models.bert import BertTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Multiple Choice")

    parser.add_argument("--not_use_mc_train", action="store_true",
                        help="whether not to use data/train_data_mc/train.json")
    parser.add_argument("--not_use_database_mc_train", action="store_true",
                        help="whether not to use data/train_database_mc/train.json")
    parser.add_argument("--valid_data", type=str,
                        default="data/train_data_mc/valid.json",
                        help="valid data")
    parser.add_argument("--tokenizer_name", type=str,
                        default="bert-base-chinese",
                        help="tokenizer name")
    parser.add_argument("--model_name_or_path", type=str,
                        default="hfl/chinese-bert-wwm-ext",
                        help="model name or path")
    parser.add_argument("--batch_size", type=int,
                        default=8,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=4,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=10,
                        help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=2e-5,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5,
                        help="weight decay")
    parser.add_argument("--lr_scheduler", type=str,
                        default="linear",
                        help="learning rate scheduler")
    parser.add_argument("--warm_up_step", type=int,
                        default=300,
                        help="number of warm up steps")
    parser.add_argument("--device_id", type=int,
                        default=0,
                        help="deivce id")
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="Option of train from scratch")
    parser.add_argument("--bf16", action="store_true",
                        help="Option of using bf16")
    parser.add_argument("--only_test", action="store_true",
                        help="Option of do only testing")

    return parser.parse_args()


class MultiModelDataset(Dataset):
    def _getitem(self, key: int | slice | str | Sequence[int], **kwargs) -> dict | list:
        """
        Can be used to index columns (by string names) or rows (by integer, slice, or list-like of integer indices)
        """
        if isinstance(key, bool):
            raise TypeError(
                "dataset index must be int, str, slice or collection of int, not bool")
        format_type = kwargs["format_type"] if "format_type" in kwargs else self._format_type
        format_columns = kwargs["format_columns"] if "format_columns" in kwargs else self._format_columns
        output_all_columns = (
            kwargs["output_all_columns"] if "output_all_columns" in kwargs else self._output_all_columns
        )
        format_kwargs = kwargs["format_kwargs"] if "format_kwargs" in kwargs else self._format_kwargs
        format_kwargs = format_kwargs if format_kwargs is not None else {}
        formatter = get_formatter(
            format_type, features=self._info.features, **format_kwargs)
        pa_subtable = query_table(
            self._data, key, indices=self._indices if self._indices is not None else None)
        formatted_output = format_table(
            pa_subtable, key, formatter=formatter, format_columns=format_columns, output_all_columns=output_all_columns
        )
        return formatted_output


def replace_and_extract_indices(input_string):
    # Regular expression to find "image{index}" patterns with mixed numbers and alphabets
    pattern = r'image{([a-zA-Z_\d]+)}'
    
    # Find all matches of the pattern in the input string
    matches = re.finditer(pattern, input_string)
    
    # Initialize a list to store the extracted indices
    indices = []
    
    # Replace all matches with "<masked>" and extract indices
    replaced_string = re.sub(pattern, 'photo', input_string)
    for match in matches:
        indices.append(match.group(1))  # Capture the index as a string
    
    return replaced_string, indices

def preprocess_mc_func(data: dict, tokenizer: AutoTokenizer, train=True) -> dict:
    """
    Reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py
    """
    # for field in ["question", "A", "B", "C", "D"]:
    #     for i in range(len(data[field])):
    #         if "\\image{" in data[field][i]:
    #             data[field][i] = data[field][i].replace("\\image{", "圖")
    #             data[field][i] = data[field][i].replace("}", "")

    first_sentences = [[context] * 4 for context in data['question']]
    years = [[context] * 4 for context in data['year']]
    second_sentences = [
        [data['A'][i], data['B'][i], data['C'][i], data['D'][i]]
        for i in range(len(data['A']))
    ]

    first_sentences = flatten_list(first_sentences)
    second_sentences = flatten_list(second_sentences)
    years = flatten_list(years)

    temp = []
    for sentence in first_sentences:
        replaced_string, indices = replace_and_extract_indices(sentence)
        temp.append((replaced_string, indices)) 
    first_sentences = [d[0] for d in temp]
    first_indices = [d[1] for d in temp]

    temp = []
    for sentence in second_sentences:
        replaced_string, indices = replace_and_extract_indices(sentence)
        temp.append((replaced_string, indices)) 
    second_sentences = [d[0] for d in temp]
    second_indices = [d[1] for d in temp]

    merged_indices = [
        first_indices[i] + second_indices[i]
        for i in range(len(second_indices))
    ]

    tokenized_data = tokenizer(
        first_sentences,
        second_sentences,
        max_length=MC_MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
    )

    MAX_NUM_EMBS = 10
    image_embeddings = []
    for i, raw_indices in enumerate(tokenized_data['input_ids']):
        embeddings = torch.zeros(MAX_NUM_EMBS, 1024)
        if len(merged_indices[i]) == 0:
            image_embeddings.append(embeddings)
            continue
        
        image_count = 0
        feature_mapping = {}
        for j in range(len(raw_indices)):
            if raw_indices[j] == 9020: # 9020 == "photo"
                photo_name = merged_indices[i][image_count]
                if photo_name not in feature_mapping:
                    current_size = len(feature_mapping)
                    feature_mapping[photo_name] = -1 - current_size

                    emb_filename = f"./data/train_data_mc/embeddings/{years[i]}_{photo_name}.pth"
                    embeddings[current_size] = torch.load(emb_filename)

                raw_indices[j] = feature_mapping[photo_name]
                raw_indices[j] = 0
                image_count += 1

        image_embeddings.append(embeddings)        
        # Use <= not == because there's a truncated case.
        assert image_count <= len(merged_indices[i]), f"{image_count} != {len(merged_indices[i])}" \
            f"{first_sentences[i]} {second_sentences[i]}"

    tokenized_data["image_embeddings"] = image_embeddings
    tokenized_data = {k: unflatten_list(v, MC_ENDING_LEN) for k, v in tokenized_data.items()}

    if train:
        tokenized_data["labels"] = data['answer']

    return tokenized_data


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()

    # Prepared datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
    )

    datasets = load_dataset("json", data_files={
        'train_data_mc': "data/train_data_mc/train.json",
        'train_database_mc': "data/train_database_mc/train.json",
        'valid': args.valid_data,
    })

    preprocess_func = partial(preprocess_mc_func, tokenizer=tokenizer)
    processed_datasets = datasets.map(
        preprocess_func,
        batched=True,
        remove_columns=datasets["train_database_mc"].column_names
    )

    if args.not_use_mc_train:
        processed_datasets["train"] = processed_datasets["train_database_mc"]
    elif args.not_use_database_mc_train:
        processed_datasets["train"] = processed_datasets["train_data_mc"]
    else:
        processed_datasets["train"] = concatenate_datasets([
            processed_datasets["train_data_mc"],
            processed_datasets["train_database_mc"]
        ])
        # ds: Dataset = processed_datasets["train"]
        # processed_datasets["train"] = MultiModelDataset.from_pandas(ds.to_pandas())

    # processed_datasets["valid"] = concatenate_datasets([processed_datasets["valid"], processed_datasets["test"]])
    train_loader = DataLoader(
        processed_datasets["train"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        processed_datasets["valid"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=8,
        shuffle=False,
    )

    # Prepared model
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    if args.train_from_scratch:
        model = AutoModelForMultipleChoice.from_config(model_config).to(device)
    else:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=False,
            config=model_config,
        ).to(device)

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay)
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.accum_grad_step)
    max_train_steps = args.epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(args.warm_up_step / args.accum_grad_step),
        num_training_steps=max_train_steps,
    )

    # Prepared logger
    wandb.init(
        project="adl-final",
        group="mc",
        name="experiment_mc",
        config={
            "tokenizer": args.tokenizer_name,
            "model": args.model_name_or_path,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "accum_grad_step": args.accum_grad_step,
            "optimizer": "adamw",
            "lr_scheduler": args.lr_scheduler,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_warmup_steps": args.warm_up_step,
        }
    )
    wandb.watch(model, log="all")

    trainer = MCTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        accum_grad_step=args.accum_grad_step,
        lr_scheduler=lr_scheduler,
        logger=wandb,
        bf16=args.bf16,
    )

    if args.only_test:
        trainer.valid_one_epoch()
    else:
        trainer.fit(epoch=args.epoch)
    wandb.finish()
