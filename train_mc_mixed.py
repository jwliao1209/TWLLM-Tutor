import math
import os
from argparse import ArgumentParser, Namespace
from functools import partial

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForMultipleChoice,
                          AutoTokenizer, default_data_collator, get_scheduler)

import wandb
from lib.lib_mc.constants import MC_DATA_FILE_WITH_DATABASE
from lib.lib_mc.preprocess import preprocess_mc_func
from lib.lib_mc.trainer import MCTrainer
from lib.optimizer import get_optimizer
from lib.utils.train_utils import set_random_seeds

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

    def _preprocess_mc_func(data, tokenizer, train=True):
        for field in ["question", "A", "B", "C", "D"]:
            for i in range(len(data[field])):
                if "\\image{" in data[field][i]:
                    data[field][i] = data[field][i].replace("\\image{", "圖")
                    data[field][i] = data[field][i].replace("}", "")

        for field in ["A", "B", "C", "D"]:
            for i in range(len(data[field])):
                data[field][i] = "答案：" + data[field][i]
        return preprocess_mc_func(data, tokenizer, train)

    preprocess_func = partial(_preprocess_mc_func, tokenizer=tokenizer)
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
