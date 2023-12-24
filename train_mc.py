import os
import math
from argparse import ArgumentParser, Namespace
from functools import partial

import json
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForMultipleChoice,
                          AutoTokenizer, default_data_collator, get_scheduler)

import wandb
from lib.lib_mc.preprocess import preprocess_mc_func
from lib.lib_mc.trainer import MCTrainer
from lib.optim.optimizer import get_optimizer
from lib.utils.train_utils import set_random_seeds


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Multiple Choice")

    parser.add_argument("--tokenizer_name", type=str,
                        default="bert-base-chinese",
                        help="tokenizer name")
    parser.add_argument("--model_name_or_path", type=str,
                        default="hfl/chinese-bert-wwm-ext",
                        help="model name or path")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        help="Optimizer")
    parser.add_argument("--batch_size", type=int,
                        default=8,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=16,
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
    parser.add_argument("--exp_name", type=str, default="adl_hw1",
                        help="Experiment name for wandb")

    parser.add_argument("--use_train_gsat_83_107", action="store_true")
    parser.add_argument("--use_train_qb_history", action="store_true")
    parser.add_argument("--use_train_qb_civics", action="store_true")
    parser.add_argument("--use_valid_gsat_all", action="store_true")
    parser.add_argument("--use_valid_gsat_history", action="store_true")
    parser.add_argument("--use_valid_gsat_civics", action="store_true")
    parser.add_argument("--use_valid_qb_history", action="store_true")
    return parser.parse_args()


def _preprocess_mc_func_from_new_format(data: dict, tokenizer: AutoTokenizer, train: bool=True) -> dict:
    """Preprocessing function for new format."""
    answer_mapping = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }
    data['answer'] = [answer_mapping[ans] for ans in data['answer']]
    return preprocess_mc_func(data, tokenizer, train)


def preprocess_files(data_files: dict[str, str], fields_to_keep: list[str], tmp_dir: str) -> dict[str, str]:
    """Preprocessing json file to required format and saved to tmp_dir."""
    os.makedirs(tmp_dir, exist_ok=True)
    output = {}
    for key, filename in data_files.items():
        tmp_filename = tmp_dir + "/" + filename.split("/")[-1]
        output[key] = tmp_filename
        data = json.load(open(filename, "r", encoding='utf-8'))
        data = [
            {
                k: sample[k]
                for k in fields_to_keep
            }
            for sample in data
        ]
        json.dump(data, open(tmp_filename, "w", encoding='utf-8'),
                  ensure_ascii=False, indent=4)
    return output


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()

    # Prepared datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
    )

    # from lib.lib_mc.constants import MC_DATA_FILE
    # datasets = load_dataset("json", data_files=MC_DATA_FILE)
    # preprocess_func = partial(preprocess_mc_func, tokenizer=tokenizer)
    data_files = {
        'train_GSAT_83_107': "data/train_data/train_GSAT_social-83-107_1221.json",
        'train_QB_history': "data/train_data/train_QB_history_9000.json",
        'train_QB_civics': "data/train_data/train_QB_civics_2035.json",
        'valid_GSAT_all': "data/train_data/valid_GSAT_social-108-112_213.json",
        'valid_GSAT_history': "data/train_data/valid_GSAT_history-108-112_97.json",
        'valid_GSAT_civics': "data/train_data/valid_GSAT_civics-108-112_79.json",
        'valid_QB_history': "data/train_data/valid_QB_history_205.json",
    }
    data_files = preprocess_files(
        data_files, ['question', 'A', 'B', 'C', 'D', 'answer'], "./tmp")
    datasets = load_dataset(
        "json",
        data_files=data_files,
    )
    preprocess_func = partial(
        _preprocess_mc_func_from_new_format, tokenizer=tokenizer)

    processed_datasets = datasets.map(
        preprocess_func,
        batched=True,
        remove_columns=datasets["train_GSAT_83_107"].column_names
    )

    training_datasets = []
    validation_datasets = []

    if args.use_train_gsat_83_107:
        training_datasets.append(processed_datasets["train_GSAT_83_107"])
    if args.use_train_qb_history:
        training_datasets.append(processed_datasets["train_QB_history"])
    if args.use_train_qb_civics:
        training_datasets.append(processed_datasets["train_QB_civics"])

    if args.use_valid_gsat_all:
        validation_datasets.append(processed_datasets["valid_GSAT_all"])
    if args.use_valid_gsat_history:
        validation_datasets.append(processed_datasets["valid_GSAT_history"])
    if args.use_valid_gsat_civics:
        validation_datasets.append(processed_datasets["valid_GSAT_civics"])
    if args.use_valid_qb_history:
        validation_datasets.append(processed_datasets["valid_QB_history"])

    processed_datasets["train"] = concatenate_datasets(
        training_datasets
    )
    processed_datasets["valid"] = concatenate_datasets(
        validation_datasets
    )

    train_loader = DataLoader(
        processed_datasets["train"],
        num_workers=2,
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )
    valid_loader = DataLoader(
        processed_datasets["valid"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
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
        model, args.optimizer, lr=args.lr, weight_decay=args.weight_decay
    )
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
        project="adl_hw1",
        group=f"mc_final_exp",
        name=args.exp_name,
        config={
            "tokenizer": args.tokenizer_name,
            "model": args.model_name_or_path,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "accum_grad_step": args.accum_grad_step,
            "optimizer": args.optimizer,
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
    )
    trainer.fit(epoch=args.epoch)
    wandb.finish()
