import math
from argparse import ArgumentParser, Namespace

import torch
from dataset import LLMMCDataset, collate_func
from optimization.optimizer import get_optimizer
from peft import PeftModel
from torch.utils.data import DataLoader
from trainer import MCTrainer
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_scheduler)
from utils.data_utils import read_json
from utils.train_utils import set_random_seeds

import wandb
from configs import get_bnb_config


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--base_model_path", type=str,
                        default="model_weight/Taiwan-LLM-7B-v2.0-chat",
                        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
                        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9).")
    parser.add_argument("--train_data_path", type=str,
                        default="data/train_data/train.json",
                        help="Path to train data.")
    parser.add_argument("--valid_data_path", type=str,
                        default="data/train_data/valid.json",
                        help="Path to validation data.")
    parser.add_argument("--batch_size", type=int,
                        default=8,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=2,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=100,
                        help="number of epochs")
    parser.add_argument("--optimizer", type=str,
                        default="adamw",
                        help="optimizer")
    parser.add_argument("--lr", type=float,
                        default=2e-4,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=0,
                        help="weight decay")
    parser.add_argument("--lr_scheduler", type=str,
                        default="constant",
                        help="learning rate scheduler: linear, constant, cosine, cosine_warmup")
    parser.add_argument("--warm_up_step", type=int,
                        default=0,
                        help="number of warm up steps")
    parser.add_argument("--lora_rank", type=int,
                        default=8,
                        help="rank of lora")
    parser.add_argument("--device_id", type=int,
                        default=0,
                        help="device id")
    return parser.parse_args()


if __name__ == "__main__":
    # Fix random seed
    set_random_seeds()

    args = parse_arguments()

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

    train_data = read_json(args.train_data_path)
    valid_data = read_json(args.valid_data_path)

    train_dataset = LLMMCDataset(train_data, tokenizer, max_length=1024)
    valid_dataset = LLMMCDataset(valid_data, tokenizer, max_length=1024)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_func)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)

    # Prepare model
    bnb_config = get_bnb_config()
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=4,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

    model = PeftModel.from_pretrained(
        model,
        args.base_model_path,
        subfolder="loftq_init",
        is_trainable=True,
    )
    model.print_trainable_parameters()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(model, lr=args.lr, optimizer_name=args.optimizer, weight_decay=args.weight_decay)
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.accum_grad_step)
    max_train_steps = args.epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(args.warm_up_step / args.accum_grad_step),
        num_training_steps=max_train_steps,
    )

    # Prepared logger
    wandb.init(
        project="adl_final_project",
        name="twllm_mc_loftQ",
        group="twllm_mc_loftQ",
        config={
            "tokenizer": args.base_model_path,
            "model": args.base_model_path,
            "epoch": args.epoch,
            "batch_size": args.batch_size,
            "accum_grad_step": args.accum_grad_step,
            "optimizer": "adamw",
            "lr_scheduler": args.lr_scheduler,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warm_up_step": args.warm_up_step,
            "lora_rank": args.lora_rank,
        }
    )
    wandb.watch(model, log="all")

    # Start training
    trainer = MCTrainer(
        tokenizer=tokenizer,
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
