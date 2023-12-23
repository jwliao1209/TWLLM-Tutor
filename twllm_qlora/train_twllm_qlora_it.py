import math
import yaml
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

import wandb
from lib.configs import get_bnb_config
from lib.dataset import AcademicDataset
from lib.optimization.optimizer import get_optimizer
from lib.trainer import InstructionTuningTrainer
from lib.utils.data_utils import collate_func, read_json
from lib.utils.train_utils import set_random_seeds


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--config_path", type=str, default="configs/twllm_qlora_it.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    # Fix random seed
    set_random_seeds()
    args = parse_arguments()

    # Split config
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    device_config = config["device"]

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["base_model_path"], use_fast=False
    )

    train_data = read_json(data_config["train_data_path"])
    valid_data = read_json(data_config["valid_data_path"])

    train_dataset = AcademicDataset(
        train_data, tokenizer,
        max_length=512,
        is_train=True,
        with_answer_details=data_config["with_answer_details"],
    )
    valid_dataset = AcademicDataset(
        valid_data, tokenizer,
        max_length=2048,
        is_train=False,
        with_answer_details=data_config["with_answer_details"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=collate_func,
        num_workers=2,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_func,
    )

    # Prepare model
    bnb_config = get_bnb_config()
    device = torch.device(f"cuda:{device_config['id']}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_config["base_model_path"],
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    peft_config = LoraConfig(
        r=model_config["lora_rank"],
        lora_alpha=model_config["lora_alpha"],
        lora_dropout=model_config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(
        model,
        optimizer_name=train_config["optimizer"],
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )
    num_update_steps_per_epoch = math.ceil(len(train_loader) / train_config["accum_grad_step"])
    max_train_steps = train_config["epoch"] * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=train_config["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=math.ceil(train_config["warm_up_step"] / train_config["accum_grad_step"]),
        num_training_steps=max_train_steps,
    )

    # Prepared logger
    wandb.init(
        project="adl_final_project",
        group=f"LLM-IT-{Path(args.train_data_path).stem}-{Path(args.valid_data_path).stem}",
        config=config,
    )
    wandb.watch(model, log="all")

    # Start training
    trainer = InstructionTuningTrainer(
        tokenizer=tokenizer,
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        accum_grad_step=train_config["accum_grad_step"],
        lr_scheduler=lr_scheduler,
        logger=wandb,
    )
    trainer.fit(epoch=train_config["epoch"])
