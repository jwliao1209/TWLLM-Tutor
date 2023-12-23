import yaml
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from easydict import EasyDict

import wandb
from lib.configs import get_bnb_config
from lib.dataset import AcademicDataset
from lib.optimization.optimizer import get_optimizer, get_lr_scheduler
from lib.trainer import InstructionTuningTrainer
from lib.utils.data_utils import collate_func, read_json, flatten_dict
from lib.utils.train_utils import set_random_seeds


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--config_path", type=str,
                        default="configs/twllm_qlora_it-train_QB_history-valid_QB_history_w_answer_details.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    # Fix random seed
    set_random_seeds()
    args = parse_arguments()

    # Split config
    config = EasyDict(yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader))

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_path, use_fast=False
    )

    train_data = read_json(config.dataset.train.data_path)
    valid_data = read_json(config.dataset.valid.data_path)

    train_dataset = AcademicDataset(
        train_data, tokenizer,
        max_length=config.dataset.train.max_length,
        is_train=True,
        with_answer_details=config.dataset.train.with_answer_details,
    )
    valid_dataset = AcademicDataset(
        valid_data, tokenizer,
        max_length=config.dataset.valid.max_length,
        is_train=False,
        with_answer_details=config.dataset.train.with_answer_details,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataloader.train.batch_size,
        shuffle=True,
        collate_fn=collate_func,
        num_workers=config.dataloader.train.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.dataloader.valid.batch_size,
        shuffle=False,
        collate_fn=collate_func,
        num_workers=config.dataloader.valid.num_workers,
    )

    # Prepare model
    bnb_config = get_bnb_config()
    device = torch.device(f"cuda:{config.device.cuda_id}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    peft_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(
        model,
        optimizer_name=config.optim.optimizer.name,
        lr=config.optim.optimizer.lr,
        weight_decay=config.optim.optimizer.weight_decay,
    )

    lr_scheduler = get_lr_scheduler(
        name=config.optim.lr_scheduler.name,
        optimizer=optimizer,
        epoch=config.trainer.epoch,
        train_loader_len=len(train_loader),
        accum_grad_step=config.trainer.accum_grad_step,
        warm_up_step=config.optim.lr_scheduler.warm_up_step,
    )

    # Prepared logger
    wandb.init(
        project="adl_final_project",
        group=f"LLM-IT-{Path(config.dataset.train.data_path).stem}-{Path(config.dataset.valid.data_path).stem}",
        config=flatten_dict(config),
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
        accum_grad_step=config.trainer.accum_grad_step,
        lr_scheduler=lr_scheduler,
        logger=wandb,
    )
    trainer.fit(epoch=config.trainer.epoch)
