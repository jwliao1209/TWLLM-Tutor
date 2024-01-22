import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.constants import PROJECT_NAME
from src.constants import TWLLM, BERT
from src.constants import INSTRUCTION_TUNING, MULTIPLE_CHOICE
from src.constants import CHECKPOINT_DIR
from src.data.dataset import InstructionDataset, TWLLMMultipleChoiceDataset, BERTMultipleChoiceDataset
from src.optim.optimizer import get_optimizer
from src.optim.lr_scheduler import get_lr_scheduler
from src.trainer import InstructionTuningTrainer, MultipleChoiceTrainer
from src.utils.data_utils import collate_func, read_json, flatten_dict, load_config
from src.utils.train_utils import set_random_seeds
from src.utils.time_utils import get_time
from src.model.prepared_model import get_model


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLM Tutor")
    parser.add_argument("--config_path", type=str,
                        default="configs/twllm_qlora_IT-train_QB_history-valid_GSAT_history_w_answer_details.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    # Fix random seed
    set_random_seeds()

    # Set config
    args = parse_arguments()
    config = load_config(args.config_path)
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, config.name, get_time())
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.name,
        use_fast=False,
    )

    if config.model.finetune_type == INSTRUCTION_TUNING:
        Dataset = InstructionDataset
    elif config.model.finetune_type == MULTIPLE_CHOICE:
        if config.model.name == TWLLM:
            Dataset = TWLLMMultipleChoiceDataset
        elif config.model.name == BERT:
            Dataset = BERTMultipleChoiceDataset
        else:
            raise ValueError(f"Unsupported model name for MULTIPLE_CHOICE: {config.model.name}")
    else:
        raise ValueError(f"Unsupported finetune type: {config.model.finetune_type}")

    train_dataset = Dataset(
        read_json(config.dataset.train.data_path),
        tokenizer,
        is_train=True,
        max_length=config.dataset.train.max_length,
        with_answer_details=config.dataset.train.with_answer_details,
    )
    valid_dataset = Dataset(
        read_json(config.dataset.valid.data_path),
        tokenizer,
        is_train=False,
        max_length=config.dataset.valid.max_length,
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
    device = torch.device(f"cuda:{config.device.cuda_id}" if torch.cuda.is_available() else "cpu")
    model = get_model(
        tokenizer=tokenizer,
        device=device,
        model_type=config.model.name,
        base_model_path=config.model.base_model_path,
        finetune_type=config.model.finetune_type,
        adapter=config.model.adapter if hasattr(config.model, "adapter") else None,
        lora_rank=config.model.lora_rank if hasattr(config.model, "lora_rank") else None,
        lora_alpha=config.model.lora_alpha if hasattr(config.model, "lora_alpha") else None,
        lora_dropout=config.model.lora_dropout if hasattr(config.model, "lora_dropout") else None,
        is_trainable=True,
    )

    # Prepared optimization tools
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
        project=PROJECT_NAME,
        group=config.name,
        name=Path(checkpoint_dir).parts[-1],
        config=flatten_dict(config),
    )
    wandb.watch(model, log="all")

    # Prepared trainer
    if config.model.finetune_type == INSTRUCTION_TUNING:
        Trainer = InstructionTuningTrainer
    elif config.model.finetune_type == MULTIPLE_CHOICE:
        Trainer = MultipleChoiceTrainer
    else:
        raise ValueError(f"Unsupported finetune type: {config.model.finetune_type}")

    trainer = Trainer(
        tokenizer=tokenizer,
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        accum_grad_step=config.trainer.accum_grad_step,
        lr_scheduler=lr_scheduler,
        logger=wandb,
        checkpoint_dir=checkpoint_dir,
        config=config,
    )

    # Start training
    trainer.fit(epoch=config.trainer.epoch)
