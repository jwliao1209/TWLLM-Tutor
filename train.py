import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, TaskType
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForMultipleChoice

from src.configs import get_bnb_config
from src.constants import PROJECT_NAME
from src.constants import TWLLM, BERT
from src.constants import INSTRUCTION_TUNING, MULTIPLE_CHOICE
from src.constants import QLORA, LOFTQ
from src.constants import CHECKPOINT_DIR, OPTION
from src.data.dataset import InstructionDataset, TWLLMMultipleChoiceDataset, BERTMultipleChoiceDataset 
from src.optim.optimizer import get_optimizer
from src.optim.lr_scheduler import get_lr_scheduler
from src.trainer import InstructionTuningTrainer, MultipleChoiceTrainer
from src.utils.data_utils import collate_func, read_json, flatten_dict, load_config
from src.utils.train_utils import set_random_seeds
from src.utils.time_utils import get_time


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="TAIWAN_LLM Fine Tuning")
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
    train_data = read_json(config.dataset.train.data_path)
    valid_data = read_json(config.dataset.valid.data_path)

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
        train_data, tokenizer,
        is_train=True,
        max_length=config.dataset.train.max_length,
        with_answer_details=config.dataset.train.with_answer_details,
    )
    valid_dataset = Dataset(
        valid_data, tokenizer,
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
    bnb_config = get_bnb_config()
    device = torch.device(f"cuda:{config.device.cuda_id}" if torch.cuda.is_available() else "cpu")

    if config.model.finetune_type == INSTRUCTION_TUNING:
        Trainer = InstructionTuningTrainer

        if config.model.adapter == QLORA:
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

        elif config.model.adapter == LOFTQ:
            # ref: https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning#load-and-train
            bnb_config.bnb_4bit_use_double_quant = False
            model = AutoModelForCausalLM.from_pretrained(
                config.model.base_model_path,
                low_cpu_mem_usage=True,
                quantization_config=bnb_config,
            )
            model = PeftModel.from_pretrained(
                model,
                config.model.base_model_path,
                subfolder="loftq_init",
                is_trainable=True,
            )
            model.print_trainable_parameters()
        
        else:
            raise ValueError(f"Unsupported adapter type: {config.model.adapter}")

    elif config.model.finetune_type == MULTIPLE_CHOICE:
        if config.model.name == TWLLM:
            Trainer = MultipleChoiceTrainer

            if config.model.adapter == QLORA:
                model = AutoModelForSequenceClassification.from_pretrained(
                    config.model.base_model_path,
                    num_labels=len(OPTION),
                    torch_dtype=torch.bfloat16,
                    quantization_config=bnb_config,
                )
                peft_config = LoraConfig(
                    r=config.model.lora_rank,
                    lora_alpha=config.model.lora_alpha,
                    lora_dropout=config.model.lora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                )
                model = prepare_model_for_kbit_training(model)
                model = get_peft_model(model, peft_config)
                model.config.pad_token_id = tokenizer.pad_token_id

            elif config.model.adapter == LOFTQ:
                # ref: https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning#load-and-train
                bnb_config.bnb_4bit_use_double_quant = False
                model = AutoModelForSequenceClassification.from_pretrained(
                    config.model.base_model_path,
                    num_labels=len(OPTION),
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    quantization_config=bnb_config,
                )
                model = PeftModel.from_pretrained(
                    model,
                    config.model.base_model_path,
                    subfolder="loftq_init",
                    is_trainable=True,
                )
                model.print_trainable_parameters()
                model.config.pad_token_id = tokenizer.pad_token_id

            else:
                raise ValueError(f"Unsupported adapter type: {config.model.adapter}")

        elif config.model.name == BERT:
            Trainer = MultipleChoiceTrainer
            model_config = AutoConfig.from_pretrained(
                config.model.base_model_path,
            )
            model = AutoModelForMultipleChoice.from_pretrained(
                config.model.base_model_path,
                trust_remote_code=False,
                config=model_config,
            )
            model.to(device)

        else:
            raise ValueError(f"Unsupported model name: {config.model.name}")
    else:
        raise ValueError(f"Unsupported fine-tune type: {config.model.finetune_type}")

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
        project=PROJECT_NAME,
        group=config.name,
        name=Path(checkpoint_dir).parts[-1],
        config=flatten_dict(config),
    )
    wandb.watch(model, log="all")

    # Start training
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
    trainer.fit(epoch=config.trainer.epoch)
