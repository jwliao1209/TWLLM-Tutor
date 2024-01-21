import os
from pathlib import Path
from typing import Optional
from itertools import product
from pprint import pprint

from easydict import EasyDict

from src.utils.data_utils import save_config

from src.constants import TRAIN_GSAT_SOCIAL
from src.constants import VALID_GSAT_SOCIAL, VALID_GSAT_HISTORY, VALID_GSAT_CIVICS
from src.constants import TRAIN_QB_SOCIAL, TRAIN_QB_GEOGRAPHY, TRAIN_QB_HISTORY, TRAIN_QB_CIVICS
from src.constants import VALID_QB_HISTORY
from src.constants import CONFIGS_DIR
from src.constants import TWLLM, BERT
from src.constants import INSTRUCTION_TUNING, MULTIPLE_CHOICE
from src.constants import QLORA, LOFTQ


def get_tokenizer_config(model_type: str = TWLLM):
    config = EasyDict()
    config.tokenizer = EasyDict()
    config.tokenizer.name = "bert-base-chinese" if model_type == BERT \
                    else "model_weight/Taiwan-LLM-7B-v2.0-chat"

    return config.tokenizer


def get_dataset_config(
    train_data: str = "data/train_data/GSAT_social/train_GSAT_social.json",
    valid_data: str = "data/train_data/GSAT_social/valid_GSAT_social.json",
    with_answer_details: bool = False,
):
    config = EasyDict()
    config.dataset = EasyDict()
    config.dataset.train = EasyDict()
    config.dataset.train.data_path = train_data
    config.dataset.train.max_length = 512
    config.dataset.train.with_answer_details = with_answer_details
    config.dataset.valid = EasyDict()
    config.dataset.valid.data_path = valid_data
    config.dataset.valid.max_length = 512
    config.dataset.valid.with_answer_details = with_answer_details

    return config.dataset


def get_dataloader_config():
    config = EasyDict()
    config.dataloader = EasyDict()
    config.dataloader.train = EasyDict()
    config.dataloader.train.batch_size = 16
    config.dataloader.train.num_workers = 2
    config.dataloader.valid = EasyDict()
    config.dataloader.valid.batch_size = 1
    config.dataloader.valid.num_workers = 1

    return config.dataloader


def get_bert_config():
    config = EasyDict()
    config.model = EasyDict()
    config.model.name = BERT
    config.model.finetune_type = MULTIPLE_CHOICE  # INSTRUCTION_TUNING, MULTIPLE_CHOICE
    config.model.base_model_path = "hfl/chinese-bert-wwm-ext"
    return config.model


def get_twllm_config(
    finetune_type: str = INSTRUCTION_TUNING,
    adapter: str = QLORA,
):
    config = EasyDict()
    config.model = EasyDict()
    config.model.name = TWLLM
    config.model.finetune_type = finetune_type  # INSTRUCTION_TUNING, MULTIPLE_CHOICE
    config.model.adapter = adapter  # QLORA, LOFTQ
    config.model.base_model_path = "model_weight/Taiwan-LLM-7B-v2.0-chat"
    config.model.lora_rank = 8
    config.model.lora_alpha = 16
    config.model.lora_dropout = 0.1

    if config.model.adapter == LOFTQ:
        config.model.nbit = 4

    return config.model


def get_device_config():
    config = EasyDict()
    config.device = EasyDict()
    config.device.cuda_id = 0

    return config.device


def get_optim_config():
    config = EasyDict()
    config.optim = EasyDict()
    config.optim.optimizer = EasyDict()
    config.optim.optimizer.name = "adamw"  # adamw, lion
    config.optim.optimizer.lr = 2e-4
    config.optim.optimizer.weight_decay = 1e-5
    config.optim.lr_scheduler = EasyDict()
    config.optim.lr_scheduler.name = "constant"  # linear, constant, cosine, cosine_warmup
    config.optim.lr_scheduler.warm_up_step = 0

    return config.optim


def get_trainer_config():
    config = EasyDict()
    config.trainer = EasyDict()
    config.trainer.epoch = 10
    config.trainer.accum_grad_step = 1

    return config.trainer


def get_config(
    model: str,
    train_data: str,
    valid_data: str,
    with_answer_details: bool = False,
    finetune_type: Optional[str] = None,
    adapter: Optional[str] = None,
):
    config = EasyDict()
    config.name = None
    config.tokenizer = get_tokenizer_config()
    config.dataset = get_dataset_config(
        train_data=train_data,
        valid_data=valid_data,
        with_answer_details=with_answer_details,
    )
    config.dataloader = get_dataloader_config()

    if model == BERT:
        config.model = get_bert_config()

    elif model == TWLLM:
        config.model = get_twllm_config(
            finetune_type=finetune_type,
            adapter=adapter,
        )

    config.device = get_device_config()
    config.optim = get_optim_config()
    config.trainer = get_trainer_config()
    config.name = f"{config.model.name}-{config.model.adapter + '_' if hasattr(config.model, 'adapter') else ''}{config.model.finetune_type}-{Path(config.dataset.train.data_path).stem}-{Path(config.dataset.valid.data_path).stem}-{'w' if config.dataset.train.with_answer_details else 'wo'}_answer_details"

    return config


def generate_config():
    config = get_config(
        model=TWLLM,
        finetune_type=INSTRUCTION_TUNING,
        adapter=QLORA,
        train_data="data/train_data/GSAT_social/train_GSAT_social.json",
        valid_data="data/train_data/GSAT_social/valid_GSAT_social.json",
        with_answer_details=True,
    )
    save_config(config, os.path.join(CONFIGS_DIR, f"{config.name}.yaml"))
    return


def generate_all_configs():
    MODEL_LIST = [
        # dict(
        #     model=BERT,
        # ),
        dict(
            model=TWLLM,
            adapter=QLORA,
            finetune_type=INSTRUCTION_TUNING,
        ),
        dict(
            model=TWLLM,
            adapter=QLORA,
            finetune_type=MULTIPLE_CHOICE,
        ),
        # dict(
        #     model=TWLLM,
        #     adapter=LOFTQ,
        #     finetune_type=INSTRUCTION_TUNING,
        # ),
        # dict(
        #     model=TWLLM,
        #     adapter=LOFTQ,
        #     finetune_type=MULTIPLE_CHOICE,
        # )
    ]

    DATA_LIST = [
        dict(
            train_data=TRAIN_GSAT_SOCIAL,
            valid_data=VALID_GSAT_SOCIAL,
            with_answer_details=False,
        ),
        dict(
            train_data=TRAIN_QB_SOCIAL,
            valid_data=VALID_GSAT_SOCIAL,
            with_answer_details=False,
        ),
        dict(
            train_data=TRAIN_QB_SOCIAL,
            valid_data=VALID_GSAT_SOCIAL,
            with_answer_details=True,
        ),
        dict(
            train_data=TRAIN_QB_HISTORY,
            valid_data=VALID_GSAT_HISTORY,
            with_answer_details=False,
        ),
        dict(
            train_data=TRAIN_QB_HISTORY,
            valid_data=VALID_GSAT_HISTORY,
            with_answer_details=True,
        ),
        dict(
            train_data=TRAIN_QB_HISTORY,
            valid_data=VALID_QB_HISTORY,
            with_answer_details=False,
        ),
        dict(
            train_data=TRAIN_QB_HISTORY,
            valid_data=VALID_QB_HISTORY,
            with_answer_details=True,
        ),
        dict(
            train_data=TRAIN_QB_CIVICS,
            valid_data=VALID_GSAT_CIVICS,
            with_answer_details=False,
        ),
        dict(
            train_data=TRAIN_QB_CIVICS,
            valid_data=VALID_GSAT_CIVICS,
            with_answer_details=True,
        ),
    ]

    for i, (model_type, data) in enumerate(product(MODEL_LIST, DATA_LIST), start=1):
        print(f"Parameter {i}:")
        pprint(model_type | data)
        print()
        config = get_config(**model_type, **data)
        save_config(config, os.path.join(CONFIGS_DIR, f"{config.name}.yaml"))
    return


if __name__ == "__main__":
    # generate_config()
    generate_all_configs()
