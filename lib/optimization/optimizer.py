import math

from torch.optim import SGD, Adam, AdamW, Optimizer
from transformers import get_scheduler

from .lion import Lion


def get_optimizer(
    model,
    optimizer_name: str,
    lr: float = 3e-5,
    weight_decay: float = 0,
) -> Optimizer:

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parames = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    match optimizer_name:
        case "sgd":
            optimizer = SGD
        case "adam":
            optimizer = Adam
        case "adamw":
            optimizer = AdamW
        case "lion":
            optimizer = Lion
        case _:
            raise TypeError("not a optimizer we support")
    return optimizer(optimizer_grouped_parames, lr=lr)


def get_max_train_steps(epoch, train_loader_len, accum_grad_step):
    num_update_steps_per_epoch = math.ceil(train_loader_len / accum_grad_step)
    return epoch * num_update_steps_per_epoch


def get_num_warmup_steps(warm_up_step, accum_grad_step):
    return math.ceil(warm_up_step / accum_grad_step)


def get_lr_scheduler(name, optimizer, epoch, train_loader_len, accum_grad_step, warm_up_step):
    num_warmup_steps = get_num_warmup_steps(warm_up_step, accum_grad_step)
    max_train_steps = get_max_train_steps(epoch, train_loader_len, accum_grad_step, warm_up_step)
    return get_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
