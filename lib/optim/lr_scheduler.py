import math

from transformers import get_scheduler


def get_max_train_steps(epoch, train_loader_len, accum_grad_step):
    num_update_steps_per_epoch = math.ceil(train_loader_len / accum_grad_step)
    return epoch * num_update_steps_per_epoch


def get_num_warmup_steps(warm_up_step, accum_grad_step):
    return math.ceil(warm_up_step / accum_grad_step)


def get_lr_scheduler(name, optimizer, epoch, train_loader_len, accum_grad_step, warm_up_step):
    num_warmup_steps = get_num_warmup_steps(warm_up_step, accum_grad_step)
    max_train_steps = get_max_train_steps(epoch, train_loader_len, accum_grad_step)
    return get_scheduler(
        name=name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )
