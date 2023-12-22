from torch.optim import AdamW, Optimizer

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
        case "adamw":
            return AdamW(optimizer_grouped_parames, lr=lr)
        case "lion":
            return Lion(optimizer_grouped_parames, lr=lr)
        case _:
            raise
