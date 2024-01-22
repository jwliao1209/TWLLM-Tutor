import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch import nn
from peft import LoftQConfig, LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from src.constants import INSTRUCTION_TUNING, MULTIPLE_CHOICE


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Quantize a model with LoftQ.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the fp32/16 model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="The access token to download model from HuggingFace Hub.",
    )
    parser.add_argument(
        "--nbit",
        type=int,
        default=4,
        help="The quantized nbit",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
        help="The alternating steps in LoftQ",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./model_weight",
        help="The directory to save the quantized model",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default=INSTRUCTION_TUNING,
        choices=[INSTRUCTION_TUNING, MULTIPLE_CHOICE],
    )
    return parser.parse_args()


class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")
    return


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)
    return


if __name__ == "__main__":
    print("Status: Quantizing and saving the model")
    args = parse_arguments()

    # Save LoftQ model
    model_name = f"{args.base_model_path.split('/')[-1]}-{args.nbit}bit-{args.lora_rank}rank-{args.task_type}"
    base_model_dir = os.path.join(args.save_dir, model_name)
    lora_model_dir = os.path.join(args.save_dir, model_name, "loftq_init")

    if not Path(base_model_dir).exists() or not Path(lora_model_dir).exists():
        # Download weights and configure LoRA
        print(f"Loading model and tokenizer from {args.base_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, token=args.token, trust_remote_code=True)

        if args.task_type == MULTIPLE_CHOICE:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.base_model_path,
                num_labels=4,
                token=args.token,
            )
            task_type = TaskType.SEQ_CLS
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

        elif args.task_type == INSTRUCTION_TUNING:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_path,
                token=args.token,
                trust_remote_code=False
            )
            task_type = TaskType.CAUSAL_LM
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

        else:
            raise ValueError(f"Unknown task type: {args.task_type}")

        # Config of LoftQ
        loftq_config = LoftQConfig(loftq_bits=args.nbit, loftq_iter=args.iter)

        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=True,
            r=args.lora_rank,
            lora_alpha=16 if task_type is TaskType.CAUSAL_LM else args.lora_rank,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights="loftq_init",
            loftq_config=loftq_config,
        )

        # Obtain LoftQ model
        lora_model = get_peft_model(model, lora_config)
        base_model = lora_model.get_base_model()

        # save lora adapters first
        lora_model.base_model.peft_config["default"].base_base_model_path = base_model_dir  # This can be a local path or Hub model id
        lora_model.base_model.peft_config["default"].init_lora_weights = True  # Don't apply LoftQ when loading again

        lora_model.save_pretrained(lora_model_dir)
        print_model(lora_model, "lora_model")

        # remove lora adapters and save the backbone
        unwrap_model(base_model)
        base_model.save_pretrained(base_model_dir)
        tokenizer.save_pretrained(base_model_dir)

        print_model(base_model, "base_model")

    print(f"Base model saved to {base_model_dir}, LoraQ model saved to {lora_model_dir}")
