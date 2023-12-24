import math
import wandb

import torch
from torch.utils.data import DataLoader

from argparse import Namespace, ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

from configs import get_bnb_config
from dataset import InstructionDataset, collate_func
from optimizer import get_optimizer
from trainer import Trainer
from utils.train_utils import set_random_seeds
from utils.data_utils import read_json


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
                        default=16,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=1,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=1,
                        help="number of epochs")
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
                        default=16,
                        help="rank of lora")
    parser.add_argument("--device_id", type=int,
                        default=0,
                        help="device id")
    parser.add_argument("--with_answer_details", action="store_true",
                        help="Option of answer details")
    return parser.parse_args()


if __name__ == "__main__":
    # Fix random seed
    set_random_seeds()

    args = parse_arguments()

    # Prepare model
    bnb_config = get_bnb_config()
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        
    )
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=args.lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    print(model["base_model"]["model"])
