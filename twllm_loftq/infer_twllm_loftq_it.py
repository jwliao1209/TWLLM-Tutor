import logging
import os
from argparse import ArgumentParser, Namespace

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.configs import get_bnb_config
from lib.constants import FEW_SHOT, LORA_FINE_TUNE
from lib.dataset import InstructionDataset
from lib.trainer import InstructionTuningTrainer
from lib.utils.data_utils import collate_func, read_json, write_json
from lib.utils.train_utils import set_random_seeds

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,  # logging.INFO, logging.DEBUG,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--method", type=str,
                        default="lora-fine-tune",
                        choices=["zero-shot", "few-shot", "lora-fine-tune"],
                        help="support method: zero-shot, few-shot, and lora-fine-tune")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoint",
                        help="Path to the checkpoint")
    parser.add_argument("--base_model_path", type=str,
                        default="model_weight/Taiwan-LLM-7B-v2.0-chat",
                        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
                        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9).")
    parser.add_argument("--subfolder",
                        type=str,
                        default="",
                        help="Path to the saved PEFT checkpoint.")
    parser.add_argument("--prompt_prefix", type=str,
                        default="", choices=["breath", "career", "die", "no_fingers", "step_by_step", "tips"],
                        help="Prompt prefix.")
    parser.add_argument("--test_data_path", type=str,
                        default="data/train_data/valid.json",
                        help="Path to test data.")
    parser.add_argument("--output_path", type=str,
                        default="prediction/prediction.json",
                        help="output path")
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    logger = logging.getLogger("ADL Final Project: TaiwanLLM Tutor")
    test_data = read_json(args.test_data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    test_data = read_json(args.test_data_path)
    prompt_prefix = args.prompt_prefix
    test_dataset = InstructionDataset(
        test_data, tokenizer, prompt_prefix,
        max_length=2048,
        is_train=False,
        with_answer_details=True,
        with_incontext=True if args.method == FEW_SHOT else False,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)

    # Prepare model
    bnb_config = get_bnb_config()
    # ref: https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning#load-and-train
    bnb_config.bnb_4bit_use_double_quant = False
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

    if args.method == LORA_FINE_TUNE:
        model = PeftModel.from_pretrained(
            model,
            args.checkpoint,
            subfolder=args.subfolder,
            is_trainable=False,
        )
    model.eval()

    trainer = InstructionTuningTrainer(
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    prediction_list = trainer.predict(test_loader)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    write_json(prediction_list, args.output_path)
