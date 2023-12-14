import logging
import os
from argparse import ArgumentParser, Namespace

import torch
from constants import FEW_SHOT, LORA_FINE_TUNE
from dataset import AcademicDataset, collate_func
from metric.accuracy import correcter
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data_utils import read_json, write_json
from utils.train_utils import dict_to_device, set_random_seeds

from configs import get_bnb_config

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,  # logging.DEBUG,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--method", type=str,
                        default="zero-shot",
                        help="support method: zero-shot, few-shot, and lora-fine-tune")
    parser.add_argument("--base_model_path", type=str,
                        default="model_weight/Taiwan-LLM-7B-v2.0-chat",
                        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
                        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9).")
    parser.add_argument("--peft_path",
                        type=str,
                        default="",
                        help="Path to the saved PEFT checkpoint.")
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
    test_dataset = AcademicDataset(
        test_data, tokenizer,
        max_length=2048,
        is_train=False,
        with_answer_details=True,
        with_incontext=True if args.method == FEW_SHOT else False,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)

    # Prepare model
    bnb_config = get_bnb_config()
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    if args.method == LORA_FINE_TUNE:
        model = PeftModel.from_pretrained(model, args.peft_path)
    model.eval()

    correct_num = 0
    prediction_list = []
    test_bar = tqdm(test_loader, desc=f"Testing")
    for _, batch_data in enumerate(test_bar, start=1):
        with torch.no_grad():
            batch_data = dict_to_device(batch_data, device)
            generated_tokens = model.generate(
                input_ids=batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_new_tokens=512,
            )
            generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            generations = generations.replace(batch_data["prompt"][0], "").strip()

            is_correct = correcter(generations, batch_data['answer'][0], batch_data['answer_description'][0])
            if is_correct:
                correct_num += 1

            test_bar.set_postfix({"correct_num": correct_num})

            logger.debug(f"Question:\n{batch_data['prompt'][0]}\n")
            logger.debug(f"Answer:\n{batch_data['answer'][0]}\n")
            logger.debug(f"Prediction:\n{generations}\n")
            logger.debug(f"Is Correct: {is_correct}")

            prediction_list.append(
                {
                    "prompt": batch_data['prompt'][0],
                    "generation": generations,
                    "answer": batch_data['answer'][0],
                    "is_correct": is_correct,
                }
            )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    write_json(prediction_list, args.output_path)
