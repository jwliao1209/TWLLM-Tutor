import os
import logging
from tqdm import tqdm
from argparse import Namespace, ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from lib.constants import FEW_SHOT, LORA_FINE_TUNE
from lib.configs import get_bnb_config
from lib.dataset import AcademicDataset
from lib.metric.accuracy import correcter
from lib.utils.data_utils import read_json, write_json, collate_func
from lib.utils.train_utils import set_random_seeds, dict_to_device


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO, # logging.INFO, logging.DEBUG,
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
    prompt_prefix=args.prompt_prefix
    test_dataset = AcademicDataset(
        test_data, tokenizer, prompt_prefix,
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
                    "id": int(batch_data['id'][0]),
                    "year": batch_data['year'][0],
                    "prompt": batch_data['prompt'][0],
                    "generation": generations,
                    "answer": batch_data['answer'][0],
                    "answer_details": batch_data['answer_description'][0],
                    "is_correct": is_correct,
                }
            )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    write_json(prediction_list, args.output_path)
