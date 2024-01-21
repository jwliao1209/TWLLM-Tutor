import os
import logging
from tqdm import tqdm
from argparse import Namespace, ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.constants import INSTRUCTION_TUNING, MULTIPLE_CHOICE, FEW_SHOT, QLORA, LOFTQ, ZERO_SHOT, CONFIG_FILE
from src.configs import get_bnb_config
from src.data.dataset import InstructionDataset, MultipleChoiceDataset
from src.metric.accuracy import correcter
from src.utils.data_utils import read_json, write_json, collate_func, load_config
from src.utils.train_utils import set_random_seeds, dict_to_device


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO, # logging.INFO, logging.DEBUG,
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoint/twllm-qlora-instruction_tuning-train_QB_history_9000-valid_GSAT_history-108-112_97/01-16-21-17-00/epoch=1_acc=0.3505")
    parser.add_argument("--prompt_prefix", type=str,
                        default="", choices=["breath", "career", "die", "no_fingers", "step_by_step", "tips"],
                        help="Prompt prefix.")
    parser.add_argument("--test_data_path", type=str,
                        default="data/train_data/GSAT_social/valid_GSAT_history-108-112_97.json",
                        help="Path to test data.")
    parser.add_argument("--output_path", type=str,
                        default="prediction/prediction_test.json",
                        help="output path")
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    logger = logging.getLogger("TAIWAN-LLM Tutor")
    config = load_config(os.path.join(args.checkpoint_path, CONFIG_FILE))

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_path,
        use_fast=False,
    )

    test_data = read_json(args.test_data_path)

    if config.model.finetune_type == MULTIPLE_CHOICE:
        Dataset = MultipleChoiceDataset
    else:
        Dataset = InstructionDataset

    test_dataset = Dataset(
        test_data, tokenizer,
        max_length=config.dataset.valid.max_length,
        is_train=False,
        with_answer_details=True,
        prompt_prefix=args.prompt_prefix,
        with_incontext=True if config.model.finetune_type == FEW_SHOT else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_func,
        num_workers=1,
    )

    # Prepare model
    bnb_config = get_bnb_config()
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    if config.model.finetune_type in [ZERO_SHOT, FEW_SHOT]:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )

    elif config.model.finetune_type == INSTRUCTION_TUNING:
        if config.model.adapter == QLORA:
            model = AutoModelForCausalLM.from_pretrained(
                config.model.base_model_path,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
            )
            model = PeftModel.from_pretrained(
                model,
                args.checkpoint_path
            )

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
                args.checkpoint_path,
                subfolder="loftq_init",
                is_trainable=False,
            )

    model.eval()
    correct_num = 0
    prediction_list = []
    test_bar = tqdm(test_loader, desc=f"Testing")
    for i, batch_data in enumerate(test_bar, start=1):
        with torch.no_grad():
            with torch.cuda.amp.autocast(
                dtype=torch.bfloat16
            ):
                batch_data = dict_to_device(batch_data, device)
                generated_tokens = model.generate(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    max_new_tokens=config.trainer.max_new_tokens,
                )
                generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                generations = generations.replace(batch_data["prompt"][0], "").strip()

                is_correct = correcter(generations, batch_data['answer'][0], batch_data['answer_description'][0])
                correct_num = (correct_num + 1) if is_correct else correct_num

                test_bar.set_postfix(
                    {
                        "correct_num": correct_num,
                        "acc": correct_num / i,
                    }
                )

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
