import os
import logging
from typing import Optional
from tqdm import tqdm
from argparse import Namespace, ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.constants import TWLLM, BERT
from src.constants import INSTRUCTION_TUNING, MULTIPLE_CHOICE
from src.constants import FEW_SHOT
from src.constants import CONFIG_FILE, LABEL_TO_OPTION
from src.data.dataset import InstructionDataset, TWLLMMultipleChoiceDataset, BERTMultipleChoiceDataset
from src.metric.accuracy import correcter, get_correct_num
from src.model.prepared_model import get_model
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
                        default="data/train_data/GSAT_social/valid_GSAT_history.json",
                        help="Path to test data.")
    parser.add_argument("--output_path", type=str,
                        default="prediction/prediction_test.json",
                        help="output path")
    return parser.parse_args()


@torch.no_grad()
def test(
    finetune_type,
    tokenizer,
    model,
    test_loader,
    fp32: bool = False,
    max_new_tokens: Optional[int] = None,
) -> None:

    model.eval()
    correct_num = 0
    prediction_list = []
    test_bar = tqdm(test_loader, desc=f"Testing")

    for i, batch_data in enumerate(test_bar, start=1):
        with torch.cuda.amp.autocast(
                dtype=torch.bfloat16 if device.type == "cuda" \
                    and not fp32 else torch.float32
        ):
            batch_data = dict_to_device(batch_data, device)

            if finetune_type == INSTRUCTION_TUNING:
                generated_tokens = model.generate(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                    max_new_tokens=max_new_tokens,
                )

                generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                generations = generations.replace(batch_data["prompt"][0], "").strip()

                is_correct = correcter(
                    generation=generations,
                    answer=batch_data['answer'][0],
                    description=batch_data['answer_description'][0]
                )
                correct_num = (correct_num + 1) if is_correct else correct_num
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

                logger.debug(f"Question:\n{batch_data['prompt'][0]}\n")
                logger.debug(f"Answer:\n{batch_data['answer'][0]}\n")
                logger.debug(f"Prediction:\n{generations}\n")
                logger.debug(f"Is Correct: {is_correct}")

            elif finetune_type == MULTIPLE_CHOICE:
                preds = model(
                    input_ids=batch_data["input_ids"],
                    attention_mask=batch_data["attention_mask"],
                ).logits.argmax(dim=-1)

                is_correct = bool(get_correct_num(preds, batch_data["labels"]))
                correct_num = (correct_num + 1) if is_correct else correct_num

                prediction_list.append(
                    {
                        "id": int(batch_data["id"][0]),
                        "year": batch_data["year"][0],
                        "question": batch_data["question"][0],
                        "generation": LABEL_TO_OPTION[int(preds)],
                        "answer": batch_data["answer"][0],
                        "answer_details": batch_data["answer_description"][0],
                        "is_correct": is_correct,
                    }
                )

            test_bar.set_postfix(
                {
                    "correct_num": correct_num,
                    "acc": correct_num / i,
                }
            )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    write_json(prediction_list, args.output_path)
    return


if __name__ == "__main__":
    logger = logging.getLogger("Taiwan-LLM Tutor")

    # Fix random seed
    set_random_seeds()

    # Set config
    args = parse_arguments()
    config = load_config(os.path.join(args.checkpoint_path, CONFIG_FILE))

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer.name,
        use_fast=False,
    )

    if config.model.finetune_type == INSTRUCTION_TUNING:
        Dataset = InstructionDataset
    elif config.model.finetune_type == MULTIPLE_CHOICE:
        if config.model.name == TWLLM:
            Dataset = TWLLMMultipleChoiceDataset
        elif config.model.name == BERT:
            Dataset = BERTMultipleChoiceDataset
        else:
            raise ValueError(f"Unsupported model name for MULTIPLE_CHOICE: {config.model.name}")
    else:
        raise ValueError(f"Unsupported finetune type: {config.model.finetune_type}")

    test_dataset = Dataset(
        read_json(args.test_data_path),
        tokenizer,
        is_train=False,
        max_length=config.dataset.valid.max_length,
        with_answer_details=config.dataset.valid.with_answer_details,
        with_incontext=True if config.model.finetune_type == FEW_SHOT else False,
        prompt_prefix=args.prompt_prefix,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataloader.valid.batch_size,
        shuffle=False,
        collate_fn=collate_func,
        num_workers=config.dataloader.valid.num_workers,
    )

    # Prepare model
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(
        tokenizer=tokenizer,
        device=device,
        checkpoint_path=args.checkpoint_path,
        model_type=config.model.name,
        base_model_path=config.model.base_model_path,
        finetune_type=config.model.finetune_type,
        adapter=config.model.adapter if hasattr(config.model, "adapter") else None,
        lora_rank=config.model.lora_rank if hasattr(config.model, "lora_rank") else None,
        lora_alpha=config.model.lora_alpha if hasattr(config.model, "lora_alpha") else None,
        lora_dropout=config.model.lora_dropout if hasattr(config.model, "lora_dropout") else None,
        is_trainable=False,
    )

    # Start testing
    test(
        finetune_type=config.model.finetune_type,
        model=model,
        tokenizer=tokenizer,
        test_loader=test_loader,
        max_new_tokens=config.trainer.max_new_tokens if hasattr(config.trainer, "max_new_tokens") else None,
    )
