import math
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Dict, List

import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoModelForMultipleChoice,
                          AutoTokenizer, default_data_collator, get_scheduler)

import re
import wandb
import json

from transformers.models.bert.modeling_bert import BertForMultipleChoice

from src.trainer import BERTMultipleChoiceTrainer
from src.utils.data_utils import flatten_list, unflatten_list
from src.optim.optimizer import get_optimizer
from src.utils.train_utils import set_random_seeds
from src.data.preprocess import preprocess_mc_func
from src.model.vision_bert import VisionBertForMultipleChoice

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MC_MAX_SEQ_LEN = 520
MC_ENDING_LEN = 4



def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Multiple Choice")

    parser.add_argument("--tokenizer_name", type=str,
                        default="bert-base-chinese",
                        help="tokenizer name")
    parser.add_argument("--model_name_or_path", type=str,
                        default="hfl/chinese-bert-wwm-ext",
                        help="model name or path")
    parser.add_argument("--batch_size", type=int,
                        default=8,
                        help="batch size")
    parser.add_argument("--accum_grad_step", type=int,
                        default=16,
                        help="accumulation gradient steps")
    parser.add_argument("--epoch", type=int,
                        default=10,
                        help="number of epochs")
    parser.add_argument("--lr", type=float,
                        default=2e-5,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-5,
                        help="weight decay")
    parser.add_argument("--lr_scheduler", type=str,
                        default="linear",
                        help="learning rate scheduler")
    parser.add_argument("--warm_up_step", type=int,
                        default=300,
                        help="number of warm up steps")
    parser.add_argument("--device_id", type=int,
                        default=0,
                        help="deivce id")
    parser.add_argument("--train_from_scratch", action="store_true",
                        help="Option of train from scratch")
    parser.add_argument("--bf16", action="store_true",
                        help="Option of using bf16")
    parser.add_argument("--exp_name", type=str, default="adl_hw1",
                        help="Experiment name for wandb")

    parser.add_argument("--use_train_qb", action="store_true")
    parser.add_argument("--vision_bert", action="store_true",
                        help="Whether to use vision bert")
    return parser.parse_args()


def replace_and_extract_indices(input_string):
    # Regular expression to find "image{index}" patterns with mixed numbers and alphabets
    pattern = r'image{([a-zA-Z_\d]+)}'

    # Find all matches of the pattern in the input string
    matches = re.finditer(pattern, input_string)

    # Initialize a list to store the extracted indices
    indices = []

    # Replace all matches with "<masked>" and extract indices
    replaced_string = re.sub(pattern, 'photo', input_string)
    for match in matches:
        indices.append(match.group(1))  # Capture the index as a string

    return replaced_string, indices


def _preprocess_mc_func_from_new_format(data: dict, tokenizer: AutoTokenizer, train: bool=True) -> dict:
    """Preprocessing function for new format."""
    answer_mapping = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }

    answer_list = []
    for ans in data['answer']:
        print(ans)
        answer_list.append(answer_mapping[ans])
    data['answer'] = answer_list

    return preprocess_mc_func(data, tokenizer, train)



def vision_preprocess_mc_func(data: dict, tokenizer: AutoTokenizer, train=True) -> dict:
    """
    Reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py
    """
    first_sentences = [[context] * 4 for context in data['question']]
    years = [[context] * 4 for context in data['year']]
    second_sentences = [
        [data['A'][i], data['B'][i], data['C'][i], data['D'][i]]
        for i in range(len(data['A']))
    ]

    first_sentences = flatten_list(first_sentences)
    second_sentences = flatten_list(second_sentences)
    years = flatten_list(years)

    temp = []
    for sentence in first_sentences:
        replaced_string, indices = replace_and_extract_indices(sentence)
        temp.append((replaced_string, indices))
    first_sentences = [d[0] for d in temp]
    first_indices = [d[1] for d in temp]

    temp = []
    for sentence in second_sentences:
        replaced_string, indices = replace_and_extract_indices(sentence)
        temp.append((replaced_string, indices))
    second_sentences = [d[0] for d in temp]
    second_indices = [d[1] for d in temp]

    merged_indices = [
        first_indices[i] + second_indices[i]
        for i in range(len(second_indices))
    ]

    tokenized_data = tokenizer(
        first_sentences,
        second_sentences,
        max_length=MC_MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
    )

    MAX_NUM_EMBS = 10
    image_embeddings = []
    image_indices = []
    for i, raw_indices in enumerate(tokenized_data['input_ids']):
        embeddings = torch.zeros(MAX_NUM_EMBS, 1024)
        image_index = [-1] * MC_MAX_SEQ_LEN
        if len(merged_indices[i]) == 0:
            image_embeddings.append(embeddings)
            image_indices.append(image_index)
            continue

        image_count = 0
        feature_mapping = {}
        for j in range(len(raw_indices)):
            if raw_indices[j] == 9020:  # 9020 == "photo"
                photo_name = merged_indices[i][image_count]
                if photo_name not in feature_mapping:
                    current_size = len(feature_mapping)
                    feature_mapping[photo_name] = current_size

                    emb_filename = f"./data/train_data/GSAT_social_with_image/embeddings/{years[i]}_{photo_name}.pth"
                    embeddings[current_size] = torch.load(emb_filename)

                image_index[j] = feature_mapping[photo_name]
                image_count += 1

        image_embeddings.append(embeddings)
        image_indices.append(image_index)
        # Use <= not == because there's a truncated case.
        assert image_count <= len(merged_indices[i]), f"{image_count} != {len(merged_indices[i])}" \
            f"{first_sentences[i]} {second_sentences[i]}"

    tokenized_data["image_embeddings"] = image_embeddings
    tokenized_data["image_indices"] = image_indices
    tokenized_data = {k: unflatten_list(
        v, MC_ENDING_LEN) for k, v in tokenized_data.items()}

    if train:
        tokenized_data["labels"] = data['answer']

    return tokenized_data


def preprocess_files(data_files: Dict[str, str], fields_to_keep: List[str], tmp_dir: str) -> Dict[str, str]:
    """Preprocessing json file to required format and saved to tmp_dir."""
    os.makedirs(tmp_dir, exist_ok=True)
    output = {}
    for key, filename in data_files.items():
        print(filename)
        tmp_filename = tmp_dir + "/" + filename.split("/")[-1]
        output[key] = tmp_filename
        data = json.load(open(filename, "r", encoding='utf-8'))

        answer_mapping = {
            '(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3,
            '(A': 0, '(B': 1, '(C': 2, '(D': 3,
            'A': 0, 'B': 1, 'C': 2, 'D': 3,
            0: 0, 1: 1, 2: 2, 3: 3,
        }

        parsed_data = []
        for i in range(len(data)):
            if not all([k in data[i] for k in fields_to_keep]):
                continue

            temp = {}
            if 'year' in data[i]:
                temp['year'] = data[i]['year']
            else:
                temp['year'] = -1
            
            for k in fields_to_keep:
                temp[k] = data[i][k]
            temp['answer'] = answer_mapping[data[i]['answer']]
            
            parsed_data.append(temp)
            
        json.dump(parsed_data, open(tmp_filename, "w", encoding='utf-8'),
                  ensure_ascii=False, indent=4)
    return output


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()

    # Prepared datasets
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=True,
        trust_remote_code=False,
    )

    data_files = {
        'train_GSAT': "data/train_data/GSAT_social_with_image/train.json",
        'train_QB_history': "data/train_data/QB_social/train_QB_history.json",
        'train_QB_civics': "data/train_data/QB_social/train_QB_civics.json",
        'train_QB_geo': "data/train_data/QB_social/train_QB_geography.json",
        'valid': "data/train_data/GSAT_social_with_image/valid.json",
    }

    data_files = preprocess_files(
        data_files, ['question', 'A', 'B', 'C', 'D', 'answer'], "./tmp")
    datasets = load_dataset("json", data_files=data_files)

    if args.vision_bert:
        preprocess_func = partial(vision_preprocess_mc_func, tokenizer=tokenizer)
    else:
        preprocess_func = partial(_preprocess_mc_func_from_new_format, tokenizer=tokenizer)

    processed_datasets = datasets.map(
        preprocess_func,
        batched=True,
        remove_columns=datasets["train_GSAT"].column_names
    )

    training_datasets = [processed_datasets["train_GSAT"]]

    if args.use_train_qb:
        training_datasets.append(processed_datasets["train_QB_history"])
        training_datasets.append(processed_datasets["train_QB_civics"])
        training_datasets.append(processed_datasets["train_QB_geo"])

    processed_datasets["train"] = concatenate_datasets(
        training_datasets
    )

    train_loader = DataLoader(
        processed_datasets["train"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        processed_datasets["valid"],
        batch_size=args.batch_size,
        collate_fn=default_data_collator,
        num_workers=8,
        shuffle=False,
    )

    # Prepared model
    device = torch.device(
        f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    if args.train_from_scratch:
        model = AutoModelForMultipleChoice.from_config(model_config).to(device)
    else:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=False,
            config=model_config,
        )

        if args.vision_bert:
            model: BertForMultipleChoice = model
            torch.save(model.state_dict(), "tmp.pth")
            state_dict = torch.load("tmp.pth")
            model = VisionBertForMultipleChoice(model.config).to(device)
            try:
                model.load_state_dict(state_dict)
            except:
                print("Some keys are missing")
                model.load_state_dict(state_dict, strict=False)

    # Prepared optimizer and learning rate scheduler
    optimizer = get_optimizer(
        model, "adamw", lr=args.lr, weight_decay=args.weight_decay)
    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.accum_grad_step)
    max_train_steps = args.epoch * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(args.warm_up_step / args.accum_grad_step),
        num_training_steps=max_train_steps,
    )

    from src.constants import PROJECT_NAME

    # Prepared logger
    wandb.init(
        project=PROJECT_NAME,
        group="mc",
        name="experiment",
        config={
            "tokenizer": args.tokenizer_name,
            "model": args.model_name_or_path,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "accum_grad_step": args.accum_grad_step,
            "optimizer": "adamw",
            "lr_scheduler": args.lr_scheduler,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "num_warmup_steps": args.warm_up_step,
        }
    )
    wandb.watch(model, log="all")

    trainer = BERTMultipleChoiceTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        accum_grad_step=args.accum_grad_step,
        lr_scheduler=lr_scheduler,
        logger=wandb,
        bf16=args.bf16,
    )

    trainer.fit(epoch=args.epoch)
    wandb.finish()
