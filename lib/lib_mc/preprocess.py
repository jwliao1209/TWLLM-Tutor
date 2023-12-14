from itertools import chain
from transformers import AutoTokenizer

from lib_mc.constants import MC_ENDING_LEN, MC_MAX_SEQ_LEN


def flatten_list(input_list: list) -> list:
    return list(chain(*input_list))


def unflatten_list(input_list: list, sub_list_num) -> list:
    return [
        input_list[i : i + sub_list_num]
        for i in range(0, len(input_list), sub_list_num)
    ]


def preprocess_mc_func(data: dict, tokenizer: AutoTokenizer, train=True) -> dict:
    """
    Reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/multiple-choice/run_swag_no_trainer.py
    """
    first_sentences = [[context] * 4 for context in data['question']]
    second_sentences = [
        [data['A'][i], data['B'][i], data['C'][i], data['D'][i]]
        for i in range(len(data['A']))
    ]

    tokenized_data = tokenizer(
        flatten_list(first_sentences),
        flatten_list(second_sentences),
        max_length=MC_MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
    )
    tokenized_data = {k: unflatten_list(v, MC_ENDING_LEN) for k, v in tokenized_data.items()}

    if train:
        tokenized_data["labels"] = data['answer']

    return tokenized_data
