import transformers
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer

from .prompt import PromptTemplate, Answer
from ..constants import OPTION, OPTION_TO_LABEL
from ..utils.data_utils import flatten_list, unflatten_list


class BasicDataset(Dataset):

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class InstructionDataset(BasicDataset):

    def __init__(
        self,
        data_list: list,
        tokenizer: AutoTokenizer,
        prompt_prefix: str = "",
        max_length: int = 512,
        is_train: bool = True,
        with_incontext: bool = False,
        with_answer_details: bool = True,
        *args, **kwargs
    ) -> None:

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.prompt_prefix = prompt_prefix
        self.with_incontext = with_incontext
        self.with_answer_details = with_answer_details
        self.data_list = self.process(data_list)

    def pad_or_truncate(self, data, padding_token=0):
        # if the length of data is less than max_length, pad it with padding_token
        if self.max_length >= len(data):
            return data + [padding_token] * (self.max_length - len(data))
        # if the length of data is greater than max_length, truncate it
        else:
            return data[:self.max_length]

    def process(self, data_list):
        prompt = PromptTemplate(prompt_prefix=self.prompt_prefix, with_incontext=self.with_incontext)
        answer = Answer(with_answer_details=self.with_answer_details)

        processed_data = []
        for data in data_list:
            if self.is_train:
                tokenized_instructions = self.tokenizer(prompt.get(data), add_special_tokens=False)
                tokenized_outputs = self.tokenizer(answer.get(data), add_special_tokens=False)

                instructions_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"]
                outputs_input_ids = tokenized_outputs["input_ids"] + [self.tokenizer.eos_token_id]
                processed_data_input_ids = instructions_input_ids + outputs_input_ids

                processed_data.append(
                    {
                        "id": str(data.get("id")),
                        "input_ids": self.pad_or_truncate(processed_data_input_ids),
                        "attention_mask": self.pad_or_truncate([1] * len(processed_data_input_ids)),
                        "labels": self.pad_or_truncate([-100] * len(instructions_input_ids) + outputs_input_ids),
                        "output_mask": self.pad_or_truncate([0] * len(instructions_input_ids) + [1] * len(outputs_input_ids)),
                    }
                )
            else:
                tokenized_instructions = self.tokenizer(prompt.get(data), add_special_tokens=False)
                processed_data_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"]
                processed_data.append(
                    {
                        "id": str(data.get("id")),
                        "year": str(data.get("year", 0)),
                        "input_ids": processed_data_input_ids,
                        "attention_mask": [1] * len(processed_data_input_ids),
                        "prompt": prompt.get(data),
                        "outputs": answer.get(data),
                        "answer": data.get("answer"),
                        "answer_description": data.get(str(data.get("answer"))),
                        "answer_details": data.get("answer_details"),
                    }
                )
        return processed_data


class TWLLMMultipleChoiceDataset(BasicDataset):

    def __init__(
        self,
        data_list: list,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        *args, **kwargs
    ) -> None:

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list = self.process(data_list)

    def process(self, data_list):
        processed_data = []
        for data in data_list:
            question = f"{data['question']} \nA.{data['A']} \nB.{data['B']} \nC.{data['C']} \nD.{data['D']}".replace(
                " ", "")
            tokenized_question = self.tokenizer(
                question,
                add_special_tokens=False,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            processed_data.append(
                {
                    "id": str(data.get("id")),
                    "year": str(data.get("year", 0)),
                    "question": question,
                    "input_ids": tokenized_question["input_ids"],
                    "attention_mask": tokenized_question["attention_mask"],
                    "labels": OPTION_TO_LABEL[data["answer"]],
                    "answer": data.get("answer"),
                    "answer_description": data.get(str(data.get("answer"))),
                    "answer_details": data.get("answer_details"),
                }
            )
        return processed_data


class BERTMultipleChoiceDataset(BasicDataset):

    def __init__(
        self,
        data_list: list,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        is_train: bool = True,
        *args, **kwargs
    ) -> None:

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.data_list = self.process(data_list)

    def process(self, data_list: dict) -> dict:
        transformers.logging.set_verbosity_error()

        first_sentences = [
            [data["question"]] * len(OPTION)
            for data in data_list
        ]
        second_sentences = [
            [data["A"], data["B"], data["C"], data["D"]]
            for data in data_list
        ]
        tokenized_data = self.tokenizer(
            flatten_list(first_sentences),
            flatten_list(second_sentences),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        tokenized_data = {
            k: unflatten_list(v, len(OPTION))
            for k, v in tokenized_data.items()
        }

        processed_data = []
        for i, data in enumerate(data_list):
            question = f"{data['question']} \nA.{data['A']} \nB.{data['B']} \nC.{data['C']} \nD.{data['D']}".replace(" ", "")

            processed_data.append(
                {
                    "id": str(data.get("id")),
                    "year": str(data.get("year", 0)),
                    "question": question,
                    "input_ids": tokenized_data["input_ids"][i],
                    "attention_mask": tokenized_data["attention_mask"][i],
                    "labels": OPTION_TO_LABEL[data["answer"]],
                    "answer": data.get("answer"),
                    "answer_description": data.get(str(data.get("answer"))),
                    "answer_details": data.get("answer_details"),
                }
            )
        return processed_data
