import torch
from torch.utils.data.dataset import Dataset
from constants import PROMPT_PREFIX_DICT


class Prompt:
    def __init__(self, prompt_prefix: str = "", with_incontext: bool = False) -> None:
        self.prompt_prefix = prompt_prefix
        self.with_incontext = with_incontext

    def get(self, data: dict) -> str:
        '''Format the instruction as a prompt for LLM.'''
        question_and_choice = f"{data['question']} \nA.{data['A']} \nB.{data['B']} \nC.{data['C']} \nD.{data['D']}".replace(" ", "")

        if self.with_incontext:
            return f"""{PROMPT_PREFIX_DICT.get(self.prompt_prefix, "")}你是一名人工智慧家教，以下的題目為高中社會科，請根據題目提供使用者正確答案的選項 A 或 B 或 C 或 D。提供你例子參考: USER: 多數國家的民族分布，存在「大分散，小聚居」的現象，如果政府欲增進不同民族間的關係和諧，下列何種作法最為適當？\nA. 經由政黨協商，自行劃定各民族的自治區\nB. 鼓勵各民族的交流，理性對待彼此的文化差異\nC. 開放不同民族間的競爭，創造最優秀的強勢民族\nD. 將相同語言、文化的民族遷徙集中到同一生活棲息地 ASSISTANT: B. 鼓勵各民族的交流，理性對待彼此的文化差異。因為國父民族主義主張民族平等，尤其複合民族國家間，要增進不同民族間之和諧，需堅持民族平等的原則，並強調文化多元主義的觀點，選項(B)，即符合此一文化多元化之理想；至於選項A、C及D，或剝奪弱勢族群的基本權利，或要求弱勢族群放棄自己的文化，接受優勢族群的文化，或多或少都具有種族中心主義的色彩，不符合民族主義所強調之文化相對論原則。現在請回答: USER: {question_and_choice} ASSISTANT:"""
        else:
            return f"""{PROMPT_PREFIX_DICT.get(self.prompt_prefix, "")}你是一名人工智慧家教，以下的題目為高中社會科，請根據題目提供使用者正確答案的選項 A 或 B 或 C 或 D。USER: {question_and_choice} ASSISTANT: 正確答案為"""


class Answer:
    def __init__(self, with_answer_details: bool = False) -> None:
        self.with_answer_details = with_answer_details

    def get(self, data: dict) -> str:
        '''Format the answer as a prompt for LLM.'''
        if self.with_answer_details:
            return f"{data['answer']}.{data[str(data['answer'])]} 原因：{data['answer_details']}"
        else:
            # Example: "A. XXXX"
            return f"{data['answer']}.{data[str(data['answer'])]}"


class AcademicDataset(Dataset):
    def __init__(
        self,
        data_list,
        tokenizer,
        prompt_prefix,
        max_length=512,
        is_train=True,
        with_incontext=False,
        with_answer_details=True,
    ) -> None:

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.with_incontext = with_incontext
        self.prompt_prefix=prompt_prefix
        self.with_answer_details = with_answer_details
        self.data_list = self.transform(data_list)

    def pad_or_truncate(self, data, padding_token=0):
        # if the length of data is less than max_length, pad it with padding_token
        if self.max_length >= len(data):
            return data + [padding_token] * (self.max_length - len(data))
        # if the length of data is greater than max_length, truncate it
        else:
            return data[:self.max_length]

    # TODO: refactor this function
    def transform(self, data_list):
        prompt = Prompt(prompt_prefix=self.prompt_prefix, with_incontext=self.with_incontext)
        answer = Answer(with_answer_details=self.with_answer_details)

        ids = [str(data["id"]) for data in data_list]
        years = [str(data.get("year", 0)) for data in data_list]
        instructions = [prompt.get(data) for data in data_list]
        tokenized_instructions = self.tokenizer(instructions, add_special_tokens=False)
        outputs = [answer.get(data) for data in data_list]
        answer = [data["answer"] for data in data_list]
        answer_description = [data[str(data['answer'])] for data in data_list]
        answer_details = [data["answer_details"] for data in data_list]

        processed_data = []
        if self.is_train:
            tokenized_outputs = self.tokenizer(outputs, add_special_tokens=False)

            for i in range(len(data_list)):
                instructions_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
                outputs_input_ids = tokenized_outputs["input_ids"][i] + [self.tokenizer.eos_token_id]
                processed_data_input_ids =  instructions_input_ids + outputs_input_ids
                processed_data.append(
                    {
                        "id": ids[i],
                        "input_ids": self.pad_or_truncate(processed_data_input_ids),
                        "attention_mask": self.pad_or_truncate([1] * len(processed_data_input_ids)),
                        "labels": self.pad_or_truncate([-100] * len(instructions_input_ids) + outputs_input_ids),
                        "output_mask": self.pad_or_truncate([0] * len(instructions_input_ids) + [1] * len(outputs_input_ids)),
                    }
                )
        else:
            for i in range(len(data_list)):
                processed_data_input_ids = [self.tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
                processed_data.append(
                    {
                        "id": ids[i],
                        "year": years[i],
                        "input_ids": processed_data_input_ids,
                        "attention_mask": [1] * len(processed_data_input_ids),
                        "prompt": instructions[i],
                        "outputs": outputs[i],
                        "answer": answer[i],
                        "answer_description": answer_description[i],
                        "answer_details": answer_details[i],
                    }
                )
        return processed_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


class LLMMCDataset(Dataset):
    def __init__(
        self,
        data_list,
        tokenizer,
        max_length=1024,
    ) -> None:

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_list = self.transform(data_list)

    def transform(self, data_list):
        ABCD_MAP = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
        }
        processed_data = []
        for data in data_list:
            question = f"{data['question']} \nA.{data['A']} \nB.{data['B']} \nC.{data['C']} \nD.{data['D']}".replace(" ", "")
            tokenized_question = self.tokenizer(question,
                                                add_special_tokens=False,
                                                truncation=True,
                                                padding="max_length",
                                                max_length=self.max_length)
            processed_data.append(
                {
                    "id": data["id"],
                    "year": str(data.get("year", 0)),
                    "input_ids": tokenized_question["input_ids"],
                    "attention_mask": tokenized_question["attention_mask"],
                    "labels": ABCD_MAP[data["answer"]],
                    "answer": data["answer"],
                    "answer_description": data[data["answer"]],
                    "answer_details": data["answer_details"],
                }
            )
        return processed_data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def collate_func(data: list) -> dict:
    # convert list of dict to dict of list
    data_list_dict = {k: [dic[k] for dic in data] for k in data[0]}

    # convert dict of list to dict of torch tensor
    data_tensor_dict = {
        k: v if isinstance(v[0], str) else torch.tensor(v)
        for k, v in data_list_dict.items()
    }
    return data_tensor_dict
