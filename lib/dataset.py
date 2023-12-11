import torch
from torch.utils.data.dataset import Dataset


# TODO: move this function to other file
def get_prompt(instruction: str, incontext: bool = False) -> str:
    '''Format the instruction as a prompt for LLM.'''
    if incontext:
        return f"""你是一名人工智慧家教，以下的題目為高中社會科，請根據題目提供使用者正確答案的選項 A 或 B 或 C 或 D。提供你例子參考:USER: 時下各國普遍流行將國營事業開放民間經營，試問這種決策的主要著眼點為何？\nA. 實現社會公平\nB. 增進社會福利\nC. 提高經營效率\nD. 揚棄共產主義  ASSISTANT: C 提高經營效率。現在請請回答: USER: {instruction} ASSISTANT:"""
    else:
        return f"""你是一名人工智慧家教，以下的題目為高中社會科，請根據題目提供使用者正確答案的選項 A 或 B 或 C 或 D。USER: {instruction} ASSISTANT:"""


class AcademicDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_length=512, is_train=True, incontext=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.incontext = incontext
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
        ids = [x["id"] for x in data_list]
        instructions = [get_prompt(x["instruction"], incontext=self.incontext) for x in data_list]
        tokenized_instructions = self.tokenizer(instructions, add_special_tokens=False)
        outputs = [x["output"] for x in data_list]

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
                        "input_ids": processed_data_input_ids,
                        "attention_mask": [1] * len(processed_data_input_ids),
                        "prompt": instructions[i],
                        "answer": outputs[i],
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
