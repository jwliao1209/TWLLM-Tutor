import json
import torch


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return


def collate_func(data: list) -> dict:
    # convert list of dict to dict of list
    data_list_dict = {k: [dic[k] for dic in data] for k in data[0]}

    # convert dict of list to dict of torch tensor
    data_tensor_dict = {
        k: v if isinstance(v[0], str) else torch.tensor(v)
        for k, v in data_list_dict.items()
    }
    return data_tensor_dict
