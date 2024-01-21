import os
import json
import yaml
from collections import OrderedDict

import torch
from easydict import EasyDict


def makedirs(func):
    def wrap(data, path, *args):
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        func(data, path, *args)
    return wrap


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@makedirs
def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return


def load_config(path):
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return EasyDict(config)


@makedirs
def save_config(data, path):
    data = convert_easydict_to_ordereddict(data)
    with open(path, "w") as f:
        yaml.add_representer(
            OrderedDict,
            lambda dumper, data: dumper.represent_mapping(
                'tag:yaml.org,2002:map',
                data.items()
            )
        )
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return


def convert_easydict_to_ordereddict(obj):
    if isinstance(obj, (EasyDict)):
        obj = OrderedDict(obj)
        for key, val in obj.items():
            obj[key] = convert_easydict_to_ordereddict(val)
    return obj


def flatten_dict(d: dict, parent_key: str = None, sep: str = "/") -> dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key is not None else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def collate_func(data: list) -> dict:
    # convert list of dict to dict of list
    data_list_dict = {k: [dic[k] for dic in data] for k in data[0]}

    # convert dict of list to dict of torch tensor
    data_tensor_dict = {
        k: v if isinstance(v[0], str) else torch.tensor(v)
        for k, v in data_list_dict.items()
    }
    return data_tensor_dict
