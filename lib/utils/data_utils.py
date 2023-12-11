import json


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return


def format_data(data):
    return {
        "subject": data["subject"],
        "year": data["year"],
        "id": data["id"],
        "instruction": f"{data['question']} \nA. {data['A']} \nB. {data['B']} \nC. {data['C']} \nD. {data['D']}",
        "output": f"{data['answer']}. {data[str(data['answer'])]}",
        "ansewr": data["answer"],
        "detailed explanation": None,
    }
