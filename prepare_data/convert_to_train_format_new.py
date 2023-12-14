import os
import re
from collections import OrderedDict
from argparse import ArgumentParser, Namespace

from constants import TRAIN_FOLDERS, VALID_FOLDERS
from utils.data_utils import read_json, write_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str,
                        default="data/public/university_exams/social_study")
    parser.add_argument("--output_folder", type=str,
                        default="data/train_data")
    return parser.parse_args()


def process_dir_data(directory):
    year = os.path.basename(directory)

    problem_path = os.path.join(directory, "content.json")
    problems = read_json(problem_path)
    problem_group = problems["question_groups"]
    problem_list = problems["questions"]

    explanation_path = os.path.join(directory, "explanation.json")
    try:
        explanation_dict = read_json(explanation_path)
    except:
        explanation_dict = {}
        print(f"{year} has no explanation file.")

    transformed_data_dict = {}
    for data in problem_list:
        if "image" in data.get("question"):
            continue
        if "table" in data.get("question"):
            continue
        if "image" in data.get("A"):
            continue
        if "table" in data.get("A"):
            continue
        if data.get("type") == "multi":
            continue
        if data.get("answer") == "無答案":
            continue
        if len(data.get("answer", "")) > 1:
            continue

        transformed_data_dict.update(
            {
                data["id"]: 
                    OrderedDict(
                        subject="social_study",
                        year=os.path.basename(year),
                        id=data["id"],
                        type=data["type"],
                        question=data["question"],
                        A=data["A"],
                        B=data["B"],
                        C=data["C"],
                        D=data["D"],
                        answer=data["answer"],
                        answer_details=explanation_dict.get(str(data["id"]), ""),
                    )
            }
        )

    for group in problem_group:
        for i in group["ids"]:
            if transformed_data_dict.get(i, None):
                if "image" in group["prefix"]:
                    transformed_data_dict.pop(i, None)
                else:
                    transformed_data_dict[i]["question"] = group["prefix"] + transformed_data_dict[i]["question"]

    return list(transformed_data_dict.values())


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs("data/train_data", exist_ok=True)

    train_dirs = [os.path.join(args.data_folder, f) for f in TRAIN_FOLDERS]
    valid_dirs = [os.path.join(args.data_folder, f) for f in VALID_FOLDERS]

    # Prepare training data
    print(f"Prepare training data")
    train_data = []
    for train_dir in train_dirs:
        train_data.extend(process_dir_data(train_dir))
    write_json(train_data, os.path.join(args.output_folder, "train.json"))

    # Prepare validation data
    valid_data = []
    print(f"Prepare validation data")
    for valid_dir in valid_dirs:
        valid_data.extend(process_dir_data(valid_dir))
    write_json(valid_data, os.path.join(args.output_folder, "valid.json"))
