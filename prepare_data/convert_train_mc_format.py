import logging
import os
from argparse import ArgumentParser, Namespace

from lib.constants import TRAIN_FOLDERS, VALID_FOLDERS
from lib.utils.data_utils import read_json, write_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str,
                        default="data/public/university_exams/social_study")
    parser.add_argument("--output_folder", type=str,
                        default="data/train_data_mc")
    return parser.parse_args()


def format_mc_data(data):
    abcd_map = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }
    return {
        "subject": data["subject"],
        "year": data["year"],
        "id": data["id"],
        "question": data['question'],
        "A": data['A'],
        "B": data['B'],
        "C": data['C'],
        "D": data['D'],
        "answer": abcd_map[data["answer"]],
    }


def process_directory(directory):
    # Paths for the input and output files
    file_path = os.path.join(directory, "content.json")
    data_list = read_json(file_path)["questions"]

    # Prepare informations of question_groups
    question_groups = read_json(file_path)["question_groups"]
    question_groups_map = {}
    for question_group in question_groups:
        for qid in question_group['ids']:
            question_groups_map[qid] = question_group['prefix']

    transformed_data = []
    for data in data_list:
        if data.get("type") == "multi":
            continue
        if data.get("answer") == "無答案":
            continue
        if len(data.get("answer", "")) > 1:
            continue
        if data.get("E"):
            logging.info("Ignore questions with 5 options (ABCDE)")
            continue

        if data["id"] in question_groups_map:
            data['question'] = (
                question_groups_map[data['id']] +
                " " +
                data['question']
            )
        transformed_data.append(
            format_mc_data(data | {"year": os.path.basename(directory), "subject": "social_study"})
        )
    return transformed_data


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_folder, exist_ok=True)

    train_dirs = [os.path.join(args.data_folder, f) for f in TRAIN_FOLDERS]
    valid_dirs = [os.path.join(args.data_folder, f) for f in VALID_FOLDERS]

    # Prepare training data
    print(f"Prepare training data")
    train_data = []
    for train_dir in train_dirs:
        train_data.extend(process_directory(train_dir))
    write_json(train_data, os.path.join(args.output_folder, "train.json"))

    # Prepare validation data
    valid_data = []
    print(f"Prepare validation data")
    for valid_dir in valid_dirs:
        valid_data.extend(process_directory(valid_dir))
    write_json(valid_data, os.path.join(args.output_folder, "valid.json"))
