import os
import re
import glob
from docx import Document
from pprint import pprint
from collections import defaultdict
from argparse import ArgumentParser, Namespace

from utils.data_utils import write_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str,
                        default="data/raw_data/university_exams/social_study/explanation")
    parser.add_argument("--output_folder", type=str,
                        default="data/public/university_exams/social_study")
    return parser.parse_args()


def extract_answer_details(file_lines, total_num):
    question_pattern = re.compile(r"^\d+\.")  # e.g., "65."
    answer_details_dict = defaultdict(str)
    question_num = 0

    for line in file_lines:
        if "單選題" in line:
            continue

        if question_pattern.match(line.lstrip()):
            question_id = question_pattern.findall(line.lstrip())[0].rstrip(".")

        answer_details = re.search(r"【試題解析】(.+)", line)
        if answer_details and (int(question_id) > question_num):
            answer_details_dict[int(question_id)] = answer_details.group(1)
            question_num += 1

        if question_num == total_num:
            break

    return answer_details_dict


if __name__ == "__main__":
    args = parse_arguments()
    file_list = glob.glob(os.path.join(args.data_folder, '*.docx'))
    total_num = 72

    for f in file_list:
        print(f"Processing file: {f}")
        file_lines = [paragraph.text for paragraph in Document(f).paragraphs]
        answer_details_dict = extract_answer_details(file_lines, total_num)
        # pprint(answer_details_dict)

        year = re.search(r"^\d+", os.path.basename(f)).group(0)
        save_path = os.path.join(args.output_folder, str(year), "explanation.json")
        write_json(answer_details_dict, save_path)
