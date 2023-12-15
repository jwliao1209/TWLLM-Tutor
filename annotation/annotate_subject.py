from pprint import pprint
from argparse import Namespace, ArgumentParser

from dataset import Prompt
from constants import GEOGRAPHY, HISTORY, CIVICS
from utils.data_utils import read_json, write_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--test_data_path", type=str,
                        default="data/train_data/valid.json",
                        help="Path to test data.")
    parser.add_argument("--output_path", type=str,
                        default="data/train_data/valid_w_subject.json",
                        help="output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    test_data = read_json(args.test_data_path)
    prompt = Prompt(with_incontext=False)
    
    correct_num = 0
    prediction_list = []
    for i, data in enumerate(test_data):
        print("===============================================================================================")
        print(f"Question {i}:")
        print(f"Subject: {data['subject']}")
        print(data["question"])
        action = input("0: 維持 1: 地理 2: 歷史 3: 公民，請選擇題目類型: ")

        match action:
            case "0":
                pass
            case "1":
                data["subject"] = GEOGRAPHY
            case "2":
                data["subject"] = HISTORY
            case "3":
                data["subject"] = CIVICS
            case _:
                raise TypeError("not a point we support")

        pprint(test_data[i])
        write_json(test_data, args.output_path)
