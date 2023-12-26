from pprint import pprint
from argparse import Namespace, ArgumentParser
from collections import OrderedDict

from dataset import Prompt
from metric.accuracy import correcter
from utils.data_utils import read_json, write_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Taiwan-LLaMa Instruction Tuning")
    parser.add_argument("--test_data_path", type=str,
                        default="data/train_data/valid.json",
                        help="Path to test data.")
    parser.add_argument("--output_path", type=str,
                        default="prediction/chatGPT.json",
                        help="output path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    test_data = read_json(args.test_data_path)
    prompt = Prompt(with_incontext=False)
    
    correct_num = 0
    prediction_list = []
    for i, data in enumerate(test_data, 1):
        print("===============================================================================================")
        print(f"Prompt:\n{prompt.get(data)}")
        print("===============================================================================================")
        generation = ""
        print("Input chatGPT generation:")
        while True:
            inputs = input()
            if inputs == "Q":
                break
            generation += inputs

        is_correct = correcter(generation, data['answer'], data[str(data['answer'])])
        prediction = OrderedDict(
            id=data["id"],
            year=data["year"],
            subject=data["subject"],
            prompt=prompt.get(data),
            generation=generation,
            answer=data['answer'],
            is_correct=is_correct,
        )
        if is_correct:
            correct_num += 1

        print("===============================================================================================")
        pprint(prediction)
        print("答對題數:", correct_num)
        print("答對率:", correct_num / i)
        prediction_list.append(prediction)
        write_json(prediction_list, args.output_path)
