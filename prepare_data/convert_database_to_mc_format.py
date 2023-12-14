from argparse import ArgumentParser, Namespace
from pathlib import Path

from sklearn.model_selection import train_test_split

from lib.utils.data_utils import read_json, write_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--database_json", type=str,
                        default="data/public/problem_database/social_study/history/problem_database.json")
    parser.add_argument("--output_folder", type=str,
                        default="data/train_database_mc")
    return parser.parse_args()


def format_mc_data(data):
    abcd_map = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }
    return {
        "subject": 'social_study',
        "year": "-1",
        "id": data['id'],
        "question": data['question'],
        "A": data['A'],
        "B": data['B'],
        "C": data['C'],
        "D": data['D'],
        "answer": abcd_map[data["answer"]],
    }


def main():
    args = parse_arguments()
    database_json = read_json(args.database_json)
    format_datas = []
    for data in database_json:
        data = format_mc_data(data)
        format_datas.append(data)
    train, valid = train_test_split(format_datas, test_size=0.1, random_state=42)
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    write_json(train, args.output_folder + "/train.json")
    write_json(valid, args.output_folder + "/valid.json")


if __name__ == "__main__":
    main()
