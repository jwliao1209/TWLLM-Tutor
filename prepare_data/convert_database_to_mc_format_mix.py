from argparse import ArgumentParser, Namespace

from lib.utils.data_utils import read_json, write_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--database_json", type=str,
                        default="data/public/problem_database/social_study/history/problem_database.json")
    parser.add_argument("--train_mc_json", type=str,
                        default="data/train_data_mc/train.json")
    parser.add_argument("--output_mc_json", type=str,
                        default="data/train_data_mc/train_with_database.json")
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
    train_json = read_json(args.train_mc_json)
    database_json = read_json(args.database_json)
    for data in database_json:
        data = format_mc_data(data)
        train_json.append(data)
    write_json(train_json, args.output_mc_json, )


if __name__ == "__main__":
    main()
