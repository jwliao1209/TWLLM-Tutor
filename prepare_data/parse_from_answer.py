import os
import re
import pdfplumber

from argparse import Namespace, ArgumentParser


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description="Parse answer from pdf file")
    parser.add_argument("-f", "--file_name", type=str,
                        default="96.pdf",
                        help="file name to parse answer")
    parser.add_argument("--input_folder", type=str,
                        default="data/raw_data/university_exams/social_study/answer_pdf",
                        help="input folder")
    parser.add_argument("--output_folder", type=str,
                        default="data/raw_data/university_exams/social_study/answer",
                        help="output folder")
    parser.add_argument("--answer_num", type=int,
                        default="72",
                        help="number of answer")
    return parser.parse_args()


def extract_answer(text):
    pattern = pattern = r'(\d+)\s+([A-D](?:\s*或[A-D])?(?:\s*\(\s*或[A-D]\s*\))?(?:\s*\（\s*或[A-D]\s*\）)?|無答案)'
    matches = re.findall(pattern, text)
    answers = {int(num): ans
               .replace("(", "")
               .replace(")", "")
               .replace(" ", "")
               .replace("（", "")
               .replace("）", "")
               for num, ans in matches
    }
    return answers


def write_txt(text, file_name):
    with open(file_name, "w") as f:
        f.write(text)
    return


if __name__ == "__main__":
    args = parse_arguments()
    with pdfplumber.open(os.path.join(args.input_folder, args.file_name)) as f:
        for page in f.pages:
            text = page.extract_text()

            # find the page with "社會" in text
            if "會" in text:
                break

    print("============================================================")
    print("Original Scan Answer:")
    print(text)
    
    print("============================================================")
    print("Cleaned Answer:")
    print(text)
    cleaned_lines = [line for line in text.split('\n') if re.search(r'\d', line)]
    cleaned_text = '\n'.join(cleaned_lines)

    print("============================================================")
    print("Exact Answer:")
    answers = extract_answer(cleaned_text)

    # Automatically fill the answer
    for i in range(1, args.answer_num + 1):
        if not answers.get(i, False):
            answers[i] = input(f"Please input the answer of question {i}: ")
        else:
            print(f"The answer of question {i} is {answers.get(i)}")

    answer_text = "\n".join([f"{i}, {answer}" for i, answer in sorted(answers.items())])
    
    print("============================================================")
    print("Final Answer:")
    print(answer_text)

    os.makedirs(args.output_folder, exist_ok=True)
    write_txt(answer_text, os.path.join(args.output_folder, f"{os.path.splitext(args.file_name)[0]}.txt"))
