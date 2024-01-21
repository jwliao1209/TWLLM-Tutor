"""Parse university exam from raw text"""
from typing import Any

import os
import re
import json
import sys



DOC_TO_TXT_CONFUSION = {
    '1': '1l',
}


def create_question_prefix_re(number: int) -> str:
    """Create regular repr for prefix of each question

    Because the doc to txt process has some errors like
    mistaking 1 (number) as l (alphabet), we use this function
    to handle this issue.

    For example:
        1  -> "1.|l."
        2  -> "2."
        10 -> "10.|l0."
    """
    number = str(number)

    outputs = ['']
    for digit in number:
        if digit in DOC_TO_TXT_CONFUSION:
            append_digit = DOC_TO_TXT_CONFUSION[digit]
        else:
            append_digit = digit

        temp = []
        for output in outputs:
            for d in append_digit:
                temp.append(output + d)
        outputs = temp

    outputs = [output + r"\." for output in outputs]
    return "|".join(outputs)


def parse_raw_qa_with_remainings(text: str, question_id: int) -> tuple[str, str]:
    """Parse a Q&A from a string.

    Returns: a tuple of strings
        1. a string representing the raw Q&A
        2. a string representing the remaining questions
            after this question
    """
    match = re.search(create_question_prefix_re(question_id), text)
    if match:
        question_w_answer = text[match.end():]

        next_question_match = re.search(
            create_question_prefix_re(question_id+1),
            question_w_answer
        )
        if next_question_match:
            remaining_text = question_w_answer[next_question_match.start():]
            question_w_answer = question_w_answer[:next_question_match.start()]
        else:
            remaining_text = ""

        return question_w_answer, remaining_text
    else:
        return "", ""


def parse_qa_dict(text: str, question_id: int) -> dict[str, Any]:
    """Parse an dictionary representing a pair of question and answer."""
    raw_qa, remaining_text = parse_raw_qa_with_remainings(
        text, question_id)

    if raw_qa:
        options_match = re.search(
            r'(.*?)\(A\)(.*?)\(B\)(.*?)\(C\)(.*?)\(D\)(.*?)$',
            raw_qa,
            re.DOTALL
        )

        question_dict = {
            "id": question_id
        }
        question_dict["question"] = options_match.group(1).strip()
        question_dict["A"] = options_match.group(2).strip()
        question_dict["B"] = options_match.group(3).strip()
        question_dict["C"] = options_match.group(4).strip()
        d_with_e = options_match.group(5).strip()

        e_match = re.search(r"\(E\)", d_with_e)
        if e_match:
            question_dict["D"] = d_with_e[:e_match.start()].strip()
            question_dict["E"] = d_with_e[e_match.end():].strip()
        else:
            question_dict["D"] = d_with_e

        return question_dict, remaining_text
    else:
        return {}, ""

def parse_question_group(text: str) -> tuple[dict[str, Any], str]:
    """Parse question groups and remove them from raw text"""
    question_groups = []
    while True:
        match = re.search(
            "為題組",
            text,
        )
        if not match:
            break

        chunk_left = match.start() - 1
        while text[chunk_left] in "0123456789-" or chunk_left < 0:
            chunk_left -= 1
        left, right = text[chunk_left+1:match.start()].split("-")

        group_match = re.search(
            left + r"\.",
            text,
        )

        left = int(left)
        right = int(right)
        group = {
            "ids": list(range(left, right+1)),
            "prefix": text[match.end():group_match.start()].strip()
        }
        question_groups.append(group)

        text = text[:chunk_left] + text[group_match.start():]

    return question_groups, text

def main():
    """main script"""
    raw_dir = sys.argv[1]
    out_dir = sys.argv[2]
    answer_dir = os.path.join(raw_dir, "answer")
    os.makedirs(out_dir, exist_ok=True)

    # for year in list(range(83, 112+1)) + ["91_bu", "92_bu"]:
    for year in [83]:
        input_filename = os.path.join(raw_dir, f"{year}.txt")
        output_filename = os.path.join(out_dir, f"{year}.json")
        answer_filename = os.path.join(answer_dir, f"{year}.answer.txt")

        answer_dict = {}
        for i, answer in enumerate(
            open(answer_filename, "r", encoding="utf-8").readlines()
        ):
            answer_dict[i+1] = answer.strip()

        questions = []

        text = open(input_filename, "r", encoding="utf-8").read()
        question_qroups, text = parse_question_group(text)

        skip_saving = False
        question_id = 1
        while len(text):
            try:
                question_dict, text = parse_qa_dict(text, question_id)
                ans = answer_dict[question_id]
                question_dict["answer"] = ans
                question_dict["type"] = 'single' if len(ans) == 1 else 'multi'
                question_dict["answer_details"] = ""
                question_id += 1
            except AttributeError as error:
                error_filename = os.path.join(out_dir, "error", f"{year}.log")
                os.makedirs(os.path.dirname(error_filename), exist_ok=True)
                error_logger = open(error_filename, "w", encoding="utf-8")
                print(
                    f"Error occurs in"
                    f"year: {year}\n"
                    f"question_id: {question_id}\n",
                )
                print(
                    f"year: {year}\n"
                    f"question_id: {question_id}\n"
                    f"{error}",
                    file=error_logger
                )
                skip_saving = True
                break

            questions.append(question_dict)
        if not skip_saving:
            output = {
                "question_qroups": question_qroups,
                "questions": questions,
            }
            json.dump(output, open(output_filename, "w",
                      encoding="utf-8"), ensure_ascii=False)


if __name__ == "__main__":
    main()
