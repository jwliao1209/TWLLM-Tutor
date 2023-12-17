import os
import json
from docx import Document

def extract_options_from_question(question):
    """
    Extracts options A, B, C, D from the question text and adds them as separate fields.
    """
    if "question" in question:
        options_start = question["question"].rfind("。") + 1
        options_text = question["question"][options_start:]

        for opt in ['A', 'B', 'C', 'D']:
            option_key = opt + ')'
            if option_key in options_text:
                option_index = options_text.index(option_key)
                next_option_index = len(options_text)
                for next_opt in ['A', 'B', 'C', 'D']:
                    next_option_key = next_opt + ')'
                    if next_option_key in options_text and options_text.index(next_option_key) > option_index:
                        next_option_index = min(next_option_index, options_text.index(next_option_key))

                question[opt] = options_text[option_index + 2 : next_option_index].strip().rstrip('　(')

def parse_qa(document, subject):
    parsed_questions = []
    in_single_choice_section = False
    question_counter = 0  # Counter for question ID
    difficulty_mapping = {'易': 'easy', '中': 'medium', '難': 'hard'}

    for para in document.paragraphs:
        text = para.text.strip()

        if text == "【單選題】":
            in_single_choice_section = True
            continue
        if text.startswith("【題組題】"):
            in_single_choice_section = False
            continue

        if in_single_choice_section and text:
            if text.startswith("編碼"):
                difficulty_code = text.split("難易度：")[1].split()[0]
                difficulty = difficulty_mapping.get(difficulty_code, 'unknown')

                current_question = {
                    "id": question_counter,
                    "difficulty": difficulty,
                    "subject": subject,
                    "type": "single",
                    "raw_question": ""
                }
                question_counter += 1
            elif text.startswith("解答"):
                current_question["answer"] = text.split()[1]
            elif text.startswith("解析"):
                current_question["answer_details"] = text.replace("解析 　", "")
                parsed_questions.append(current_question)
            else:
                if "question" not in current_question:
                    current_question["raw_question"] = text
                    current_question["question"] = text
                else:
                    option_key = text[0]
                    current_question[option_key] = text[2:].strip()

    for question in parsed_questions:
        extract_options_from_question(question)

    # Save to JSON file
    os.makedirs('test', exist_ok=True)
    with open('test/parsed_questions.json', 'w', encoding='utf-8') as file:
        # json.dump(parsed_data, file, ensure_ascii=False, indent=4)
        json.dump(parsed_questions, file, ensure_ascii=False, indent=4)

    return parsed_questions

# File path
file_path = 'data/raw_data/university_exams/social_study/problem_database/civics/CH7科技、永續與全球關連.docx'
document = Document(file_path)

# Extracting text for questions
parsed_qa = parse_qa(document, 'civics')
# print(parsed_qa)
# print(parsed_qa[-2:])
# print(parsed_qa[:2])
print(len(parsed_qa))