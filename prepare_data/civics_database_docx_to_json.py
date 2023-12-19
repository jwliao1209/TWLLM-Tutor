import os
import sys
import json
from docx import Document


def list_files(path="."):
    # List all files in the directory
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def extract_options_from_question(question):
    """
    Extracts options A, B, C, D from the question text and adds them as separate fields.
    """
    if "raw_question" in question:
        options_start = question["raw_question"].rfind("。") + 1
        options_text = question["raw_question"][options_start:]

        for opt in ['A', 'B', 'C', 'D']:
            option_key = opt + ')'
            if option_key in options_text:
                option_index = options_text.index(option_key)
                next_option_index = len(options_text)
                for next_opt in ['A', 'B', 'C', 'D']:
                    next_option_key = next_opt + ')'
                    if next_option_key in options_text and options_text.index(next_option_key) > option_index:
                        next_option_index = min(
                            next_option_index, options_text.index(next_option_key))

                question[opt] = options_text[option_index +
                                             2: next_option_index].strip().rstrip('　(')


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
                if not ("附圖" in current_question["raw_question"] or "圖片" in current_question["raw_question"] or "下圖" in current_question["raw_question"] or "附表" in current_question["raw_question"] or "表格" in current_question["raw_question"] or "下表" in current_question["raw_question"]):
                    parsed_questions.append(current_question)
            else:
                if "raw_question" not in current_question or not current_question["raw_question"]:
                    current_question["raw_question"] = text
                    question_end = text.find("？") + 1
                    current_question["question"] = text[:question_end]
                else:
                    option_key = text[0]
                    current_question[option_key] = text[2:].strip()

    for question in parsed_questions:
        extract_options_from_question(question)

    # # Save to JSON file
    # os.makedirs('test', exist_ok=True)
    # with open('test/parsed_questions.json', 'w', encoding='utf-8') as file:
    #     # json.dump(parsed_data, file, ensure_ascii=False, indent=4)
    #     json.dump(parsed_questions, file, ensure_ascii=False, indent=4)

    return parsed_questions


def save_json_file(json_file_path, data):
    # Save the data as JSON file
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def create_directory(path):
    # Create a test directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)


# Example usage
raw_dir = sys.argv[1]
out_dir = sys.argv[2]

# Ensure the output directory exists
create_directory(out_dir)

# Initialize an empty list to hold all parsed data
all_parsed_data = []

# Process each file in the raw directory
raw_dir_files = list_files(raw_dir)
for file in raw_dir_files:
    print(f"Processing file: {file}")
    # file_path = raw_dir+file
    file_path = os.path.join(raw_dir, file)
    document = Document(file_path)

    # Parse the document and append to all_parsed_data
    parsed_qa = parse_qa(document, 'civics')
    all_parsed_data.extend(parsed_qa)

# Save all parsed data to a single JSON file in the output directory
json_file_path = os.path.join(out_dir, 'civics_problem_database.json')
save_json_file(json_file_path, all_parsed_data)
