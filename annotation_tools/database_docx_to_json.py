import os
import re
import sys
import json
from docx import Document


def list_files(path="."):
    # List all files in the directory
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def parse_qa(document):
    parsed_data = []
    current_question = None
    current_answer = None
    current_explanation = None
    current_difficulty = None
    question_id = 0  # Counter for question ID
    collecting_question = False
    collecting_explanation = False

    difficulty_mapping = {'易': 'easy', '中': 'medium', '難': 'hard'}

    for para in document.paragraphs:
        text = para.text.strip()
        if text.startswith("題號："):
            # Check if it is a '題組' question
            if '題組' in text:
                current_question = None
                continue

            # Save the previous question-answer pair if it exists and if there's a single answer
            if current_question is not None and current_answer is not None and len(re.findall(r'\([A-E]\)', current_answer)) == 1:
                output = current_answer
                if current_explanation is not None:
                    output += ' , ' + current_explanation
                parsed_data.append({'id': question_id, 'instruction': current_question,
                                   'output': output, 'difficulty': current_difficulty})
                question_id += 1

            difficulty_match = re.search(r'難易度：(\w+)', text)
            current_difficulty = difficulty_mapping[difficulty_match.group(
                1)] if difficulty_match else None

            # Initialize variables for a new question
            current_question = ''
            current_answer = ''
            current_explanation = ''
            collecting_question = True
            collecting_explanation = False
        elif text.startswith("答案："):
            current_answer = text.replace("答案：", "").strip()
            collecting_question = False
            collecting_explanation = True
        elif text.startswith("解析："):
            current_explanation = text.replace("解析：", "").strip()
        elif collecting_question:
            current_question += text + ' '
        elif collecting_explanation and not text.startswith("題號："):
            current_explanation += ' ' + text

    # Adding the last question-answer pair if exists and if there's a single answer
    if current_question is not None and current_answer is not None and len(re.findall(r'\([A-E]\)', current_answer)) == 1:
        output = current_answer
        if current_explanation is not None:
            output += ' , ' + current_explanation
        parsed_data.append({'id': question_id, 'instruction': current_question,
                           'output': output, 'difficulty': current_difficulty})

    return parsed_data


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
    parsed_qa = parse_qa(document)
    all_parsed_data.extend(parsed_qa)

# Save all parsed data to a single JSON file in the output directory
json_file_path = os.path.join(out_dir, 'problem_database_original.json')
save_json_file(json_file_path, all_parsed_data)
