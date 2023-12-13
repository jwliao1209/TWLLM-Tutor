import os
import re
import sys
import json
from docx import Document


def list_files(path="."):
    # List all files in the directory
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def parse_qa(document, subject):
    parsed_data = []
    question_id = 0  # Counter for question ID
    current_question = ''
    current_choices = {}
    current_answer = ''
    current_answer_details = ''
    current_difficulty = None
    collecting_choices = False
    collecting_answer_details = False

    difficulty_mapping = {'易': 'easy', '中': 'medium', '難': 'hard'}

    for para in document.paragraphs:
        text = para.text.strip()
        if "附圖" in text:
            current_question = ''
            continue  # Skip questions that contain "附圖"
        if text.startswith("題號："):
            if current_question and current_answer and re.fullmatch(r'\([A-D]\)', current_answer):
                parsed_data.append({
                    'id': question_id,
                    'question': current_question,
                    'A': current_choices.get('A', ''),
                    'B': current_choices.get('B', ''),
                    'C': current_choices.get('C', ''),
                    'D': current_choices.get('D', ''),
                    'answer': current_answer.replace('(', '').replace(')', ''),  # Remove parentheses from answer
                    'type': 'single',
                    'answer_details': current_answer_details,
                    'subject': subject,
                    'difficulty': current_difficulty
                })
                question_id += 1

            current_question = ''
            current_choices = {}
            current_answer = ''
            current_answer_details = ''
            collecting_choices = True
            collecting_answer_details = False

            difficulty_match = re.search(r'難易度：(\w+)', text)
            current_difficulty = difficulty_mapping[difficulty_match.group(1)] if difficulty_match else None
        elif text.startswith("答案："):
            current_answer = text.replace("答案：", "").strip()
            collecting_choices = False
            collecting_answer_details = True
        elif text.startswith("解析："):
            current_answer_details = text.replace("解析：", "").strip()
            collecting_answer_details = True
        elif collecting_choices:
            choice_match = re.match(r'([A-D])\.\s*(.*)', text)
            if choice_match:
                current_choices[choice_match.group(1)] = choice_match.group(2).strip()
            else:
                current_question += text + ' '
        elif collecting_answer_details:
            current_answer_details += ' ' + text

    # Add the last question-answer pair if it meets the criteria
    if current_question and current_answer and re.fullmatch(r'\([A-D]\)', current_answer):
        parsed_data.append({
            'id': question_id,
            'question': current_question,
            'A': current_choices.get('A', ''),
            'B': current_choices.get('B', ''),
            'C': current_choices.get('C', ''),
            'D': current_choices.get('D', ''),
            'answer': current_answer.replace('(', '').replace(')', ''),  # Remove parentheses from answer
            'type': 'single',
            'answer_details': current_answer_details,
            'subject': subject,
            'difficulty': current_difficulty
        })

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
    parsed_qa = parse_qa(document, 'history')
    all_parsed_data.extend(parsed_qa)

# Save all parsed data to a single JSON file in the output directory
json_file_path = os.path.join(out_dir, 'problem_database.json')
save_json_file(json_file_path, all_parsed_data)
