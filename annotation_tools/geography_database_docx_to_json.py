import os
import sys
import json
from docx import Document

def preprocess_data(data):
    processed_questions = []
    question_id = 0
    difficulty_mapping = {'易': 'easy', '中': 'medium', '難': 'hard'}

    for i in range(0, len(data), 4):
        if i + 3 < len(data) and "題號：" in data[i] and "解析：" in data[i + 3]:
            # Check if the line with "題號：" contains "出處：學測試題"
            if "出處：學測試題" in data[i]:
                continue  # Skip this question

            # Check if the question text contains "附圖" or "附表"
            if "附圖" in data[i + 1] or "附表" in data[i + 1]:
                continue  # Skip this question

            # Check if the answer text contains more than one "(" or ")"
            if data[i + 2].count('(') > 1 or data[i + 2].count(')') > 1:
                continue  # Skip this question

            # Extract the text before the first "\n" as the question and remove '\u3000'
            question_text = data[i + 1].split('\n')[0].replace('\u3000', ' ')

            question_dict = {
                "id": question_id,
                "type": "single",
                "subject": "geography",
                "raw_question": data[i + 1],  # Only include the question text
                "question": question_text.replace('\n', ' '),  # Remove newlines and '\u3000'
                "answer": data[i + 2].split('：')[-1].strip(),
                "answer_details": data[i + 3].split('：', 1)[-1].strip()
            }

            # Extracting difficulty
            difficulty = data[i].split('難易度：')[-1].split('　')[0]
            question_dict["difficulty"] = difficulty_mapping.get(difficulty, 'unknown')

            # Extracting options A, B, C, D
            options = data[i + 1].split('\n')[-1].split('　')
            for opt in options:
                if opt.startswith('(A)'):
                    question_dict["A"] = opt.split('(A)')[-1].strip()
                elif opt.startswith('(B)'):
                    question_dict["B"] = opt.split('(B)')[-1].strip()
                elif opt.startswith('(C)'):
                    question_dict["C"] = opt.split('(C)')[-1].strip()
                elif opt.startswith('(D)'):
                    question_dict["D"] = opt.split('(D)')[-1].strip()

            processed_questions.append(question_dict)
            question_id += 1

    return processed_questions

def list_files(path="."):
    # List all files in the directory
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def parse_text(document):
    parsed_text = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        parsed_text.append(text)
    return parsed_text

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
    parsed_text = parse_text(document)
    all_parsed_data.extend(parsed_text)

# Save all parsed data to a single JSON file in the output directory
json_file_path = os.path.join(out_dir, 'geography_problem_database_raw.json')
save_json_file(json_file_path, all_parsed_data)

# Read the file content
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Perform preprocessing
preprocessed_data = preprocess_data(data)

# Write the preprocessed data to the new file
cleaned_file_path = os.path.join(out_dir, 'geography_problem_database.json')
save_json_file(cleaned_file_path, preprocessed_data)