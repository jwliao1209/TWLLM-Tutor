import os
import json
from docx import Document

# Function to preprocess the data


def preprocess_data(data):
    processed_questions = []
    question_id = 0
    difficulty_mapping = {'易': 'easy', '中': 'medium', '難': 'hard'}

    for i in range(0, len(data), 4):
        if i + 3 < len(data) and "題號：" in data[i] and "解析：" in data[i + 3]:
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


# File path
file_path = 'data/public/university_exams/social_study/problem_database/geography/geography_problem_database_raw.json'
cleaned_file_path = 'data/public/university_exams/social_study/problem_database/geography/geography_problem_database_cleaned.json'
# Read the file content
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Perform preprocessing
preprocessed_data = preprocess_data(data)

# Write the preprocessed data to the new file
with open(cleaned_file_path, 'w', encoding='utf-8') as file:
    json.dump(preprocessed_data, file, ensure_ascii=False, indent=4)
