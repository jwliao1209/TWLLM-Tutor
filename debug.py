import os
import re
import json
from docx import Document

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

    # Save to JSON file
    os.makedirs('test', exist_ok=True)
    with open('test/parsed_data.json', 'w', encoding='utf-8') as file:
        json.dump(parsed_data, file, ensure_ascii=False, indent=4)

    return parsed_data

# File path
file_path = './data/raw_data/problem_database/social_study/history/07_高中歷史(三)題本-第7章(亞非古文明的興起).docx'
document = Document(file_path)

# Extracting text for questions
parsed_qa = parse_qa(document, 'history')
# print(parsed_qa)
# print(parsed_qa[-2:])
# print(parsed_qa[:2])
print(len(parsed_qa))