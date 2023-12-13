import re
from docx import Document

def extract_first_paragraph_after_marker(text, marker):
    # Split the text at the first occurrence of the marker
    parts = text.split(marker, 1)
    if len(parts) > 1:
        # Further split the second part at the first period and return the first paragraph
        paragraph = parts[1].split('。', 1)[0] + '。'
        # Remove the character "】" from the paragraph
        paragraph = paragraph.replace("】", "")
        return paragraph
    else:
        return "No paragraph found after the marker"

def extract_text_for_questions(file_path, start_marker, explanation_marker, num_questions):
    # Load the document
    document = Document(file_path)

    # Extract the text of all paragraphs
    lines = [para.text for para in document.paragraphs]

    # Flag to indicate the start of extraction
    start_extraction = False

    # Counter for the number of questions extracted
    question_count = 0

    # List to hold the explanations along with question numbers
    explanations = []

    # Temporary string to hold the current question's text
    current_question_text = ''

    # Regular expression pattern for question numbers (e.g., "65.")
    question_pattern = re.compile(r"^\d+\.")

    # Iterate over each line and extract explanations for the specified number of questions
    for line in lines:
        if start_marker in line:
            start_extraction = True
            continue  # Skip the line containing the start_marker

        if start_extraction and question_pattern.match(line):
            question_number = question_pattern.findall(line)[0].rstrip('.')
            if question_number == '1':
                continue  # Skip the question number 1

            if question_count >= num_questions:
                break  # Stop extraction after the specified number of questions
            
            question_count += 1

            # Process the current question's text for explanation
            if current_question_text:
                explanation = extract_first_paragraph_after_marker(
                    current_question_text, explanation_marker)
                explanations.append((int(question_number)-1, explanation))
                current_question_text = line  # Start new question text
            else:
                current_question_text = line
        elif start_extraction:
            current_question_text += ' ' + line

    # Process the last question's text
    if current_question_text:
        explanation = extract_first_paragraph_after_marker(
            current_question_text, explanation_marker)
        explanations.append((int(question_number), explanation))

    return explanations

# File path
file_path = './data/raw_data/university_exams/social_study/explanation/105學測(社會考科)_試題解析.docx'

# Markers and number of questions
start_marker = "單選題"
explanation_marker = "試題解析"
num_questions = 72  # Number of multiple choice questions to extract

# Extracting text for questions
extracted_explanations = extract_text_for_questions(file_path, start_marker, explanation_marker, num_questions)

# print(extracted_explanations[-2:])
# print(extracted_explanations[:2])
print(len(extracted_explanations))