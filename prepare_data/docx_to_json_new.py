import os
import re
from docx import Document
from argparse import ArgumentParser, Namespace

from utils.data_utils import read_json, write_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str,
                        default="data/raw_data/university_exams/social_study/explanation")
    parser.add_argument("--output_folder", type=str,
                        default="data/public/university_exams/social_study")
    return parser.parse_args()


def list_files(path="."):
    # List all files in the directory
    file_names = [f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]
    return file_names


def list_directories(path="."):
    """
    Lists all directories in the specified path.
    Default is the current directory.
    """
    directories = [d for d in os.listdir(
        path) if os.path.isdir(os.path.join(path, d))]
    return directories


def extract_leading_numbers(file_names):
    numbers = []
    for name in file_names:
        num_str = ""
        for char in name:
            if char.isdigit():
                num_str += char
            else:
                break
        numbers.append(int(num_str))
    return numbers


def extract_first_paragraph_after_marker(text, marker):
    # Split the text at the first occurrence of the marker
    parts = text.split(marker, 1)
    if len(parts) > 1:
        # Further split the second part at the first period and return the first paragraph
        paragraph = parts[1].split("。", 1)[0] + "。"
        # Remove the character "】" from the paragraph
        paragraph = paragraph.replace("】", "")
        return paragraph
    else:
        return "" #"No paragraph found after the marker"


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

    # Temporary string to hold the current question"s text
    current_question_text = ""

    # Regular expression pattern for question numbers (e.g., "65.")
    question_pattern = re.compile(r"^\d+\.")

    # Iterate over each line and extract explanations for the specified number of questions
    for line in lines:
        if start_marker in line:
            start_extraction = True
            continue  # Skip the line containing the start_marker

        if start_extraction and question_pattern.match(line):
            if question_count > num_questions:
                break  # Stop extraction after the specified number of questions
            question_number = question_pattern.findall(line)[0].rstrip(".")
            question_count += 1
            # Process the current question"s text for explanation
            if current_question_text:
                explanation = extract_first_paragraph_after_marker(
                    current_question_text, explanation_marker)
                if int(question_number) > 1:
                    explanations.append((str(int(question_number)-1), explanation))
                    current_question_text = line  # Start new question text
            else:
                current_question_text = line
        elif start_extraction:
            current_question_text += " " + line

    # Process the last question"s text
    if current_question_text:
        explanation = extract_first_paragraph_after_marker(
            current_question_text, explanation_marker)
        explanations.append((question_number, explanation))

    return explanations


def update_json_file(json_file_path, data_to_update):
    # Load the existing JSON data
    json_data = read_json(json_file_path)

    # Update the JSON data
    for item in json_data:
        for question_number, explanation in data_to_update:
            if item["id"] == int(question_number) -1 :
                item["detailed explanation"] = explanation

    # Save the updated JSON data
    write_json(json_data, json_file_path)
    return


if __name__ == "__main__":
    args = parse_arguments()

    raw_dir_directories = list_files(args.data_folder)
    years = extract_leading_numbers(raw_dir_directories)
    start_marker = "單選題"
    explanation_marker = "試題解析"
    num_questions = 72  # Number of multiple choice questions to extract

    for f in raw_dir_directories:
        print(f"Processing file: {f}")
        file_path = os.path.join(args.data_folder, f)
        year = extract_leading_numbers([f])[0]

        extracted_text = extract_text_for_questions(
            file_path, start_marker, explanation_marker, num_questions
        )

        save_path = os.path.join(args.output_folder, str(year), "explanation.json")
        explanation_dict = {
            int(problem_id): explanation for problem_id, explanation in extracted_text
        }
        write_json(explanation_dict, save_path)
