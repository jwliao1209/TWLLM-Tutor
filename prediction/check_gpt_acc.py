import os
import json
from collections import defaultdict
from utils.data_utils import read_json


def save_json_file(json_file_path, data):
    # Save the data as JSON file
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


gpt_json = read_json("./prediction/chatGPT.json")
val_w_subject_json = read_json("./data/train_data/valid_w_subject.json")

# Extract 'id' and 'year' pairs from chatGPT.json and save them into a list.
id_year_pairs = [(entry['id'], entry['year']) for entry in gpt_json]

# Use each 'id' and 'year' pair to find the corresponding entry in valid_w_subject.json.
# Extract the 'subject' from the found entry and write it back to chatGPT.json.
for chatgpt_entry in gpt_json:
    for valid_entry in val_w_subject_json:
        if chatgpt_entry['id'] == valid_entry['id'] and chatgpt_entry['year'] == valid_entry['year']:
            # Add 'subject' to the chatGPT entry
            chatgpt_entry['subject'] = valid_entry['subject']
            break  # Once found, no need to continue searching

# Save the updated chatGPT data back to a new file.
save_json_file("./prediction/chatGPT.json", gpt_json)

# Initialize dictionaries to count correct and total answers for each subject
correct_counts = defaultdict(int)
total_counts = defaultdict(int)

# Iterate over each entry in the updated chatGPT data
for entry in gpt_json:
    if 'subject' in entry and 'is_correct' in entry:
        subject = entry['subject']
        total_counts[subject] += 1
        if entry['is_correct']:
            correct_counts[subject] += 1

# Calculate accuracy for each subject
accuracy_by_subject = {
    subject: correct_counts[subject] / total_counts[subject] for subject in total_counts}
formatted_accuracy_by_subject = {
    subject: f"{accuracy * 100:.2f}%" for subject, accuracy in accuracy_by_subject.items()
}

# Creating a dictionary to store total count of questions and correct answers for each subject
subject_stats = {
    subject: {
        'correct_answers': correct_counts[subject],
        'total_questions': total_counts[subject]
    } for subject in total_counts
}

print("Accuracy by subject:")
for subject in formatted_accuracy_by_subject:
    print(f'{subject}: {subject_stats[subject]}')
    print(f"{subject}: {formatted_accuracy_by_subject[subject]}")
