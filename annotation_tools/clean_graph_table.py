import os
import sys
import json


def filter_json_file(input_file_path, output_file_path):
    # Read the JSON file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Filter out entries where "instruction" contains "附圖" or "附表"
    filtered_data = [entry for entry in data if "附圖" not in entry["instruction"] and "附表" not in entry["instruction"]
                     and "如圖" not in entry["instruction"] and "如表" not in entry["instruction"] and "圖二" not in entry["instruction"]
                     and "表格" not in entry["instruction"] and "右圖" not in entry["instruction"] and "左圖" not in entry["instruction"]
                     and "右表" not in entry["instruction"] and "左表" not in entry["instruction"]]

    # Write the filtered data back to the file or a new file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=4)


original_dir = sys.argv[1]
ori_json_file_path = os.path.join(
    original_dir, 'problem_database_original.json')
filtered_json_file_path = os.path.join(
    original_dir, 'problem_database_filtered.json')
filter_json_file(ori_json_file_path, filtered_json_file_path)
