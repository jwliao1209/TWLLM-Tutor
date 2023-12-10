import os
import json

def transform_dict(original):
    return {
        "id": original["id"],
        "instruction": f"問題：{original['question']} \nA. {original['A']} \nB. {original['B']} \nC. {original['C']} \nD. {original['D']} \n答案：",
        "output": f"{original['answer']} {original[str(original['answer'])]}"
    }

def process_directory(directory):
    # Paths for the input and output files
    file_path = os.path.join(directory, "content.json")
    output_path = os.path.join(directory, "content_hw3.json")

    # Reading the input file
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    # Transforming the data
    transformed_dicts = [
        transform_dict(original) 
        for original in data["questions"] 
        if original.get("type") != "multi" and
        original.get("answer") != "無答案" and
        len(original.get("answer", "")) <= 1
    ]

    # Writing the results to a file
    with open(output_path, 'w', encoding="utf-8") as file:
        json.dump(transformed_dicts, file, indent=4, ensure_ascii=False)


def list_directories(path='.'):
    """
    Lists all directories in the specified path.
    Default is the current directory.
    """
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

# Example usage
current_dir_directories = list_directories()
for directory in current_dir_directories:
    print(f"Processing directory: {directory}")
    process_directory(directory)