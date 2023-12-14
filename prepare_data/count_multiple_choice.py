import json
import os


def list_directories(path='.'):
    """
    Lists all directories in the specified path.
    Default is the current directory.
    """
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories


current_dir_directories = list_directories()
count = []
for directory in current_dir_directories:
    # print(f"Processing directory: {directory}")
    file_path = os.path.join(directory, "content_hw3.json")
    # Reading the input file
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        # print(f"Number of records: {len(data)}")
        count.append(len(data))

print(f"Total number of records: {sum(count)}")
print(f"Average number of records: {sum(count)/len(count)}")
