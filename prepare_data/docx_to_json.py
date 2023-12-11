import os
import sys
import docx
import json


def list_files(path="."):
    # List all files in the directory
    file_names = [f for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]
    return file_names


def list_directories(path='.'):
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
        num_str = ''
        for char in name:
            if char.isdigit():
                num_str += char
            else:
                break
        numbers.append(int(num_str))
    return numbers


raw_dir = sys.argv[1]
out_dir = sys.argv[2]

raw_dir_directories = list_files(raw_dir)
years = extract_leading_numbers(raw_dir_directories)
output_dir_directories = list_directories(out_dir)

print(raw_dir_directories[0])
document = docx.Document(raw_dir+raw_dir_directories[0])
lines = [para.text for para in document.paragraphs]
# Flag to indicate whether "單選題" has been found
found = False
end = False

# Iterate over the lines
for line in lines:
    if found:
        print(line)
    if "單選題" in line:
        found = True
    if "【公民與社會科】" in line:
        end = True
    if end:
        break

# for file in raw_dir_directories:
#     print(f"Processing file: {file}")
#     document = docx.Document(raw_dir+file)
#     lines = [para.text for para in document.paragraphs]
#     print(lines)
#     # json_result = json.dumps(lines)
