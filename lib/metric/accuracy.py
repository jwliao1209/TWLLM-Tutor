import re
import numpy as np


def get_correct_num(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return (y_pred == y_true).float().sum()


def correcter(generation: str, answer: str, description: str) -> bool:
    generation = re.sub(r"\s*原因：[\s\S]*", "", generation)
    generated_answer = re.findall(r'(?<![A-Za-z])[A-D](?![A-Za-z])', generation)
    if (not generated_answer) and description in generation:
        return True
    if (len(set(generated_answer)) == 1) and (generated_answer[0] == answer):
        return True

    generated_answer = re.search(r'答案.*?(?<![A-Za-z])[A-D](?![A-Za-z])', generation)
    if generated_answer and (generated_answer.group(0) == answer):
        return True

    return False
