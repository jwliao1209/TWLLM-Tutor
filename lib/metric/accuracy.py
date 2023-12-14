import re

import numpy as np

"""
implement mutiple choice accuracy
"""


def get_correct_num(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return (y_pred == y_true).float().sum()


# Todo: implement accuracy for multiple choice
# class Accuracy:
#     def __init__(self):
#         pass

#     def __call__(self):
#         pass

# class Correcter:
#     def __call__(self, generation, answer, description):
#         if description in generation:
#             self.num += 1
#             return

#         generated_answer = re.findall(r'[A-D]', generation)
#         if len(generated_answer) == 1:
#             if generated_answer[0] == answer:
#                 self.num += 1
#         return

def correcter(generation, answer, description):
    if description in generation:
        return True

    generated_answer = re.findall(r'[A-D]', generation)
    if len(generated_answer) == 1:
        if generated_answer[0] == answer:
            return True
    return False
