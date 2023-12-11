import torch
import numpy as np


class Perplexity:
    def __init__(self):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

    def __call__(self, pred_logits, labels, output_masks):
        shift_logits = pred_logits[..., :-1, :].contiguous().transpose(1, 2)
        shift_labels = labels[..., 1:].contiguous()
        shift_output_masks = output_masks[..., 1:].contiguous()
        length = shift_output_masks.sum(dim=1)
        scores = torch.exp((self.cross_entropy(shift_logits, shift_labels) * shift_output_masks).sum(dim=1) / length).tolist()
        return np.mean(scores)
