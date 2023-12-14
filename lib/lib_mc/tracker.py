import torch
from collections import OrderedDict


class MetricTracker:
    def reset(self, keys):
        self.keys = keys
        self.record = OrderedDict(
            [
                (k, OrderedDict([(stat, 0)for stat in ["total", "counts", "avg"]])) for k in self.keys
            ]
        )
        return

    def update(self, key, value, num=1):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        self.record[key]["total"] += value * num
        self.record[key]["counts"] += num
        self.record[key]["avg"] = self.record[key]["total"] / self.record[key]["counts"]
        return
    
    def result(self):
        return dict([(k, self.record[k]["avg"]) for k in self.keys])
