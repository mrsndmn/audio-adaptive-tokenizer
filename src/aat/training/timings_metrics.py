
from time import perf_counter
from typing import Dict

class RecordTimings:
    def __enter__(self, metrics_dict: Dict, metric_name: str):
        self.metrics_dict = metrics_dict
        self.metric_name = metric_name
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start
        self.metrics_dict[self.metric_name] = self.time

