from typing import List
from data_detect.base import BaseModel

class BatchProcessor:
    def __init__(self, models: List[BaseModel]):
        self.models = models

    def run_all(self, input_data: dict):
        results = []
        for m in self.models:
            res = m.predict(input_data)
            results.append(res)
        return results
