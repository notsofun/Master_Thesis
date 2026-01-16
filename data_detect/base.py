from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseModel(ABC):

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        out = []
        for t in texts:
            try:
                lbl = int(self.score(t))
                out.append({"label": lbl, "prob": float(1.0)})
            except Exception:
                out.append({"label": 0, "prob": 0.0})
        return out

    @abstractmethod
    def score(self, text: str) -> int:
        """所有子类必须实现此方法，且签名一致"""
        pass
