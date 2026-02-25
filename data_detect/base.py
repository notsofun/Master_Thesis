from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class BaseModel(ABC):

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """批量预测，返回统一格式: [{"label": int, "prob": float}, ...]"""
        out = []
        for t in texts:
            try:
                result = self.score(t)
                
                # score 方法应返回 dict: {"label": int, "prob": float}
                # 或兼容返回 int（老的实现）
                if isinstance(result, dict):
                    label = int(result.get("label", 0))
                    prob = float(result.get("prob", 0.5))
                else:
                    # 向后兼容：如果返回 int，按原方式处理
                    label = int(result)
                    prob = 1.0
                
                out.append({"label": label, "prob": prob})
            except Exception as e:
                out.append({"label": 0, "prob": 0.0})
        return out
    
    def _get_clean_device(self, device):
            """统一返回 torch.device 对象，自动处理 cuda 探测"""
            if device == "cpu" and torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device(device)
    
    @abstractmethod
    def score(self, text: str) -> Any:
        """
        子类必须实现此方法。
        
        返回值可以是：
        - dict: {"label": int (0/1), "prob": float (0.0-1.0)}
        - int: 标签值 0 或 1（向后兼容，prob 将设为 1.0）
        """
        pass
