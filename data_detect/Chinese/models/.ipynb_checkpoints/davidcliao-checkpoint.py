import sys
import os

# 获取当前脚本的根目录（Master_Thesis）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
from data_detect.base import BaseModel
import requests
from flair.data import Sentence
from flair.models import TextClassifier

class DavidcliaoModel(BaseModel):
    """Davidcliao Flair 模型 wrapper (自动下载并加载)"""
    
    def __init__(self, model_path="best-model.pt", device="cpu"):
        self.device = device
        self.model_path = model_path
        self.url = "https://github.com/davidycliao/taiwan-political-hatespeech-detection/raw/main/ch-hs-model/best-model.pt"
        
        # 自动触发检查与下载
        self._ensure_model_exists()
        
        # 加载模型
        print(f"Loading model from {self.model_path}...")
        self.classifier = TextClassifier.load(self.model_path)

    def _ensure_model_exists(self):
        """检查模型是否存在，不存在则下载"""
        if not os.path.exists(self.model_path):
            print(f"Model not found. Downloading from {self.url}...")
            try:
                response = requests.get(self.url, stream=True)
                response.raise_for_status() # 检查请求是否成功
                
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download complete.")
            except Exception as e:
                raise ConnectionError(f"Failed to download model: {e}")
        else:
            print("Model file already exists. Skipping download.")

    def score(self, text: str) -> dict:
        """
        返回预测标签和置信度
        Returns: {"label": 0/1, "prob": float(0.0-1.0)}
        """
        sentence = Sentence(text)
        self.classifier.predict(sentence)
        
        if not sentence.labels:
            return {"label": 0, "prob": 0.0}
            
        label_str = sentence.labels[0].value
        confidence = sentence.labels[0].score
        
        # 将 Flair 的标签映射到 0/1
        if label_str == 'Hate Speech':
            label = 1
        else:
            label = 0
        
        return {"label": label, "prob": float(confidence)}