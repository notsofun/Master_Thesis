import sys
import os

# 获取当前脚本的根目录（Master_Thesis）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName, ModelInfo, HateScore
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np
import torch

class LukeModel(BaseModel):
    def __init__(self, device="cpu"):
        model_info = ModelName.LUKE.value
        
        self.device = self._get_clean_device(device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_info.model, 
            trust_remote_code=True
        )
        
        # 3. 直接移动到 device 对象上
        self.model.to(self.device)
        self.model.eval()

    def score(self, text: str) -> dict:
            # 1. 编码
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
                
            with torch.no_grad():
                # 2. 修改这里：不要用 **inputs，而是显式传参
                # 这里的顺序必须和模型 forward 定义的一致（通常是 input_ids 在前，mask 在后）
                outputs = self.model(
                    inputs['input_ids'], 
                    inputs['attention_mask']
                )
                
                # 如果 outputs 直接就是 tensor，就按你成功的代码来
                # 如果 outputs 是 ModelOutput 对象，则用 outputs.logits
                if hasattr(outputs, "logits"):
                    logits = outputs.logits[0][:3]
                else:
                    logits = outputs[0][:3]
    
            # 3. 后续处理逻辑（保持你原来的 numpy 转换和归一化）
            logits_np = logits.detach().cpu().numpy()
            
            # ... 剩下的归一化和 Label 逻辑 ...
            minimum = np.min(logits_np)
            if minimum < 0:
                logits_np = logits_np - minimum
            
            probs = logits_np / np.sum(logits_np)
            
            # 组装返回结果
            return {
                "label": 1 if np.argmax(probs) == 2 else 0, # 假设 2 是 attack
                "prob": float(probs[2] if np.argmax(probs) == 2 else probs[0])
            }