import sys
import os
from pathlib import Path
# 获取当前脚本的根目录（Master_Thesis）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)
    
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_detect.base import BaseModel
from data_detect.Japanese.constants import ModelName

class YukiModel(BaseModel):
    def __init__(self, device="cpu"):
        model_info = ModelName.YUKI.value
        
        # 1. 统一获取设备对象
        self.device = self._get_clean_device(device)
        
        # 2. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_info.model,
            use_fast=True,
            trust_remote_code=True,
        )

        # 3. 加载 CausalLM 模型 (核心改动：从 SequenceClassification 换成 CausalLM)
        self.current_dir = Path(__file__).parent.absolute()
        self.pth_path = self.current_dir.parent / "classification_head.pth"
        is_cuda = self.device.type == "cuda"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_info.model,
            torch_dtype=torch.float32,
            device_map="auto" if is_cuda else None,
            trust_remote_code=True,
        )

        if not is_cuda:
            self.model.to(self.device)
            
        self.model.eval()

        # 4. 初始化分类头 (提前加载，避免 score 循环中重复读取文件)
        if not self.pth_path.exists():
            raise FileNotFoundError(f"找不到权重文件: {self.pth_path}")
            
        head_weights = torch.load(self.pth_path, map_location=self.device)
        # 获取权重维度，通常 GPT-NeoX 是 32000
        in_features = head_weights.shape[1] if len(head_weights.shape) > 1 else head_weights.shape[0]
        
        self.head = torch.nn.Linear(in_features, 1, bias=False).to(self.device)
        self.head.weight.data = head_weights.view(1, in_features)
        self.head.eval()

    def score(self, text: str) -> dict:
        """
        保持原本的输入输出不变
        Returns: {"label": 0/1, "prob": float(0.0-1.0)}
        """
        # 对齐输入
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)

        with torch.no_grad():
            # CausalLM 的 logits 形状是 [Batch, Seq_len, Vocab_size]
            out = self.model(**inputs).logits
            
            # 取最后一个 token 的 32000 维特征，并对齐 dtype
            # 形状从 [1, seq_len, 32000] 变为 [1, 32000]
            last_token_logits = out[:, -1, :].to(torch.float32)
            
            # 经过分类头得到最终 logits
            logits = self.head(last_token_logits)
            
            # 使用 sigmoid 获取置信度
            confidence = torch.sigmoid(logits[0]).item()
            
            # 判断标签（阈值为 0.5）
            label = 1 if confidence > 0.5 else 0

        return {"label": label, "prob": float(confidence)}