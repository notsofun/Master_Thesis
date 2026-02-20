from transformers import AutoModel
import torch.nn as nn

# ==========================================
# 3. 多任务模型抽象 (双头结构)
# ==========================================
class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name):
        super(MultiTaskClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # 两个独立的分类头
        self.rel_header = nn.Linear(hidden_size, 1)  # 宗教相关性头
        self.hate_header = nn.Linear(hidden_size, 1) # 仇恨检测头
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 的输出
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        pooled_output = self.dropout(pooled_output)
        
        rel_logits = self.rel_header(pooled_output).squeeze(-1)
        hate_logits = self.hate_header(pooled_output).squeeze(-1)
        
        return rel_logits, hate_logits