from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class ModelInfo:
    name: str
    model: str
    score_method: str = "default_score"

@dataclass
class HateScore:
    non_attack: float
    gray_zone: float
    attack: float

class ChineseModelName(Enum):
    """中文仇恨言论检测模型列表"""
    
    THUCOAI = ModelInfo(
        name="thu_coai_roberta",
        model="thu-coai/roberta-base-cold",
        score_method="thucoai_score"
    )

    DAVIDCLIAO = ModelInfo(
        name="davidcliao_flair",
        model="best-model.pt",  # 需要本地下载
        score_method="davidcliao_score"
    )

    MORIT = ModelInfo(
        name="morit_xlm_xnli",
        model="morit/chinese_xlm_xnli",
        score_method="morit_score"
    )