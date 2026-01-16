from dataclasses import dataclass
from enum import Enum

@dataclass(frozen=True)
class ModelInfo:
    name: str
    tokenizer: str
    model: str
    score_method: str

@dataclass
class HateScore:
    non_attack: float
    gray_zone: float
    attack: float

class ModelName(Enum):
    LUKE = ModelInfo(
        name="luke_offensiveness",
        tokenizer="studio-ousia/luke-japanese-base-lite",
        model="TomokiFujihara/luke-japanese-base-lite-offensiveness-estimation",
        score_method= "luke_score_text"
    )

    DUO_GUARD = ModelInfo(
        name="DuoGuard",
        tokenizer="Qwen/Qwen2.5-1.5B",
        model="DuoGuard/DuoGuard-1.5B-transfer",
        score_method="duo_guard_score"
    )

    KUBOTA = ModelInfo(
        name="kubota",
        model="kubota/luke-large-defamation-detection-japanese",
        tokenizer= '',
        score_method="kubota_score",
    )

    YUKI = ModelInfo(
        name="yuki",
        tokenizer="yukismd/HateSpeechClassification-japanese-gpt-neox-3-6b-instruction-ppo",
        model="yukismd/HateSpeechClassification-japanese-gpt-neox-3-6b-instruction-ppo",
        score_method='yuki_score'
    )

    KIT = ModelInfo(
        name="kit",
        tokenizer="kit-nlp/electra-small-japanese-discriminator-cyberbullying",
        model="kit-nlp/electra-small-japanese-discriminator-cyberbullying",
        score_method="kit_score"
    )

    YACIS = ModelInfo(
        name="yacis",
        tokenizer="ptaszynski/yacis-electra-small-japanese-cyberbullying",
        model="ptaszynski/yacis-electra-small-japanese-cyberbullying",
        score_method='kit_score' # 标准的transformer分类模型，可以用一套分类方法
    )