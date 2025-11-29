# hate_detector.py
# 这个是可扩展的
from dataclasses import dataclass
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM
from tqdm import tqdm
from enum import Enum
import torch, os
from huggingface_hub import hf_hub_download

current_dir = os.path.dirname(os.path.abspath(__file__))
@dataclass
class HateScore:
    non_attack: float
    gray_zone: float
    attack: float

@dataclass(frozen=True)
class ModelInfo:
    name: str
    tokenizer: str
    model: str
    score_method: str

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

class ModelWrapper:
    def __init__(self, logger, model_info: ModelInfo, device="cpu"):
        """
        base_model_name: tokenizer base (e.g., 'studio-ousia/luke-japanese-base-lite')
        fine_tuned_model: if provided, path or HF id to the model weights to use for classification
        """
        self.logger = logger
        self.device = device
        self.model_info = model_info
        if model_info == ModelName.DUO_GUARD.value:
            self.tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # 2. Load the DuoGuard-0.5B model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_info.model,
                torch_dtype=torch.bfloat16
            ).to(self.device)
        elif model_info == ModelName.KUBOTA.value:
            self.pipe = pipeline(model= model_info.model)
        elif model_info == ModelName.YUKI.value:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_info.model,
                use_fast=True,
                trust_remote_code=True,
            )
            hf_hub_download(repo_id=self.model_info.model, filename="classification_head.pth", local_dir=current_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_info.model,
                torch_dtype=torch.float32,   # 强制全精度
                device_map={"": device},      # 绑定到 GPU/CPU
                trust_remote_code=True,
            ).eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_info.tokenizer)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_info.model, trust_remote_code=True)

        try:
            self.model.to(self.device)
        except Exception:
            self.logger.warning(f"This model_wrapper uses pipeline {self.model_info.name}")
            
        self.score_fn = getattr(self, model_info.score_method)

    def score_text(self, text: str) -> int:
        # 截断到 256 字符
        if len(text) > 256:
            text = text[:256]
        return self.score_fn(text)
    
    def luke_score_text(self, text: str) -> int:
        inputs = self.tokenizer.encode_plus(text, return_tensors="pt")
        logits = self.model(
            inputs["input_ids"],
            inputs["attention_mask"]
        ).detach().numpy()[0][:3]

        minimum = np.min(logits)
        if minimum < 0:
            logits = logits - minimum
        score = logits / np.sum(logits)
        score = HateScore(non_attack=float(score[0]), gray_zone=float(score[1]), attack=float(score[2]))
        max_score = max(score.non_attack, score.gray_zone, score.attack)
        return 1 if max_score == score.attack else 0

    def duo_guard_score(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.sigmoid(logits)

        category_names = [
            "Violent crimes",
            "Non-violent crimes",
            "Sex-related crimes",
            "Child sexual exploitation",
            "Specialized advice",
            "Privacy",
            "Intellectual property",
            "Indiscriminate weapons",
            "Hate",
            "Suicide and self-harm",
            "Sexual content",
            "Jailbreak prompts",
        ]

        prob_vector = probabilities[0].tolist()
        max_index = prob_vector.index(max(prob_vector))
        hate_index = category_names.index("Hate")

        return 1 if max_index == hate_index else 0

    def kubota_score(self, text: str) -> int:
        result = self.pipe(text)[0]   # 取第一个结果 dict
        rule = {
            "中傷性のない発言": 0,
            "脅迫的な発言": 1,
            "侮蔑的な発言": 2,
            "名誉を低下させる発言": 3,
        }

        label = result["label"]
        class_id = rule.get(label, 0)   # 找不到就当作 0

        # 如果类别是 2 或 3，则判为 1，否则 0
        return 1 if class_id in (2, 3) else 0

    def yuki_score(self, text:str) -> int:
        head_weights = torch.load("classification_head.pth", map_location=self.device)
        head = torch.nn.Linear(1, 1, bias=False).to(self.device)
        head.weight.data = head_weights

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)

        out = self.model(**inputs).logits
        out = out.to(head.weight.dtype)   # dtype 对齐
        logits = head(out[:, -1])

        threshold = 0
        is_hate = int(logits.item() > threshold)

        return is_hate

    def kit_score(self, text:str) -> int:
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)

        result = self.model(**inputs).logits

        pred_label = torch.argmax(result, dim=-1).item()

        return pred_label

class HateSpeechDetector:
    def __init__(self,logger, model_specs: list[ModelInfo], device="cpu"):
        """
        model_specs: list of dicts, each dict: {"name": "friendly name", "tokenizer_base": "...", "model": "..."}
        """
        import torch
        self.torch = torch
        self.device = device
        self.models = {}
        self.logger = logger
        for spec in model_specs:
            self.logger.info(f"[INFO] Loading model {spec.name} -> {spec.model or spec.tokenizer}")
            wrapper = ModelWrapper(logger=self.logger,model_info=spec,device=self.device)
            self.models[spec.name] = wrapper

    def run_on_texts(self, texts):
        """
        texts: iterable of str
        return: dict of DataFrames keyed by model name with columns: text, HS
        """
        outputs = {}
        total = len(texts)

        for name, wrapper in self.models.items():
            self.logger.info(f"Start running model: {name} on {total} texts")

            rows = []
            for idx, text in enumerate(tqdm(texts, desc=f"Running {name}", leave=False, ncols=100)):
                try:
                    hs = wrapper.score_text(text)
                    rows.append({"text": text, "HS": hs})
                except Exception as e:
                    self.logger.warning(
                        f"[{name}] Error processing text idx={idx}: {text[:50]}... - {e}"
                    )
                    rows.append({"text": text, "HS": None, "error": str(e)})

            df = pd.DataFrame(rows)
            outputs[name] = df

            try:
                hs_series = pd.to_numeric(df["HS"], errors="coerce").fillna(0).astype(int)
                hs_count = hs_series.sum()
                valid = df["HS"].notna().sum()
                ratio = (hs_count / valid * 100) if valid > 0 else 0
            except Exception as e:
                self.logger.error(f"[{name}] Error computing summary stats: {e}")
                hs_count = valid = ratio = 0

            self.logger.info(
                f"[{name}] Finished. Processed: {valid}/{total} valid texts. "
                f"Hate Speech: {hs_count} ({ratio:.2f}%)."
            )

        return outputs

