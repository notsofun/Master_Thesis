from data_detect.Japanese.constants import ModelName
from data_detect.Japanese.models import NormalModel, YukiModel, DuoGuardModel, KubotaModel, LukeModel
from data_detect.base import BaseModel

class ModelFactory:

    _mapping = {
        ModelName.LUKE: LukeModel,
        ModelName.DUO_GUARD: DuoGuardModel,
        ModelName.KUBOTA: KubotaModel,
        ModelName.YUKI: YukiModel,
        ModelName.KIT: NormalModel,
        ModelName.YACIS: NormalModel,
    }

    @classmethod
    def create_model(cls, logger, name: ModelName, device) -> BaseModel:
        model_class = cls._mapping.get(name)
        logger.info(f"now we are creating {model_class}")
        if not model_class:
            raise ValueError(f"Model {name} not supported")
        if model_class == NormalModel:
            return NormalModel(model_info=name.value, device=device)
        else:
            return model_class(device=device)
