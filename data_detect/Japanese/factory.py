from data_detect.Japanese.constants import ModelName
from data_detect.Japanese.models import normal, Yuki, Duo_Guard, Kubota, Luke
from data_detect.base import BaseModel

class ModelFactory:
    _mapping = {
        ModelName.LUKE: Luke,
        ModelName.DUO_GUARD: Duo_Guard,
        ModelName.KUBOTA: Kubota,
        ModelName.YUKI: Yuki,
        ModelName.KIT: normal,
        ModelName.YACIS: normal,
    }

    @classmethod
    def create_model(cls, name: ModelName) -> BaseModel:
        model_class = cls._mapping.get(name)
        if not model_class:
            raise ValueError(f"Model {name} not supported")
        elif model_class is normal:
            return model_class(name.value)
        return model_class()
