from data_detect.Chinese.constants import ChineseModelName
from data_detect.Chinese.models.thucoai import THUCOAIModel
from data_detect.Chinese.models.davidcliao import DavidcliaoModel
from data_detect.Chinese.models.morit import MoritModel
from data_detect.base import BaseModel

class ChineseModelFactory:
    """中文模型工厂类"""

    _mapping = {
        ChineseModelName.THUCOAI: THUCOAIModel,
        ChineseModelName.DAVIDCLIAO: DavidcliaoModel,
        ChineseModelName.MORIT: MoritModel,
    }

    @classmethod
    def create_model(cls, logger, name: ChineseModelName, device="cpu") -> BaseModel:
        """
        创建模型实例
        
        Args:
            logger: 日志记录器
            name: 模型名称 (ChineseModelName 枚举)
            device: 计算设备 (cpu 或 cuda)
        
        Returns:
            BaseModel 的子类实例
        """
        model_class = cls._mapping.get(name)
        logger.info(f"Creating model: {model_class}")
        
        if not model_class:
            raise ValueError(f"Model {name} not supported")
        
        try:
            return model_class(device=device)
        except Exception as e:
            logger.error(f"Failed to create model {name}: {e}")
            raise
