"""模型实现模块 - 包含所有具体的模型实现"""

from data_detect.Japanese.models.normal import NormalModel
from data_detect.Japanese.models.Yuki import YukiModel
from data_detect.Japanese.models.Luke import LukeModel
from data_detect.Japanese.models.Kubota import KubotaModel
from data_detect.Japanese.models.Duo_Guard import DuoGuardModel

# 为了与工厂兼容，创建别名
normal = NormalModel
Yuki = YukiModel
Luke = LukeModel
Kubota = KubotaModel
Duo_Guard = DuoGuardModel

__all__ = [
    'NormalModel',
    'YukiModel',
    'LukeModel',
    'KubotaModel',
    'DuoGuardModel',
    # 别名
    'normal',
    'Yuki',
    'Luke',
    'Kubota',
    'Duo_Guard',
]
