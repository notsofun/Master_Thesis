"""数据检测模块 - 处理仇恨言论的检测和分析"""

# 延迟导入以避免依赖问题
def __getattr__(name):
    if name == 'BaseModel':
        from data_detect.base import BaseModel
        return BaseModel
    elif name == 'HatePipeline':
        from data_detect.pipeline import HatePipeline
        return HatePipeline
    elif name == 'compute_sample_size':
        from data_detect.utils import compute_sample_size
        return compute_sample_size
    elif name == 'random_sample_indices':
        from data_detect.utils import random_sample_indices
        return random_sample_indices
    elif name == 'ensure_dir':
        from data_detect.utils import ensure_dir
        return ensure_dir
    elif name == 'save_annotation_csv':
        from data_detect.utils import save_annotation_csv
        return save_annotation_csv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'BaseModel',
    'HatePipeline',
    'compute_sample_size',
    'random_sample_indices',
    'ensure_dir',
    'save_annotation_csv',
]

