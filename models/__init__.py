from .model import Model
from .gs3_module import (
    GS3Module,                      # 兼容性别名（实际指向SymmetricGS3Module）
    SymmetricGS3Module,            # 对称解耦版本
    SymmetricBranchAttention,      # 对称双分支注意力
    DualMambaStream,               # 双Mamba流
)
from .semantic_guidance import SemanticGuidedDecoupling  # CLIP语义引导

__factory = {
    'Model': Model,
    'GS3Module': GS3Module,
    'SymmetricGS3Module': SymmetricGS3Module,
    'SemanticGuidedDecoupling': SemanticGuidedDecoupling,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError(f"Unknown model: {name}")
    return __factory[name](**kwargs)
