from .model import Model
from .gs3_module import (
    GS3Module,                      # 兼容性别名（实际指向SymmetricGS3Module）
    SymmetricGS3Module,            # 对称解耦版本
    SymmetricBranchAttention,      # 对称双分支注意力
    DualMambaStream,               # 双Mamba流
)
from .fshd_module import FSHDModule  # 新增：频域-空域联合解耦模块
from .frequency_module import DCTFrequencySplitter, WaveletFrequencySplitter  # 频域分解
from .hybrid_stream import HybridDualStream, MultiScaleCNN, FrequencyGuidedAttention  # 异构双流
from .semantic_guidance import SemanticGuidedDecoupling  # CLIP语义引导

__factory = {
    'Model': Model,
    'FSHDModule': FSHDModule,
    'GS3Module': GS3Module,
    'SymmetricGS3Module': SymmetricGS3Module,
    'SemanticGuidedDecoupling': SemanticGuidedDecoupling,
    'DCTFrequencySplitter': DCTFrequencySplitter,
    'WaveletFrequencySplitter': WaveletFrequencySplitter,
    'HybridDualStream': HybridDualStream,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError(f"Unknown model: {name}")
    return __factory[name](**kwargs)
