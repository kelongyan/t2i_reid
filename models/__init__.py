from .model import Model
from .fshd_module import FSHDModule, GS3Module, SymmetricGS3Module
from .frequency_module import DCTFrequencySplitter
from .hybrid_stream import HybridDualStream, MultiScaleCNN, FrequencyGuidedAttention
from .semantic_guidance import SemanticGuidedDecoupling

__factory = {
    'Model': Model,
    'FSHDModule': FSHDModule,
    'GS3Module': GS3Module,
    'SymmetricGS3Module': SymmetricGS3Module,
    'SemanticGuidedDecoupling': SemanticGuidedDecoupling,
    'DCTFrequencySplitter': DCTFrequencySplitter,
    'HybridDualStream': HybridDualStream,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError(f"Unknown model: {name}")
    return __factory[name](**kwargs)