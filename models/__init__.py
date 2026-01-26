from .model import Model
from .ahnet_module import AHNetModule, FSHDModule
from .ahnet_streams import IDStructureStream, AttributeTextureStream
from .semantic_guidance import SemanticGuidedDecoupling
from .fusion import get_fusion_module

__factory = {
    'Model': Model,
    'AHNetModule': AHNetModule,
    'FSHDModule': FSHDModule, # Kept for backward compatibility
    'IDStructureStream': IDStructureStream,
    'AttributeTextureStream': AttributeTextureStream,
    'SemanticGuidedDecoupling': SemanticGuidedDecoupling,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError(f"Unknown model: {name}")
    return __factory[name](**kwargs)
