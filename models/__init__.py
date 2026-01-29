from .model import Model
from .ahnet_module import AHNetModule
from .ahnet_streams import IDStructureStream, AttributeTextureStream
from .semantic_guidance import SemanticGuidedDecoupling
from .fusion import get_fusion_module

# 模型组件工厂
__factory = {
    'Model': Model,
    'AHNetModule': AHNetModule,
    'IDStructureStream': IDStructureStream,
    'AttributeTextureStream': AttributeTextureStream,
    'SemanticGuidedDecoupling': SemanticGuidedDecoupling,
}


def names():
    """返回所有已注册的模型组件名称"""
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    """根据名称创建对应的模型组件实例"""
    if name not in __factory:
        raise KeyError(f"未知的模型组件: {name}")
    return __factory[name](**kwargs)
