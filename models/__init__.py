from .model import Model
from .gs3_module import GS3Module, OrthogonalProjectionAttention, ContentAwareMambaFilter

__factory = {
    'Model': Model,
    'GS3Module': GS3Module,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError(f"Unknown model: {name}")
    return __factory[name](**kwargs)
