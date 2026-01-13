from .model import Model

__factory = {
    'Model': Model,
}


def names():
    return sorted(__factory.keys())


def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError(f"Unknown model: {name}")
    return __factory[name](**kwargs)
