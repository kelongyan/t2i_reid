from .trainer import Trainer


class TrainerFactory:
    @staticmethod
    def create(model_name, model, task_info, num_classes, is_distributed=False):
        if model_name == 'Model':
            return Trainer(model, task_info)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
