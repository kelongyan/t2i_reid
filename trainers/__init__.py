from .trainer import Trainer

class TrainerFactory:
    # 训练器工厂类：根据模型名称创建对应的训练器实例
    @staticmethod
    def create(model_name, model, task_info, num_classes, is_distributed=False):
        # model_name: 模型类名
        # model: 模型实例
        # task_info: 任务配置信息
        # num_classes: 类别总数
        # is_distributed: 是否启用分布式训练
        if model_name == 'Model':
            return Trainer(model, task_info)
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
