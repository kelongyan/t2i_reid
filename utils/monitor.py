"""
训练过程监控工具
用于监控模型训练过程中的详细信息，包括关键模块权重、数据流动等
并将调试信息写入日志文件
"""
import os
import logging
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional


class TrainingMonitor:
    """
    训练过程监控器
    用于监控模型训练过程中的详细信息，包括关键模块权重、数据流动等
    """

    def __init__(self, dataset_name: str = "unknown", log_dir: str = "log"):
        """
        初始化监控器

        Args:
            dataset_name: 数据集名称 (cuhk_pedes, rstp, icfg)
            log_dir: 日志根目录
        """
        self.dataset_name = dataset_name
        self.log_dir = Path(log_dir) / dataset_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志记录器
        self.setup_loggers()

        # 记录初始化时间
        self.logger.info(f"Training monitor initialized for dataset: {dataset_name}")
        self.logger.info(f"Log directory: {self.log_dir}")

    def setup_loggers(self):
        """设置日志记录器"""
        # 主要日志记录器 (log.txt)
        self.logger = logging.getLogger(f"training_{self.dataset_name}")
        self.logger.setLevel(logging.INFO)

        # 清除现有处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 主要日志文件处理器
        main_log_file = self.log_dir / "log.txt"
        main_handler = logging.FileHandler(main_log_file, mode='a', encoding='utf-8')
        main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        main_handler.setFormatter(main_formatter)
        self.logger.addHandler(main_handler)

        # 调试日志记录器 (debug.txt)
        self.debug_logger = logging.getLogger(f"debug_{self.dataset_name}")
        self.debug_logger.setLevel(logging.DEBUG)

        # 清除现有处理器
        for handler in self.debug_logger.handlers[:]:
            self.debug_logger.removeHandler(handler)

        # 调试日志文件处理器
        debug_log_file = self.log_dir / "debug.txt"
        debug_handler = logging.FileHandler(debug_log_file, mode='a', encoding='utf-8')
        debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')
        debug_handler.setFormatter(debug_formatter)
        self.debug_logger.addHandler(debug_handler)

    def log_training_start(self, model: nn.Module, args: Any):
        """记录训练开始信息"""
        self.logger.info("=" * 60)
        self.logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Dataset: {self.dataset_name}")
        self.logger.info(f"Model architecture: {model.__class__.__name__}")

        # 记录训练参数
        self.logger.info(f"Training parameters:")
        self.logger.info(f"  - Batch size: {getattr(args, 'batch_size', 'Unknown')}")
        self.logger.info(f"  - Learning rate: {getattr(args, 'lr', 'Unknown')}")
        self.logger.info(f"  - Epochs: {getattr(args, 'epochs', 'Unknown')}")
        self.logger.info(f"  - Weight decay: {getattr(args, 'weight_decay', 'Unknown')}")
        self.logger.info(f"  - Mixed precision: {getattr(args, 'fp16', 'Unknown')}")

        # 记录模型参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")

        self.logger.info("-" * 60)

    def log_training_end(self, final_metrics: Dict[str, float]):
        """记录训练结束信息"""
        self.logger.info("-" * 60)
        self.logger.info(f"Training ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Final metrics: {final_metrics}")
        self.logger.info("=" * 60)

    def log_epoch_info(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """记录每个epoch的信息"""
        self.logger.info(f"Epoch [{epoch}/{total_epochs}] - Metrics: {metrics}")

    def log_batch_info(self, epoch: int, batch_idx: int, total_batches: int,
                      loss_info: Dict[str, float], learning_rate: float = None):
        """记录每个批次的信息"""
        if batch_idx % 100 == 0:  # 每100个批次记录一次
            info_str = f"Epoch {epoch}, Batch [{batch_idx}/{total_batches}]"
            if learning_rate is not None:
                info_str += f", LR: {learning_rate:.6f}"
            info_str += f", Losses: {loss_info}"
            self.logger.info(info_str)

    def log_model_weights(self, model: nn.Module, step: str = "initial"):
        """记录模型关键模块的权重信息"""
        self.debug_logger.debug(f"Model weights {step} state:")

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                weight_info = self._get_tensor_stats(module.weight, f"{name}.weight")
                bias_info = self._get_tensor_stats(module.bias, f"{name}.bias") if module.bias is not None else "None"

                self.debug_logger.debug(f"  {name}:")
                self.debug_logger.debug(f"    Weight: {weight_info}")
                self.debug_logger.debug(f"    Bias: {bias_info}")

    def log_gradients(self, model: nn.Module, step: str = "after_backward"):
        """记录模型梯度信息"""
        self.debug_logger.debug(f"Gradients {step} state:")

        total_grad_norm = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                total_grad_norm += grad_norm
                param_count += 1

                if param_count <= 10:  # 只记录前10个参数的梯度信息
                    self.debug_logger.debug(f"  {name}: grad_norm={grad_norm:.6f}")

        avg_grad_norm = total_grad_norm / param_count if param_count > 0 else 0.0
        self.debug_logger.debug(f"  Average gradient norm: {avg_grad_norm:.6f}")

    def log_feature_statistics(self, features: torch.Tensor, name: str = "features"):
        """记录特征统计信息"""
        if features is not None:
            stats = self._get_tensor_stats(features, name)
            self.debug_logger.debug(f"{name} statistics: {stats}")

    def log_loss_components(self, loss_dict: Dict[str, torch.Tensor]):
        """记录损失组件信息"""
        loss_info = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                loss_info[key] = value.item()
            else:
                loss_info[key] = value
        self.debug_logger.debug(f"Loss components: {loss_info}")

    def log_memory_usage(self):
        """记录GPU内存使用情况"""
        if torch.cuda.is_available():
            memory_info = {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated(),
                'max_cached': torch.cuda.max_memory_reserved()
            }
            self.debug_logger.debug(f"GPU Memory usage: {memory_info}")

    def _get_tensor_stats(self, tensor: torch.Tensor, name: str = "") -> Dict[str, float]:
        """获取张量统计信息"""
        if tensor is None:
            return {"value": "None"}

        # 确保tensor在CPU上以便计算统计信息
        tensor_cpu = tensor.detach().cpu()

        stats = {
            "mean": tensor_cpu.mean().item(),
            "std": tensor_cpu.std().item(),
            "min": tensor_cpu.min().item(),
            "max": tensor_cpu.max().item(),
            "shape": list(tensor_cpu.shape),
            "requires_grad": tensor.requires_grad if hasattr(tensor, 'requires_grad') else False
        }
        return stats

    def log_attention_weights(self, attention_weights: torch.Tensor, layer_name: str = "attention"):
        """记录注意力权重信息"""
        if attention_weights is not None:
            stats = self._get_tensor_stats(attention_weights, f"{layer_name}_weights")
            self.debug_logger.debug(f"{layer_name} weights: {stats}")

    def log_gate_weights(self, gate_weights: torch.Tensor, module_name: str = "gate"):
        """记录门控权重信息"""
        if gate_weights is not None:
            stats = self._get_tensor_stats(gate_weights, f"{module_name}_weights")
            self.debug_logger.debug(f"{module_name} weights: {stats}")

    def log_fusion_info(self, fused_features: torch.Tensor, gate_weights: torch.Tensor = None):
        """记录融合模块信息"""
        self.debug_logger.debug("Fusion module information:")
        self.log_feature_statistics(fused_features, "fused_features")
        if gate_weights is not None:
            self.log_gate_weights(gate_weights, "fusion_gate")

    def log_disentangle_info(self, id_features: torch.Tensor, cloth_features: torch.Tensor,
                           gate: torch.Tensor = None):
        """记录解耦模块信息"""
        self.debug_logger.debug("Disentangle module information:")
        self.log_feature_statistics(id_features, "identity_features")
        self.log_feature_statistics(cloth_features, "clothing_features")
        if gate is not None:
            self.log_gate_weights(gate, "disentangle_gate")
    
    def log_gs3_module_info(self, id_seq: torch.Tensor, cloth_seq: torch.Tensor, 
                           saliency_score: torch.Tensor = None, 
                           id_filtered: torch.Tensor = None):
        """记录 G-S3 模块详细信息"""
        self.debug_logger.debug("=" * 50)
        self.debug_logger.debug("G-S3 Module Debug Info:")
        
        # OPA 输出
        if id_seq is not None:
            stats = self._get_tensor_stats(id_seq, "id_seq_after_opa")
            self.debug_logger.debug(f"  ID sequence (after OPA): {stats}")
        
        if cloth_seq is not None:
            stats = self._get_tensor_stats(cloth_seq, "cloth_seq_after_opa")
            self.debug_logger.debug(f"  Cloth sequence (after OPA): {stats}")
        
        # 正交性检查
        if id_seq is not None and cloth_seq is not None:
            import torch.nn.functional as F
            id_norm = F.normalize(id_seq, dim=-1)
            cloth_norm = F.normalize(cloth_seq, dim=-1)
            cosine_sim = (id_norm * cloth_norm).sum(dim=-1).mean().item()
            self.debug_logger.debug(f"  Orthogonality (cosine sim): {cosine_sim:.6f} (should be close to 0)")
        
        # 显著性分数
        if saliency_score is not None:
            stats = self._get_tensor_stats(saliency_score, "saliency_score")
            self.debug_logger.debug(f"  Saliency score: {stats}")
            # 统计高/中/低显著性的比例
            flat_score = saliency_score.flatten()
            high_ratio = (flat_score > 0.7).float().mean().item()
            mid_ratio = ((flat_score > 0.3) & (flat_score <= 0.7)).float().mean().item()
            low_ratio = (flat_score <= 0.3).float().mean().item()
            self.debug_logger.debug(f"  Saliency distribution: High={high_ratio:.2%}, Mid={mid_ratio:.2%}, Low={low_ratio:.2%}")
        
        # Mamba 过滤后
        if id_filtered is not None:
            stats = self._get_tensor_stats(id_filtered, "id_filtered_after_mamba")
            self.debug_logger.debug(f"  ID features (after Mamba filter): {stats}")
        
        self.debug_logger.debug("=" * 50)
    
    def log_loss_breakdown(self, loss_dict: Dict[str, torch.Tensor], epoch: int, batch_idx: int):
        """详细记录每个损失分量"""
        self.debug_logger.debug(f"Loss breakdown [Epoch {epoch}, Batch {batch_idx}]:")
        
        total_loss = loss_dict.get('total', torch.tensor(0.0)).item()
        self.debug_logger.debug(f"  Total loss: {total_loss:.6f}")
        
        for key, value in sorted(loss_dict.items()):
            if key != 'total' and isinstance(value, torch.Tensor):
                loss_val = value.item()
                percentage = (loss_val / total_loss * 100) if total_loss > 0 else 0
                self.debug_logger.debug(f"    - {key}: {loss_val:.6f} ({percentage:.2f}%)")
    
    def log_model_architecture(self, model: torch.nn.Module):
        """记录模型详细架构"""
        self.debug_logger.debug("=" * 80)
        self.debug_logger.debug("Model Architecture:")
        self.debug_logger.debug(f"  Model class: {model.__class__.__name__}")
        
        # 记录主要模块
        for name, module in model.named_children():
            module_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            self.debug_logger.debug(f"  - {name}: {module.__class__.__name__}")
            self.debug_logger.debug(f"      Params: {module_params:,} (Trainable: {trainable_params:,})")
        
        self.debug_logger.debug("=" * 80)
    
    def log_data_batch_info(self, batch_data: Dict[str, torch.Tensor], batch_idx: int):
        """记录数据批次信息"""
        self.debug_logger.debug(f"Batch {batch_idx} data info:")
        
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                self.debug_logger.debug(f"  {key}: shape={list(value.shape)}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                self.debug_logger.debug(f"  {key}: type={type(value).__name__}, len={len(value)}")
    
    def log_optimizer_state(self, optimizer: torch.optim.Optimizer, epoch: int):
        """记录优化器状态"""
        self.debug_logger.debug(f"Optimizer state [Epoch {epoch}]:")
        
        for i, param_group in enumerate(optimizer.param_groups):
            self.debug_logger.debug(f"  Param group {i}:")
            self.debug_logger.debug(f"    LR: {param_group['lr']:.8f}")
            self.debug_logger.debug(f"    Weight decay: {param_group.get('weight_decay', 0):.6f}")
            self.debug_logger.debug(f"    Num params: {len(param_group['params'])}")
    
    def log_gradient_flow(self, model: torch.nn.Module):
        """记录梯度流动情况，检测梯度消失/爆炸"""
        self.debug_logger.debug("Gradient flow analysis:")
        
        ave_grads = []
        max_grads = []
        layers = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                layers.append(name)
                ave_grads.append(param.grad.abs().mean().item())
                max_grads.append(param.grad.abs().max().item())
        
        # 检测异常梯度
        for i, (layer, ave_grad, max_grad) in enumerate(zip(layers, ave_grads, max_grads)):
            if i < 5 or i >= len(layers) - 5:  # 只记录前5层和后5层
                status = ""
                if ave_grad < 1e-7:
                    status = " [WARNING: Vanishing gradient]"
                elif ave_grad > 1e2:
                    status = " [WARNING: Exploding gradient]"
                
                self.debug_logger.debug(f"  {layer}: avg={ave_grad:.8f}, max={max_grad:.8f}{status}")
    
    def log_checkpoint_save(self, checkpoint_path: str, epoch: int, metrics: Dict[str, float]):
        """记录检查点保存信息"""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.debug_logger.debug(f"Checkpoint details [Epoch {epoch}]:")
        self.debug_logger.debug(f"  Path: {checkpoint_path}")
        self.debug_logger.debug(f"  Metrics: {metrics}")
        self.debug_logger.debug(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def get_monitor_for_dataset(dataset_name: str, log_dir: str = "log") -> TrainingMonitor:
    """
    根据数据集名称获取对应的监控器

    Args:
        dataset_name: 数据集名称 ('cuhk_pedes', 'rstp', 'icfg', 或其他)
        log_dir: 日志根目录

    Returns:
        TrainingMonitor实例
    """
    # 规范化数据集名称 - 使用与检查点保存一致的命名规则
    if 'cuhk' in dataset_name.lower():
        normalized_name = 'cuhk'
    elif 'rstp' in dataset_name.lower():
        normalized_name = 'rstp'
    elif 'icfg' in dataset_name.lower():
        normalized_name = 'icfg'
    else:
        normalized_name = dataset_name.lower()

    return TrainingMonitor(dataset_name=normalized_name, log_dir=log_dir)


# 示例使用方法
if __name__ == "__main__":
    # 创建监控器示例
    monitor = TrainingMonitor(dataset_name="cuhk_pedes")

    # 记录一些示例信息
    monitor.logger.info("This is a test log entry")
    monitor.debug_logger.debug("This is a test debug entry")

    print(f"Log directory created at: {monitor.log_dir}")
    print("Monitor initialization completed!")