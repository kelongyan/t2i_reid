# src/utils/monitor.py
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import json
import os
import logging
from pathlib import Path

# 创建logs目录
LOG_DIR = Path("log")
DEBUG_LOG_DIR = LOG_DIR / "debug"
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)

# 配置logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# 创建debug logger (详细日志)
debug_logger = logging.getLogger('train.debug')
debug_logger.setLevel(logging.DEBUG)

# 配置文件处理器
file_handler = logging.FileHandler(LOG_DIR / "log.txt", mode='a', encoding='utf-8')
debug_file_handler = logging.FileHandler(DEBUG_LOG_DIR / "debug.txt", mode='a', encoding='utf-8')

# 配置格式化输出
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

file_handler.setFormatter(formatter)
debug_file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
debug_logger.addHandler(debug_file_handler)
logger.addHandler(logging.StreamHandler())
debug_logger.addHandler(logging.StreamHandler())

class TrainingMonitor:
    """训练监控器，记录训练过程中的各种信息"""
    
    def __init__(self, dataset_name: str, log_dir: str = "log"):
        self.dataset_name = dataset_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建数据集特定的日志目录
        dataset_log_dir = self.log_dir / dataset_name
        dataset_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志文件
        self.log_file = dataset_log_dir / "log.txt"
        self.debug_log_file = dataset_log_dir / "debug.txt"
        self.metrics_file = dataset_log_dir / "metrics.json"
        
        # 清理旧日志
        if self.log_file.exists():
            self.log_file.unlink()
        if self.debug_log_file.exists():
            self.debug_log_file.unlink()
        
        # 设置logger
        self.logger = logging.getLogger(f"train.{dataset_name}")
        self.logger.setLevel(logging.INFO)
        
        self.debug_logger = logging.getLogger(f"train.{dataset_name}.debug")
        self.debug_logger.setLevel(logging.DEBUG)
        
        # 配置handler
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        debug_file_handler = logging.FileHandler(self.debug_log_file, mode='a', encoding='utf-8')
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        debug_file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(logging.StreamHandler())
        
        self.debug_logger.addHandler(debug_file_handler)
        self.debug_logger.addHandler(logging.StreamHandler())
        
        # 存储训练指标
        self.metrics = []
        
    def log_feature_statistics(self, features: torch.Tensor, name: str):
        """记录特征统计信息"""
        stats = self._get_tensor_stats(features, name)
        self.debug_logger.debug(f"{name} statistics: {stats}")
    
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
    
    def log_gate_weights(self, gate_weights: torch.Tensor, module_name: str):
        """记录门控权重信息"""
        if gate_weights is not None:
            stats = self._get_tensor_stats(gate_weights, f"{module_name}_weights")
            self.debug_logger.debug(f"{module_name} weights: {stats}")
    
    def log_loss_components(self, loss_dict: Dict[str, torch.Tensor]):
        """【修复】记录损失组件信息 - 只在debug模式下输出"""
        # 只在debug模式下输出详细信息，避免与_format_loss_display重复
        if self.debug_logger.level >= 10:  # DEBUG level
            loss_info = {}
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    loss_info[key] = value.item()
                else:
                    loss_info[key] = value
            self.debug_logger.debug(f"Loss components: {loss_info}")
        # 不在info级别输出，避免与_format_loss_display重复
    
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
    
    def log_batch_info(self, epoch: int, batch_idx: int, total_batches: int,
                       loss_meters: Dict[str, float], lr: float):
        """记录批次信息"""
        loss_str = ', '.join([f"{k}: {v.avg:.4f}" for k, v in loss_meters.items()])
        self.logger.debug(f"Epoch {epoch}, Batch {batch_idx}/{total_batches}, LR: {lr:.6f}, {loss_str}")
    
    def log_epoch_info(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """记录epoch信息"""
        # 记录到metrics字典
        self.metrics.append({
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })
        
        # 保存到文件
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
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
    
    def log_gradients(self, model, step_name: str):
        """记录梯度信息"""
        self.debug_logger.debug(f"Gradients epoch_{step_name} state:")
        gradient_info = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_info[name] = f"grad_norm={grad_norm:.4f}"
        self.debug_logger.debug("Gradients epoch_{step_name} state:")
        for name, value in gradient_info.items():
            self.debug_logger.debug(f"   {name}: {value}")
    
    def log_gradient_flow(self, model):
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
            status = ""
            if i < 5 or i >= len(layers) - 5:  # 只记录前5层和后5层
                status = ""
                if ave_grad < 1e-7:
                    status = " [WARNING: Vanishing gradient]"
                if max_grad > 100:
                    status = " [WARNING: Exploding gradient]"
                if status:
                    self.debug_logger.debug(f"   {name}: grad_avg={ave_grad:.6f}, max={max_grad:.6f}{status}")
        
        # 检查CLIP文本编码器的bias梯度
        clip_bias_grads = []
        for name, param in model.named_parameters():
            if 'text_encoder' in name and 'bias' in name and param.grad is not None:
                if param.grad.abs().max().item() < 1e-7:
                    clip_bias_grads.append(name)
        
        if clip_bias_grads:
            self.debug_logger.warning(f"CLIP text encoder bias with vanishing gradient: {clip_bias_grads[:3]}")
    
    def log_loss_breakdown(self, loss_dict, epoch: int, batch_idx: int):
        """记录详细损失分解"""
        total = loss_dict['total'].item() if isinstance(loss_dict['total'], torch.Tensor) else loss_dict['total']
        loss_info = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                loss_info[key] = value.item()
            else:
                loss_info[key] = value
            
        # 计算百分比
        breakdown = f"Loss breakdown [Epoch {epoch}, Batch {batch_idx}]:"
        breakdown += f"   Total loss: {total:.4f}"
        
        # 按百分比排序
        loss_percentages = []
        for key, value in loss_dict.items():
            if key != 'total' and isinstance(value, torch.Tensor):
                percent = value.item() / total * 100
                loss_percentages.append((percent, key))
        
        loss_percentages.sort(key=lambda x: -x[0])
        for percent, key in loss_percentages[:10]:  # 显示前10个
            breakdown += f"     - {key}: {loss_dict[key].item():.4f} ({percent:.2f}%)"
        
        self.debug_logger.debug(breakdown)
    
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
        
        self.logger.info("=" * 60)
        self.logger.info(f"Epoch {epoch} Validation Loss : {self._format_loss_display(loss_meters)}")
        self.logger.info("=" * 60)
    
    def _format_loss_display(self, loss_meters):
        """【修复】格式化损失显示"""
        display_order = ['info_nce', 'cls', 'cloth_semantic', 'orthogonal', 'gate_adaptive', 'total']
        hidden_losses = set()
        
        avg_losses = []
        for key in display_order:
            if key in loss_meters and loss_meters[key].count > 0:
                avg_losses.append(f"{key}={loss_meters[key].avg:.4f}")
        
        return avg_losses
    
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
    
    def log_attention_weights(self, attention_weights: torch.Tensor, layer_name: str):
        """记录注意力权重信息"""
        if attention_weights is not None:
            stats = self._get_tensor_stats(attention_weights, f"{layer_name}_weights")
            self.debug_logger.debug(f"{layer_name} weights: {stats}")
    
    def log_gate_weights(self, gate_weights: torch.Tensor, module_name: str):
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
        """记录G-S3模块详细信息"""
        self.debug_logger.debug("=" * 50)
        self.debug_logger.debug("G-S3 Module Debug Info:")
        
        # OPA输出
        if id_seq is not None:
            stats = self._get_tensor_stats(id_seq, "id_seq_after_opa")
            self.debug_logger.debug(f" ID sequence (after OPA): {stats}")
        
        if cloth_seq is not None:
            stats = self._get_tensor_stats(cloth_seq, "cloth_seq_after_opa")
            self.debug_logger.debug(f" Cloth sequence (after OPA): {stats}")
        
        # 正交性检查
        if id_seq is not None and cloth_seq is not None:
            import torch.nn.functional as F
            id_norm = F.normalize(id_seq, dim=-1)
            cloth_norm = F.normalize(cloth_seq, dim=-1)
            cosine_sim = (id_norm * cloth_norm).sum(dim=-1).mean().item()
            self.debug_logger.debug(f" Orthogonality (cosine sim): {cosine_sim:.6f} (should be close to 0)")
        
        # 显著性分数
        if saliency_score is not None:
            stats = self._get_tensor_stats(saliency_score, "saliency_score")
            self.debug_logger.debug(f" Saliency score: {stats}")
            # 统计高/中/低显著性的比例
            flat_score = saliency_score.flatten()
            high_salient = (flat_score > 0.7).sum().item() / flat_score.numel()
            low_salient = (flat_score < 0.3).sum().item() / flat_score.numel()
            self.debug_logger.debug(f" High salient: {high_salient:.2%}, Low salient: {low_salient:.2%}")
        
        # 过滤后的ID序列
        if id_filtered is not None:
            stats = self._get_tensor_stats(id_filtered, "id_seq_filtered")
            self.debug_logger.debug(f" ID sequence (after Mamba): {stats}")
    
    def log_gradients(self, model, step_name: str):
        """记录梯度信息"""
        self.debug_logger.debug(f"Gradients epoch_{step_name} state:")
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.debug_logger.debug(f"   {name}: grad_norm={grad_norm:.4f}")
    
    def log_gradient_flow(self, model):
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
            status = ""
            if i < 5 or i >= len(layers) - 5:
                status = ""
                if ave_grad < 1e-7:
                    status = " [WARNING: Vanishing gradient]"
                if max_grad > 100:
                    status = " [WARNING: Exploding gradient]"
                if status:
                    self.debug_logger.debug(f"   {name}: grad_avg={ave_grad:.6f}, max={max_grad:.6f}{status}")
        
        # 检查CLIP文本编码器的bias梯度
        clip_bias_grads = []
        for name, param in model.named_parameters():
            if 'text_encoder' in name and 'bias' in name and param.grad is not None:
                if param.grad.abs().max().item() < 1e-7:
                    clip_bias_grads.append(name)
        
        if clip_bias_grads:
            self.debug_logger.warning(f"CLIP text encoder bias with vanishing gradient: {clip_bias_grads[:3]}")
    
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
            "debug_logger.debug(f"{name} statistics: {stats}")
            "std": tensor_cpu.std().item(),
            "min": tensor_cpu.min().item(),
            "max": tensor_cpu.max().item(),
            "shape": list(tensor_cpu.shape),
            "requires_grad": tensor.requires_grad if hasattr(tensor, 'requires_grad') else False
        }
        return stats
    
    def log_attention_weights(self, attention_weights: torch.Tensor, layer_name: str):
        """记录注意力权重信息"""
        if attention_weights is not None:
            stats = self._get_tensor_stats(attention_weights, f"{layer_name}_weights")
            self.debug_logger.debug(f"{layer_name} weights: {stats}")
    
    def log_gate_weights(self, gate_weights: torch.Tensor, module_name: str):
        """记录门控权重信息"""
        if gate_weights is not None:
            stats = self._get_tensor_stats(gate_weights, f"{module_name}_weights")
            self.debug_logger.debug(f"{module_name} weights: {stats}")
    
    def log_fusion_info(self, fused_features: torch.Tensor, gate_weights: torch.Tensor = None):
        """记录融合模块信息"""
        self.debug_logger.debug("Fusion module information:")
        self.debug_logger.debug(f"Fused features stats: {self._get_tensor_stats(fused_features, 'fused_features')}")
        if gate_weights is not None:
            self.debug_logger.debug(f"Fusion gate weights: {self._get_tensor_stats(gate_weights, 'fused_gate_weights')}")
    
    def log_disentangle_info(self, id_features: torch.Tensor, cloth_features: torch.Tensor,
                           gate: torch.Tensor = None):
        """记录解耦模块信息"""
        self.debug_logger.debug("Disentangle module information:")
        self.debug_logger.debug(f"ID features stats: {self._get_tensor_stats(id_features, 'id_features')}")
        self.debug_logger.debug(f"Cloth features stats: {self._get_tensor_stats(cloth_features, 'clothing_features')}")
        if gate is not None:
            self.debug_logger.debug(f"Disentangle gate weights: {self._get_tensor_stats(gate, 'gate_weights')}")
    
    def log_gs3_module_info(self, id_seq: torch.Tensor, cloth_seq: torch.Tensor, 
                           saliency_score: torch.Tensor = None, 
                           id_filtered: torch.Tensor = None):
        """记录G-S3模块详细信息"""
        self.debug_logger.debug("=" * 50)
        self.debug_logger.debug("G-S3 Module Debug Info:")
        if id_seq is not None:
            self.debug_logger.debug(f"ID sequence stats: {self._get_tensor_stats(id_seq, 'id_seq')}")
        if cloth_seq is not None:
            self.debug_logger.debug(f"Cloth sequence stats: {self._get_tensor_stats(cloth_seq, 'cloth_seq')}")
        if saliency_score is not None:
            self.debug_logger.debug(f"Saliency score stats: {self._get_tensor_stats(saliency_score, 'saliency_score')}")
        if id_filtered is not None:
            self.debug_logger.debug(f"Filtered ID sequence stats: {self._get_tensor_stats(id_filtered, 'id_seq_filtered')}")
        if id_seq is not None and cloth_seq is not None:
            id_norm = F.normalize(id_seq, dim=-1)
            cloth_norm = F.normalize(cloth_seq, dim=-1)
            cosine_sim = (id_norm * cloth_norm).sum(dim=-1).mean().item()
            self.debug_logger.debug(f"Orthogonality check: {cosine_sim:.6f}")
        
        # 记录G-S3模块参数
        if hasattr(self, 'disentangle'):
            self.debug_logger.debug("G-S3 module: {self.disentangle}")
        
        # 注意：G-S3模块现在返回gate_stats字典而不是单个gate张量
        if hasattr(self, 'gs3_debug_info'):
            self.debug_logger.debug(f"G-S3 debug info: {self.gs3_debug_info}")
    
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
    
    @staticmethod
    def _get_tensor_stats(tensor: torch.Tensor, name: str = "") -> Dict[str, float]:
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

# 示例使用方法
if __name__ == "__main__":
    # 创建监控器示例
    monitor = TrainingMonitor(dataset_name="cuhk_pedes")
    
    # 记录一些示例信息
    monitor.logger.info("This is a test log entry")
    monitor.debug_logger.debug("This is a test debug entry")
    print(f"Log directory created at: {monitor.log_dir}")
    print("Monitor initialization completed!")
