# src/utils/monitor.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
import json
import logging
from pathlib import Path
import torch.nn.functional as F

class TrainingMonitor:
    """
    训练监控器：旨在使训练过程透明化
    功能：
    1. 记录特征统计信息（单行紧凑格式）
    2. 梯度健康度分析（摘要 + 异常检测）
    3. 关键模块（G-S3, Fusion）内部状态监控
    4. 自动记录指标到 JSON
    """
    
    def __init__(self, dataset_name: str, log_dir: str = "log"):
        self.dataset_name = dataset_name
        self.log_dir = Path(log_dir)
        self.dataset_log_dir = self.log_dir / dataset_name
        self.dataset_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.log_file = self.dataset_log_dir / "log.txt"
        self.debug_log_file = self.dataset_log_dir / "debug.txt"
        self.metrics_file = self.dataset_log_dir / "metrics.json"
        
        # 1. 设置主 Logger (Console + File)
        self.logger = logging.getLogger(f"train.{dataset_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._setup_handler(self.logger, self.log_file, level=logging.INFO, console=True)
        
        # 2. 设置调试 Logger (File Only)
        self.debug_logger = logging.getLogger(f"train.{dataset_name}.debug")
        self.debug_logger.setLevel(logging.DEBUG)
        self.debug_logger.propagate = False
        self._setup_handler(self.debug_logger, self.debug_log_file, level=logging.DEBUG, console=False)
        
        self.metrics_history = []

    def _setup_handler(self, logger, log_path, level, console=False):
        if logger.hasHandlers():
            logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File Handler
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
        
        # Optional Console Handler
        if console:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            ch.setLevel(level)
            logger.addHandler(ch)

    # --- 1. 特征统计 (透明化数据流) ---
    
    def log_feature_statistics(self, features: torch.Tensor, name: str):
        """记录特征统计信息 (Compact format)"""
        if features is None: return
        t = features.detach().cpu().float()
        
        stats_str = (
            f"[{name}] shape={list(t.shape)} | μ={t.mean():.4f} σ={t.std():.4f} | "
            f"min={t.min():.4f} max={t.max():.4f}"
        )
        
        if torch.isnan(t).any() or torch.isinf(t).any():
            self.debug_logger.warning(f"⚠️  NAN/INF DETECTED: {stats_str}")
        else:
            self.debug_logger.debug(stats_str)

    # --- 2. 梯度健康度 (透明化训练稳定性) ---

    def log_gradients(self, model, step_name: str):
        """记录梯度摘要和异常"""
        grads = []
        names = []
        for n, p in model.named_parameters():
            if p.grad is not None:
                g_norm = p.grad.norm().item()
                grads.append(g_norm)
                names.append(n)
        
        if not grads: return
        grads = np.array(grads)
        
        # 摘要记录
        self.debug_logger.debug(
            f"Grad Summary [{step_name}]: Count={len(grads)} | Mean={grads.mean():.6f} | "
            f"Max={grads.max():.4f} | Min={grads.min():.8f}"
        )
        
        # 异常检测
        exploding = [(n, g) for n, g in zip(names, grads) if g > 5.0]
        if exploding:
            self.debug_logger.warning(f"⚠️  EXPLODING Gradients detected in {len(exploding)} layers!")
            for n, g in sorted(exploding, key=lambda x: x[1], reverse=True)[:3]:
                self.debug_logger.warning(f"   - {n}: {g:.4f}")

        # Top 活跃层 (确认哪些层在学)
        top_idx = grads.argsort()[::-1][:3]
        top_str = " | ".join([f"{names[i]}={grads[i]:.4f}" for i in top_idx])
        self.debug_logger.debug(f"🔥 Top Active Layers: {top_str}")

    def log_gradient_flow(self, model):
        """保持接口兼容，逻辑已并入 log_gradients"""
        pass

    # --- 3. 损失与批次 (透明化进度) ---

    def log_batch_info(self, epoch: int, batch_idx: int, total_batches: int,
                       loss_meters: Dict[str, float], lr: float):
        """记录每一批次的简要状态"""
        loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in loss_meters.items() if 'total' not in k])
        self.debug_logger.info(
            f"Epoch {epoch} [{batch_idx}/{total_batches}] | LR: {lr:.2e} | "
            f"Total: {loss_meters.get('total', 0):.4f} | {loss_str}"
        )

    def log_loss_breakdown(self, loss_dict: Dict[str, torch.Tensor], epoch: int, batch_idx: int):
        """记录损失占比"""
        total = loss_dict['total'].item() if isinstance(loss_dict['total'], torch.Tensor) else loss_dict['total']
        if total == 0: return
        
        parts = []
        for k, v in loss_dict.items():
            if k == 'total': continue
            val = v.item() if isinstance(v, torch.Tensor) else v
            parts.append((val / total * 100, k, val))
        
        parts.sort(key=lambda x: -x[0])
        msg = f"Loss Breakdown E{epoch}B{batch_idx}: Total={total:.4f} | "
        msg += " | ".join([f"{k}:{v:.3f}({p:.1f}%)" for p, k, v in parts[:5]])
        self.debug_logger.debug(msg)

    def log_epoch_info(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """保存指标到历史记录"""
        entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(entry)
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    # --- 4. 模块特定状态 (透明化模型内部) ---

    def log_gs3_module_info(self, id_feat, cloth_feat, gate_stats=None):
        """监控 G-S3 解耦质量"""
        self.debug_logger.debug("--- G-S3 Internal State ---")
        self.log_feature_statistics(id_feat, "GS3_ID_Final")
        self.log_feature_statistics(cloth_feat, "GS3_Cloth_Final")
        
        # 检查正交性
        if id_feat is not None and cloth_feat is not None:
            cos_sim = F.cosine_similarity(id_feat, cloth_feat, dim=-1).abs().mean().item()
            self.debug_logger.debug(f"[Decouple] Absolute Cosine Similarity: {cos_sim:.6f} (target: 0.0)")
            
        if isinstance(gate_stats, dict):
            g_str = " | ".join([f"{k}={v:.4f}" for k, v in gate_stats.items()])
            self.debug_logger.debug(f"[Gate] {g_str}")

    def log_gate_weights(self, weights: torch.Tensor, name: str):
        """记录门控权重分布"""
        if weights is None: return
        w = weights.detach().cpu().numpy()
        self.debug_logger.debug(f"[{name}] distribution: mean={w.mean():.4f}, std={w.std():.4f}, min={w.min():.4f}, max={w.max():.4f}")

    def log_fusion_info(self, fused_feat, gate_weights=None):
        self.log_feature_statistics(fused_feat, "Fused_Embeds")
        if gate_weights is not None:
            self.log_gate_weights(gate_weights, "Fusion_Gate")

    # --- 5. 系统与辅助 ---

    def log_memory_usage(self):
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            max_alloc = torch.cuda.max_memory_allocated() / 1024**2
            self.debug_logger.debug(f"GPU Memory: {alloc:.0f}MB / Max: {max_alloc:.0f}MB")

    def log_optimizer_state(self, optimizer, epoch):
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.debug_logger.debug(f"Optimizer LRs [Epoch {epoch}]: {lrs}")

    def log_loss_components(self, loss_dict):
        """仅在 Debug 中记录原始 Loss"""
        info = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
        self.debug_logger.debug(f"Raw Loss: {info}")

    def log_data_batch_info(self, batch_data, batch_idx):
        self.debug_logger.debug(f"Batch {batch_idx} data shapes: { {k: list(v.shape) for k, v in batch_data.items() if hasattr(v, 'shape')} }")

    def log_attention_weights(self, weights, name):
        self.log_feature_statistics(weights, f"Attn_{name}")

    def log_disentangle_info(self, id_feat, cloth_feat, gate=None):
        self.log_gs3_module_info(id_feat, cloth_feat, gate_stats=gate if isinstance(gate, dict) else None)

def get_monitor_for_dataset(dataset_name: str, log_dir: str = "log") -> "TrainingMonitor":
    return TrainingMonitor(dataset_name=dataset_name, log_dir=log_dir)