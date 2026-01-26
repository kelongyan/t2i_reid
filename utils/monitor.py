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
        
        # === 新的目录结构 ===
        # log/dataset_name/ (日志文件)
        # log/dataset_name/model/ (模型文件)
        self.dataset_log_dir = self.log_dir / dataset_name
        self.model_dir = self.dataset_log_dir / "model"
        
        # 创建目录
        self.dataset_log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.log_file = self.dataset_log_dir / "log.txt"
        self.debug_log_file = self.dataset_log_dir / "debug.txt"
        self.metrics_file = self.dataset_log_dir / "metrics.json"
        
        # 1. 设置主 Logger (Console + File)
        self.logger = logging.getLogger(f"train.{dataset_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._setup_handler(self.logger, self.log_file, level=logging.INFO, console=True)

        # [New] 设置仅文件 Logger (用于后台记录 batch 信息)
        self.file_logger = logging.getLogger(f"train.{dataset_name}.file_only")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.propagate = False
        self._setup_handler(self.file_logger, self.log_file, level=logging.INFO, console=False)
        
        # 2. 设置调试 Logger (File Only)
        self.debug_logger = logging.getLogger(f"train.{dataset_name}.debug")
        self.debug_logger.setLevel(logging.DEBUG)
        self.debug_logger.propagate = False
        self._setup_handler(self.debug_logger, self.debug_log_file, level=logging.DEBUG, console=False)
        
        self.metrics_history = []

    def _setup_handler(self, logger, log_path, level, console=False):
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # File Handler: 完整格式（带时间戳和级别）
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File Handler
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(file_formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
        
        # Optional Console Handler: 简洁格式（仅消息内容）
        if console:
            console_formatter = logging.Formatter('%(message)s')  # 只显示消息内容
            ch = logging.StreamHandler()
            ch.setFormatter(console_formatter)
            ch.setLevel(level)
            logger.addHandler(ch)

    # --- 1. 特征统计 (透明化数据流) ---
    
    def log_feature_statistics(self, features: torch.Tensor, name: str):
        """记录特征统计信息到debug.txt (仅文件，不显示终端)"""
        if features is None: 
            self.debug_logger.debug(f"[{name}] Feature is None, skipped")
            return
        
        t = features.detach().cpu().float()
        
        # 计算详细统计信息
        stats_str = (
            f"[{name}] shape={list(t.shape)} | "
            f"μ={t.mean().item():.6f} σ={t.std().item():.6f} | "
            f"min={t.min().item():.6f} max={t.max().item():.6f} | "
            f"norm={t.norm().item():.6f}"
        )
        
        # 检测异常值
        if torch.isnan(t).any():
            nan_count = torch.isnan(t).sum().item()
            self.debug_logger.warning(f"⚠️  NAN DETECTED in {name}: {nan_count} values | {stats_str}")
        elif torch.isinf(t).any():
            inf_count = torch.isinf(t).sum().item()
            self.debug_logger.warning(f"⚠️  INF DETECTED in {name}: {inf_count} values | {stats_str}")
        else:
            self.debug_logger.debug(stats_str)

    # --- 2. 梯度健康度 (透明化训练稳定性) ---

    def log_gradients(self, model, step_name: str):
        """记录梯度摘要和异常到debug.txt (仅文件)"""
        grads = []
        names = []
        nan_params = []
        zero_grad_params = []
        
        for n, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad
                if torch.isnan(g).any():
                    nan_params.append(n)
                g_norm = g.norm().item()
                grads.append(g_norm)
                names.append(n)
                if g_norm < 1e-7:
                    zero_grad_params.append(n)
        
        if not grads: 
            self.debug_logger.debug(f"[{step_name}] No gradients found")
            return
        
        grads = np.array(grads)
        
        # 详细摘要记录
        self.debug_logger.debug(
            f"Grad Summary [{step_name}]: Count={len(grads)} | "
            f"Mean={grads.mean():.8f} Std={grads.std():.8f} | "
            f"Max={grads.max():.6f} Min={grads.min():.10f} | "
            f"Median={np.median(grads):.8f}"
        )
        
        # NaN检测
        if nan_params:
            self.debug_logger.error(f"❌ NaN Gradients in {len(nan_params)} params: {nan_params[:5]}")
        
        # 异常检测 - 梯度爆炸
        exploding = [(n, g) for n, g in zip(names, grads) if g > 5.0]
        if exploding:
            self.debug_logger.warning(
                f"⚠️  EXPLODING Gradients detected in {len(exploding)}/{len(grads)} layers!"
            )
            for n, g in sorted(exploding, key=lambda x: x[1], reverse=True)[:5]:
                self.debug_logger.warning(f"   - {n}: norm={g:.6f}")
        
        # 异常检测 - 梯度消失
        if zero_grad_params:
            self.debug_logger.warning(
                f"⚠️  VANISHING Gradients in {len(zero_grad_params)} params (norm<1e-7)"
            )
            if len(zero_grad_params) <= 10:
                for n in zero_grad_params:
                    self.debug_logger.warning(f"   - {n}")

        # Top 活跃层
        if len(grads) >= 5:
            top_idx = grads.argsort()[::-1][:5]
            self.debug_logger.debug("🔥 Top 5 Active Layers:")
            for i in top_idx:
                self.debug_logger.debug(f"   - {names[i]}: norm={grads[i]:.6f}")

    def log_gradient_flow(self, model):
        """保持接口兼容，逻辑已并入 log_gradients"""
        pass

    # --- 3. 损失与批次 (透明化进度) ---

    def log_batch_info(self, epoch: int, batch_idx: int, total_batches: int,
                       loss_meters: Dict[str, float], lr: float, print_to_console=True):
        """记录每一批次的状态到log.txt (显示终端) 和 debug.txt (仅文件，详细版)"""
        # 简要版本
        loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in loss_meters.items() if 'total' not in k])
        msg = (
            f"E{epoch} [{batch_idx}/{total_batches}] LR:{lr:.2e} | "
            f"Total:{loss_meters.get('total', 0):.4f} | {loss_str}"
        )
        
        if print_to_console:
            self.logger.info(msg)
        else:
            self.file_logger.info(msg)
        
        # 详细版本 - 仅写入debug.txt
        self.debug_logger.debug(
            f"Batch Detail - Epoch:{epoch} Batch:{batch_idx}/{total_batches} | LR:{lr:.2e}"
        )
        for k, v in loss_meters.items():
            self.debug_logger.debug(f"  └─ {k}: {v:.6f}")

    def log_loss_breakdown(self, loss_dict: Dict[str, torch.Tensor], epoch: int, batch_idx: int):
        """记录损失占比到debug.txt (仅文件)"""
        total = loss_dict['total'].item() if isinstance(loss_dict['total'], torch.Tensor) else loss_dict['total']
        if total == 0: 
            self.debug_logger.debug(f"Loss Breakdown E{epoch}B{batch_idx}: Total=0, skipped")
            return
        
        parts = []
        for k, v in loss_dict.items():
            if k == 'total': continue
            val = v.item() if isinstance(v, torch.Tensor) else v
            ratio = (val / total * 100) if total > 0 else 0
            parts.append((ratio, k, val))
        
        parts.sort(key=lambda x: -x[0])
        
        # 详细记录每个损失项
        self.debug_logger.debug(f"Loss Breakdown - Epoch:{epoch} Batch:{batch_idx} Total={total:.6f}")
        for ratio, k, val in parts:
            self.debug_logger.debug(f"  └─ {k}: {val:.6f} ({ratio:.2f}%)")

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
        """监控 G-S3/FSHD 解耦质量到debug.txt (仅文件)"""
        self.debug_logger.debug("=== Disentangle Module Internal State ===")
        self.log_feature_statistics(id_feat, "ID_Feature")
        self.log_feature_statistics(cloth_feat, "Cloth_Feature")
        
        # 检查正交性
        if id_feat is not None and cloth_feat is not None:
            id_norm = F.normalize(id_feat, dim=-1, eps=1e-8)
            cloth_norm = F.normalize(cloth_feat, dim=-1, eps=1e-8)
            
            cos_sim = (id_norm * cloth_norm).sum(dim=-1)
            abs_cos_sim = cos_sim.abs()
            
            self.debug_logger.debug(
                f"[Orthogonality] Cosine Similarity: "
                f"mean={cos_sim.mean().item():.6f} std={cos_sim.std().item():.6f} | "
                f"abs_mean={abs_cos_sim.mean().item():.6f} (target: 0.0)"
            )
            
            # 检查是否有严重的非正交情况
            high_sim_count = (abs_cos_sim > 0.5).sum().item()
            if high_sim_count > 0:
                self.debug_logger.warning(
                    f"⚠️  {high_sim_count}/{id_feat.size(0)} samples have high correlation (>0.5)"
                )
            
        # 记录门控统计
        if isinstance(gate_stats, dict):
            self.debug_logger.debug("[Gate Statistics]")
            for k, v in gate_stats.items():
                if isinstance(v, (int, float)):
                    self.debug_logger.debug(f"  └─ {k}: {v:.6f}")
                else:
                    self.debug_logger.debug(f"  └─ {k}: {v}")

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
        """记录GPU内存使用情况到debug.txt (仅文件)"""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            max_alloc = torch.cuda.max_memory_allocated() / 1024**2
            free = reserved - alloc
            
            self.debug_logger.debug(
                f"[GPU Memory] Allocated:{alloc:.1f}MB | Reserved:{reserved:.1f}MB | "
                f"Free:{free:.1f}MB | Peak:{max_alloc:.1f}MB"
            )

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

    def log_conflict_score(self, conflict_score, step_name=""):
        """
        🔥 方案书 Phase 3: Conflict Score 日志追踪

        核心指标：衡量 ID 和 Attr 注意力图的空间重叠程度
        - conflict_score 高 → 解耦失败 → 图像特征不可信
        - conflict_score 低 → 解耦成功 → 图像特征可信

        Args:
            conflict_score: [B] 冲突分数
            step_name: 步骤名称 (用于日志区分)
        """
        if conflict_score is None:
            return

        # 转为 numpy 便于统计
        if isinstance(conflict_score, torch.Tensor):
            scores = conflict_score.detach().cpu().numpy()
        else:
            scores = conflict_score

        # 统计信息
        mean_score = scores.mean()
        std_score = scores.std()
        min_score = scores.min()
        max_score = scores.max()

        # 分档统计
        low_conflict = (scores < 0.01).sum()   # < 1% 重叠 → 优秀
        mid_conflict = (scores >= 0.01) & (scores < 0.05)  # 1-5% → 良好
        high_conflict = (scores >= 0.05) & (scores < 0.1)  # 5-10% → 一般
        severe_conflict = (scores >= 0.1)  # > 10% → 差

        # 记录到 debug.txt
        self.debug_logger.debug(
            f"[Conflict Score{step_name}] "
            f"mean={mean_score:.6f} std={std_score:.6f} | "
            f"min={min_score:.6f} max={max_score:.6f}"
        )
        self.debug_logger.debug(
            f"  Distribution: "
            f"Excellent(<1%)={low_conflict} Good(1-5%)={mid_conflict} "
            f"Fair(5-10%)={high_conflict} Poor(>10%)={severe_conflict}"
        )

        # 异常检测：如果平均冲突分数过高，发出警告
        if mean_score > 0.1:
            self.debug_logger.warning(
                f"⚠️  [Conflict Score{step_name}] Average conflict too high: {mean_score:.4f} "
                f"(Expected < 0.05). Decoupling quality is poor!"
            )
        elif mean_score < 0.02:
            self.debug_logger.info(
                f"✅ [Conflict Score{step_name}] Excellent decoupling quality: {mean_score:.4f}"
            )

def get_monitor_for_dataset(dataset_name: str, log_dir: str = "log") -> "TrainingMonitor":
    return TrainingMonitor(dataset_name=dataset_name, log_dir=log_dir)