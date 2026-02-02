# utils/monitor.py
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
    训练监控器：管理日志记录和性能指标保存
    日志结构：
    - log/{dataset}/log.txt      # 训练日志
    - log/{dataset}/res.txt      # 最佳性能指标（用于论文）
    - log/{dataset}/model/       # 模型检查点
    """
    
    def __init__(self, dataset_name: str, log_dir: str = "log"):
        self.dataset_name = dataset_name
        self.log_dir = Path(log_dir)
        
        # 目录结构：log/dataset_name/
        self.dataset_log_dir = self.log_dir / dataset_name
        self.model_dir = self.dataset_log_dir / "model"
        
        # 创建目录
        self.dataset_log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件路径
        self.log_file = self.dataset_log_dir / "log.txt"
        self.res_file = self.dataset_log_dir / "res.txt"
        
        # 设置主 Logger (Console + File)
        self.logger = logging.getLogger(f"train.{dataset_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self._setup_handler(self.logger, self.log_file, level=logging.INFO, console=True)
        
        # 仅文件 Logger (用于批量记录)
        self.file_logger = logging.getLogger(f"train.{dataset_name}.file_only")
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.propagate = False
        self._setup_handler(self.file_logger, self.log_file, level=logging.INFO, console=False)
        
        # 性能指标历史
        self.metrics_history = []
        self.best_metrics = {
            'epoch': 0,
            'mAP': 0.0,
            'rank1': 0.0,
            'rank5': 0.0,
            'rank10': 0.0
        }

    def _setup_handler(self, logger, log_path, level, console=False):
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # File Handler
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setFormatter(file_formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
        
        # Console Handler
        if console:
            console_formatter = logging.Formatter('%(message)s')
            ch = logging.StreamHandler()
            ch.setFormatter(console_formatter)
            ch.setLevel(level)
            logger.addHandler(ch)

    def log_batch_info(self, epoch: int, batch_idx: int, total_batches: int,
                       loss_meters: Dict[str, float], lr: float, print_to_console=True):
        """记录每一批次的状态"""
        loss_str = ', '.join([f"{k}: {v:.4f}" for k, v in loss_meters.items() if 'total' not in k])
        msg = (
            f"E{epoch} [{batch_idx}/{total_batches}] LR:{lr:.2e} | "
            f"Total:{loss_meters.get('total', 0):.4f} | {loss_str}"
        )
        
        if print_to_console:
            self.logger.info(msg)
        else:
            self.file_logger.info(msg)

    def log_epoch_summary(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """记录每个 epoch 的摘要信息"""
        msg = f"Epoch {epoch}/{total_epochs} Summary: "
        msg += f"mAP={metrics.get('mAP', 0):.4f} | "
        msg += f"Rank-1={metrics.get('rank1', 0):.4f} | "
        msg += f"Rank-5={metrics.get('rank5', 0):.4f} | "
        msg += f"Rank-10={metrics.get('rank10', 0):.4f}"
        self.logger.info(msg)

    def update_best_metrics(self, epoch: int, metrics: Dict[str, float]):
        """更新并保存最佳性能指标到 res.txt"""
        current_map = metrics.get('mAP', 0)
        
        if current_map > self.best_metrics['mAP']:
            self.best_metrics = {
                'epoch': epoch,
                'mAP': current_map,
                'rank1': metrics.get('rank1', 0),
                'rank5': metrics.get('rank5', 0),
                'rank10': metrics.get('rank10', 0)
            }
            self._save_res_file()
            return True
        return False

    def _save_res_file(self):
        """保存最佳性能指标到 res.txt"""
        res_content = [
            "=" * 70,
            "最佳性能指标",
            "=" * 70,
            f"数据集: {self.dataset_name}",
            f"Epoch: {self.best_metrics['epoch']}",
            f"mAP: {self.best_metrics['mAP']:.3f}",
            f"Rank-1: {self.best_metrics['rank1']:.3f}",
            f"Rank-5: {self.best_metrics['rank5']:.3f}",
            f"Rank-10: {self.best_metrics['rank10']:.3f}",
            "=" * 70,
        ]
        
        with open(self.res_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(res_content))

    def log_training_start(self, config: Dict):
        """记录训练开始信息"""
        self.logger.info("=" * 70)
        self.logger.info(f"开始训练 - 数据集: {self.dataset_name}")
        self.logger.info(f"配置: {json.dumps(config, indent=2)}")
        self.logger.info("=" * 70)

    def log_training_end(self, total_epochs: int):
        """记录训练结束信息"""
        self.logger.info("=" * 70)
        self.logger.info("训练结束")
        self.logger.info(f"总 Epoch: {total_epochs}")
        self.logger.info(f"最佳性能:")
        self.logger.info(f"  - Epoch: {self.best_metrics['epoch']}")
        self.logger.info(f"  - mAP: {self.best_metrics['mAP']:.4f}")
        self.logger.info(f"  - Rank-1: {self.best_metrics['rank1']:.4f}")
        self.logger.info(f"详细结果保存至: {self.res_file}")
        self.logger.info("=" * 70)

    def save_checkpoint(self, state: Dict, is_best: bool, filename: str = ""):
        """保存模型检查点"""
        if not filename:
            filename = f"checkpoint_{self.dataset_name}.pth"
        
        checkpoint_path = self.model_dir / filename
        torch.save(state, checkpoint_path)
        
        if is_best:
            best_path = self.model_dir / f"best_{self.dataset_name}.pth"
            torch.save(state, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")


def get_monitor_for_dataset(dataset_name: str, log_dir: str = "log") -> TrainingMonitor:
    return TrainingMonitor(dataset_name=dataset_name, log_dir=log_dir)
