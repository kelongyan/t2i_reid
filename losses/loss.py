import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.loss_logger import LossLogger


class SymmetricReconstructionLoss(nn.Module):
    """
    简化的重构损失 - 仅关注信息完整性

    核心思想：F_input ≈ F_id + F_attr
    确保解耦后的两个特征能够重建原始输入，防止信息丢失

    重构说明：
    - 移除复杂的Cosine、Diversity、Energy损失
    - 仅保留MSE损失，让模型自然学习重构
    - 与orthogonal_loss配合，避免特征重叠
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, id_feat, attr_feat, original_feat):
        """
        Args:
            id_feat: ID特征 [B, dim]
            attr_feat: Attr特征 [B, dim]
            original_feat: 原始特征（解耦前的全局特征）[B, dim]

        Returns:
            loss: 重构损失（MSE）
        """
        # 简单加法重建
        reconstructed = id_feat + attr_feat  # [B, dim]

        # 仅使用MSE损失
        mse_loss = self.mse_loss(reconstructed, original_feat)

        return mse_loss


class EnhancedOrthogonalLoss(nn.Module):
    """
    增强正交损失 (Enhanced Orthogonal Loss)

    改进：增加交叉批次约束，让不同样本的ID和Attr特征空间也趋向正交
    """
    def __init__(self):
        super().__init__()

    def forward(self, id_embeds, attr_embeds, cross_batch=True):
        """
        Args:
            id_embeds: ID特征 [B, dim]
            attr_embeds: Attr特征 [B, dim]
            cross_batch: 是否启用交叉批次正交约束

        Returns:
            loss: 正交损失
        """
        # 归一化
        id_norm = F.normalize(id_embeds, dim=-1, eps=1e-8)     # [B, dim]
        attr_norm = F.normalize(attr_embeds, dim=-1, eps=1e-8) # [B, dim]

        # === 批次内正交约束（样本自己的ID和Attr正交）===
        # 余弦相似度：应该接近0
        intra_cosine = (id_norm * attr_norm).sum(dim=-1)  # [B]
        intra_cosine = torch.clamp(intra_cosine, min=-1.0, max=1.0)
        intra_loss = intra_cosine.pow(2).mean()

        # === 交叉批次正交约束（所有ID vs 所有Attr）===
        if cross_batch and id_embeds.size(0) > 1:
            # 计算全局相似度矩阵 [B, B]
            cross_sim = torch.matmul(id_norm, attr_norm.t())
            # 最小化所有元素的平方和（让整个矩阵趋向0）
            cross_loss = cross_sim.pow(2).mean()

            return intra_loss + 0.5 * cross_loss
        else:
            return intra_loss


class Loss(nn.Module):
    """
    === FSHD损失函数模块 (重构版 - Phase 3: 做减法) ===

    核心改进 (基于日志诊断)：
    1. 彻底移除重构损失 (Reconstruction Loss)：消除"线性重构"与"语义流形"的数学冲突。
    2. 降级正交约束 (Orthogonal Loss)：权重降至 0.05，避免破坏特征的内在语义联系。
    3. 缩放分类损失 (CLS Scaling)：Logits / 20.0，解决分类损失数值过大(8.0+)主导梯度的问题。
    4. 激活服装语义 (Cloth Semantic)：权重提升至 0.5，强迫模型学习属性对齐。

    保留的5个核心损失：
    - InfoNCE (1.0): 主任务
    - IdTriplet (1.0): 身份一致性
    - ClothSemantic (0.5): 属性对齐
    - Orthogonal (0.05): 弱解耦约束
    - AntiCollapse (1.0): 基础正则
    - Cls (0.05): 弱分类辅助
    """

    def __init__(self, temperature=0.1, weights=None, num_classes=None, logger=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.logger = logger

        # Label Smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.2)

        # === 核心损失模块 ===
        # 移除 SymmetricReconstructionLoss
        self._orthogonal_loss_module = EnhancedOrthogonalLoss()

        # === 初始化LossLogger ===
        self.loss_logger = LossLogger(logger.debug_logger) if logger else None

        # === Phase 3 推荐权重配置 ===
        self.weights = weights if weights is not None else {
            # === 核心任务 ===
            'info_nce': 1.0,              # 主任务
            'id_triplet': 1.0,            # 身份一致性 (增强)
            'cloth_semantic': 0.5,        # 🔥 激活：属性对齐 (大幅提升)

            # === 约束与正则 ===
            'cls': 0.05,                  # 🔥 降低：配合Logit Scaling使用
            'orthogonal': 0.05,           # 🔥 降级：弱约束，避免破坏语义
        }

        # 动态权重调整参数
        self.current_epoch = 0
        self.enable_dynamic_weights = True

        # 注册dummy参数用于获取设备
        self.register_buffer('_dummy', torch.zeros(1))

        # 调试计数器
        if logger:
            self.debug_logger = logger.debug_logger
            self._batch_counter = 0

    def set_semantic_guidance(self, semantic_guidance_module):
        pass

    def _get_device(self):
        return self._dummy.device

    def update_epoch(self, epoch):
        """
        === Phase 3: 简化的两阶段策略 ===
        不再进行激进的权重波动，保持稳定的优化目标。
        """
        self.current_epoch = epoch

        if not self.enable_dynamic_weights:
            return

        # 动态策略仅微调，不再改变主次关系
        if epoch <= 5:
            # Warmup: 稍微降低 cloth_semantic，让 ID 特征先成型
            self.weights['cloth_semantic'] = 0.2
            self.weights['orthogonal'] = 0.0  # 前5个epoch完全关闭正交，先学特征
        else:
            # Full Regime
            self.weights['cloth_semantic'] = 0.5
            self.weights['orthogonal'] = 0.05 # 开启弱正交

        # 记录权重变化
        if self.logger and epoch in [1, 6]:
            self.debug_logger.info(f"🔥 Loss weights updated at epoch {epoch}:")
            for k, v in self.weights.items():
                if v > 0:
                    self.debug_logger.info(f"   - {k}: {v:.4f}")

    def info_nce_loss(self, image_embeds, text_embeds, fused_embeds=None):
        """InfoNCE对比学习损失"""
        if image_embeds is None or text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())

        visual_embeds = fused_embeds if fused_embeds is not None else image_embeds

        bsz = visual_embeds.size(0)
        visual_embeds = F.normalize(visual_embeds, dim=-1, eps=1e-8)
        text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-8)

        sim = torch.matmul(visual_embeds, text_embeds.t()) / self.temperature
        sim = torch.clamp(sim, min=-50, max=50)

        labels = torch.arange(bsz, device=sim.device, dtype=torch.long)
        loss_i2t = self.ce_loss(sim, labels)
        loss_t2i = self.ce_loss(sim.t(), labels)

        total_loss = (loss_i2t + loss_t2i) / 2

        if self.logger and self.loss_logger and self.loss_logger.should_log('info_nce'):
            self.loss_logger.log_info_nce_stats(visual_embeds, text_embeds, total_loss, self.temperature)

        return total_loss

    def id_classification_loss(self, id_logits, pids):
        """
        身份分类损失 (Fixed: Remove Logit Scaling)
        
        Logits / 20.0 导致 Softmax 分布平坦，CE Loss 维持在 ln(C) ~ 8.2。
        移除缩放，允许模型学习尖峰分布以降低损失。
        """
        if id_logits is None or pids is None:
            return torch.tensor(0.0, device=self._get_device())

        # 移除手动缩放，仅保留数值稳定性的裁剪
        scaled_logits = torch.clamp(id_logits, min=-50, max=50)

        ce_loss = self.ce_loss(scaled_logits, pids)

        if self.logger and self.loss_logger and self.loss_logger.should_log('cls'):
            self.loss_logger.log_cls_stats(scaled_logits, pids, ce_loss)

        return ce_loss

    def cloth_semantic_loss(self, cloth_image_embeds, cloth_text_embeds):
        """服装语义损失"""
        if cloth_image_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())

        bsz = cloth_image_embeds.size(0)
        cloth_image_norm = F.normalize(cloth_image_embeds, dim=-1, eps=1e-8)
        cloth_text_norm = F.normalize(cloth_text_embeds, dim=-1, eps=1e-8)

        sim = torch.matmul(cloth_image_norm, cloth_text_norm.t()) / self.temperature
        sim = torch.clamp(sim, min=-50, max=50)

        labels = torch.arange(bsz, device=sim.device, dtype=torch.long)
        loss_i2t = self.ce_loss(sim, labels)
        loss_t2i = self.ce_loss(sim.t(), labels)

        total_loss = (loss_i2t + loss_t2i) / 2

        if self.logger and self.loss_logger and self.loss_logger.should_log('cloth_semantic'):
            self.loss_logger.log_cloth_semantic_stats(cloth_image_norm, cloth_text_norm, total_loss, self.temperature)

        return total_loss

    def orthogonal_loss(self, id_embeds, cloth_embeds):
        """正交约束损失"""
        if id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=self._get_device())

        ortho_loss = self._orthogonal_loss_module(id_embeds, cloth_embeds, cross_batch=True)

        if self.logger and self.loss_logger and self.loss_logger.should_log('orthogonal'):
            self.loss_logger.log_orthogonality_stats(id_embeds, cloth_embeds, ortho_loss)

        return ortho_loss

    def triplet_loss(self, embeds, pids, margin=0.3):
        """ID 一致性 Triplet Loss"""
        if embeds is None or pids is None:
            return torch.tensor(0.0, device=self._get_device())

        n = embeds.size(0)
        dist = torch.pow(embeds, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(embeds, embeds.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = pids.expand(n, n).eq(pids.expand(n, n).t())
        dist_ap, _ = torch.max(dist * mask.float(), dim=1)
        dist_an, _ = torch.min(dist * (1. - mask.float()) + 1e6 * mask.float(), dim=1)

        loss = F.relu(dist_ap - dist_an + margin).mean()

        if self.logger and self.loss_logger and self.loss_logger.should_log('id_triplet'):
            self.loss_logger.log_triplet_stats(embeds, pids, loss, margin)

        return loss

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids,
                is_matched=None, epoch=None, gate=None,
                id_seq_features=None, cloth_seq_features=None, saliency_score=None,
                id_cls_features=None, original_feat=None, freq_info=None):
        """前向传播：计算所有损失 (Phase 3)"""
        losses = {}

        if epoch is not None:
            self.update_epoch(epoch)

        # 1. InfoNCE (主任务)
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds, fused_embeds)

        # 2. Classification (缩放后)
        losses['cls'] = self.id_classification_loss(id_logits, pids)

        # 3. Cloth Semantic (激活)
        losses['cloth_semantic'] = self.cloth_semantic_loss(cloth_image_embeds, cloth_text_embeds)

        # 4. Orthogonal (弱约束)
        losses['orthogonal'] = self.orthogonal_loss(id_embeds, cloth_embeds)

        # 5. Triplet (ID一致性)
        losses['id_triplet'] = self.triplet_loss(id_embeds, pids)

        # === 兼容性占位符 (已删除的损失返回0，但不加入total) ===
        losses['gate_adaptive'] = torch.tensor(0.0, device=self._get_device())
        losses['semantic_alignment'] = torch.tensor(0.0, device=self._get_device())
        losses['freq_consistency'] = torch.tensor(0.0, device=self._get_device())
        losses['freq_separation'] = torch.tensor(0.0, device=self._get_device())
        losses['anti_collapse'] = torch.tensor(0.0, device=self._get_device())
        losses['reconstruction'] = torch.tensor(0.0, device=self._get_device())

        # === NaN检测与求和 ===
        total_loss = torch.tensor(0.0, device=self._get_device())
        for key, value in losses.items():
            if key == 'total': continue
            if torch.isnan(value).any() or torch.isinf(value).any():
                if self.logger:
                    self.debug_logger.warning(f"⚠️  Loss '{key}' is NaN/Inf! Resetting to 0.")
                losses[key] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
            
            weight = self.weights.get(key, 0.0)
            if weight > 0:
                total_loss += weight * losses[key]

        losses['total'] = total_loss

        # 日志记录
        if self.logger and self.loss_logger:
            self._batch_counter += 1
            if self._batch_counter % 100 == 0:
                self.loss_logger.log_weighted_loss_summary(losses, self.weights)

        return losses
