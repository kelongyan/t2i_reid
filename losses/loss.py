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
    === FSHD损失函数模块 (重构版 - Phase 2) ===

    核心改进：
    1. 简化损失体系：11个损失 → 5个核心损失
    2. 修复关键bug：anti_collapse的target_norm自适应检测
    3. 优化权重配置：提升CLS和关键辅助损失的权重
    4. 移除失效/冲突的损失：freq_consistency、freq_separation、gate_adaptive、semantic_alignment

    保留的5个核心损失：
    - InfoNCE: 主对比学习损失（图文匹配）
    - Orthogonal: ID-Attr正交约束（解耦核心）
    - AntiCollapse: 防止特征坍缩（范数+方差约束）
    - IdTriplet: 同ID一致性（服装变化下的身份不变性）
    - ClothSemantic: 服装语义对齐（辅助任务）

    删除的6个损失：
    - freq_consistency: 与orthogonal功能重叠
    - freq_separation: 与orthogonal功能重叠
    - reconstruction: 简化为仅MSE，作为可选监督
    - gate_adaptive: 过于复杂，效果有限
    - semantic_alignment: 与cloth_semantic重叠
    - cls: 可选，建议降低权重或删除
    """

    def __init__(self, temperature=0.1, weights=None, num_classes=None, logger=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.logger = logger

        # 🔥 增强Label Smoothing，降低分类损失的初始值和敏感度
        # 0.1 → 0.2: 更强的正则化，避免过拟合
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.2)

        # === 核心损失模块（简化版）===
        self._reconstruction_loss = SymmetricReconstructionLoss()
        self._orthogonal_loss_module = EnhancedOrthogonalLoss()

        # === 初始化LossLogger ===
        self.loss_logger = LossLogger(logger.debug_logger) if logger else None

        # === 🔥 紧急修复版权重配置 ===
        # 策略：降低CLS和cloth_semantic主导，增强对比学习和解耦
        self.weights = weights if weights is not None else {
            # === 核心任务损失 ===
            'info_nce': 1.0,              # 对比学习 - 主任务
            'cls': 0.15,                  # 🔥 大幅降低 (0.3 → 0.15)，避免过拟合
            'cloth_semantic': 0.2,        # 🔥 大幅降低 (0.5 → 0.2)，减少与orthogonal冲突

            # === 解耦与约束损失 ===
            'orthogonal': 0.3,            # 🔥 提升 (0.15 → 0.3)，强化解耦
            'id_triplet': 0.8,            # ID一致性（保持）
            'anti_collapse': 1.5,         # 🔥 提升 (1.0 → 1.5)，修复后激活

            # === 辅助监督损失 ===
            'reconstruction': 0.2,        # 🔥 降低 (0.3 → 0.2)

            # === 已删除的损失（保持权重为0，兼容性）===
            'gate_adaptive': 0.0,         # 已删除（过于复杂）
            'semantic_alignment': 0.0,    # 已删除（与cloth_semantic重叠）
            'freq_consistency': 0.0,      # 已删除（与orthogonal重叠）
            'freq_separation': 0.0,       # 已删除（与orthogonal重叠）
        }

        # 动态权重调整参数
        self.current_epoch = 0
        self.enable_dynamic_weights = True

        # 语义引导模块（外部传入，可选）
        self.semantic_guidance_module = None

        # 注册dummy参数用于获取设备
        self.register_buffer('_dummy', torch.zeros(1))

        # 调试计数器
        if logger:
            self.debug_logger = logger.debug_logger
            self._log_counter_ortho = 0
            self._log_counter_triplet = 0
            self._log_counter_anti_collapse = 0
            self._log_counter_info_nce = 0
            self._batch_counter = 0

    def set_semantic_guidance(self, semantic_guidance_module):
        """
        设置语义引导模块（已废弃，保持接口兼容）
        """
        self.semantic_guidance_module = semantic_guidance_module
        if self.logger:
            self.debug_logger.info("⚠️  Semantic guidance module attached (DEPRECATED - will not be used)")

    def _get_device(self):
        """安全获取设备"""
        return self._dummy.device

    def update_epoch(self, epoch):
        """
        === 🔥 紧急修复版：激进的权重调整策略 ===

        策略：
        - Stage 1 (Epoch 1-10): 快速建立基础特征，降低CLS主导
        - Stage 2 (Epoch 11-30): 强化解耦，逐步激活cloth_semantic
        - Stage 3 (Epoch 31+): 对比学习主导，分类最小化
        """
        self.current_epoch = epoch

        if not self.enable_dynamic_weights:
            return

        # === 🔥 激进的三阶段权重调整 ===
        if epoch <= 10:
            # Stage 1: 基础学习 - 降低CLS，关闭cloth_semantic
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.2          # 🔥 降低 (原0.5)
            self.weights['cloth_semantic'] = 0.0  # 🔥 完全禁用，避免冲突
            self.weights['orthogonal'] = 0.3   # 🔥 提升，优先建立正交
            self.weights['anti_collapse'] = 1.5
            self.weights['reconstruction'] = 0.3
        elif epoch <= 30:
            # Stage 2: 精细解耦 - 逐步激活cloth_semantic
            self.weights['info_nce'] = 1.2     # 🔥 增强主任务
            self.weights['cls'] = 0.15         # 🔥 继续降低
            # 🔥 线性增长：epoch 11→0.05, epoch 20→0.15, epoch 30→0.2
            cloth_weight = 0.05 + (epoch - 10) * 0.0075
            self.weights['cloth_semantic'] = min(cloth_weight, 0.2)
            self.weights['orthogonal'] = 0.4   # 🔥 继续增强
            self.weights['anti_collapse'] = 2.0
            self.weights['reconstruction'] = 0.2
        else:
            # Stage 3: 对比学习主导
            self.weights['info_nce'] = 1.5     # 🔥 最大化
            self.weights['cls'] = 0.05         # 🔥 最小化
            self.weights['cloth_semantic'] = 0.15  # 🔥 保持低权重
            self.weights['orthogonal'] = 0.3
            self.weights['anti_collapse'] = 2.0
            self.weights['reconstruction'] = 0.15

        # 记录权重变化（仅关键epoch）
        if self.logger and epoch in [1, 11, 31]:
            self.debug_logger.info(f"🔥 Loss weights updated at epoch {epoch}:")
            for k, v in self.weights.items():
                if v > 0:  # 只显示激活的损失
                    self.debug_logger.info(f"   - {k}: {v:.4f}")

    def anti_collapse_loss(self, cloth_embeds, target_norm=None, margin_ratio=0.8):
        """
        [修复版] 防坍缩正则：确保特征存在，防止零和博弈

        核心修复：
        1. 使用EMA追踪目标范数（避免自适应导致loss=0的BUG）
        2. 固定margin策略，确保损失始终有效
        3. 添加方差正则，防止维度坍缩

        Args:
            cloth_embeds: 衣服/ID特征 [B, D]
            target_norm: 目标范数（None表示使用EMA追踪）
            margin_ratio: margin比例（0.8表示容忍20%的范数下降）
        """
        if cloth_embeds is None:
            return torch.tensor(0.0, device=self._get_device())

        # 【关键修复】使用EMA追踪目标范数，避免自适应BUG
        current_mean_norm = torch.norm(cloth_embeds, p=2, dim=-1).mean().item()
        
        if target_norm is None:
            # 初始化或更新EMA
            if not hasattr(self, '_target_norm_ema'):
                # 首次初始化：如果当前范数合理则使用，否则用默认值
                if current_mean_norm > 1.0:
                    self._target_norm_ema = current_mean_norm * 1.2  # 初始目标略高于当前
                else:
                    self._target_norm_ema = 8.0  # 默认目标
            else:
                # EMA更新：90%旧值 + 10%新值
                self._target_norm_ema = 0.9 * self._target_norm_ema + 0.1 * current_mean_norm
            
            # 目标范数：EMA的1.2倍（鼓励特征适度增长）
            target_norm = self._target_norm_ema * 1.2
        
        # 【关键修复】margin必须小于target_norm，确保损失有效
        # 使用target_norm的80%作为下界，低于此值将受到惩罚
        adaptive_margin = target_norm * margin_ratio

        # 计算L2范数
        norms = torch.norm(cloth_embeds, p=2, dim=-1)  # [B]
        # 惩罚模长过小的向量
        norm_loss = F.relu(adaptive_margin - norms).mean()

        # 【修复】方差正则：防止特征坍缩到少数维度
        # 计算每个维度的标准差
        feature_std = cloth_embeds.std(dim=0)  # [D]
        # 惩罚标准差过小的维度（说明该维度信息量低）
        std_threshold = 0.01  # 最小标准差阈值
        collapse_loss = F.relu(std_threshold - feature_std).mean()

        # 组合两种损失
        total_loss = norm_loss + 0.5 * collapse_loss

        # 调试信息
        if self.logger and self.loss_logger:
            if self.loss_logger.should_log('anti_collapse'):
                self.loss_logger.log_anti_collapse_stats(
                    cloth_embeds, target_norm, margin_ratio, total_loss
                )

        return total_loss

    def info_nce_loss(self, image_embeds, text_embeds, fused_embeds=None):
        """
        InfoNCE对比学习损失

        修复：支持使用fused_embeds参与对比学习
        让Fusion模块真正影响主任务，避免梯度死亡
        """
        if image_embeds is None or text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())

        # 优先使用fused_embeds（融合后的特征）
        # 如果没有fusion或fusion未激活，则使用image_embeds
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

        # 调试信息
        if self.logger and self.loss_logger:
            if self.loss_logger.should_log('info_nce'):
                self.loss_logger.log_info_nce_stats(
                    visual_embeds, text_embeds, total_loss, self.temperature
                )

        return total_loss

    def id_classification_loss(self, id_logits, pids):
        """
        身份分类损失

        === 修复方案 ===
        1. 移除温度缩放 - 让分类器正常学习
        2. 保留logits裁剪防止数值爆炸
        3. 通过动态权重控制学习速度，而非温度缩放
        """
        if id_logits is None or pids is None:
            return torch.tensor(0.0, device=self._get_device())

        # 裁剪防止数值爆炸
        id_logits_clipped = torch.clamp(id_logits, min=-50, max=50)

        # 直接计算CE损失，不使用温度缩放
        ce_loss = self.ce_loss(id_logits_clipped, pids)

        # 调试信息
        if self.logger and self.loss_logger:
            if self.loss_logger.should_log('cls'):
                self.loss_logger.log_cls_stats(
                    id_logits_clipped, pids, ce_loss
                )

        return ce_loss

    def cloth_semantic_loss(self, cloth_image_embeds, cloth_text_embeds):
        """
        === 修复方案：简化的cloth_semantic损失 ===
        移除去ID正则，让模型专注于服装语义对齐
        """
        if cloth_image_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())

        bsz = cloth_image_embeds.size(0)

        # 标准对比学习损失（与InfoNCE一致）
        cloth_image_norm = F.normalize(cloth_image_embeds, dim=-1, eps=1e-8)
        cloth_text_norm = F.normalize(cloth_text_embeds, dim=-1, eps=1e-8)

        sim = torch.matmul(cloth_image_norm, cloth_text_norm.t()) / self.temperature
        sim = torch.clamp(sim, min=-50, max=50)

        labels = torch.arange(bsz, device=sim.device, dtype=torch.long)
        loss_i2t = self.ce_loss(sim, labels)
        loss_t2i = self.ce_loss(sim.t(), labels)

        total_loss = (loss_i2t + loss_t2i) / 2

        # 调试信息
        if self.logger and self.loss_logger:
            if self.loss_logger.should_log('cloth_semantic'):
                self.loss_logger.log_cloth_semantic_stats(
                    cloth_image_norm, cloth_text_norm, total_loss, self.temperature
                )

        return total_loss

    def orthogonal_loss(self, id_embeds, cloth_embeds):
        """
        === 对称解耦改进：使用增强正交损失 ===
        启用交叉批次正交约束，让特征空间更彻底分离
        """
        if id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=self._get_device())

        # 使用增强版正交损失
        ortho_loss = self._orthogonal_loss_module(
            id_embeds, cloth_embeds, cross_batch=True
        )

        # 调试信息
        if self.logger and self.loss_logger:
            if self.loss_logger.should_log('orthogonal'):
                self.loss_logger.log_orthogonality_stats(
                    id_embeds, cloth_embeds, ortho_loss
                )

        return ortho_loss

    def triplet_loss(self, embeds, pids, margin=0.3):
        """
        ID 一致性损失：确保同一 ID 在不同衣服下的特征一致性
        """
        if embeds is None or pids is None:
            return torch.tensor(0.0, device=self._get_device())

        n = embeds.size(0)
        # 计算欧氏距离矩阵
        dist = torch.pow(embeds, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(embeds, embeds.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        # Hard Mining Mask
        mask = pids.expand(n, n).eq(pids.expand(n, n).t())

        # dist_ap: 每个anchor对应的最远正样本距离
        dist_ap, _ = torch.max(dist * mask.float(), dim=1)
        # dist_an: 每个anchor对应的最近负样本距离 (mask为0的地方加个大数1e6)
        dist_an, _ = torch.min(dist * (1. - mask.float()) + 1e6 * mask.float(), dim=1)

        loss = F.relu(dist_ap - dist_an + margin).mean()

        # 调试信息
        if self.logger and self.loss_logger:
            if self.loss_logger.should_log('id_triplet'):
                self.loss_logger.log_triplet_stats(
                    embeds, pids, loss, margin
                )

        return loss

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids,
                is_matched=None, epoch=None, gate=None,
                id_seq_features=None, cloth_seq_features=None, saliency_score=None,
                id_cls_features=None, original_feat=None, freq_info=None):
        """
        前向传播：计算所有损失（Phase 2重构版）

        新增参数：
            original_feat: 解耦前的原始特征，用于重构监督（可选）
            freq_info: 频域信息字典（已废弃，保持接口兼容）
        """
        losses = {}

        # === P1: 动态权重更新 ===
        if epoch is not None:
            self.update_epoch(epoch)

        # === 核心损失计算（简化版 - 5个核心损失）===
        # 1. InfoNCE损失（主任务）
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds, fused_embeds) \
            if image_embeds is not None and id_text_embeds is not None \
            else torch.tensor(0.0, device=self._get_device())

        # 2. 分类损失（可选，建议降低权重）
        losses['cls'] = self.id_classification_loss(id_logits, pids) \
            if id_logits is not None and pids is not None \
            else torch.tensor(0.0, device=self._get_device())

        # 3. 服装语义损失（辅助任务）
        losses['cloth_semantic'] = self.cloth_semantic_loss(
            cloth_image_embeds, cloth_text_embeds
        )

        # 4. 正交约束损失（解耦核心）
        losses['orthogonal'] = self.orthogonal_loss(id_embeds, cloth_embeds)

        # 5. ID 一致性 Triplet
        losses['id_triplet'] = self.triplet_loss(id_embeds, pids)

        # 6. 防坍缩正则（修复版 - 自动检测target_norm）
        # 对ID和Attr特征都应用防坍缩约束
        id_collapse_loss = self.anti_collapse_loss(id_embeds) if id_embeds is not None \
            else torch.tensor(0.0, device=self._get_device())
        cloth_collapse_loss = self.anti_collapse_loss(cloth_embeds) if cloth_embeds is not None \
            else torch.tensor(0.0, device=self._get_device())

        losses['anti_collapse'] = (id_collapse_loss + cloth_collapse_loss) / 2

        # 7. 重构损失（可选 - 简化为MSE）
        if original_feat is not None and id_embeds is not None and cloth_embeds is not None:
            losses['reconstruction'] = self._reconstruction_loss(
                id_embeds, cloth_embeds, original_feat
            )

            # 调试信息
            if self.logger and self.loss_logger and self.loss_logger.should_log('reconstruction'):
                self.loss_logger.log_reconstruction_stats(
                    id_embeds, cloth_embeds, original_feat, losses['reconstruction']
                )
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

        # === 已删除的损失（保持接口兼容，返回0）===
        # gate_adaptive - 已删除
        losses['gate_adaptive'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

        # semantic_alignment - 已删除
        losses['semantic_alignment'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

        # freq_consistency - 已删除
        losses['freq_consistency'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

        # freq_separation - 已删除
        losses['freq_separation'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

        # === NaN/Inf检查 ===
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    if self.logger:
                        self.debug_logger.warning(f"⚠️  WARNING: Loss '{key}' contains NaN/Inf! Resetting to 0.0.")
                    losses[key] = torch.tensor(0.0, device=value.device, requires_grad=True)

        # === 加权求和 ===
        total_loss = sum(self.weights.get(k, 0) * losses[k]
                        for k in losses.keys() if k != 'total')

        # 最终检查
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)

        losses['total'] = total_loss

        # 记录加权损失摘要（每100个batch）
        if self.logger and self.loss_logger:
            self._batch_counter += 1
            if self._batch_counter % 100 == 0:
                self.loss_logger.log_weighted_loss_summary(losses, self.weights)

        return losses
