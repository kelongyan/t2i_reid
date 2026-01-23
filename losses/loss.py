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


class FrequencyAlignmentLoss(nn.Module):
    """
    频域对齐损失 (Frequency Alignment Loss) - 方案B

    核心思想：
    1. ID特征应该与低频成分高度相关
    2. Attr特征应该与高频成分高度相关
    3. 避免频域混叠导致的身份信息泄漏

    设计理念：
    - 充分利用FSHD模块的频域分解能力
    - 强化频域-空域联合建模的有效性
    - 防止ID和Attr特征在频域上混叠
    - 与检索任务完全一致，使用L2归一化特征
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, id_feat, attr_feat, freq_info):
        """
        Args:
            id_feat: [B, dim] - ID特征
            attr_feat: [B, dim] - Attr特征
            freq_info: dict - 包含频域信息（可能为None）
                - low_freq_energy: [B] - 低频能量（可选）
                - high_freq_energy: [B] - 高频能量（可选）
                - energy_ratio: [B] - 高频能量比率（high_freq / total）
                - freq_magnitude: [B] - DCT系数幅度
                - freq_coeff: [B, D, H, W] - DCT系数张量（用于高级分析）

        Returns:
            loss: 频域对齐损失
        """
        B = id_feat.shape[0]
        device = id_feat.device
        
        # === 🔥 修复：如果freq_info为None，使用默认值 ===
        if freq_info is None:
            freq_info = {}  # 创建空字典，后续会使用默认值
        
        # 归一化特征（使用L2归一化，与检索任务一致）
        id_norm = F.normalize(id_feat, dim=-1, eps=1e-8)     # [B, dim]
        attr_norm = F.normalize(attr_feat, dim=-1, eps=1e-8) # [B, dim]
        
        # 提取频域信息（提供默认值以防缺失）
        energy_ratio = freq_info.get('energy_ratio', torch.ones(B, device=device) * 0.5)
        freq_magnitude = freq_info.get('freq_magnitude', torch.ones(B, device=device))
        freq_coeff = freq_info.get('freq_coeff', None)
        
        # ===== 损失1：特征能量一致性 =====
        # ID特征应该有更大的能量（身份信息更丰富）
        id_energy = torch.sum(id_norm.pow(2), dim=-1)  # [B]
        attr_energy = torch.sum(attr_norm.pow(2), dim=-1)  # [B]
        
        # ID能量应该大于Attr能量
        energy_gap_loss = F.relu(attr_energy - id_energy).mean() * 0.3
        
        # ===== 损失2：频域能量相关性 =====
        # ID特征应该主导低频，Attr特征应该主导高频
        # 使用energy_ratio作为指导：低ratio表示更多低频，高ratio表示更多高频
        
        # ID特征应该与低频主导的样本（energy_ratio低）有更高的能量
        id_energy_weighted = id_energy * (1.0 - energy_ratio)
        # Attr特征应该与高频主导的样本（energy_ratio高）有更高的能量
        attr_energy_weighted = attr_energy * energy_ratio
        
        # 最大化加权能量
        freq_energy_loss = (1.0 - id_energy_weighted.mean()).abs() * 0.5
        freq_energy_loss += (1.0 - attr_energy_weighted.mean()).abs() * 0.5
        
        # ===== 损失3：频域结构保持 =====
        # 如果提供了DCT系数，计算频域结构的相似性
        freq_structure_loss = torch.tensor(0.0, device=device)
        if freq_coeff is not None:
            # freq_coeff: [B, D, H, W]
            # 计算每个样本的频域结构（沿通道维度的方差）
            freq_structure = torch.var(freq_coeff, dim=1).mean(dim=(1, 2))  # [B]
            
            # ID特征应该与更平滑的频域结构相关（低频主导）
            # Attr特征应该与更复杂的频域结构相关（高频主导）
            
            # 使用频域幅度作为参考
            freq_structure_loss = (freq_magnitude - freq_structure).abs().mean() * 0.2
        
        # ===== 损失4：ID-Attr频域分离 =====
        # ID和Attr特征在频域上的投影应该正交
        # 计算ID和Attr特征与"虚拟"低频/高频向量的相似度
        
        # 虚拟低频向量：假设低频的特征（平滑、稳定）
        # 我们使用特征均值作为"全局低频"的代理
        id_mean = id_norm.mean(dim=0, keepdim=True)  # [1, dim]
        attr_mean = attr_norm.mean(dim=0, keepdim=True)  # [1, dim]
        
        # ID特征应该更接近全局ID特征（表示身份的一致性）
        id_consistency = (id_norm * id_mean).sum(dim=-1)  # [B]
        id_consistency_loss = (1.0 - id_consistency).mean() * 0.4
        
        # Attr特征应该更接近全局Attr特征（表示属性的一致性）
        attr_consistency = (attr_norm * attr_mean).sum(dim=-1)  # [B]
        attr_consistency_loss = (1.0 - attr_consistency).mean() * 0.4
        
        # ===== 损失5：梯度分离（防止混叠）=====
        # ID和Attr特征的梯度方向应该不同
        # 这里我们使用特征的空间分布差异来近似
        
        # ID特征：应该更加集中（身份明确）
        id_variance = id_norm.var(dim=1)  # [B]
        
        # Attr特征：可以更加分散（属性多样）
        attr_variance = attr_norm.var(dim=1)  # [B]
        
        # ID方差应该相对较小，Attr方差可以相对较大
        variance_gap_loss = F.relu(id_variance - attr_variance).mean() * 0.2
        
        # ===== 总损失 =====
        total_loss = (
            energy_gap_loss +           # 能量一致性
            freq_energy_loss +          # 频域能量相关性
            freq_structure_loss +       # 频域结构保持
            id_consistency_loss +       # ID一致性
            attr_consistency_loss +     # Attr一致性
            variance_gap_loss           # 梯度分离
        )
        
        # NaN检测
        if torch.isnan(total_loss).any():
            total_loss = torch.tensor(0.0, device=device)
        
        return total_loss


class Loss(nn.Module):
    """
    === FSHD损失函数模块 (方案B：频域对齐损失版) ===

    核心改进：
    1. 移除CLS损失：解决CLS损失无法下降的问题
    2. 新增频域对齐损失：充分利用FSHD架构的频域分解能力
    3. 强化身份一致性：提升Triplet损失权重
    4. 简化权重策略：移除动态权重调整，使用固定权重

    保留的5个核心损失：
    - InfoNCE (1.0): 主任务
    - IdTriplet (1.0): 身份一致性
    - ClothSemantic (0.5): 属性对齐
    - Orthogonal (0.05): 弱解耦约束
    - FrequencyAlignment (0.3): 频域对齐（新增）
    """

    def __init__(self, temperature=0.1, weights=None, num_classes=None, logger=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.logger = logger

        # Label Smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.2)

        # === 核心损失模块 ===
        self._orthogonal_loss_module = EnhancedOrthogonalLoss()
        self._frequency_alignment_module = FrequencyAlignmentLoss()

        # === 初始化LossLogger ===
        self.loss_logger = LossLogger(logger.debug_logger) if logger else None

        # === 方案B推荐权重配置（固定权重，无动态调整）===
        self.weights = weights if weights is not None else {
            # === 核心任务 ===
            'info_nce': 1.0,               # 主任务
            'id_triplet': 1.0,             # 身份一致性
            'cloth_semantic': 0.5,         # 属性对齐

            # === 约束与正则 ===
            'orthogonal': 0.05,            # 弱解耦约束
            'frequency_alignment': 0.3,     # 频域对齐（新增，替代CLS）
        }

        # 移除动态权重调整（方案B使用固定权重）
        self.current_epoch = 0
        self.enable_dynamic_weights = False

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
        方案B：移除动态权重调整策略

        原因：
        - 频域对齐损失与检索任务一致，无需特殊调整
        - 固定权重更稳定，便于调试和对比
        - 简化训练逻辑，减少超参搜索空间
        """
        self.current_epoch = epoch
        
        # 记录当前权重（用于监控）
        if self.logger and epoch % 10 == 1:
            self.debug_logger.info(f"📊 Fixed loss weights at epoch {epoch}:")
            for k, v in sorted(self.weights.items(), key=lambda x: -x[1]):
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

    def frequency_alignment_loss(self, id_feat, attr_feat, freq_info):
        """
        频域对齐损失（新增）

        Args:
            id_feat: [B, dim] - ID特征
            attr_feat: [B, dim] - Attr特征
            freq_info: dict - 频域信息（可能为None）

        Returns:
            loss: 频域对齐损失
        """
        if id_feat is None or attr_feat is None:
            return torch.tensor(0.0, device=self._get_device())
        
        # === 🔥 修复：freq_info可能为None ===
        if freq_info is None:
            # 如果没有频域信息，返回0损失或使用简化版本
            if self.logger and self._batch_counter % 200 == 0:
                self.debug_logger.warning("⚠️  freq_info is None, frequency_alignment_loss disabled for this batch")
            return torch.tensor(0.0, device=self._get_device(), requires_grad=True)

        loss = self._frequency_alignment_module(id_feat, attr_feat, freq_info)

        if self.logger and self._batch_counter % 200 == 0:
            # 记录频域对齐损失的统计信息
            if freq_info is not None:
                energy_ratio = freq_info.get('energy_ratio', torch.tensor([0.5]))
                freq_magnitude = freq_info.get('freq_magnitude', torch.tensor([0.0]))
                self.debug_logger.debug(
                    f"Frequency Alignment Loss: {loss.item():.6f}, "
                    f"energy_ratio={energy_ratio.mean().item():.4f}, "
                    f"freq_magnitude={freq_magnitude.mean().item():.4f}"
                )

        return loss

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids,
                is_matched=None, epoch=None, gate=None,
                id_seq_features=None, cloth_seq_features=None, saliency_score=None,
                id_cls_features=None, original_feat=None, freq_info=None):
        """
        前向传播：计算所有损失 (方案B：频域对齐版)

        注意：保留了id_logits和id_cls_features参数以保持向后兼容，
        但不使用这些参数进行损失计算。
        """
        losses = {}

        if epoch is not None:
            self.update_epoch(epoch)

        # 1. InfoNCE (主任务)
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds, fused_embeds)

        # 2. Cloth Semantic (属性对齐)
        losses['cloth_semantic'] = self.cloth_semantic_loss(cloth_image_embeds, cloth_text_embeds)

        # 3. Orthogonal (弱约束)
        losses['orthogonal'] = self.orthogonal_loss(id_embeds, cloth_embeds)

        # 4. Triplet (ID一致性)
        losses['id_triplet'] = self.triplet_loss(id_embeds, pids)

        # 5. Frequency Alignment (频域对齐，新增，替代CLS)
        losses['frequency_alignment'] = self.frequency_alignment_loss(
            id_embeds, cloth_embeds, freq_info
        )

        # === 兼容性占位符 (已删除的损失返回0，但不加入total) ===
        losses['gate_adaptive'] = torch.tensor(0.0, device=self._get_device())
        losses['semantic_alignment'] = torch.tensor(0.0, device=self._get_device())
        losses['freq_consistency'] = torch.tensor(0.0, device=self._get_device())
        losses['freq_separation'] = torch.tensor(0.0, device=self._get_device())
        losses['anti_collapse'] = torch.tensor(0.0, device=self._get_device())
        losses['reconstruction'] = torch.tensor(0.0, device=self._get_device())
        losses['cls'] = torch.tensor(0.0, device=self._get_device())  # CLS损失已废弃

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
