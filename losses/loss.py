import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyConsistencyLoss(nn.Module):
    """
    频域一致性损失
    
    目标：
    - ID特征应该与低频特征对齐
    - Attr特征应该与高频特征对齐
    
    这是FSHD-Net的核心监督信号
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, id_feat, attr_feat, low_freq_feat, high_freq_feat):
        """
        Args:
            id_feat: ID特征 [B, D]
            attr_feat: Attr特征 [B, D]
            low_freq_feat: 低频特征（全局池化后） [B, D]
            high_freq_feat: 高频特征（全局池化后） [B, D]
        Returns:
            loss: 频域一致性损失
        """
        # 归一化
        id_norm = F.normalize(id_feat, dim=-1, eps=1e-8)
        attr_norm = F.normalize(attr_feat, dim=-1, eps=1e-8)
        low_norm = F.normalize(low_freq_feat, dim=-1, eps=1e-8)
        high_norm = F.normalize(high_freq_feat, dim=-1, eps=1e-8)
        
        # ID特征应该与低频特征相似（余弦相似度应接近1）
        id_low_sim = (id_norm * low_norm).sum(dim=-1)  # [B]
        loss_id_low = (1.0 - id_low_sim).mean()
        
        # Attr特征应该与高频特征相似
        attr_high_sim = (attr_norm * high_norm).sum(dim=-1)
        loss_attr_high = (1.0 - attr_high_sim).mean()
        
        # 总损失
        return loss_id_low + loss_attr_high


class FrequencySeparationLoss(nn.Module):
    """
    频域分离损失（可选）
    
    目标：
    - ID特征应该远离高频特征
    - Attr特征应该远离低频特征
    
    这是一个辅助约束，增强频域分离的纯净度
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, id_feat, attr_feat, low_freq_feat, high_freq_feat):
        """
        Args:
            id_feat, attr_feat: [B, D]
            low_freq_feat, high_freq_feat: [B, D]
        Returns:
            loss: 分离损失
        """
        id_norm = F.normalize(id_feat, dim=-1, eps=1e-8)
        attr_norm = F.normalize(attr_feat, dim=-1, eps=1e-8)
        low_norm = F.normalize(low_freq_feat, dim=-1, eps=1e-8)
        high_norm = F.normalize(high_freq_feat, dim=-1, eps=1e-8)
        
        # ID应该远离高频（相似度应接近0）
        id_high_sim = torch.abs((id_norm * high_norm).sum(dim=-1))
        
        # Attr应该远离低频
        attr_low_sim = torch.abs((attr_norm * low_norm).sum(dim=-1))
        
        # 惩罚相似度（越接近0越好）
        return id_high_sim.mean() + attr_low_sim.mean()


class SymmetricReconstructionLoss(nn.Module):
    """
    对称重构损失 (Symmetric Reconstruction Loss)
    
    核心思想：F_input ≈ F_id + F_attr
    确保解耦后的两个特征能够重建原始输入，防止信息丢失
    
    改进：相比TextGuidedDecouplingLoss，直接约束视觉特征的重构
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
    
    def forward(self, id_feat, attr_feat, original_feat):
        """
        Args:
            id_feat: ID特征 [B, dim]
            attr_feat: Attr特征 [B, dim]
            original_feat: 原始特征（解耦前的全局特征）[B, dim]
            
        Returns:
            loss: 重构损失
        """
        # 简单加法重建
        reconstructed = id_feat + attr_feat  # [B, dim]
        
        # 方案1：MSE Loss（L2距离）
        mse_loss = self.mse_loss(reconstructed, original_feat)
        
        # 方案2：Cosine Similarity Loss（方向一致性）
        # CosineEmbeddingLoss需要target为1（表示相似）
        target = torch.ones(id_feat.size(0), device=id_feat.device)
        cos_loss = self.cosine_loss(
            F.normalize(reconstructed, dim=-1),
            F.normalize(original_feat, dim=-1),
            target
        )
        
        # 组合两种损失
        return mse_loss + 0.5 * cos_loss


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
    === FSHD损失函数模块 (Frequency-Spatial Hybrid Decoupling Loss System) ===
    
    核心损失：
    - InfoNCE: 主对比学习损失
    - SymmetricReconstructionLoss: 特征重构
    - EnhancedOrthogonalLoss: 正交约束
    - FrequencyConsistencyLoss: 频域一致性 (FSHD核心)
    - FrequencySeparationLoss: 频域分离 (FSHD辅助)
    
    动态权重调整：3-stage策略
    """
    def __init__(self, temperature=0.1, weights=None, num_classes=None, logger=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.logger = logger
        
        # 使用Label Smoothing降低分类损失的初始值
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # === 核心损失模块 ===
        self.symmetric_reconstruction = SymmetricReconstructionLoss()
        self.enhanced_orthogonal = EnhancedOrthogonalLoss()
        
        # === FSHD频域损失 ===
        self.frequency_consistency = FrequencyConsistencyLoss()
        self.frequency_separation = FrequencySeparationLoss()
        
        # === FSHD权重配置（优化版）===
        # 移除冗余损失，保留核心损失
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,              # 对比学习 - 主导
            'cls': 0.05,                  # 分类损失
            'cloth_semantic': 1.0,        # 衣服语义
            'orthogonal': 0.1,            # 正交约束（降低，防止过度解耦）
            'id_triplet': 0.5,            # ID一致性
            'anti_collapse': 1.0,         # 防坍缩
            'gate_adaptive': 0.02,        # 门控自适应（降低）
            'reconstruction': 0.5,        # 对称重构
            'semantic_alignment': 0.1,    # CLIP语义对齐（可选）
            'freq_consistency': 0.5,      # 【FSHD】频域一致性
            'freq_separation': 0.2,       # 【FSHD】频域分离
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
            self._log_counter_gate = 0
    
    def set_semantic_guidance(self, semantic_guidance_module):
        """
        设置语义引导模块
        
        Args:
            semantic_guidance_module: SemanticGuidedDecoupling实例
        """
        self.semantic_guidance_module = semantic_guidance_module
        if self.logger:
            self.logger.debug_logger.info("✅ Semantic guidance module attached to Loss")
    
    def _get_device(self):
        """安全获取设备"""
        return self._dummy.device
    
    def update_epoch(self, epoch):
        """
        === 动态权重调整（3-Stage策略）===
        优化：简化逻辑，减少不必要的调整
        """
        self.current_epoch = epoch
        
        if not self.enable_dynamic_weights:
            return
        
        # Stage 1 (Epoch 0-10): 激活期 - 让双分支都能提取有效特征
        if epoch <= 10:
            self.weights['info_nce'] = 1.2
            self.weights['cls'] = 0.02
            self.weights['orthogonal'] = 0.15         # 适中的正交约束
            self.weights['reconstruction'] = 0.8      # 强化重构
            self.weights['semantic_alignment'] = 0.05
            self.weights['anti_collapse'] = 1.5
            self.weights['freq_consistency'] = 0.3    # 逐步引入频域监督
            self.weights['freq_separation'] = 0.1
            
        # Stage 2 (Epoch 11-40): 语义对齐期 - 优化特征质量
        elif epoch <= 40:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.05
            self.weights['cloth_semantic'] = 1.2      # 增强cloth语义
            self.weights['orthogonal'] = 0.1          # 降低正交约束
            self.weights['reconstruction'] = 0.5
            self.weights['semantic_alignment'] = 0.1
            self.weights['freq_consistency'] = 0.5    # 完整频域监督
            self.weights['freq_separation'] = 0.2
            
        # Stage 3 (Epoch 41+): 精细微调期 - 稳定收敛
        else:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.1                 # 轻微增加分类权重
            self.weights['cloth_semantic'] = 1.0
            self.weights['orthogonal'] = 0.08         # 进一步降低
            self.weights['reconstruction'] = 0.4
            self.weights['semantic_alignment'] = 0.12
            self.weights['freq_consistency'] = 0.5
            self.weights['freq_separation'] = 0.2
            
        # 记录权重变化
        if self.logger and epoch in [0, 11, 41]:
            self.debug_logger.info(f"📊 Loss weights updated at epoch {epoch}:")
            for k, v in self.weights.items():
                self.debug_logger.info(f"   - {k}: {v:.4f}")
    
    def gate_adaptive_loss_v2(self, gate_stats, id_embeds, cloth_embeds, pids):
        """
        === 门控自适应损失（简化版）===
        目标：使同类样本的ID特征更相似（类内紧凑）
        优化：简化计算，减少冗余逻辑
        """
        if gate_stats is None or id_embeds is None:
            return torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
        batch_size = id_embeds.size(0)
        
        # 小batch跳过复杂计算
        if batch_size <= 1 or pids is None:
            return torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
        # 同类样本mask
        mask = (pids.unsqueeze(0) == pids.unsqueeze(1)).float()
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # 如果没有同类样本，跳过
        if mask.sum() < 1e-6:
            return torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
        # 归一化特征
        id_norm = F.normalize(id_embeds, dim=-1, eps=1e-8)
        id_sim = torch.matmul(id_norm, id_norm.t())
        
        # 类内紧凑度
        intra_class_sim = (id_sim * mask).sum() / (mask.sum() + 1e-8)
        compact_loss = 1.0 - intra_class_sim
        
        # 门控正则（防止极端值）
        gate_id_mean = gate_stats.get('gate_id_mean', 0.5)
        gate_regularization = 0.0
        if gate_id_mean < 0.25 or gate_id_mean > 0.85:
            gate_regularization = 0.05 * ((gate_id_mean - 0.55) ** 2)
        
        total_loss = compact_loss + gate_regularization
        
        # 定期记录调试信息
        if self.logger:
            self._log_counter_gate = getattr(self, '_log_counter_gate', 0) + 1
            if self._log_counter_gate % 500 == 0:
                self.debug_logger.debug(
                    f"[Gate Adaptive] intra_sim={intra_class_sim:.4f} | "
                    f"loss={compact_loss:.6f} | gate_mean={gate_id_mean:.4f}"
                )
        
        return torch.clamp(total_loss, min=0.0, max=5.0)
    
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
        
        return (loss_i2t + loss_t2i) / 2
    
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
        # 原因：温度缩放会严重抑制学习速度
        # 通过调整loss权重（0.08→0.3）来控制学习进度更合理
        ce_loss = self.ce_loss(id_logits_clipped, pids)
        
        return ce_loss
    
    def cloth_semantic_loss_v2(self, cloth_image_embeds, cloth_text_embeds, id_embeds_768=None):
        """
        === 修复方案：简化的cloth_semantic损失 ===
        移除去ID正则，让G-S3模块专注于解耦任务
        原因：
        1. 增加额外投影层会增加训练难度
        2. 去ID惩罚与orthogonal_loss功能重复
        3. 实验显示cloth_semantic占总损失83-95%，说明基础损失就已经很高
        """
        if cloth_image_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        
        bsz = cloth_image_embeds.size(0)
        
        # === 标准对比学习损失（与InfoNCE一致）=== 
        cloth_image_norm = F.normalize(cloth_image_embeds, dim=-1, eps=1e-8)
        cloth_text_norm = F.normalize(cloth_text_embeds, dim=-1, eps=1e-8)
        
        sim = torch.matmul(cloth_image_norm, cloth_text_norm.t()) / self.temperature
        sim = torch.clamp(sim, min=-50, max=50)
        
        labels = torch.arange(bsz, device=sim.device, dtype=torch.long)
        loss_i2t = self.ce_loss(sim, labels)
        loss_t2i = self.ce_loss(sim.t(), labels)
        
        # 不再添加去ID正则，让损失保持简洁
        # orthogonal_loss会负责身份-服装的解耦
        return (loss_i2t + loss_t2i) / 2
    
    def orthogonal_loss_v2(self, id_embeds, cloth_embeds):
        """
        === 对称解耦改进：使用增强正交损失 ===
        启用交叉批次正交约束，让特征空间更彻底分离
        """
        if id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        
        # 使用增强版正交损失
        ortho_loss = self.enhanced_orthogonal(
            id_embeds, cloth_embeds, cross_batch=True
        )
        
        # 调试信息
        if self.logger and hasattr(self, '_log_counter_ortho'):
            self._log_counter_ortho = getattr(self, '_log_counter_ortho', 0) + 1
            if self._log_counter_ortho % 200 == 0:
                id_norm = F.normalize(id_embeds, dim=-1, eps=1e-8)
                cloth_norm = F.normalize(cloth_embeds, dim=-1, eps=1e-8)
                cosine_sim = (id_norm * cloth_norm).sum(dim=-1)
                self.logger.debug_logger.debug(
                    f"Enhanced Orthogonal: cosine_sim mean={cosine_sim.mean().item():.4f}, "
                    f"std={cosine_sim.std().item():.4f}, ortho_loss={ortho_loss.item():.6f}"
                )
        
        return ortho_loss
    
    def triplet_loss(self, embeds, pids, margin=0.3):
        """[方案 C] ID 一致性损失：确保同一 ID 在不同衣服下的特征一致性"""
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
        return loss

    def anti_collapse_loss(self, cloth_embeds, margin=1.0):
        """[基础保障] 防坍缩正则：确保衣服特征存在，打破零和博弈"""
        if cloth_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        
        # 计算 L2 范数
        norms = torch.norm(cloth_embeds, p=2, dim=-1)
        # 惩罚模长过小的向量
        loss = F.relu(margin - norms).mean()
        return loss

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids, 
                is_matched=None, epoch=None, gate=None,
                id_seq_features=None, cloth_seq_features=None, saliency_score=None,
                id_cls_features=None, original_feat=None, freq_info=None):
        """
        前向传播：计算所有损失（FSHD版本）
        
        新增参数：
            original_feat: 解耦前的原始特征，用于重构监督
            freq_info: 频域信息字典（包含low_freq和high_freq）
        """
        losses = {}
        
        # === P1: 动态权重更新 ===
        if epoch is not None:
            self.update_epoch(epoch)
        
        # === 核心损失计算 ===
        # 1. InfoNCE损失
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds, fused_embeds) \
            if image_embeds is not None and id_text_embeds is not None \
            else torch.tensor(0.0, device=self._get_device())
        
        # 2. 分类损失
        losses['cls'] = self.id_classification_loss(id_logits, pids) \
            if id_logits is not None and pids is not None \
            else torch.tensor(0.0, device=self._get_device())
        
        # 3. 服装语义损失
        losses['cloth_semantic'] = self.cloth_semantic_loss_v2(
            cloth_image_embeds, cloth_text_embeds, id_embeds
        )
        
        # 4. 正交约束损失（使用增强版）
        losses['orthogonal'] = self.orthogonal_loss_v2(id_embeds, cloth_embeds)
        
        # 5. ID 一致性 Triplet
        losses['id_triplet'] = self.triplet_loss(id_embeds, pids)
        
        # 6. 防坍缩正则
        if id_embeds is not None:
            id_collapse_loss = self.anti_collapse_loss(id_embeds, margin=1.0)
        else:
            id_collapse_loss = torch.tensor(0.0, device=self._get_device())
        
        if cloth_embeds is not None:
            cloth_collapse_loss = self.anti_collapse_loss(cloth_embeds, margin=1.0)
        else:
            cloth_collapse_loss = torch.tensor(0.0, device=self._get_device())
        
        losses['anti_collapse'] = (id_collapse_loss + cloth_collapse_loss) / 2
        
        # 7. 门控自适应
        losses['gate_adaptive'] = self.gate_adaptive_loss_v2(
            gate, id_embeds, cloth_embeds, pids
        )
        
        # 8. 对称重构损失
        if original_feat is not None and id_embeds is not None and cloth_embeds is not None:
            losses['reconstruction'] = self.symmetric_reconstruction(
                id_embeds, cloth_embeds, original_feat
            )
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
        # 9. CLIP语义对齐损失
        if self.semantic_guidance_module is not None and \
           id_embeds is not None and cloth_embeds is not None:
            losses['semantic_alignment'] = self.semantic_guidance_module(
                id_embeds, cloth_embeds, use_cross_separation=False
            )
        else:
            losses['semantic_alignment'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
        # 10. 【新增】频域一致性损失
        if freq_info is not None and 'low_freq' in freq_info and 'high_freq' in freq_info:
            # 从freq_info提取频域特征（需要池化为全局特征）
            low_freq_seq = freq_info['low_freq']  # [B, N, D]
            high_freq_seq = freq_info['high_freq']
            
            # 全局平均池化
            low_freq_global = low_freq_seq.mean(dim=1)  # [B, D]
            high_freq_global = high_freq_seq.mean(dim=1)
            
            if id_embeds is not None and cloth_embeds is not None:
                losses['freq_consistency'] = self.frequency_consistency(
                    id_embeds, cloth_embeds, low_freq_global, high_freq_global
                )
            else:
                losses['freq_consistency'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        else:
            losses['freq_consistency'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
        # 11. 【新增】频域分离损失（可选，默认权重较小）
        if freq_info is not None and 'low_freq' in freq_info and 'high_freq' in freq_info:
            low_freq_global = freq_info['low_freq'].mean(dim=1)
            high_freq_global = freq_info['high_freq'].mean(dim=1)
            
            if id_embeds is not None and cloth_embeds is not None:
                losses['freq_separation'] = self.frequency_separation(
                    id_embeds, cloth_embeds, low_freq_global, high_freq_global
                )
            else:
                losses['freq_separation'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        else:
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
        
        return losses