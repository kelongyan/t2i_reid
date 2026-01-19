import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGuidedDecouplingLoss(nn.Module):
    """
    文本引导的解耦损失 (Text-Guided Consistency Loss)
    
    目标：利用 CLIP 文本编码器的语义信息作为监督，约束视觉特征解耦的语义完整性。
    逻辑：视觉 ID 特征 + 视觉衣服特征 重新组合后，应该能重建其对应的语义表达。
    """
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, vis_id_feat, vis_cloth_feat, target_feat):
        """
        Args:
            vis_id_feat: 视觉 ID 特征
            vis_cloth_feat: 视觉衣服特征
            target_feat: 目标语义特征 (通常是融合后的特征 fused_embeds 或 文本特征)
        """
        # 简单的加和重建
        vis_reconstructed = vis_id_feat + vis_cloth_feat
        
        # 归一化后计算距离，更关注方向一致性
        vis_reconstructed = F.normalize(vis_reconstructed, dim=-1)
        target_feat = F.normalize(target_feat, dim=-1)
        
        reconstruction_loss = self.loss_fn(vis_reconstructed, target_feat)
        return reconstruction_loss


class Loss(nn.Module):
    """
    === 深度重构的损失函数模块 ===
    实施方案：
    - P0: 修复权重失衡（提高cls权重4倍）
    - P1: 动态权重调整（根据训练阶段自适应）
    - P2: 重新设计gate_adaptive（使用对比学习）
    - P3: 新增文本引导的一致性重构损失 (TextGuidedDecouplingLoss)
    - 修复cloth_semantic去ID正则（添加维度转换）
    """
    def __init__(self, temperature=0.1, weights=None, num_classes=None, logger=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.logger = logger
        
        # 使用Label Smoothing降低分类损失的初始值
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 新增：重构损失
        self.reconstruction_loss = TextGuidedDecouplingLoss()
        
        # === 修复方案：合理的初始权重配置 ===
        # 核心原则：让所有加权损失在同一数量级（~1.0）
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,        # 对比学习 - 主导
            'cls': 0.5,             # 分类损失 (适当提高)
            'cloth_semantic': 2.0,  # 衣服语义 (强力拉扯)
            'id_triplet': 1.0,      # ID 一致性
            'anti_collapse': 1.5,   # 防坍缩 (底线)
            'gate_adaptive': 0.05,  # 辅助
            'reconstruction': 0.1,  # 辅助
        }
        
        # 动态权重调整参数
        self.current_epoch = 0
        self.enable_dynamic_weights = True
        
        # 移除额外的投影层，简化cloth_semantic
        # 原因：增加不必要的复杂度和训练难度
        self.use_decouple_penalty = False  # 禁用去ID正则
        
        # 注册一个dummy参数用于获取设备
        self.register_buffer('_dummy', torch.zeros(1))
    
    def _get_device(self):
        """安全获取设备"""
        return self._dummy.device
    
    def _initialize_projection_layers(self, device):
        """动态初始化投影层"""
        if not self.initialized:
            self.id_to_256 = nn.Linear(768, 256).to(device)
            self.initialized = True
    
    def update_epoch(self, epoch):
        """
        === 优化后的动态权重调整 (Relax & Constrain Schedule) ===
        """
        self.current_epoch = epoch
        
        if not self.enable_dynamic_weights:
            return
        
        # Stage 1 (Epoch 0-5): 强制生存期
        # 目标：激活特征，防止全零坍缩
        if epoch <= 5:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.5
            self.weights['anti_collapse'] = 2.0  # 极高权重，强制特征存在
            self.weights['cloth_semantic'] = 0.5 # 先不要求太准
            self.weights['id_triplet'] = 0.5     # 辅助
            
        # Stage 2 (Epoch 6-30): 语义对齐期
        # 目标：利用 CLIP 赋予特征语义
        elif epoch <= 30:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.5
            self.weights['anti_collapse'] = 1.0  # 降低约束，允许特征变化
            self.weights['cloth_semantic'] = 2.0 # 强力语义监督
            self.weights['id_triplet'] = 1.0     # 增强 ID 一致性
            
        # Stage 3 (Epoch 31+): 精细微调期
        # 目标：平衡解耦与识别
        else:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.5
            self.weights['anti_collapse'] = 1.0
            self.weights['cloth_semantic'] = 1.5 # 适度降低
            self.weights['id_triplet'] = 1.0     # 保持 ID 约束
    
    def gate_adaptive_loss_v2(self, gate_stats, id_embeds, cloth_embeds, pids):
        """
        === 软门控版本：处理gate_stats字典 ===
        gate_stats是dict，包含gate_id和gate_cloth的统计信息
        目标：好的gate应该使同类样本的id特征更相似（类内紧凑）
        """
        if gate_stats is None or id_embeds is None or cloth_embeds is None:
            if id_embeds is not None:
                return id_embeds.sum() * 0.0
            elif cloth_embeds is not None:
                return cloth_embeds.sum() * 0.0
            else:
                return torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
        batch_size = id_embeds.size(0)
        
        # 从gate_stats提取平均门控值（用于正则）
        # 注意：gate_stats中的值是标量，我们需要从原始gate_id/gate_cloth计算
        # 但这里我们只用统计信息做监控，loss还是基于id_embeds
        
        # === 核心：基于对比学习的gate优化 ===
        if batch_size > 1 and pids is not None:
            # 同类样本mask
            mask = (pids.unsqueeze(0) == pids.unsqueeze(1)).float()
            mask = mask - torch.eye(batch_size, device=mask.device)
            
            # 归一化特征
            id_norm = F.normalize(id_embeds, dim=-1, eps=1e-8)
            id_sim = torch.matmul(id_norm, id_norm.t())
            
            # 类内紧凑度
            if mask.sum() > 0:
                intra_class_sim = (id_sim * mask).sum() / (mask.sum() + 1e-8)
            else:
                intra_class_sim = id_sim.mean()
            
            # 最大化类内相似度
            compact_loss = 1.0 - intra_class_sim
            
            # gate平滑正则（基于统计信息）
            # 鼓励gate_id保持合理范围（不要太接近0或1）
            gate_id_mean = gate_stats.get('gate_id_mean', 0.5)
            gate_regularization = 0.0
            if gate_id_mean < 0.3 or gate_id_mean > 0.9:
                # 如果gate过于极端，添加惩罚
                gate_regularization = 0.1 * ((gate_id_mean - 0.6) ** 2)
            
            total_loss = compact_loss + gate_regularization
            
            # 调试信息
            if self.logger and hasattr(self, '_log_counter_gate'):
                self._log_counter_gate = getattr(self, '_log_counter_gate', 0) + 1
                if self._log_counter_gate % 200 == 0:
                    self.debug_logger.debug(
                        f"Gate_adaptive: intra_sim={intra_class_sim:.4f}, "
                        f"compact_loss={compact_loss:.6f}, gate_id_mean={gate_id_mean:.4f}, "
                        f"total={total_loss:.6f}"
                    )
        else:
            # batch太小时，基于gate统计信息的简单正则
            gate_id_mean = gate_stats.get('gate_id_mean', 0.5)
            total_loss = (gate_id_mean - 0.6).pow(2)  # 鼓励gate_id稍微偏向身份
        
        return torch.clamp(total_loss, min=0.0, max=10.0)
    
    def info_nce_loss(self, image_embeds, text_embeds):
        """InfoNCE对比学习损失"""
        if image_embeds is None or text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        
        bsz = image_embeds.size(0)
        image_embeds = F.normalize(image_embeds, dim=-1, eps=1e-8)
        text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-8)
        
        sim = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
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
        === 修复方案：简化的正交约束 ===
        核心目标：最小化 cos(id, cloth)^2
        移除复杂的跨样本约束，避免梯度混乱
        """
        if id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        
        batch_size = id_embeds.size(0)
        
        # 归一化特征
        id_norm = F.normalize(id_embeds, dim=-1, eps=1e-8)
        cloth_norm = F.normalize(cloth_embeds, dim=-1, eps=1e-8)
        
        # 批次内正交约束：最小化余弦相似度的平方
        cosine_sim = (id_norm * cloth_norm).sum(dim=-1)
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
        ortho_loss = cosine_sim.pow(2).mean()
        
        # 调试信息：记录余弦相似度
        if self.logger and hasattr(self, '_log_counter'):
            self._log_counter = getattr(self, '_log_counter', 0) + 1
            if self._log_counter % 200 == 0:  # 每200次记录一次
                self.debug_logger.debug(
                    f"Orthogonal loss: cosine_sim mean={cosine_sim.mean().item():.4f}, "
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
                id_cls_features=None):
        """
        前向传播：计算所有损失
        """
        losses = {}
        
        # === P1: 动态权重更新 ===
        if epoch is not None:
            self.update_epoch(epoch)
        
        # === 核心损失计算 ===
        # 1. InfoNCE损失
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds) \
            if image_embeds is not None and id_text_embeds is not None \
            else torch.tensor(0.0, device=self._get_device())
        
        # 2. 分类损失
        losses['cls'] = self.id_classification_loss(id_logits, pids) \
            if id_logits is not None and pids is not None \
            else torch.tensor(0.0, device=self._get_device())
        
        # 3. 服装语义损失 (权重提升)
        losses['cloth_semantic'] = self.cloth_semantic_loss_v2(
            cloth_image_embeds, cloth_text_embeds, id_embeds
        )
        
        # 4. ID 一致性 Triplet (新增)
        losses['id_triplet'] = self.triplet_loss(id_embeds, pids)
        
        # 5. 防坍缩正则 (新增)
        losses['anti_collapse'] = self.anti_collapse_loss(cloth_embeds, margin=1.0)
        
        # 6. 门控自适应 (保留)
        losses['gate_adaptive'] = self.gate_adaptive_loss_v2(
            gate, id_embeds, cloth_embeds, pids
        )
        
        # 7. 重构损失 (保留，作为辅助)
        # 修复维度不匹配问题：使用投影后的 256d 特征进行重构
        # image_embeds (256d) + cloth_image_embeds (256d) -> fused_embeds (256d)
        if image_embeds is not None and cloth_image_embeds is not None and fused_embeds is not None:
            losses['reconstruction'] = self.reconstruction_loss(
                image_embeds, cloth_image_embeds, fused_embeds
            )
        else:
             losses['reconstruction'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
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
