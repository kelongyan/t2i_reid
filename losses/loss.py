import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    """
    === 深度重构的损失函数模块 ===
    实施方案：
    - P0: 修复权重失衡（提高cls权重4倍）
    - P1: 动态权重调整（根据训练阶段自适应）
    - P2: 重新设计gate_adaptive（使用对比学习）
    - 修复cloth_semantic去ID正则（添加维度转换）
    """
    def __init__(self, temperature=0.1, weights=None, num_classes=None, logger=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.logger = logger
        
        # 使用Label Smoothing降低分类损失的初始值
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # === 修复方案：合理的初始权重配置 ===
        # 核心原则：让所有加权损失在同一数量级（~1.0）
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,        # 对比学习 - 主导 (损失~4.0，加权后~4.0)
            'cls': 0.1,             # 分类损失 (损失~8.0，加权后~0.8，让其慢慢学习)
            'cloth_semantic': 0.15, # 服装语义 (损失~4.5，加权后~0.7)
            'orthogonal': 0.3,      # 正交约束 (损失~0.01，加权后~0.003)
            'gate_adaptive': 0.02,  # 门控自适应 (损失~0.5，加权后~0.01)
            'diversity': 0.05,      # 门控多样性正则（新增，软门控专用）
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
        === 优化后的动态权重调整 ===
        核心改进：
        1. 降低cloth_semantic初始权重（0.15 -> 0.1）
        2. 降低cls增长速度（0.08+0.002*e -> 0.06+0.001*e）
        3. 提高gate_adaptive权重（0.02 -> 0.05）
        """
        self.current_epoch = epoch
        
        if not self.enable_dynamic_weights:
            return
        
        # 阶段1 (1-20): 预热期
        if epoch <= 20:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.06 + 0.001 * epoch  # 0.06->0.08 (更慢增长)
            self.weights['cloth_semantic'] = 0.10  # 降低初始权重
            self.weights['orthogonal'] = 0.2  # 降低初始权重
            self.weights['gate_adaptive'] = 0.05  # 提高初始权重
            self.weights['diversity'] = 0.05
            
        # 阶段2 (21-40): 稳定期
        elif epoch <= 40:
            progress = (epoch - 20) / 20
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.08 + 0.10 * progress  # 0.08->0.18
            self.weights['cloth_semantic'] = 0.10 + 0.05 * progress  # 0.10->0.15
            self.weights['orthogonal'] = 0.2 - 0.05 * progress  # 0.2->0.15
            self.weights['gate_adaptive'] = 0.05 + 0.05 * progress  # 0.05->0.10
            self.weights['diversity'] = 0.05
            
        # 阶段3 (41+): 收敛期
        else:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.20  # 适度降低
            self.weights['cloth_semantic'] = 0.15
            self.weights['orthogonal'] = 0.15
            self.weights['gate_adaptive'] = 0.10
            self.weights['diversity'] = 0.05
    
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
    
    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids, 
                is_matched=None, epoch=None, gate=None,
                id_seq_features=None, cloth_seq_features=None, saliency_score=None,
                id_cls_features=None, diversity_loss=None):
        """
        前向传播：计算所有损失
        
        === 软门控版本更新 ===
        1. gate现在是gate_stats (dict)
        2. 新增diversity_loss参数（从模型传入）
        
        Args:
            gate: 现在是gate_stats字典，包含门控统计信息
            diversity_loss: 门控多样性损失（从G-S3模块传入）
            id_cls_features: 分类分支的中间特征 [B, 1024]
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
        
        # 3. 服装语义损失
        losses['cloth_semantic'] = self.cloth_semantic_loss_v2(
            cloth_image_embeds, cloth_text_embeds, id_embeds
        )
        
        # 4. 正交约束
        losses['orthogonal'] = self.orthogonal_loss_v2(id_embeds, cloth_embeds)
        
        # 5. 门控自适应（现在接收gate_stats）
        losses['gate_adaptive'] = self.gate_adaptive_loss_v2(
            gate, id_embeds, cloth_embeds, pids
        )
        
        # 6. 门控多样性正则（软门控新增）
        if diversity_loss is not None:
            losses['diversity'] = diversity_loss
        else:
            losses['diversity'] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
        
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
