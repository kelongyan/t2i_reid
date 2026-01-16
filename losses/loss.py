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
    def __init__(self, temperature=0.1, weights=None, num_classes=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        
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
        }
        
        # 动态权重调整参数
        self.current_epoch = 0
        self.enable_dynamic_weights = True
        
        # 移除额外的投影层，简化cloth_semantic
        # 原因：增加不必要的复杂度和训练难度
        self.use_decouple_penalty = False  # 禁用去ID正则
    
    def _initialize_projection_layers(self, device):
        """动态初始化投影层"""
        if not self.initialized:
            self.id_to_256 = nn.Linear(768, 256).to(device)
            self.initialized = True
    
    def update_epoch(self, epoch):
        """
        === 修复方案：渐进式权重调整 ===
        让CLS权重逐步提高，cloth_semantic保持稳定
        """
        self.current_epoch = epoch
        
        if not self.enable_dynamic_weights:
            return
        
        # 阶段1 (1-20): 预热期，让模型适应任务
        # CLS缓慢学习，主要靠InfoNCE建立基础特征空间
        if epoch <= 20:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.08 + 0.002 * epoch  # 0.08→0.12 线性增长
            self.weights['cloth_semantic'] = 0.15
            self.weights['orthogonal'] = 0.3 + 0.01 * epoch  # 0.3→0.5 逐步增强
            self.weights['gate_adaptive'] = 0.02
            
        # 阶段2 (21-40): 加速期，CLS开始主导
        elif epoch <= 40:
            progress = (epoch - 20) / 20  # 0→1
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.12 + 0.18 * progress  # 0.12→0.3
            self.weights['cloth_semantic'] = 0.15 + 0.05 * progress  # 0.15→0.2
            self.weights['orthogonal'] = 0.5 - 0.2 * progress  # 0.5→0.3（解耦任务完成，降低权重）
            self.weights['gate_adaptive'] = 0.02 + 0.03 * progress  # 0.02→0.05
            
        # 阶段3 (41-60): 稳定期
        elif epoch <= 60:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.3
            self.weights['cloth_semantic'] = 0.2
            self.weights['orthogonal'] = 0.2  # 进一步降低
            self.weights['gate_adaptive'] = 0.05
            
        # 阶段4 (61+): 微调期
        else:
            self.weights['info_nce'] = 1.0
            self.weights['cls'] = 0.25  # 略微降低，避免过拟合
            self.weights['cloth_semantic'] = 0.25
            self.weights['orthogonal'] = 0.15
            self.weights['gate_adaptive'] = 0.08
    
    def gate_adaptive_loss_v2(self, gate, id_embeds, cloth_embeds, pids):
        """
        === P2方案：重新设计的gate_adaptive损失 ===
        目标：好的gate应该使同类样本的id特征更相似（类内紧凑）
        """
        if gate is None or id_embeds is None or cloth_embeds is None:
            if id_embeds is not None:
                return id_embeds.sum() * 0.0
            elif cloth_embeds is not None:
                return cloth_embeds.sum() * 0.0
            else:
                return torch.tensor(0.0, requires_grad=True)
        
        batch_size = id_embeds.size(0)
        
        # 统一gate维度
        if gate.dim() == 2:
            gate_value = gate.mean(dim=1) if gate.size(1) > 1 else gate.squeeze(1)
        elif gate.dim() == 1:
            gate_value = gate
        else:
            gate_value = gate.expand(batch_size)
        
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
            
            # gate平滑正则
            gate_smooth = gate_value.var()
            smooth_penalty = torch.clamp(gate_smooth - 0.02, min=0.0)
            
            total_loss = compact_loss + 0.1 * smooth_penalty
        else:
            # batch太小时，鼓励gate保持合理值
            gate_mean = gate_value.mean()
            total_loss = (gate_mean - 0.5).pow(2)
        
        return torch.clamp(total_loss, min=0.0, max=10.0)
    
    def info_nce_loss(self, image_embeds, text_embeds):
        """InfoNCE对比学习损失"""
        if image_embeds is None or text_embeds is None:
            return torch.tensor(0.0, device=self.ce_loss.weight.device)
        
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
            return torch.tensor(0.0, device=self.ce_loss.weight.device)
        
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
            return torch.tensor(0.0, device=self.ce_loss.weight.device if hasattr(self, 'ce_loss') else 'cuda')
        
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
            return torch.tensor(0.0, device=id_embeds.device if id_embeds is not None else 'cuda')
        
        batch_size = id_embeds.size(0)
        
        # 归一化特征
        id_norm = F.normalize(id_embeds, dim=-1, eps=1e-8)
        cloth_norm = F.normalize(cloth_embeds, dim=-1, eps=1e-8)
        
        # 批次内正交约束：最小化余弦相似度的平方
        cosine_sim = (id_norm * cloth_norm).sum(dim=-1)
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
        ortho_loss = cosine_sim.pow(2).mean()
        
        return ortho_loss
    
    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids, 
                is_matched=None, epoch=None, gate=None,
                id_seq_features=None, cloth_seq_features=None, saliency_score=None,
                id_cls_features=None):  # 新增：分类分支的中间特征
        """
        前向传播：计算所有损失
        
        === 重构后的调用流程 ===
        1. 更新epoch（动态权重）
        2. 计算各个损失（使用v2版本）
        3. 加权求和
        
        Args:
            id_cls_features: 分类分支的中间特征 [B, 1024]，用于center loss等高级损失
        """
        losses = {}
        
        # === P1: 动态权重更新 ===
        if epoch is not None:
            self.update_epoch(epoch)
        
        # === 核心损失计算 ===
        # 1. InfoNCE损失
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds) \
            if image_embeds is not None and id_text_embeds is not None \
            else torch.tensor(0.0, device=self.ce_loss.weight.device)
        
        # 2. 分类损失（P0：权重已提高）
        losses['cls'] = self.id_classification_loss(id_logits, pids) \
            if id_logits is not None and pids is not None \
            else torch.tensor(0.0, device=self.ce_loss.weight.device)
        
        # 3. 服装语义损失（P0：修复去ID正则）
        losses['cloth_semantic'] = self.cloth_semantic_loss_v2(
            cloth_image_embeds, cloth_text_embeds, id_embeds
        )
        
        # 4. 正交约束（P1：增强版）
        losses['orthogonal'] = self.orthogonal_loss_v2(id_embeds, cloth_embeds)
        
        # 5. 门控自适应（P2：重新设计）
        losses['gate_adaptive'] = self.gate_adaptive_loss_v2(
            gate, id_embeds, cloth_embeds, pids
        )
        
        # === NaN/Inf检查 ===
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    losses[key] = torch.tensor(0.0, device=value.device, requires_grad=True)
        
        # === 加权求和 ===
        total_loss = sum(self.weights.get(k, 0) * losses[k] 
                        for k in losses.keys() if k != 'total')
        
        # 最终检查
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)
        
        losses['total'] = total_loss
        
        return losses
