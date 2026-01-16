import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, temperature=0.1, weights=None, num_classes=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        
        # 使用Label Smoothing降低分类损失的初始值
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # === P0方案：提高cls权重，修复权重失衡 ===
        # === P1方案：动态权重调整的初始值 ===
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,        # 对比学习权重 - 主导损失
            'cls': 0.2,             # 🔥 P0: 从0.05提高到0.2（4倍）
            'cloth_semantic': 0.15, # 适度降低，避免竞争
            'orthogonal': 0.5,      # 🔥 P1: 从0.3提高到0.5，加强解耦
            'gate_adaptive': 0.1,   # 简化后降低权重
        }
        
        # 动态权重调整参数
        self.current_epoch = 0
        self.enable_dynamic_weights = True  # 是否启用动态权重
        
        # 维度转换层（用于cloth_semantic的去ID正则）
        # 这些层会在第一次forward时动态初始化
        self.id_to_256 = None  # 将768维id_embeds投影到256维
        self.initialized = False

    
    def _initialize_projection_layers(self, device):
        """动态初始化投影层"""
        if not self.initialized:
            # 768维 -> 256维的投影层
            self.id_to_256 = nn.Linear(768, 256).to(device)
            self.initialized = True
    
    def update_epoch(self, epoch):
        """
        === P1方案：动态权重调整 ===
        根据训练阶段自适应调整损失权重
        """
        self.current_epoch = epoch
        
        if not self.enable_dynamic_weights:
            return
        
        # 阶段1 (Epoch 1-10): 快速降低cls，强化解耦
        if epoch <= 10:
            self.weights['cls'] = 0.25           # 更高的cls权重
            self.weights['cloth_semantic'] = 0.1  # 降低cloth权重
            self.weights['orthogonal'] = 0.6      # 非常强的正交约束
            self.weights['gate_adaptive'] = 0.05  # 门控晚期介入
            
        # 阶段2 (Epoch 11-30): 平衡优化
        elif epoch <= 30:
            self.weights['cls'] = 0.15
            self.weights['cloth_semantic'] = 0.15
            self.weights['orthogonal'] = 0.5
            self.weights['gate_adaptive'] = 0.1
            
        # 阶段3 (Epoch 31-50): 精细调优
        elif epoch <= 50:
            self.weights['cls'] = 0.1
            self.weights['cloth_semantic'] = 0.2
            self.weights['orthogonal'] = 0.4
            self.weights['gate_adaptive'] = 0.15
            
        # 阶段4 (Epoch 51+): 最终微调
        else:
            self.weights['cls'] = 0.08
            self.weights['cloth_semantic'] = 0.25
            self.weights['orthogonal'] = 0.3
            self.weights['gate_adaptive'] = 0.15
        """
        修复后的门控自适应损失
        目标: 根据特征质量动态平衡gate值，确保门控机制正常工作
        """
        if gate is None or id_embeds is None or cloth_embeds is None:
            # 返回一个可微分的零张量
            if id_embeds is not None:
                # 使用id_embeds创建一个可微分的零值
                return id_embeds.sum() * 0.0
            elif cloth_embeds is not None:
                return cloth_embeds.sum() * 0.0
            else:
                # 这种情况理论上不应该发生
                return torch.tensor(0.0, requires_grad=True)
        
        batch_size = id_embeds.size(0)
        
        # === 修复1: 统一gate维度处理 ===
        if gate.dim() == 2:
            if gate.size(1) > 1:
                # gate形状为[B, dim]，取平均得到[B]
                gate_value = gate.mean(dim=1)
            else:
                # gate形状为[B, 1]，squeeze得到[B]
                gate_value = gate.squeeze(1)
        elif gate.dim() == 1:
            # gate形状已经是[B]
            gate_value = gate
        else:
            # gate是标量，扩展为[B]
            gate_value = gate.expand(batch_size)
        
        # === 修复2: 重新设计特征质量度量 ===
        # 使用batch内的方差作为判别性度量，而不是特征维度的方差
        if batch_size < 2:
            # 批次太小时，使用特征的L2范数作为质量指标
            id_quality = id_embeds.norm(dim=1).mean()
            cloth_quality = cloth_embeds.norm(dim=1).mean()
        else:
            # 计算batch内特征的标准差（每个特征维度上样本的标准差）
            # 标准差大说明该维度的判别性强
            id_quality = id_embeds.std(dim=0).mean()    # 对所有特征维度的方差取平均
            cloth_quality = cloth_embeds.std(dim=0).mean()
        
        # 防止数值不稳定，使用更严格的范围
        id_quality = torch.clamp(id_quality, min=0.01, max=10.0)
        cloth_quality = torch.clamp(cloth_quality, min=0.01, max=10.0)
        
        # === 修复3: 动态目标gate计算 ===
        # gate应该反映id特征的重要性相对于总重要性的比例
        total_quality = id_quality + cloth_quality + 1e-6  # 加eps防止除零
        target_gate_value = id_quality / total_quality
        
        # 将标量扩展为[B]以匹配gate_value的形状
        if target_gate_value.dim() == 0:
            target_gate_value = target_gate_value.detach().expand(batch_size)
        else:
            target_gate_value = target_gate_value.detach()
        
        # === 损失组成 ===
        # 1. MSE损失: 使gate接近目标值
        mse_loss = F.mse_loss(gate_value, target_gate_value)
        
        # 2. 熵正则: 防止gate过于极端（过于接近0或1）
        gate_clamp = torch.clamp(gate_value, min=1e-6, max=1-1e-6)
        entropy = -(gate_clamp * torch.log(gate_clamp) + 
                    (1 - gate_clamp) * torch.log(1 - gate_clamp))
        entropy_reg = -entropy.mean()  # 负号表示最大化熵（鼓励不确定性）
        
        # 3. 稳定性约束: 防止gate在batch内变化过大
        if batch_size > 1:
            gate_var = gate_value.var()
            # 期望方差小于0.01，过大则惩罚
            stability_loss = torch.clamp(gate_var - 0.01, min=0.0)
        else:
            # 使用一个可微分的零值
            stability_loss = gate_value.mean() * 0.0
        
        # 总损失: 加权组合三个部分
        # 降低熵正则权重，避免总损失变负
        total_loss = mse_loss + 0.01 * entropy_reg + 0.05 * stability_loss
        
        # 最终裁剪，确保损失在合理范围内
        # 不再使用min=0.0，允许小的负值（熵正则可能略大）
        total_loss = torch.clamp(total_loss, min=-1.0, max=10.0)
        # 但最终返回时确保非负
        total_loss = torch.relu(total_loss)
        
        return total_loss

    def info_nce_loss(self, image_embeds, text_embeds):
        bsz = image_embeds.size(0)
        # 确保特征已归一化
        image_embeds = F.normalize(image_embeds, dim=-1, eps=1e-8)
        text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-8)
        
        # 计算相似度矩阵
        sim = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        
        # 防止数值溢出（softmax稳定性）
        sim = torch.clamp(sim, min=-50, max=50)
        
        labels = torch.arange(bsz, device=sim.device, dtype=torch.long)
        
        # 计算双向对比损失
        loss_i2t = self.ce_loss(sim, labels)
        loss_t2i = self.ce_loss(sim.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2

    def id_classification_loss(self, id_logits, pids):
        return self.ce_loss(id_logits, pids)

    def cloth_semantic_loss(self, cloth_image_embeds, cloth_text_embeds, id_embeds=None):
        """
        改进的服装语义损失
        目标: 对齐服装特征，同时添加去ID正则，避免cloth特征包含身份信息
        """
        if cloth_image_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=cloth_image_embeds.device if cloth_image_embeds is not None else 'cuda')
        
        bsz = cloth_image_embeds.size(0)
        
        # === 标准对比学习损失 ===
        # 确保特征已归一化
        cloth_image_norm = F.normalize(cloth_image_embeds, dim=-1, eps=1e-8)
        cloth_text_norm = F.normalize(cloth_text_embeds, dim=-1, eps=1e-8)
        
        # 计算相似度矩阵
        sim = torch.matmul(cloth_image_norm, cloth_text_norm.t()) / self.temperature
        sim = torch.clamp(sim, min=-50, max=50)
        
        labels = torch.arange(bsz, device=sim.device, dtype=torch.long)
        
        # 计算双向损失
        loss_i2t = self.ce_loss(sim, labels)
        loss_t2i = self.ce_loss(sim.t(), labels)
        
        base_loss = (loss_i2t + loss_t2i) / 2
        
        # === 改进: 添加去ID正则 ===
        # 确保cloth特征不包含id信息，防止信息泄漏
        if id_embeds is not None:
            id_norm = F.normalize(id_embeds, dim=-1, eps=1e-8)
            
            # cloth特征不应该与id特征相似
            # 计算cloth_image和id的余弦相似度
            cloth_id_sim = (cloth_image_norm * id_norm).sum(dim=-1)
            
            # 最小化相似度的绝对值（希望cloth和id正交）
            de_id_penalty = cloth_id_sim.abs().mean()
            
            # 将去ID正则加入总损失，权重为0.2
            return base_loss + 0.2 * de_id_penalty
        
        return base_loss

    def orthogonal_loss(self, id_embeds, cloth_embeds):
        """
        增强的正交约束损失
        目标: 强制id和cloth特征正交，减少信息泄漏，加强解耦效果
        """
        if id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=id_embeds.device if id_embeds is not None else 'cuda')
        
        batch_size = id_embeds.size(0)
        
        # 归一化特征向量
        id_norm = F.normalize(id_embeds, dim=-1, eps=1e-8)
        cloth_norm = F.normalize(cloth_embeds, dim=-1, eps=1e-8)
        
        # === 改进1: 批次内正交约束（样本内解耦）===
        # 计算余弦相似度
        cosine_sim = (id_norm * cloth_norm).sum(dim=-1)  # [B]
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
        
        # 使用平方损失而不是绝对值，梯度更稳定
        ortho_loss_batch = cosine_sim.pow(2).mean()
        
        # === 改进2: 添加跨样本正交约束（样本间独立性）===
        # 确保不同样本的id和cloth特征也相互独立
        if batch_size > 1:
            # 计算Gram矩阵: 每对样本之间的相似度
            id_gram = torch.matmul(id_norm, id_norm.t())      # [B, B]
            cloth_gram = torch.matmul(cloth_norm, cloth_norm.t())  # [B, B]
            
            # 计算id和cloth的Gram矩阵的逐元素乘积
            # 对角线是自相关(已经在ortho_loss_batch处理)，非对角线应该接近0
            # 创建掩码，移除对角线元素
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=id_gram.device)
            
            # 计算非对角线元素的交叉相关
            cross_correlation = (id_gram[mask] * cloth_gram[mask]).abs().mean()
            
            # === 改进3: 添加自相关惩罚 ===
            # 确保同模态内不同样本也保持多样性
            # 非对角线元素不应过大（避免特征坍缩）
            id_self_corr = id_gram[mask].abs().mean()
            cloth_self_corr = cloth_gram[mask].abs().mean()
            
            # 期望自相关在合理范围内（不要太大，否则特征相似；不要太小，否则过度分散）
            self_corr_penalty = torch.clamp(id_self_corr - 0.5, min=0.0) + \
                               torch.clamp(cloth_self_corr - 0.5, min=0.0)
        else:
            cross_correlation = torch.tensor(0.0, device=id_embeds.device)
            self_corr_penalty = torch.tensor(0.0, device=id_embeds.device)
        
        # 总损失: 加权组合三个部分
        # 主要约束是批次内正交(1.0)，跨样本交叉相关次之(0.1)，自相关惩罚最小(0.05)
        total_loss = ortho_loss_batch + 0.1 * cross_correlation + 0.05 * self_corr_penalty
        
        return total_loss
    
    def opa_alignment_loss(self, id_seq_features, cloth_seq_features):
        """
        OPA 对齐损失（G-S3 专用）
        确保 OPA 输出的身份和服装序列特征正交
        """
        if id_seq_features is None or cloth_seq_features is None:
            return torch.tensor(0.0, device=self.ce_loss.weight.device)
        
        id_norm = F.normalize(id_seq_features, dim=-1)
        cloth_norm = F.normalize(cloth_seq_features, dim=-1)
        cosine_sim = (id_norm * cloth_norm).sum(dim=-1)
        
        return cosine_sim.abs().mean()
    
    def mamba_filter_quality_loss(self, filtered_features, saliency_score):
        """
        Mamba 过滤质量损失（G-S3 专用）
        确保高显著性区域的特征被有效抑制
        """
        if filtered_features is None or saliency_score is None:
            return torch.tensor(0.0, device=self.ce_loss.weight.device)
        
        feature_strength = filtered_features.norm(dim=-1, keepdim=True)
        suppression_loss = (feature_strength * saliency_score).mean()
        
        return suppression_loss

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids, is_matched=None, epoch=None, gate=None,
                id_seq_features=None, cloth_seq_features=None, saliency_score=None):
        
        losses = {}
        
        # 核心损失
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds) if image_embeds is not None and id_text_embeds is not None else torch.tensor(0.0, device=self.ce_loss.weight.device)
        losses['cls'] = self.id_classification_loss(id_logits, pids) if id_logits is not None and pids is not None else torch.tensor(0.0, device=self.ce_loss.weight.device)
        
        # 服装语义损失（改进版，添加去ID正则）
        # 注意: cloth_image_embeds是投影后的256维，需要使用同样投影后的image_embeds
        losses['cloth_semantic'] = self.cloth_semantic_loss(cloth_image_embeds, cloth_text_embeds, image_embeds)
        
        # 正交约束损失（增强版）
        losses['orthogonal'] = self.orthogonal_loss(id_embeds, cloth_embeds)
        
        # 自适应门控正则（修复版）
        losses['gate_adaptive'] = self.gate_adaptive_loss(gate, id_embeds, cloth_embeds)
        
        # G-S3 专用损失（可选）
        if 'opa_alignment' in self.weights and id_seq_features is not None and cloth_seq_features is not None:
            losses['opa_alignment'] = self.opa_alignment_loss(id_seq_features, cloth_seq_features)
        
        if 'mamba_quality' in self.weights and id_seq_features is not None and saliency_score is not None:
            losses['mamba_quality'] = self.mamba_filter_quality_loss(id_seq_features, saliency_score)
        
        # 检查NaN/Inf并替换为0（避免训练崩溃）
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    losses[key] = torch.tensor(0.0, device=value.device, requires_grad=True)
        
        # 简单加权求和
        total_loss = sum(self.weights.get(k, 0) * losses[k] for k in losses.keys() if k != 'total')
        
        # 最终检查
        if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
            total_loss = torch.tensor(0.0, device=total_loss.device, requires_grad=True)
        
        losses['total'] = total_loss
        
        # 调试模式：记录损失梯度和数值稳定性
        if hasattr(self, '_debug_mode') and self._debug_mode:
            self._debug_loss_info = {
                'loss_values': {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)},
                'loss_requires_grad': {k: v.requires_grad for k, v in losses.items() if isinstance(v, torch.Tensor)},
                'has_nan': any(torch.isnan(v).any() for v in losses.values() if isinstance(v, torch.Tensor)),
                'has_inf': any(torch.isinf(v).any() for v in losses.values() if isinstance(v, torch.Tensor))
            }
        
        return losses
