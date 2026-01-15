import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, temperature=0.1, weights=None):
        super().__init__()
        self.temperature = temperature
        # 使用Label Smoothing降低分类损失的初始值
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 优化后的损失权重：大幅降低cls权重，避免其主导总损失
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,        # 提高对比学习权重
            'cls': 0.1,             # 大幅降低分类权重（从0.5降到0.1）
            'cloth_semantic': 0.5,  # 提高服装语义权重
            'orthogonal': 0.1,
            'gate_adaptive': 0.01,
        }

    def gate_adaptive_loss(self, gate, id_embeds, cloth_embeds):
        """自适应门控正则化：根据特征质量动态调整目标"""
        if gate is None or id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=self.ce_loss.weight.device if self.ce_loss.weight is not None else 'cuda')
        
        # 计算特征质量（方差作为判别性的代理指标）
        batch_size = id_embeds.size(0)
        if batch_size < 2:
            # 批次太小，使用固定目标
            target_gate = torch.tensor(0.5, device=id_embeds.device, dtype=id_embeds.dtype)
        else:
            id_quality = torch.clamp(id_embeds.var(dim=0, unbiased=False).mean().detach(), min=0.0, max=10.0)
            cloth_quality = torch.clamp(cloth_embeds.var(dim=0, unbiased=False).mean().detach(), min=0.0, max=10.0)
            total_quality = id_quality + cloth_quality + 1e-6
            target_gate = id_quality / total_quality
        
        # MSE 损失
        gate_mean = gate.mean(dim=-1) if gate.dim() > 1 else gate
        if gate_mean.dim() == 0:
            gate_mean = gate_mean.unsqueeze(0)
        
        # 确保target_gate和gate_mean形状一致
        if target_gate.dim() == 0:
            target_gate = target_gate.expand_as(gate_mean)
        
        mse_loss = F.mse_loss(gate_mean, target_gate)
        
        # 熵正则：避免门控过于极端（0或1）
        # H = -p*log(p) - (1-p)*log(1-p)
        gate_clamp = torch.clamp(gate_mean, min=1e-6, max=1-1e-6)
        entropy = -(gate_clamp * torch.log(gate_clamp) + (1 - gate_clamp) * torch.log(1 - gate_clamp))
        entropy_reg = -entropy.mean()  # 最大化熵（负号）
        
        # 限制总损失幅度，防止负值
        total_loss = torch.clamp(mse_loss + 0.1 * entropy_reg, min=0.0, max=1.0)
        
        return total_loss

    def info_nce_loss(self, image_embeds, text_embeds):
        bsz = image_embeds.size(0)
        image_embeds = F.normalize(image_embeds, dim=-1, eps=1e-6)
        text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-6)
        sim = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        
        # 防止数值溢出
        sim = torch.clamp(sim, min=-20, max=20)
        
        labels = torch.arange(bsz, device=sim.device)
        return (self.ce_loss(sim, labels) + self.ce_loss(sim.t(), labels)) / 2

    def id_classification_loss(self, id_logits, pids):
        return self.ce_loss(id_logits, pids)

    def cloth_semantic_loss(self, cloth_image_embeds, cloth_text_embeds):
        """服装语义损失：合并 cloth 和 cloth_match，避免冗余"""
        if cloth_image_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=self.ce_loss.weight.device if self.ce_loss.weight is not None else 'cuda')
        
        bsz = cloth_image_embeds.size(0)
        cloth_image_embeds = F.normalize(cloth_image_embeds, dim=-1, eps=1e-6)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1, eps=1e-6)
        
        sim = torch.matmul(cloth_image_embeds, cloth_text_embeds.t()) / self.temperature
        sim = torch.clamp(sim, min=-20, max=20)
        
        labels = torch.arange(bsz, device=sim.device)
        
        return (self.ce_loss(sim, labels) + self.ce_loss(sim.t(), labels)) / 2

    def orthogonal_loss(self, id_embeds, cloth_embeds):
        """正交约束损失：直接约束特征向量正交，比 Gram 矩阵更高效"""
        if id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=self.ce_loss.weight.device if self.ce_loss.weight is not None else 'cuda')
        
        id_norm = F.normalize(id_embeds, dim=-1, eps=1e-6)
        cloth_norm = F.normalize(cloth_embeds, dim=-1, eps=1e-6)
        cosine_sim = (id_norm * cloth_norm).sum(dim=-1)
        
        # 裁剪余弦相似度，防止梯度爆炸
        cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
        
        return cosine_sim.abs().mean()
    
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
        
        # 服装语义损失（合并版）
        losses['cloth_semantic'] = self.cloth_semantic_loss(cloth_image_embeds, cloth_text_embeds)
        
        # 正交约束损失（改进版）
        losses['orthogonal'] = self.orthogonal_loss(id_embeds, cloth_embeds)
        
        # 自适应门控正则
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
