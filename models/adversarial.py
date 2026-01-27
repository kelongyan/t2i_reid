# models/adversarial.py
"""
Adversarial Decoupling Module
对抗式解耦：通过判别器强制ID特征无法预测服装属性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    梯度反转层 (Gradient Reversal Layer)
    
    前向传播：y = x
    反向传播：dy/dx = -lambda * grad_output
    
    用于对抗训练，让特征提取器"欺骗"判别器
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """梯度反转层包装器"""
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        """动态调整反转强度"""
        self.lambda_ = lambda_


class AttributeDiscriminator(nn.Module):
    """
    属性判别器
    
    目标：判断特征中是否包含服装属性信息
    
    训练策略：
    - Discriminator Loss: 最大化分类准确率（让判别器学会识别属性）
    - Feature Extractor Loss: 最小化分类准确率（通过GRL让特征无法被识别）
    
    Args:
        dim: 输入特征维度
        num_attributes: 属性类别数（动态计算，或使用虚拟标签）
        hidden_dims: 隐藏层维度列表
        dropout: Dropout比例
    """
    def __init__(self, dim=768, num_attributes=128, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        self.dim = dim
        self.num_attributes = num_attributes
        
        # 梯度反转层
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # 判别器网络 (Multi-layer MLP)
        layers = []
        in_dim = dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, num_attributes))
        
        self.discriminator = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """🔥 改进的权重初始化，降低初始损失"""
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                # 🔥 使用更小的gain，降低判别器初始能力
                # 让对抗训练从更平衡的状态开始
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features, reverse_grad=True):
        """
        Args:
            features: [B, D] 输入特征（通常是ID特征）
            reverse_grad: 是否反转梯度（训练特征提取器时True，训练判别器时False）
        
        Returns:
            logits: [B, num_attributes] 属性分类logits
        """
        if reverse_grad:
            features = self.grl(features)
        
        logits = self.discriminator(features)
        return logits
    
    def set_lambda(self, lambda_):
        """动态调整梯度反转强度"""
        self.grl.set_lambda(lambda_)


class DomainDiscriminator(nn.Module):
    """
    域判别器 (可选)
    
    判断特征来自ID分支还是Attr分支
    用于强制两个分支学习不同的表征
    """
    def __init__(self, dim=768, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        self.discriminator = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2)  # Binary: ID or Attr
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features, reverse_grad=True):
        """
        Args:
            features: [B, D]
            reverse_grad: True表示训练特征提取器，False表示训练判别器
        
        Returns:
            logits: [B, 2] (0=ID分支, 1=Attr分支)
        """
        if reverse_grad:
            features = self.grl(features)
        
        logits = self.discriminator(features)
        return logits
    
    def set_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)


def compute_attribute_pseudo_labels(cloth_embeds, num_clusters=128):
    """
    🔥 改进的伪标签生成方法

    改进：
    1. 使用多个维度的加权组合（而非简单哈希）
    2. 添加随机扰动，避免伪标签过于固定
    3. 确保每个batch有足够的类别多样性

    Args:
        cloth_embeds: [B, D] 服装特征
        num_clusters: 聚类数量（伪属性类别数）

    Returns:
        pseudo_labels: [B] 伪标签
    """
    with torch.no_grad():
        # 归一化
        cloth_embeds_norm = F.normalize(cloth_embeds, dim=-1, eps=1e-8)

        # 🔥 改进1：使用更多维度，增加多样性
        n_dims = min(16, cloth_embeds_norm.shape[1])  # 使用前16个维度

        # 🔥 改进2：加权组合，而非简单的二进制
        # 使用质数作为权重，减少碰撞
        weights = torch.tensor([
            1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
        ], device=cloth_embeds.device)[:n_dims]

        # 将特征值离散化为-1, 0, 1三个级别
        discretized = torch.sign(cloth_embeds_norm[:, :n_dims])

        # 加权求和
        pseudo_labels = (discretized * weights).sum(dim=1)

        # 🔥 改进3：取模确保在有效范围内
        pseudo_labels = pseudo_labels % num_clusters

        # 🔥 修复：提前转换为 long 类型，避免后续类型不匹配
        pseudo_labels = pseudo_labels.long()

        # 🔥 改进4：添加微小的随机扰动，增加训练动态性
        # 仅在训练时添加（使用0.01的概率翻转5%的标签）
        if torch.rand(1).item() < 0.01:
            flip_mask = torch.rand(pseudo_labels.shape[0], device=pseudo_labels.device) < 0.05
            if flip_mask.any():
                # 随机翻转标签
                pseudo_labels[flip_mask] = torch.randint(
                    0, num_clusters, (flip_mask.sum().item(),), device=pseudo_labels.device
                )

    return pseudo_labels  # 已经是 long 类型


class AdversarialDecoupler(nn.Module):
    """
    对抗式解耦模块（整合）
    
    包含：
    1. Attribute Discriminator: 强制ID特征不包含服装信息
    2. Domain Discriminator (可选): 强制ID/Attr特征来自不同分布
    """
    def __init__(self, dim=768, num_attributes=128, use_domain_disc=False, logger=None):
        super().__init__()
        self.logger = logger
        self.use_domain_disc = use_domain_disc
        
        # 属性判别器
        self.attr_disc = AttributeDiscriminator(
            dim=dim, 
            num_attributes=num_attributes,
            hidden_dims=[512, 256],
            dropout=0.3
        )
        
        # 域判别器（可选）
        if use_domain_disc:
            self.domain_disc = DomainDiscriminator(dim=dim, hidden_dim=512, dropout=0.3)
        
        # 🔥 梯度反转强度调度器（更平缓的增长曲线）
        # 从0.0缓慢增长到1.0，避免早期对抗过强
        # 使用sigmoid函数，在训练中期达到0.5
        self.lambda_schedule = lambda p: 1.0 / (1.0 + torch.exp(torch.tensor(-5.0 * (p - 0.5))))
    
    def update_lambda(self, progress):
        """
        更新梯度反转强度
        
        Args:
            progress: 训练进度 [0, 1]
        """
        lambda_ = self.lambda_schedule(progress).item()
        self.attr_disc.set_lambda(lambda_)
        if self.use_domain_disc:
            self.domain_disc.set_lambda(lambda_)
        
        if self.logger and hasattr(self, '_log_counter'):
            self._log_counter = getattr(self, '_log_counter', 0) + 1
            if self._log_counter % 500 == 0:
                self.logger.debug_logger.debug(f"[Adversarial] Lambda updated: {lambda_:.4f}")
    
    def forward(self, id_feat, cloth_feat, training_phase='feature'):
        """
        Args:
            id_feat: [B, D] ID特征
            cloth_feat: [B, D] 服装特征
            training_phase: 'feature' or 'discriminator'
        
        Returns:
            losses: dict of adversarial losses
        """
        losses = {}
        
        # 生成服装伪标签
        pseudo_labels = compute_attribute_pseudo_labels(cloth_feat, num_clusters=self.attr_disc.num_attributes)
        
        # 1. 属性判别器损失
        if training_phase == 'feature':
            # 训练特征提取器：让ID特征"欺骗"判别器（梯度反转）
            attr_logits = self.attr_disc(id_feat, reverse_grad=True)
            # 交叉熵损失（但梯度被反转）
            loss_attr_adv = F.cross_entropy(attr_logits, pseudo_labels)
            losses['adversarial_attr'] = loss_attr_adv
        else:
            # 训练判别器：让判别器正确预测服装属性（无梯度反转）
            attr_logits = self.attr_disc(cloth_feat, reverse_grad=False)
            loss_attr_disc = F.cross_entropy(attr_logits, pseudo_labels)
            losses['discriminator_attr'] = loss_attr_disc
        
        # 2. 域判别器损失（可选）
        if self.use_domain_disc:
            if training_phase == 'feature':
                # 让判别器无法区分ID/Attr特征
                domain_logits_id = self.domain_disc(id_feat, reverse_grad=True)
                domain_logits_attr = self.domain_disc(cloth_feat, reverse_grad=True)
                
                # 目标：让判别器输出接近0.5（无法判断）
                domain_labels = torch.cat([
                    torch.zeros(id_feat.size(0), dtype=torch.long, device=id_feat.device),
                    torch.ones(cloth_feat.size(0), dtype=torch.long, device=cloth_feat.device)
                ])
                domain_logits = torch.cat([domain_logits_id, domain_logits_attr], dim=0)
                loss_domain_adv = F.cross_entropy(domain_logits, domain_labels)
                losses['adversarial_domain'] = loss_domain_adv
            else:
                # 训练判别器：正确区分ID/Attr特征
                domain_logits_id = self.domain_disc(id_feat, reverse_grad=False)
                domain_logits_attr = self.domain_disc(cloth_feat, reverse_grad=False)
                
                domain_labels = torch.cat([
                    torch.zeros(id_feat.size(0), dtype=torch.long, device=id_feat.device),
                    torch.ones(cloth_feat.size(0), dtype=torch.long, device=cloth_feat.device)
                ])
                domain_logits = torch.cat([domain_logits_id, domain_logits_attr], dim=0)
                loss_domain_disc = F.cross_entropy(domain_logits, domain_labels)
                losses['discriminator_domain'] = loss_domain_disc
        
        return losses
