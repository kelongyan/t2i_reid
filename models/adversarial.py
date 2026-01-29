# models/adversarial.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    # 梯度反转函数：在前向传播时保持输入不变，在反向传播时将梯度乘以负的缩放因子
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    # 梯度反转层封装类，用于在网络中方便地调用梯度反转功能
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        # 动态调整梯度反转的缩放强度
        self.lambda_ = lambda_


class AttributeDiscriminator(nn.Module):
    # 属性判别器：通过对抗训练强制身份特征（ID Feature）无法预测服装属性，实现特征解耦
    def __init__(self, dim=768, num_attributes=128, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        self.dim = dim
        self.num_attributes = num_attributes
        
        # 梯度反转层
        self.grl = GradientReversalLayer(lambda_=1.0)
        
        # 判别器网络：多层 MLP 结构
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
        
        # 输出层：分类到指定的属性类别数
        layers.append(nn.Linear(in_dim, num_attributes))
        
        self.discriminator = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        # 权重初始化：使用较小的 gain 降低判别器初始强度，使对抗训练更平稳
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features, reverse_grad=True):
        # features: 输入特征 [B, D]
        # reverse_grad: 是否开启梯度反转（训练特征提取器时设为 True，训练判别器自身时设为 False）
        if reverse_grad:
            features = self.grl(features)
        
        logits = self.discriminator(features)
        return logits
    
    def set_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)


class DomainDiscriminator(nn.Module):
    # 域判别器：判断特征来自 ID 分支还是属性分支，强制两分支学习互斥的特征表征
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
            nn.Linear(256, 2)  # 二分类：ID 分支或属性分支
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.discriminator.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features, reverse_grad=True):
        if reverse_grad:
            features = self.grl(features)
        
        logits = self.discriminator(features)
        return logits
    
    def set_lambda(self, lambda_):
        self.grl.set_lambda(lambda_)


def compute_attribute_pseudo_labels(cloth_embeds, num_clusters=128):
    # 为服装特征生成伪标签，用于对抗性属性判别器的监督信号
    with torch.no_grad():
        # L2 归一化
        cloth_embeds_norm = F.normalize(cloth_embeds, dim=-1, eps=1e-8)

        # 使用部分维度组合生成伪标签，增加多样性
        n_dims = min(16, cloth_embeds_norm.shape[1])

        # 使用质数权重进行加权组合，减少标签碰撞
        weights = torch.tensor([
            1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53
        ], device=cloth_embeds.device)[:n_dims]

        # 离散化特征值并计算加权和
        discretized = torch.sign(cloth_embeds_norm[:, :n_dims])
        pseudo_labels = (discretized * weights).sum(dim=1)

        # 取模确保标签在指定范围内，并转换为长整型
        pseudo_labels = pseudo_labels % num_clusters
        pseudo_labels = pseudo_labels.long()

        # 训练时以极小概率添加随机扰动，提升模型鲁棒性
        if torch.rand(1).item() < 0.01:
            flip_mask = torch.rand(pseudo_labels.shape[0], device=pseudo_labels.device) < 0.05
            if flip_mask.any():
                pseudo_labels[flip_mask] = torch.randint(
                    0, num_clusters, (flip_mask.sum().item(),), device=pseudo_labels.device
                )

    return pseudo_labels


class AdversarialDecoupler(nn.Module):
    # 对抗式解耦集成模块：管理属性判别器、域判别器以及梯度反转强度的动态调度
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
        
        # 域判别器
        if use_domain_disc:
            self.domain_disc = DomainDiscriminator(dim=dim, hidden_dim=512, dropout=0.3)
        
        # 梯度反转强度调度：随训练进度平滑增长
        self.lambda_schedule = lambda p: 1.0 / (1.0 + torch.exp(torch.tensor(-5.0 * (p - 0.5))))
    
    def update_lambda(self, progress):
        # 根据训练进度 [0, 1] 更新判别器的梯度反转强度
        lambda_ = self.lambda_schedule(progress).item()
        self.attr_disc.set_lambda(lambda_)
        if self.use_domain_disc:
            self.domain_disc.set_lambda(lambda_)
        
        if self.logger and hasattr(self, '_log_counter'):
            self._log_counter = getattr(self, '_log_counter', 0) + 1
            if self._log_counter % 500 == 0:
                self.logger.debug_logger.debug(f"[Adversarial] Lambda updated: {lambda_:.4f}")
    
    def forward(self, id_feat, cloth_feat, training_phase='feature'):
        # id_feat: 身份特征, cloth_feat: 服装特征
        # training_phase: 'feature' (训练提取器以解耦) 或 'discriminator' (训练判别器以提高辨别力)
        losses = {}
        
        # 为服装特征生成伪属性标签
        pseudo_labels = compute_attribute_pseudo_labels(cloth_feat, num_clusters=self.attr_disc.num_attributes)
        
        # 1. 属性判别器损失计算
        if training_phase == 'feature':
            # 开启梯度反转，计算让 ID 特征无法识别属性的损失
            attr_logits = self.attr_disc(id_feat, reverse_grad=True)
            loss_attr_adv = F.cross_entropy(attr_logits, pseudo_labels)
            losses['adversarial_attr'] = loss_attr_adv
        else:
            # 正常训练判别器，识别服装特征本身的属性
            attr_logits = self.attr_disc(cloth_feat, reverse_grad=False)
            loss_attr_disc = F.cross_entropy(attr_logits, pseudo_labels)
            losses['discriminator_attr'] = loss_attr_disc
        
        # 2. 域判别器损失计算（如果启用）
        if self.use_domain_disc:
            if training_phase == 'feature':
                # 强制 ID 和属性特征分布趋于一致
                domain_logits_id = self.domain_disc(id_feat, reverse_grad=True)
                domain_logits_attr = self.domain_disc(cloth_feat, reverse_grad=True)
                
                domain_labels = torch.cat([
                    torch.zeros(id_feat.size(0), dtype=torch.long, device=id_feat.device),
                    torch.ones(cloth_feat.size(0), dtype=torch.long, device=cloth_feat.device)
                ])
                domain_logits = torch.cat([domain_logits_id, domain_logits_attr], dim=0)
                loss_domain_adv = F.cross_entropy(domain_logits, domain_labels)
                losses['adversarial_domain'] = loss_domain_adv
            else:
                # 训练判别器区分 ID 特征和属性特征
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
