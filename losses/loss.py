import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.loss_logger import LossLogger


class HardNegativeTripletLoss(nn.Module):
    """
    🔥 Cosine Triplet Loss with Angular Margin（修复版）

    修复问题：
    1. 使用余弦相似度替代欧氏距离（适配归一化特征）
    2. 添加角度margin，更符合ReID任务
    3. 移除temperature scaling（对余弦相似度不需要）
    4. 改进hard negative mining策略

    Args:
        margin: 角度margin（弧度），默认0.3（约17度）
        hard_mining: 是否使用hard mining
        hard_ratio: Hard样本比例（0-1）
    """
    def __init__(self, margin=0.3, hard_mining=True, hard_ratio=0.5):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.hard_ratio = hard_ratio

    def forward(self, embeds, pids):
        """
        Args:
            embeds: [B, D] L2归一化后的特征向量
            pids: [B] 身份标签

        Returns:
            loss: Scalar tensor
        """
        if embeds is None or pids is None:
            return torch.tensor(0.0, device='cuda')

        # NaN检测
        if torch.isnan(embeds).any():
            return torch.tensor(0.0, device=embeds.device)

        # 确保特征已归一化
        embeds = F.normalize(embeds, p=2, dim=1, eps=1e-8)
        n = embeds.size(0)

        # 计算余弦相似度矩阵 [B, B]
        # sim[i, j] = cos(embeds[i], embeds[j])
        sim_matrix = torch.mm(embeds, embeds.t())

        # 构建mask
        mask = pids.expand(n, n).eq(pids.expand(n, n).t())

        # 为每个样本找hard positive和hard negative
        sim_ap = []  # positive similarity（应该大）
        sim_an = []  # negative similarity（应该小）

        for i in range(n):
            # 正样本：排除自己
            pos_mask = mask[i].clone()
            pos_mask[i] = False

            if pos_mask.sum() > 0:
                # Hard positive: 选择相似度最小的正样本（最难的正样本）
                if self.hard_mining:
                    sim_ap_i = torch.min(sim_matrix[i][pos_mask])
                else:
                    sim_ap_i = sim_matrix[i][pos_mask].mean()
                sim_ap.append(sim_ap_i)
            else:
                # 如果没有正样本，使用1.0（完美匹配）
                sim_ap.append(torch.tensor(1.0, device=embeds.device))

            # 负样本
            neg_mask = ~mask[i]
            if neg_mask.sum() > 0:
                # Hard negative: 选择相似度最大的负样本（最难的负样本）
                if self.hard_mining:
                    if self.hard_ratio < 1.0:
                        k = max(1, int(neg_mask.sum() * self.hard_ratio))
                        sim_an_i, _ = torch.topk(sim_matrix[i][neg_mask], k, largest=True)
                        sim_an.append(sim_an_i.mean())
                    else:
                        sim_an_i = torch.max(sim_matrix[i][neg_mask])
                        sim_an.append(sim_an_i)
                else:
                    sim_an.append(sim_matrix[i][neg_mask].mean())
            else:
                # 如果没有负样本，使用0.0（完全不匹配）
                sim_an.append(torch.tensor(0.0, device=embeds.device))

        sim_ap = torch.stack(sim_ap)
        sim_an = torch.stack(sim_an)

        # 🔥 Cosine Triplet Loss
        # loss = ReLU(cos(an) - cos(ap) + margin)
        # 当 cos(an) - cos(ap) + margin > 0 时有损失
        # 目标：让 cos(ap) 尽可能大，cos(an) 尽可能小
        loss = F.relu(sim_an - sim_ap + self.margin).mean()

        # NaN检测
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return torch.tensor(0.0, device=embeds.device)

        return loss


class ReconstructionLoss(nn.Module):
    """
    AH-Net 特征重构损失（改进版）
    目标：利用 Attr 分支（和被阻断梯度的 ID 分支）重构原始特征，强迫 Attr 分支捕捉纹理细节。
    
    改进：
    1. 使用 Cosine Embedding Loss 替代 MSE，更适合归一化特征
    2. 添加 L1 正则化，鼓励稀疏重构
    """
    def __init__(self, use_cosine=True, l1_weight=0.01):
        super().__init__()
        self.use_cosine = use_cosine
        self.l1_weight = l1_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, recon_feat, target_feat):
        """
        Args:
            recon_feat: [B, D] 解码器输出
            target_feat: [B, D] 目标特征 (原始特征均值)
        """
        if self.use_cosine:
            # 使用余弦相似度损失（更适合训练初期）
            recon_norm = F.normalize(recon_feat, dim=-1, eps=1e-8)
            target_norm = F.normalize(target_feat, dim=-1, eps=1e-8)
            # 1 - cosine_similarity
            cosine_loss = 1.0 - (recon_norm * target_norm).sum(dim=-1).mean()
            
            # L1 正则化
            l1_reg = torch.abs(recon_feat).mean()
            
            return cosine_loss + self.l1_weight * l1_reg
        else:
            return self.mse_loss(recon_feat, target_feat)


class SpatialOrthogonalLoss(nn.Module):
    """
    🔥 改进的空间互斥损失 (Spatial Orthogonal Loss)
    目标：最小化 ID Attention Map 和 Attr Attention Map 的空间重叠

    改进：
    1. 添加温度参数防止注意力饱和
    2. 使用KL散度增强惩罚
    3. 添加归一化防止数值不稳定
    4. 🔥 添加NaN检测和数值范围限制
    5. 🔥 优化temperature参数（从2.0改为5.0）
    """
    def __init__(self, temperature=5.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, map_id, map_attr):
        """
        Args:
            map_id: [B, 1, H, W]
            map_attr: [B, 1, H, W]
        """
        # 🔥 添加输入NaN检测
        if torch.isnan(map_id).any() or torch.isnan(map_attr).any():
            return torch.tensor(0.0, device=map_id.device, requires_grad=True)
        
        # 🔥 改进1: 添加温度缩放，防止注意力图过于尖锐
        map_id_temp = map_id / self.temperature
        map_attr_temp = map_attr / self.temperature
        
        # 🔥 添加数值范围限制，防止数值不稳定
        map_id_temp = torch.clamp(map_id_temp, min=-10, max=10)
        map_attr_temp = torch.clamp(map_attr_temp, min=-10, max=10)

        # 重新归一化
        map_id_temp_flat = map_id_temp.reshape(map_id_temp.shape[0], -1)
        map_attr_temp_flat = map_attr_temp.reshape(map_attr_temp.shape[0], -1)
        
        # 🔥 使用稳定的softmax实现
        map_id_temp = F.softmax(map_id_temp_flat, dim=-1)
        map_id_temp = map_id_temp.reshape_as(map_id)
        
        map_attr_temp = F.softmax(map_attr_temp_flat, dim=-1)
        map_attr_temp = map_attr_temp.reshape_as(map_attr)

        # 🔥 改进2: 计算KL散度（衡量分布差异）
        # 使用较小的epsilon防止log(0)
        eps = 1e-8
        
        # 🔥 防止除零和log(0)
        ratio = torch.clamp(map_id_temp / (map_attr_temp + eps), min=eps, max=1.0/eps)
        log_ratio = torch.log(ratio)
        
        kl_div = map_id_temp * log_ratio

        # 🔥 改进3: 同时计算直接重叠作为辅助
        overlap = map_id_temp * map_attr_temp

        # 组合损失：KL散度 + 直接重叠
        loss_kl = kl_div.sum(dim=(2, 3)).mean()
        loss_overlap = overlap.sum(dim=(2, 3)).mean()
        
        # 🔥 添加最终NaN检测
        if torch.isnan(loss_kl).any() or torch.isnan(loss_overlap).any():
            return torch.tensor(0.0, device=map_id.device, requires_grad=True)
        
        return loss_kl + 0.5 * loss_overlap


class Loss(nn.Module):
    """
    Complete Loss Module with Curriculum Learning Support

    包含：
    - InfoNCE (主任务)
    - Hard Negative Triplet (身份一致性)
    - Cloth Semantic (属性对齐)
    - Reconstruction (结构重构) - 已移除
    - Spatial Orthogonal (空间互斥)
    - Semantic Alignment (语义对齐)
    - Adversarial Losses (对抗式解耦)

    🔥 优化后：
    1. temperature参数增大，防止梯度爆炸
    2. 添加损失缩放，平衡各损失值范围
    """

    def __init__(self, temperature=0.07, weights=None, num_classes=None, logger=None,
                 semantic_guidance=None, adversarial_decoupler=None):
        """
        Args:
            temperature: 对比学习温度参数（修复：0.2→0.07，更标准的值）
            weights: 损失权重字典
            semantic_guidance: SemanticGuidedDecoupling 模块
            adversarial_decoupler: AdversarialDecoupler 模块（新增）
        """
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.logger = logger
        self.semantic_guidance = semantic_guidance
        self.adversarial_decoupler = adversarial_decoupler

        # Label Smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # === 核心损失模块 ===
        # 🔥 修复：使用余弦Triplet Loss
        self._hard_triplet = HardNegativeTripletLoss(
            margin=0.3,  # 角度margin（弧度）
            hard_mining=True,
            hard_ratio=0.5
        )
        # 🔥 优化temperature参数：2.0→5.0
        self._ortho_loss = SpatialOrthogonalLoss(temperature=5.0)

        # === 初始化LossLogger ===
        self.loss_logger = LossLogger(logger.debug_logger) if logger else None

        # === 优化后的权重配置（将由CurriculumScheduler动态更新）===
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,
            'id_triplet': 50.0,  # Phase 1极大权重
            'cloth_semantic': 0.001,
            'spatial_orthogonal': 0.0,
            'semantic_alignment': 0.0,
            'ortho_reg': 0.0,
            'adversarial_attr': 0.0,
            'adversarial_domain': 0.0,
            'discriminator_attr': 0.0,
            'discriminator_domain': 0.0
        }

        self.register_buffer('_dummy', torch.zeros(1))
        self._batch_counter = 0
        if logger: self.debug_logger = logger.debug_logger
    
    def update_weights(self, new_weights):
        """由CurriculumScheduler动态更新权重"""
        self.weights.update(new_weights)
        if self.logger and self._batch_counter % 500 == 0:
            self.logger.debug_logger.debug(f"[Loss] Weights updated: {new_weights}")

    def _get_device(self):
        return self._dummy.device

    def info_nce_loss(self, image_embeds, text_embeds, fused_embeds=None):
        if image_embeds is None or text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        visual_embeds = fused_embeds if fused_embeds is not None else image_embeds
        bsz = visual_embeds.size(0)
        
        # 🔥 添加NaN检测
        if torch.isnan(visual_embeds).any() or torch.isnan(text_embeds).any():
            return torch.tensor(0.0, device=self._get_device())
        
        visual_embeds = F.normalize(visual_embeds, dim=-1, eps=1e-8)
        text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-8)
        sim = torch.matmul(visual_embeds, text_embeds.t()) / self.temperature
        
        # 🔥 限制相似度范围，防止数值不稳定
        sim = torch.clamp(sim, min=-50, max=50)
        
        labels = torch.arange(bsz, device=sim.device, dtype=torch.long)
        loss_i2t = self.ce_loss(sim, labels)
        loss_t2i = self.ce_loss(sim.t(), labels)
        
        # 🔥 添加损失NaN检测
        if torch.isnan(loss_i2t).any() or torch.isnan(loss_t2i).any():
            return torch.tensor(0.0, device=self._get_device())
        
        return (loss_i2t + loss_t2i) / 2

    def cloth_semantic_loss(self, cloth_image_embeds, cloth_text_embeds):
        if cloth_image_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        
        # 🔥 添加NaN检测
        if torch.isnan(cloth_image_embeds).any() or torch.isnan(cloth_text_embeds).any():
            return torch.tensor(0.0, device=self._get_device())
        
        bsz = cloth_image_embeds.size(0)
        cloth_image_norm = F.normalize(cloth_image_embeds, dim=-1, eps=1e-8)
        cloth_text_norm = F.normalize(cloth_text_embeds, dim=-1, eps=1e-8)
        sim = torch.matmul(cloth_image_norm, cloth_text_norm.t()) / self.temperature
        
        # 🔥 限制相似度范围，防止数值不稳定
        sim = torch.clamp(sim, min=-50, max=50)
        
        labels = torch.arange(bsz, device=sim.device, dtype=torch.long)
        loss_img2t = self.ce_loss(sim, labels)
        loss_t2img = self.ce_loss(sim.t(), labels)
        
        # 🔥 添加损失NaN检测
        if torch.isnan(loss_img2t).any() or torch.isnan(loss_t2img).any():
            return torch.tensor(0.0, device=self._get_device())
        
        return (loss_img2t + loss_t2img) / 2

    def triplet_loss(self, embeds, pids):
        """使用Hard Negative Mining Triplet Loss"""
        return self._hard_triplet(embeds, pids)

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids,
                is_matched=None, epoch=None, aux_info=None, training_phase='feature'):
        """
        Compute total loss with Adversarial Training Support.

        Args:
            image_embeds: 图像嵌入 [B, D]
            id_text_embeds: ID文本嵌入 [B, D]
            fused_embeds: 融合嵌入 [B, D]
            id_logits: 分类logits (已废弃,保持兼容性)
            id_embeds: ID特征 [B, D]
            cloth_embeds: 属性特征 [B, D]
            cloth_text_embeds: 属性文本嵌入 [B, D]
            cloth_image_embeds: 属性图像嵌入 [B, D]
            pids: 人员ID标签 [B]
            is_matched: 匹配标签 [B]
            epoch: 当前训练epoch
            aux_info: Auxiliary info from AHNetModule
            training_phase: 'feature' or 'discriminator' (新增)

        🔥 优化后：
        1. 添加损失缩放，平衡各损失值范围
        2. 优化损失权重配置
        """
        losses = {}

        # === 基础损失 ===
        # 🔥 修复：同时计算 Unimodal Contrastive Loss (ITC) 和 Fused Loss
        # 1. Unimodal Alignment (训练 image_mlp 和 text_mlp，用于推理)
        loss_unimodal = self.info_nce_loss(image_embeds, id_text_embeds, fused_embeds=None)

        # 2. Fused Alignment (训练 Fusion 模块)
        if fused_embeds is not None:
            loss_fused = self.info_nce_loss(image_embeds, id_text_embeds, fused_embeds=fused_embeds)
            # 融合损失作为辅助，权重设为 0.5 (可调)
            losses['info_nce'] = loss_unimodal + 0.5 * loss_fused
        else:
            losses['info_nce'] = loss_unimodal

        losses['cloth_semantic'] = self.cloth_semantic_loss(cloth_image_embeds, cloth_text_embeds)
        losses['id_triplet'] = self.triplet_loss(id_embeds, pids)

        # === 语义引导损失 ===
        if self.semantic_guidance is not None and id_embeds is not None and cloth_embeds is not None:
            losses['semantic_alignment'] = self.semantic_guidance(
                id_feat=id_embeds,
                attr_feat=cloth_embeds,
                use_cross_separation=False
            )
        else:
            losses['semantic_alignment'] = torch.tensor(0.0, device=self._get_device())

        # === AH-Net 解耦损失 ===
        if aux_info:
            losses['spatial_orthogonal'] = self._ortho_loss(aux_info['map_id'], aux_info['map_attr'])

            # Query正交性正则化
            if 'ortho_reg' in aux_info:
                losses['ortho_reg'] = aux_info['ortho_reg']
            else:
                losses['ortho_reg'] = torch.tensor(0.0, device=self._get_device())
        else:
            losses['spatial_orthogonal'] = torch.tensor(0.0, device=self._get_device())
            losses['ortho_reg'] = torch.tensor(0.0, device=self._get_device())

        # === 对抗式解耦损失（新增）===
        if self.adversarial_decoupler is not None and id_embeds is not None and cloth_embeds is not None:
            adv_losses = self.adversarial_decoupler(
                id_feat=id_embeds,
                cloth_feat=cloth_embeds,
                training_phase=training_phase
            )
            losses.update(adv_losses)
        else:
            # 如果没有对抗模块，设置为0
            losses['adversarial_attr'] = torch.tensor(0.0, device=self._get_device())
            losses['adversarial_domain'] = torch.tensor(0.0, device=self._get_device())
            losses['discriminator_attr'] = torch.tensor(0.0, device=self._get_device())
            losses['discriminator_domain'] = torch.tensor(0.0, device=self._get_device())

        # === 🔥 修复：移除不必要的损失缩放 ===
        # 原来的除以10操作会导致损失值过小，影响训练
        # 现在通过权重来控制各损失的重要性，不再额外缩放
        # losses['info_nce'] = losses['info_nce'] / 10.0  # 已移除
        # losses['cloth_semantic'] = losses['cloth_semantic'] / 10.0  # 已移除
        # losses['semantic_alignment'] = losses['semantic_alignment'] / 10.0  # 已移除

        # === 计算总损失 ===
        total_loss = torch.tensor(0.0, device=self._get_device())
        for key, value in losses.items():
            if key == 'total':
                continue

            # NaN检测
            if torch.isnan(value).any():
                losses[key] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)

            # 根据训练阶段选择性累加损失
            if training_phase == 'discriminator':
                # 训练判别器时，只累加判别器损失
                if key.startswith('discriminator_'):
                    weight = self.weights.get(key, 0.0)
                    if weight > 0:
                        total_loss += weight * losses[key]
            else:
                # 训练特征提取器时，累加所有非判别器损失
                if not key.startswith('discriminator_'):
                    weight = self.weights.get(key, 0.0)
                    if weight > 0:
                        total_loss += weight * losses[key]

        losses['total'] = total_loss

        # 日志记录
        if self.logger and self.loss_logger and self._batch_counter % 100 == 0:
            self._batch_counter += 1

        return losses