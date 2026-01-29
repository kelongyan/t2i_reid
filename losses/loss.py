import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.loss_logger import LossLogger


class HardNegativeTripletLoss(nn.Module):
    # 带角度边界的余弦三元组损失 (Angular Margin Cosine Triplet Loss)
    # 使用余弦相似度替代欧氏距离，更适配 L2 归一化特征
    def __init__(self, margin=0.3, hard_mining=True, hard_ratio=0.5):
        super().__init__()
        self.margin = margin
        self.hard_mining = hard_mining
        self.hard_ratio = hard_ratio

    def forward(self, embeds, pids):
        # 计算特征间的余弦相似度并执行难样本挖掘
        if embeds is None or pids is None:
            return torch.tensor(0.0, device='cuda')

        if torch.isnan(embeds).any():
            return torch.tensor(0.0, device=embeds.device)

        # 确保特征已进行 L2 归一化
        embeds = F.normalize(embeds, p=2, dim=1, eps=1e-8)
        n = embeds.size(0)

        # 计算余弦相似度矩阵 [B, B]
        sim_matrix = torch.mm(embeds, embeds.t())

        # 基于 PID 构建同类样本掩码
        mask = pids.expand(n, n).eq(pids.expand(n, n).t())

        sim_ap = []  # 正样本对相似度
        sim_an = []  # 负样本对相似度

        for i in range(n):
            # 提取正样本（排除自身）
            pos_mask = mask[i].clone()
            pos_mask[i] = False

            if pos_mask.sum() > 0:
                # 难正样本挖掘：选择相似度最小的正样本
                if self.hard_mining:
                    sim_ap_i = torch.min(sim_matrix[i][pos_mask])
                else:
                    sim_ap_i = sim_matrix[i][pos_mask].mean()
                sim_ap.append(sim_ap_i)
            else:
                sim_ap.append(torch.tensor(1.0, device=embeds.device))

            # 提取负样本
            neg_mask = ~mask[i]
            if neg_mask.sum() > 0:
                # 难负样本挖掘：根据 hard_ratio 选择相似度最大的负样本
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
                sim_an.append(torch.tensor(0.0, device=embeds.device))

        sim_ap = torch.stack(sim_ap)
        sim_an = torch.stack(sim_an)

        # 计算最终三元组损失：使正样本相似度尽可能大，负样本相似度尽可能小
        loss = F.relu(sim_an - sim_ap + self.margin).mean()

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            return torch.tensor(0.0, device=embeds.device)

        return loss


class SpatialOrthogonalLoss(nn.Module):
    # 空间互斥损失：最小化 ID 注意力图与 Attr 注意力图在空间上的重叠
    def __init__(self, temperature=5.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, map_id, map_attr):
        # 计算空间分布的 KL 散度及直接重叠度
        if torch.isnan(map_id).any() or torch.isnan(map_attr).any():
            return torch.tensor(0.0, device=map_id.device, requires_grad=True)
        
        # 使用温度缩放平滑注意力分布
        map_id_temp = map_id / self.temperature
        map_attr_temp = map_attr / self.temperature
        
        # 数值截断防止溢出
        map_id_temp = torch.clamp(map_id_temp, min=-10, max=10)
        map_attr_temp = torch.clamp(map_attr_temp, min=-10, max=10)

        # 转换为概率分布
        map_id_temp_flat = map_id_temp.reshape(map_id_temp.shape[0], -1)
        map_attr_temp_flat = map_attr_temp.reshape(map_attr_temp.shape[0], -1)
        
        map_id_temp = F.softmax(map_id_temp_flat, dim=-1).reshape_as(map_id)
        map_attr_temp = F.softmax(map_attr_temp_flat, dim=-1).reshape_as(map_attr)

        # 计算 KL 散度衡量空间分布差异
        eps = 1e-8
        ratio = torch.clamp(map_id_temp / (map_attr_temp + eps), min=eps, max=1.0/eps)
        kl_div = map_id_temp * torch.log(ratio)

        # 计算直接点乘重叠作为辅助惩罚
        overlap = map_id_temp * map_attr_temp

        loss_kl = kl_div.sum(dim=(2, 3)).mean()
        loss_overlap = overlap.sum(dim=(2, 3)).mean()
        
        if torch.isnan(loss_kl).any() or torch.isnan(loss_overlap).any():
            return torch.tensor(0.0, device=map_id.device, requires_grad=True)
        
        return loss_kl + 0.5 * loss_overlap


class Loss(nn.Module):
    # 综合损失管理模块，支持课程学习下的动态权重更新
    def __init__(self, temperature=0.07, weights=None, num_classes=None, logger=None,
                 semantic_guidance=None, adversarial_decoupler=None, base_lr=None):
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.logger = logger
        self.semantic_guidance = semantic_guidance
        self.adversarial_decoupler = adversarial_decoupler
        self.base_lr = base_lr

        # 使用带标签平滑的交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 初始化核心损失子模块
        self._hard_triplet = HardNegativeTripletLoss(margin=0.3, hard_mining=True, hard_ratio=0.5)
        self._ortho_loss = SpatialOrthogonalLoss(temperature=5.0)

        self.loss_logger = LossLogger(logger.debug_logger) if logger else None
        
        # 默认损失权重配置（初始阶段侧重 ID 一致性）
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,
            'id_triplet': 50.0,
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
        # 响应 CurriculumScheduler 的权重更新指令
        self.weights.update(new_weights)
        if self.logger and self._batch_counter % 500 == 0:
            self.logger.debug_logger.debug(f"[Loss] Weights updated: {new_weights}")

    def _get_device(self):
        return self._dummy.device

    def info_nce_loss(self, image_embeds, text_embeds, fused_embeds=None):
        # 计算跨模态 InfoNCE 对比损失（ITC）
        if image_embeds is None or text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        
        visual_embeds = fused_embeds if fused_embeds is not None else image_embeds
        if torch.isnan(visual_embeds).any() or torch.isnan(text_embeds).any():
            return torch.tensor(0.0, device=self._get_device())
        
        visual_embeds = F.normalize(visual_embeds, dim=-1, eps=1e-8)
        text_embeds = F.normalize(text_embeds, dim=-1, eps=1e-8)
        sim = torch.matmul(visual_embeds, text_embeds.t()) / self.temperature
        
        sim = torch.clamp(sim, min=-50, max=50)
        labels = torch.arange(visual_embeds.size(0), device=sim.device, dtype=torch.long)
        
        loss_i2t = self.ce_loss(sim, labels)
        loss_t2i = self.ce_loss(sim.t(), labels)
        
        if torch.isnan(loss_i2t).any() or torch.isnan(loss_t2i).any():
            return torch.tensor(0.0, device=self._get_device())
        
        return (loss_i2t + loss_t2i) / 2

    def cloth_semantic_loss(self, cloth_image_embeds, cloth_text_embeds):
        # 计算属性（衣服）级别的语义对齐损失
        if cloth_image_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        
        if torch.isnan(cloth_image_embeds).any() or torch.isnan(cloth_text_embeds).any():
            return torch.tensor(0.0, device=self._get_device())
        
        cloth_image_norm = F.normalize(cloth_image_embeds, dim=-1, eps=1e-8)
        cloth_text_norm = F.normalize(cloth_text_embeds, dim=-1, eps=1e-8)
        sim = torch.matmul(cloth_image_norm, cloth_text_norm.t()) / self.temperature
        
        sim = torch.clamp(sim, min=-50, max=50)
        labels = torch.arange(cloth_image_norm.size(0), device=sim.device, dtype=torch.long)
        
        loss_img2t = self.ce_loss(sim, labels)
        loss_t2img = self.ce_loss(sim.t(), labels)
        
        return (loss_img2t + loss_t2img) / 2

    def triplet_loss(self, embeds, pids):
        # 封装三元组损失调用
        return self._hard_triplet(embeds, pids)

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids,
                is_matched=None, epoch=None, aux_info=None, training_phase='feature'):
        # 核心计算逻辑：根据训练阶段（特征提取或判别器）聚合各项加权损失
        losses = {}

        # 1. 计算跨模态对齐损失（包括单模态和融合特征）
        loss_unimodal = self.info_nce_loss(image_embeds, id_text_embeds, fused_embeds=None)
        if fused_embeds is not None:
            loss_fused = self.info_nce_loss(image_embeds, id_text_embeds, fused_embeds=fused_embeds)
            losses['info_nce'] = loss_unimodal + 0.5 * loss_fused
        else:
            losses['info_nce'] = loss_unimodal

        # 2. 计算属性语义及身份一致性损失
        losses['cloth_semantic'] = self.cloth_semantic_loss(cloth_image_embeds, cloth_text_embeds)
        losses['id_triplet'] = self.triplet_loss(id_embeds, pids)

        # 3. 计算语义引导对齐损失
        if self.semantic_guidance is not None and id_embeds is not None and cloth_embeds is not None:
            losses['semantic_alignment'] = self.semantic_guidance(id_feat=id_embeds, attr_feat=cloth_embeds, use_cross_separation=False)
        else:
            losses['semantic_alignment'] = torch.tensor(0.0, device=self._get_device())

        # 4. 计算 AH-Net 空间解耦相关损失
        if aux_info:
            losses['spatial_orthogonal'] = self._ortho_loss(aux_info['map_id'], aux_info['map_attr'])
            losses['ortho_reg'] = aux_info.get('ortho_reg', torch.tensor(0.0, device=self._get_device()))
        else:
            losses['spatial_orthogonal'] = torch.tensor(0.0, device=self._get_device())
            losses['ortho_reg'] = torch.tensor(0.0, device=self._get_device())

        # 5. 计算对抗解耦损失
        if self.adversarial_decoupler is not None and id_embeds is not None and cloth_embeds is not None:
            adv_losses = self.adversarial_decoupler(id_feat=id_embeds, cloth_feat=cloth_embeds, training_phase=training_phase)
            losses.update(adv_losses)
        else:
            for k in ['adversarial_attr', 'adversarial_domain', 'discriminator_attr', 'discriminator_domain']:
                losses[k] = torch.tensor(0.0, device=self._get_device())

        # 6. 按权重聚合总损失
        total_loss = torch.tensor(0.0, device=self._get_device())
        for key, value in losses.items():
            if key == 'total' or torch.isnan(value).any():
                continue

            if training_phase == 'discriminator':
                # 判别器训练阶段：仅累加判别器特有损失
                if key.startswith('discriminator_'):
                    total_loss += self.weights.get(key, 0.0) * value
            else:
                # 特征提取器训练阶段：累加除判别器外的所有损失
                if not key.startswith('discriminator_'):
                    total_loss += self.weights.get(key, 0.0) * value

        losses['total'] = total_loss

        if self.logger and self.loss_logger and self._batch_counter % 100 == 0:
            self._batch_counter += 1

        return losses
