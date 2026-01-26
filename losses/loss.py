import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.loss_logger import LossLogger


class ReconstructionLoss(nn.Module):
    """
    AH-Net 特征重构损失
    目标：利用 Attr 分支（和被阻断梯度的 ID 分支）重构原始特征，强迫 Attr 分支捕捉纹理细节。
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, recon_feat, target_feat):
        """
        Args:
            recon_feat: [B, D] 解码器输出
            target_feat: [B, D] 目标特征 (原始特征均值)
        """
        return self.mse_loss(recon_feat, target_feat)


class SpatialOrthogonalLoss(nn.Module):
    """
    空间互斥损失 (Spatial Orthogonal Loss)
    目标：最小化 ID Attention Map 和 Attr Attention Map 的空间重叠
    """
    def __init__(self):
        super().__init__()

    def forward(self, map_id, map_attr):
        """
        Args:
            map_id: [B, 1, H, W]
            map_attr: [B, 1, H, W]
        """
        # 计算重叠: sum(map_id * map_attr)
        # 假设 maps 已经经过 softmax 或归一化，值在 [0,1] 之间
        # 由于 Cross Attention 输出的是 softmax 后的权重，直接相乘即可
        
        overlap = map_id * map_attr # [B, 1, H, W]
        loss = overlap.sum(dim=(2, 3)).mean() # Sum spatial, Mean batch
        return loss


class Loss(nn.Module):
    """
    AH-Net 损失函数配置
    包含：
    - InfoNCE (主任务)
    - ID Triplet (身份一致性)
    - Cloth Semantic (属性对齐)
    - Reconstruction (结构重构)
    - Spatial Orthogonal (空间互斥)
    - Semantic Alignment (语义对齐, 可选)
    """

    def __init__(self, temperature=0.1, weights=None, num_classes=None, logger=None,
                 semantic_guidance=None):
        """
        Args:
            semantic_guidance: SemanticGuidedDecoupling 模块 (方案书 Phase 3 可选增强)
        """
        super().__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.logger = logger
        self.semantic_guidance = semantic_guidance

        # Label Smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.2)

        # === 核心损失模块 ===
        self._recon_loss = ReconstructionLoss()
        self._ortho_loss = SpatialOrthogonalLoss()

        # === 初始化LossLogger ===
        self.loss_logger = LossLogger(logger.debug_logger) if logger else None

        # === 权重配置 ===
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,
            'id_triplet': 1.0,
            'cloth_semantic': 0.5,
            'reconstruction': 0.5,        # 结构重构
            'spatial_orthogonal': 0.1,    # 空间互斥
            'semantic_alignment': 0.1     # 语义对齐 (方案书 Phase 3, 可选)
        }

        self.register_buffer('_dummy', torch.zeros(1))
        self._batch_counter = 0
        if logger: self.debug_logger = logger.debug_logger

    def _get_device(self):
        return self._dummy.device

    def update_epoch(self, epoch):
        pass # 固定权重

    def info_nce_loss(self, image_embeds, text_embeds, fused_embeds=None):
        if image_embeds is None or text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
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

    def cloth_semantic_loss(self, cloth_image_embeds, cloth_text_embeds):
        if cloth_image_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=self._get_device())
        bsz = cloth_image_embeds.size(0)
        cloth_image_norm = F.normalize(cloth_image_embeds, dim=-1, eps=1e-8)
        cloth_text_norm = F.normalize(cloth_text_embeds, dim=-1, eps=1e-8)
        sim = torch.matmul(cloth_image_norm, cloth_text_norm.t()) / self.temperature
        sim = torch.clamp(sim, min=-50, max=50)
        labels = torch.arange(bsz, device=sim.device, dtype=torch.long)
        return (self.ce_loss(sim, labels) + self.ce_loss(sim.t(), labels)) / 2

    def triplet_loss(self, embeds, pids, margin=0.3):
        if embeds is None or pids is None:
            return torch.tensor(0.0, device=self._get_device())
        # Ensure float32 for numerical stability in mixed precision training
        embeds = embeds.to(torch.float32)
        n = embeds.size(0)
        # Compute pairwise Euclidean distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a*b
        dist = torch.pow(embeds, 2).sum(dim=1, keepdim=True).expand(n, n)  # [n, n]
        dist = dist + dist.t()
        dist = dist - 2 * torch.mm(embeds, embeds.t())
        dist = dist.clamp(min=1e-12).sqrt()
        mask = pids.expand(n, n).eq(pids.expand(n, n).t())
        dist_ap, _ = torch.max(dist * mask.float(), dim=1)
        dist_an, _ = torch.min(dist * (1. - mask.float()) + 1e6 * mask.float(), dim=1)
        return F.relu(dist_ap - dist_an + margin).mean()

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids,
                is_matched=None, epoch=None, gate=None, freq_info=None):
        """
        Compute total loss.

        Args:
            gate: Legacy name, kept for compatibility. Alias for aux_info.
            freq_info: Auxiliary info from AHNetModule (contains attention maps, reconstruction features).
        """
        # freq_info is actually 'aux_info' from AHNetModule
        # Compatible with cases where aux_info is passed as 'gate'
        aux_info = freq_info if freq_info is not None else gate

        losses = {}
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds, fused_embeds)
        losses['cloth_semantic'] = self.cloth_semantic_loss(cloth_image_embeds, cloth_text_embeds)
        losses['id_triplet'] = self.triplet_loss(id_embeds, pids)

        # AH-Net Losses
        if aux_info:
            losses['reconstruction'] = self._recon_loss(aux_info['recon_feat'], aux_info['target_feat'])
            losses['spatial_orthogonal'] = self._ortho_loss(aux_info['map_id'], aux_info['map_attr'])
        else:
            losses['reconstruction'] = torch.tensor(0.0, device=self._get_device())
            losses['spatial_orthogonal'] = torch.tensor(0.0, device=self._get_device())

        # 🔥 方案书 Phase 3: 语义对齐损失 (可选增强)
        # 利用 CLIP 语义原型引导特征分离
        if self.semantic_guidance is not None and id_embeds is not None and cloth_embeds is not None:
            losses['semantic_alignment'] = self.semantic_guidance(id_embeds, cloth_embeds)
        else:
            losses['semantic_alignment'] = torch.tensor(0.0, device=self._get_device())

        total_loss = torch.tensor(0.0, device=self._get_device())
        for key, value in losses.items():
            if key == 'total': continue
            if torch.isnan(value).any():
                losses[key] = torch.tensor(0.0, device=self._get_device(), requires_grad=True)
            weight = self.weights.get(key, 0.0)
            if weight > 0:
                total_loss += weight * losses[key]

        losses['total'] = total_loss

        if self.logger and self.loss_logger and self._batch_counter % 100 == 0:
            self._batch_counter += 1
            # Simple logging
            pass

        return losses