"""
Loss Logger - 专门用于损失函数的调试日志工具
添加关键调试信息，便于追踪训练过程中的损失变化
"""
import torch
import torch.nn.functional as F


class LossLogger:
    """
    损失函数专用日志记录器
    """
    def __init__(self, debug_logger):
        self.debug_logger = debug_logger
        self.log_intervals = {
            'info_nce': 500,
            'cloth_semantic': 500,
            'id_triplet': 500,
            'reconstruction': 500,
            'spatial_orthogonal': 200,
        }
        self._counters = {k: 0 for k in self.log_intervals.keys()}

    def should_log(self, loss_type):
        """检查是否应该记录该损失的日志"""
        if loss_type not in self._counters:
            self._counters[loss_type] = 0
            # Default interval if not specified
            self.log_intervals[loss_type] = 500 
            
        self._counters[loss_type] += 1
        return self._counters[loss_type] % self.log_intervals[loss_type] == 0

    def log_spatial_orthogonality_stats(self, map_id, map_attr, loss_value):
        """记录空间互斥统计信息 (AH-Net)"""
        if map_id is None or map_attr is None:
            return

        # map_id, map_attr: [B, 1, H, W]
        # Calculate overlap
        overlap = map_id * map_attr
        
        # Calculate mean activation
        mean_id = map_id.mean().item()
        mean_attr = map_attr.mean().item()
        
        # Calculate overlap ratio relative to activations
        overlap_mean = overlap.mean().item()
        
        self.debug_logger.debug(
            f"[SpatialOrtho] mean_id={mean_id:.4f} | "
            f"mean_attr={mean_attr:.4f} | "
            f"overlap={overlap_mean:.6f} | "
            f"loss={loss_value.item():.6f}"
        )

    def log_reconstruction_stats(self, recon_feat, target_feat, loss_value):
        """记录重构统计信息 (AH-Net)"""
        # 计算重构误差
        mse_error = F.mse_loss(recon_feat, target_feat)

        # 计算能量保留率
        recon_energy = torch.norm(recon_feat, p=2, dim=-1).mean()
        target_energy = torch.norm(target_feat, p=2, dim=-1).mean()
        energy_ratio = (recon_energy / (target_energy + 1e-8)).item()

        # 计算方向一致性
        recon_norm = F.normalize(recon_feat, dim=-1, eps=1e-8)
        target_norm = F.normalize(target_feat, dim=-1, eps=1e-8)
        cosine_sim = (recon_norm * target_norm).sum(dim=-1).mean().item()

        self.debug_logger.debug(
            f"[Reconstruction] mse={mse_error.item():.6f} | "
            f"energy_ratio={energy_ratio:.4f} | "
            f"cosine_sim={cosine_sim:.4f} | "
            f"total_loss={loss_value.item():.6f}"
        )

    def log_triplet_stats(self, embeds, pids, loss_value, margin=0.3):
        """记录Triplet损失统计"""
        n = embeds.size(0)
        dist = torch.pow(embeds, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(embeds, embeds.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = pids.expand(n, n).eq(pids.expand(n, n).t())
        dist_ap, _ = torch.max(dist * mask.float(), dim=1)
        dist_an, _ = torch.min(dist * (1. - mask.float()) + 1e6 * mask.float(), dim=1)

        self.debug_logger.debug(
            f"[Triplet] margin={margin:.2f} | "
            f"dist_ap_mean={dist_ap.mean().item():.4f} | "
            f"dist_an_mean={dist_an.mean().item():.4f} | "
            f"hard_positives={((dist_ap - dist_an + margin) > 0).sum().item()}/{n} | "
            f"loss={loss_value.item():.6f}"
        )

    def log_info_nce_stats(self, image_embeds, text_embeds, loss_value, temperature=0.1):
        """记录InfoNCE统计"""
        image_norm = F.normalize(image_embeds, dim=-1, eps=1e-8)
        text_norm = F.normalize(text_embeds, dim=-1, eps=1e-8)

        sim = torch.matmul(image_norm, text_norm.t()) / temperature
        pos_sim = torch.diagonal(sim).mean().item()
        neg_sim = (sim - torch.eye(sim.size(0), device=sim.device) * 1e9).mean().item()

        self.debug_logger.debug(
            f"[InfoNCE] temp={temperature:.3f} | "
            f"pos_sim={pos_sim:.4f} | "
            f"neg_sim={neg_sim:.4f} | "
            f"pos_neg_gap={pos_sim - neg_sim:.4f} | "
            f"loss={loss_value.item():.4f}"
        )

    def log_cloth_semantic_stats(self, cloth_image, cloth_text, loss_value, temperature=0.1):
        """记录服装语义损失统计"""
        cloth_image_norm = F.normalize(cloth_image, dim=-1, eps=1e-8)
        cloth_text_norm = F.normalize(cloth_text, dim=-1, eps=1e-8)

        sim = torch.matmul(cloth_image_norm, cloth_text_norm.t()) / temperature
        pos_sim = torch.diagonal(sim).mean().item()
        neg_sim = (sim - torch.eye(sim.size(0), device=sim.device) * 1e9).mean().item()

        self.debug_logger.debug(
            f"[ClothSemantic] temp={temperature:.3f} | "
            f"pos_sim={pos_sim:.4f} | "
            f"neg_sim={neg_sim:.4f} | "
            f"pos_neg_gap={pos_sim - neg_sim:.4f} | "
            f"loss={loss_value.item():.4f}"
        )

    def log_weighted_loss_summary(self, losses, weights):
        """记录加权损失摘要"""
        weighted_losses = {k: weights.get(k, 0) * v.item() if isinstance(v, torch.Tensor) else weights.get(k, 0) * v
                          for k, v in losses.items() if k != 'total'}

        total = losses.get('total', sum(weighted_losses.values()))
        if isinstance(total, torch.Tensor):
            total = total.item()

        self.debug_logger.debug("=== Weighted Loss Summary ===")
        # AH-Net Losses
        for k in ['info_nce', 'id_triplet', 'cloth_semantic', 'reconstruction', 'spatial_orthogonal']:
            if k in weighted_losses and weights.get(k, 0) > 0:
                val = weighted_losses[k]
                ratio = (val / total * 100) if total > 0 else 0
                self.debug_logger.debug(
                    f"  {k:20s}: {val:.6f} (weight={weights.get(k, 0):.2f}, {ratio:5.2f}%)"
                )
        self.debug_logger.debug(f"  {'total':20s}: {total:.6f}")