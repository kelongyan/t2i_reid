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
            'cls': 500,
            'cloth_semantic': 500,
            'orthogonal': 200,
            'id_triplet': 500,
            'anti_collapse': 500,
            'reconstruction': 500,
        }
        self._counters = {k: 0 for k in self.log_intervals.keys()}

    def should_log(self, loss_type):
        """检查是否应该记录该损失的日志"""
        self._counters[loss_type] += 1
        return self._counters[loss_type] % self.log_intervals[loss_type] == 0

    def log_feature_norms(self, features, name):
        """记录特征范数统计"""
        if features is None:
            self.debug_logger.debug(f"[{name}] Feature is None")
            return

        norms = torch.norm(features, p=2, dim=-1)
        self.debug_logger.debug(
            f"[{name}] mean_norm={norms.mean().item():.4f} | "
            f"std_norm={norms.std().item():.4f} | "
            f"min_norm={norms.min().item():.4f} | "
            f"max_norm={norms.max().item():.4f}"
        )

    def log_orthogonality_stats(self, id_embeds, cloth_embeds, loss_value):
        """记录正交性统计信息"""
        id_norm = F.normalize(id_embeds, dim=-1, eps=1e-8)
        cloth_norm = F.normalize(cloth_embeds, dim=-1, eps=1e-8)
        cosine_sim = (id_norm * cloth_norm).sum(dim=-1)

        self.debug_logger.debug(
            f"[Orthogonal] cosine_sim: mean={cosine_sim.mean().item():.4f} | "
            f"std={cosine_sim.std().item():.4f} | "
            f"min={cosine_sim.min().item():.4f} | "
            f"max={cosine_sim.max().item():.4f} | "
            f"loss={loss_value.item():.6f}"
        )

    def log_anti_collapse_stats(self, features, target_norm, margin_ratio, loss_value):
        """记录防坍缩统计信息"""
        norms = torch.norm(features, p=2, dim=-1)
        feature_std = features.std(dim=0)

        self.debug_logger.debug(
            f"[AntiCollapse] target_norm={target_norm:.2f} | "
            f"actual_mean_norm={norms.mean().item():.2f} | "
            f"margin={target_norm * margin_ratio:.2f} | "
            f"norm_loss={F.relu(target_norm * margin_ratio - norms).mean().item():.6f} | "
            f"collapse_loss={F.relu(0.01 - feature_std).mean().item():.6f} | "
            f"total_loss={loss_value.item():.6f}"
        )

    def log_reconstruction_stats(self, id_feat, attr_feat, original_feat, loss_value):
        """记录重构统计信息"""
        reconstructed = id_feat + attr_feat

        # 计算重构误差
        mse_error = F.mse_loss(reconstructed, original_feat)

        # 计算能量保留率
        recon_energy = torch.norm(reconstructed, p=2, dim=-1).mean()
        orig_energy = torch.norm(original_feat, p=2, dim=-1).mean()
        energy_ratio = (recon_energy / orig_energy).item()

        # 计算方向一致性
        recon_norm = F.normalize(reconstructed, dim=-1, eps=1e-8)
        orig_norm = F.normalize(original_feat, dim=-1, eps=1e-8)
        cosine_sim = (recon_norm * orig_norm).sum(dim=-1).mean().item()

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

    def log_cls_stats(self, logits, pids, loss_value):
        """记录分类损失统计"""
        # 计算预测准确率
        preds = torch.argmax(logits, dim=-1)
        accuracy = (preds == pids).float().mean().item()

        # 计算Top-5准确率
        _, top5_preds = torch.topk(logits, k=5, dim=-1)
        top5_accuracy = (top5_preds == pids.unsqueeze(1)).any(dim=1).float().mean().item()

        # 计算置信度
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()

        # 计算熵
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

        self.debug_logger.debug(
            f"[CLS] accuracy={accuracy:.4f} | "
            f"top5_accuracy={top5_accuracy:.4f} | "
            f"confidence={confidence:.4f} | "
            f"entropy={entropy:.4f} | "
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

        total = weighted_losses['total'] if 'total' in weighted_losses else sum(weighted_losses.values())

        self.debug_logger.debug("=== Weighted Loss Summary ===")
        for k in ['info_nce', 'cls', 'cloth_semantic', 'orthogonal', 'id_triplet', 'anti_collapse', 'reconstruction']:
            if k in weighted_losses and weights.get(k, 0) > 0:
                ratio = (weighted_losses[k] / total * 100) if total > 0 else 0
                self.debug_logger.debug(
                    f"  {k:15s}: {weighted_losses[k]:.6f} (weight={weights.get(k, 0):.2f}, {ratio:5.2f}%)"
                )
        self.debug_logger.debug(f"  {'total':15s}: {total:.6f}")
