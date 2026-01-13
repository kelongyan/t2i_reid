import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, temperature=0.1, weights=None):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.id_embed_projector = nn.Linear(768, 256)
        self.cloth_embed_projector = nn.Linear(768, 256)

        # 初始化损失权重（可被 GradNorm 学习动态调整）
        self.weights = weights if weights is not None else {
            'info_nce': 1.0,
            'cls': 1.0,
            'cloth': 0.5,
            'cloth_adv': 0.1,
            'cloth_match': 1.0,
            'decouple': 0.2,
            'gate_regularization': 0.01,
            'projection_l2': 1e-4,
            'uniformity': 0.01,
        }

        # 需进行 GradNorm 动态平衡的任务列表
        self.task_list = ['info_nce', 'cls', 'cloth', 'cloth_adv', 'cloth_match', 'decouple']
        self.log_vars = nn.Parameter(torch.zeros(len(self.task_list)))  # 学习的 log σ² 权重
        self.gradnorm_alpha = 1.5  # GradNorm 调整因子
        self.initial_losses = None  # 初始参考损失

    def gate_regularization_loss(self, gate):
        target = torch.full_like(gate, 0.5)
        return F.mse_loss(gate, target)

    def info_nce_loss(self, image_embeds, text_embeds):
        bsz = image_embeds.size(0)
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        sim = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        labels = torch.arange(bsz, device=sim.device)
        return (self.ce_loss(sim, labels) + self.ce_loss(sim.t(), labels)) / 2

    def id_classification_loss(self, id_logits, pids):
        return self.ce_loss(id_logits, pids)

    def cloth_contrastive_loss(self, cloth_embeds, cloth_text_embeds):
        if cloth_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        bsz = cloth_embeds.size(0)
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        sim = torch.matmul(cloth_embeds, cloth_text_embeds.t()) / self.temperature
        labels = torch.arange(bsz, device=sim.device)
        return self.ce_loss(sim, labels)

    def cloth_adversarial_loss(self, cloth_embeds, cloth_text_embeds, epoch=None):
        if cloth_embeds is None or cloth_text_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        cloth_embeds = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        sim = torch.matmul(cloth_embeds, cloth_text_embeds.t()) / self.temperature
        sim = sim - torch.diag(torch.diagonal(sim))
        neg_loss = -F.log_softmax(sim, dim=1).mean()
        if epoch is not None:
            adv_weight = min(1.0, 0.2 + epoch * 0.05)
            neg_loss *= adv_weight
        return neg_loss

    def compute_cloth_matching_loss(self, cloth_image_embeds, cloth_text_embeds, is_matched):
        if cloth_image_embeds is None or cloth_text_embeds is None or is_matched is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        bsz = cloth_image_embeds.size(0)
        cloth_image_embeds = F.normalize(cloth_image_embeds, dim=-1)
        cloth_text_embeds = F.normalize(cloth_text_embeds, dim=-1)
        sim = torch.matmul(cloth_image_embeds, cloth_text_embeds.t()) / self.temperature
        labels = torch.arange(bsz, device=sim.device)
        return (self.ce_loss(sim, labels) + self.ce_loss(sim.t(), labels)) / 2

    def compute_decoupling_loss(self, id_embeds, cloth_embeds):
        if id_embeds is None or cloth_embeds is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        id_proj = F.normalize(self.id_embed_projector(id_embeds), dim=-1)
        cloth_proj = F.normalize(self.cloth_embed_projector(cloth_embeds), dim=-1)
        id_kernel = torch.matmul(id_proj, id_proj.t())
        cloth_kernel = torch.matmul(cloth_proj, cloth_proj.t())
        id_kernel -= torch.diag(torch.diagonal(id_kernel))
        cloth_kernel -= torch.diag(torch.diagonal(cloth_kernel))
        return torch.mean(id_kernel * cloth_kernel) / (id_proj.size(0) - 1)

    def projection_l2_regularization(self):
        return torch.norm(self.id_embed_projector.weight, p=2) + torch.norm(self.cloth_embed_projector.weight, p=2)

    def uniformity_loss(self, embeds):
        embeds = F.normalize(embeds, dim=-1)
        return (embeds @ embeds.t()).exp().mean().log()

    def forward(self, image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, pids, is_matched, epoch=None, gate=None):
        
        losses = {}
        losses['info_nce'] = self.info_nce_loss(image_embeds, id_text_embeds) if image_embeds is not None and id_text_embeds is not None else torch.tensor(0.0, device=self.ce_loss.weight.device)
        losses['cls'] = self.id_classification_loss(id_logits, pids) if id_logits is not None and pids is not None else torch.tensor(0.0, device=self.ce_loss.weight.device)
        losses['cloth'] = self.cloth_contrastive_loss(cloth_embeds, cloth_text_embeds)
        losses['cloth_adv'] = self.cloth_adversarial_loss(cloth_embeds, cloth_text_embeds, epoch)
        losses['cloth_match'] = self.compute_cloth_matching_loss(cloth_image_embeds, cloth_text_embeds, is_matched)
        losses['decouple'] = self.compute_decoupling_loss(id_embeds, cloth_embeds)
        losses['gate_regularization'] = self.gate_regularization_loss(gate) if gate is not None else torch.tensor(0.0, device=self.ce_loss.weight.device)
        losses['projection_l2'] = self.projection_l2_regularization()
        losses['uniformity'] = self.uniformity_loss(id_embeds) if id_embeds is not None else torch.tensor(0.0, device=self.ce_loss.weight.device)

        # GradNorm 权重自动平衡计算
        if self.initial_losses is None:
            self.initial_losses = {k: v.item() + 1e-8 for k, v in losses.items() if k in self.task_list}

        task_losses = torch.stack([losses[k] for k in self.task_list])
        weights = torch.exp(-self.log_vars)
        weighted_losses = weights * task_losses
        total_loss = torch.sum(weighted_losses)

        avg_loss = task_losses.mean().detach()
        relative_rates = task_losses.detach() / torch.tensor([self.initial_losses[k] for k in self.task_list], device=task_losses.device)
        inverse_rate = relative_rates / relative_rates.mean()
        gradnorm_term = (weights * inverse_rate).sum() * self.gradnorm_alpha
        total_loss += gradnorm_term

        # 加入固定加权的正则项
        for key in ['gate_regularization', 'projection_l2', 'uniformity']:
            total_loss += self.weights[key] * losses[key]

        losses['total'] = total_loss
        return losses
