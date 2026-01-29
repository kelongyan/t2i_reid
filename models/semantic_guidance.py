# models/semantic_guidance.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticGuidedDecoupling(nn.Module):
    # 语义引导特征解耦模块：利用 CLIP 的语言先验引导身份和属性特征的语义分离
    def __init__(self, text_encoder, tokenizer, dim=768, logger=None):
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.dim = dim
        self.logger = logger
        
        # 身份相关 Prompt：侧重于人体结构、轮廓等不变特征
        self.id_prompts = [
            "person's body structure",
            "human silhouette",
            "pedestrian figure",
            "person walking",
            "individual standing",
            "unique person identity",
            "pedestrian appearance",
        ]
        
        # 属性相关 Prompt：侧重于服装颜色、风格及配饰描述
        self.attr_prompts = [
            "red shirt", "blue shirt", "black shirt", "white shirt",
            "gray shirt", "yellow shirt", "green shirt", "pink shirt",
            "blue jeans", "black pants", "gray pants", "white pants",
            "casual clothes", "formal attire", "sportswear",
            "wearing backpack", "carrying handbag", "wearing hat",
        ]
        
        # 注册用于缓存 Prompt 嵌入向量的 buffer
        self.register_buffer('id_prompt_embeds', torch.zeros(len(self.id_prompts), dim))
        self.register_buffer('attr_prompt_embeds', torch.zeros(len(self.attr_prompts), dim))
        self._initialized = False
        
        # 投影层：将 CLIP 输出维度映射到统一的系统维度
        clip_dim = text_encoder.config.hidden_size
        if clip_dim != dim:
            self.prompt_proj = nn.Sequential(
                nn.Linear(clip_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU()
            )
        else:
            self.prompt_proj = nn.Identity()
    
    def _initialize_prompt_embeddings(self, device):
        # 初始化 Prompt 嵌入向量：首次调用时执行，使用预训练文本编码器进行特征提取
        if self._initialized:
            return
        
        with torch.no_grad():
            # 编码身份相关 Prompts
            id_tokens = self.tokenizer(
                self.id_prompts, padding='max_length', max_length=77, truncation=True, return_tensors='pt'
            ).to(device)
            id_outputs = self.text_encoder(**id_tokens)
            id_embeds_raw = id_outputs.last_hidden_state.mean(dim=1)
            id_embeds = self.prompt_proj(id_embeds_raw)
            self.id_prompt_embeds.copy_(F.normalize(id_embeds, dim=-1))
            
            # 编码属性相关 Prompts
            attr_tokens = self.tokenizer(
                self.attr_prompts, padding='max_length', max_length=77, truncation=True, return_tensors='pt'
            ).to(device)
            attr_outputs = self.text_encoder(**attr_tokens)
            attr_embeds_raw = attr_outputs.last_hidden_state.mean(dim=1)
            attr_embeds = self.prompt_proj(attr_embeds_raw)
            self.attr_prompt_embeds.copy_(F.normalize(attr_embeds, dim=-1))
        
        self._initialized = True
        
        if self.logger:
            self.logger.debug_logger.info(
                f"✅ Semantic Guidance Initialized: "
                f"{len(self.id_prompts)} ID Prompts, {len(self.attr_prompts)} Attr Prompts"
            )
    
    def compute_semantic_alignment_loss(self, id_feat, attr_feat):
        # 计算语义对齐损失：拉近视觉特征与其对应语义 Prompt 的距离
        if not self._initialized:
            self._initialize_prompt_embeddings(id_feat.device)
        
        # 归一化输入特征
        id_feat_norm = F.normalize(id_feat, dim=-1, eps=1e-8)
        attr_feat_norm = F.normalize(attr_feat, dim=-1, eps=1e-8)
        
        # 数值稳定性检测
        if torch.isnan(id_feat_norm).any() or torch.isnan(attr_feat_norm).any():
            return torch.tensor(0.0, device=id_feat.device, requires_grad=True)
        
        # 计算身份特征与 ID Prompts 的最大余弦相似度
        id_sim = torch.matmul(id_feat_norm, self.id_prompt_embeds.t())
        id_max_sim, _ = torch.max(id_sim, dim=1)
        
        # 计算属性特征与属性 Prompts 的最大余弦相似度
        attr_sim = torch.matmul(attr_feat_norm, self.attr_prompt_embeds.t())
        attr_max_sim, _ = torch.max(attr_sim, dim=1)
        
        # 使用 L2 距离形式的损失，增强训练稳定性
        loss_id = (1.0 - id_max_sim).mean()
        loss_attr = (1.0 - attr_max_sim).mean()
        
        if torch.isnan(loss_id).any() or torch.isnan(loss_attr).any():
            return torch.tensor(0.0, device=id_feat.device, requires_grad=True)
        
        total_loss = loss_id + loss_attr
        
        if self.logger and hasattr(self, '_log_counter'):
            self._log_counter = getattr(self, '_log_counter', 0) + 1
            if self._log_counter % 200 == 0:
                self.logger.debug_logger.debug(
                    f"Semantic Alignment: ID_sim={id_max_sim.mean():.4f}, "
                    f"Attr_sim={attr_max_sim.mean():.4f}, Loss={total_loss.item():.6f}"
                )
        
        return total_loss
    
    def compute_cross_separation_loss(self, id_feat, attr_feat):
        # 计算交叉分离损失：身份特征应远离属性 Prompt，属性特征应远离身份 Prompt
        if not self._initialized:
            self._initialize_prompt_embeddings(id_feat.device)
        
        id_feat_norm = F.normalize(id_feat, dim=-1, eps=1e-8)
        attr_feat_norm = F.normalize(attr_feat, dim=-1, eps=1e-8)
        
        id_to_attr_sim = torch.matmul(id_feat_norm, self.attr_prompt_embeds.t())
        id_to_attr_max, _ = torch.max(id_to_attr_sim, dim=1)
        
        attr_to_id_sim = torch.matmul(attr_feat_norm, self.id_prompt_embeds.t())
        attr_to_id_max, _ = torch.max(attr_to_id_sim, dim=1)
        
        return id_to_attr_max.mean() + attr_to_id_max.mean()
    
    def forward(self, id_feat, attr_feat, use_cross_separation=False):
        # 前向传播：计算语义对齐损失及可选的交叉分离损失
        align_loss = self.compute_semantic_alignment_loss(id_feat, attr_feat)
        
        if use_cross_separation:
            sep_loss = self.compute_cross_separation_loss(id_feat, attr_feat)
            return align_loss + 0.5 * sep_loss
        else:
            return align_loss
