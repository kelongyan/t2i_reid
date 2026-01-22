# models/semantic_guidance.py
"""
CLIP语义引导模块
利用CLIP的语言先验知识，引导ID和Attribute特征的语义分离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticGuidedDecoupling(nn.Module):
    """
    语义引导的特征解耦模块
    
    核心思想：
    - 利用CLIP Text Encoder预先定义的ID和Attribute Prompts
    - 通过语义对齐损失，让视觉特征向对应的语义空间靠拢
    - ID特征 → "a person", "pedestrian" 等身份相关描述
    - Attr特征 → "red clothes", "backpack" 等属性相关描述
    """
    
    def __init__(self, text_encoder, tokenizer, dim=768, logger=None):
        """
        Args:
            text_encoder: CLIP Text Encoder实例
            tokenizer: CLIP Tokenizer实例
            dim (int): 特征维度（需与ID/Attr特征对齐）
            logger: TrainingMonitor实例
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.dim = dim
        self.logger = logger
        
        # === 🔥 改进的Prompt模板（更具体、更多样化）===
        # ID Prompts: 强调身份结构、体态、不变特征
        self.id_prompts = [
            # 结构类
            "person's body structure",
            "human silhouette",
            "pedestrian figure",
            # 动作类
            "person walking",
            "individual standing",
            # 抽象身份
            "unique person identity",
            "pedestrian appearance",
        ]
        
        # Attribute Prompts: 细粒度服装描述
        self.attr_prompts = [
            # 上衣颜色
            "red shirt", "blue shirt", "black shirt", "white shirt",
            "gray shirt", "yellow shirt", "green shirt", "pink shirt",
            # 下装颜色
            "blue jeans", "black pants", "gray pants", "white pants",
            # 风格
            "casual clothes", "formal attire", "sportswear",
            # 配饰
            "wearing backpack", "carrying handbag", "wearing hat",
        ]
        
        # 预计算并缓存CLIP Embeddings（避免重复编码）
        self.register_buffer('id_prompt_embeds', torch.zeros(len(self.id_prompts), dim))
        self.register_buffer('attr_prompt_embeds', torch.zeros(len(self.attr_prompts), dim))
        self._initialized = False
        
        # 投影层：将CLIP输出维度（512）映射到系统维度（768）
        # 这与主模型的text_proj保持一致
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
        """
        初始化Prompt Embeddings（首次调用时执行）
        使用预训练的CLIP Text Encoder编码固定Prompts
        """
        if self._initialized:
            return
        
        with torch.no_grad():
            # 编码ID Prompts
            id_tokens = self.tokenizer(
                self.id_prompts,
                padding='max_length',
                max_length=77,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            id_outputs = self.text_encoder(**id_tokens)
            # 使用pooler_output（[CLS] token）或last_hidden_state的均值
            id_embeds_raw = id_outputs.last_hidden_state.mean(dim=1)  # [num_prompts, 512]
            id_embeds = self.prompt_proj(id_embeds_raw)  # [num_prompts, 768]
            self.id_prompt_embeds.copy_(F.normalize(id_embeds, dim=-1))
            
            # 编码Attr Prompts
            attr_tokens = self.tokenizer(
                self.attr_prompts,
                padding='max_length',
                max_length=77,
                truncation=True,
                return_tensors='pt'
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
        """
        计算语义对齐损失（简化版本）
        
        修复：
        1. 使用L2距离替代负对数，避免梯度爆炸
        2. 添加NaN检测
        3. 降低损失敏感度
        
        目标：
        - ID特征应该与ID Prompts更相似
        - Attr特征应该与Attr Prompts更相似
        
        Args:
            id_feat: ID特征 [batch_size, dim]
            attr_feat: Attr特征 [batch_size, dim]
            
        Returns:
            loss: 语义对齐损失（标量）
        """
        # 初始化Prompt Embeddings（如果未初始化）
        if not self._initialized:
            self._initialize_prompt_embeddings(id_feat.device)
        
        # 归一化特征
        id_feat_norm = F.normalize(id_feat, dim=-1, eps=1e-8)      # [B, dim]
        attr_feat_norm = F.normalize(attr_feat, dim=-1, eps=1e-8)  # [B, dim]
        
        # NaN检测
        if torch.isnan(id_feat_norm).any() or torch.isnan(attr_feat_norm).any():
            return torch.tensor(0.0, device=id_feat.device, requires_grad=True)
        
        # === ID特征与ID Prompts的对齐 ===
        # 计算余弦相似度矩阵 [B, num_id_prompts]
        id_sim = torch.matmul(id_feat_norm, self.id_prompt_embeds.t())
        # 取最大相似度（最接近的prompt）
        id_max_sim, _ = torch.max(id_sim, dim=1)  # [B]
        
        # === Attr特征与Attr Prompts的对齐 ===
        attr_sim = torch.matmul(attr_feat_norm, self.attr_prompt_embeds.t())
        attr_max_sim, _ = torch.max(attr_sim, dim=1)  # [B]
        
        # === 损失：使用L2距离替代负对数 ===
        # 相似度越高（接近1）损失越小
        loss_id = (1.0 - id_max_sim).mean()
        loss_attr = (1.0 - attr_max_sim).mean()
        
        # NaN检测
        if torch.isnan(loss_id).any() or torch.isnan(loss_attr).any():
            return torch.tensor(0.0, device=id_feat.device, requires_grad=True)
        
        # 总损失
        total_loss = loss_id + loss_attr
        
        # 调试信息
        if self.logger and hasattr(self, '_log_counter'):
            self._log_counter = getattr(self, '_log_counter', 0) + 1
            if self._log_counter % 200 == 0:
                self.logger.debug_logger.debug(
                    f"Semantic Alignment: ID_sim={id_max_sim.mean():.4f}, "
                    f"Attr_sim={attr_max_sim.mean():.4f}, Loss={total_loss.item():.6f}"
                )
        
        return total_loss
    
    def compute_cross_separation_loss(self, id_feat, attr_feat):
        """
        计算交叉分离损失（可选）
        
        目标：
        - ID特征应远离Attr Prompts
        - Attr特征应远离ID Prompts
        
        这是一个辅助约束，增强两个特征空间的分离
        """
        if not self._initialized:
            self._initialize_prompt_embeddings(id_feat.device)
        
        id_feat_norm = F.normalize(id_feat, dim=-1, eps=1e-8)
        attr_feat_norm = F.normalize(attr_feat, dim=-1, eps=1e-8)
        
        # ID特征与Attr Prompts的相似度（应该低）
        id_to_attr_sim = torch.matmul(id_feat_norm, self.attr_prompt_embeds.t())
        id_to_attr_max, _ = torch.max(id_to_attr_sim, dim=1)
        
        # Attr特征与ID Prompts的相似度（应该低）
        attr_to_id_sim = torch.matmul(attr_feat_norm, self.id_prompt_embeds.t())
        attr_to_id_max, _ = torch.max(attr_to_id_sim, dim=1)
        
        # 损失：相似度越高惩罚越大
        loss = id_to_attr_max.mean() + attr_to_id_max.mean()
        
        return loss
    
    def forward(self, id_feat, attr_feat, use_cross_separation=False):
        """
        前向传播：计算语义引导损失
        
        Args:
            id_feat: ID特征 [B, dim]
            attr_feat: Attr特征 [B, dim]
            use_cross_separation: 是否使用交叉分离损失
            
        Returns:
            loss: 语义引导损失
        """
        # 主损失：语义对齐
        align_loss = self.compute_semantic_alignment_loss(id_feat, attr_feat)
        
        # 可选：交叉分离损失
        if use_cross_separation:
            sep_loss = self.compute_cross_separation_loss(id_feat, attr_feat)
            return align_loss + 0.5 * sep_loss  # 权重可调
        else:
            return align_loss
