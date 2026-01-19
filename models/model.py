# src/models/model.py
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel, ViTModel
from safetensors.torch import load_file
from utils.serialization import copy_state_dict
from .fusion import get_fusion_module
from .gs3_module import GS3Module  # 兼容性别名，实际使用SymmetricGS3Module
from .semantic_guidance import SemanticGuidedDecoupling  # 新增CLIP语义引导
# from .residual_classifier import ResidualClassifier, DeepResidualClassifier  # Deprecated in Optimization Plan
from .vim import VisionMamba

# 设置transformers库日志级别
import logging as _logging
import warnings
_logging.getLogger("transformers").setLevel(_logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=(), mid_cls=True, logger=None):
    """
    Rescale the grid of position embeddings when loading from state_dict. Adapted from DEIT/Vim.
    
    Args:
        posemb: Pretrained position embedding tensor. [1, 197, 384]
        posemb_new: Target position embedding tensor. [1, 129, 384]
        num_tokens: Number of special tokens (CLS, etc.).
        gs_new: Target grid size (h, w).
        mid_cls: Whether the CLS token is in the middle (Vim style).
        logger: Logger instance for debugging
    """
    if logger:
        debug_logger.debug(f"Resizing position embedding: {posemb.shape} -> {posemb_new.shape}")
    
    ntok_new = posemb_new.shape[1]
    
    # 1. 解析输入权重 (Handle Input CLS position)
    # 如果加载的是 vim_midclstok 权重，CLS token 通常已经在中间了
    if num_tokens:
        if mid_cls:
            # 假设预训练权重也是 Mid-CLS 结构
            old_cls_idx = posemb.shape[1] // 2
            posemb_tok = posemb[:, old_cls_idx:old_cls_idx+num_tokens]
            # 拼接除了 CLS 以外的部分
            posemb_grid = torch.cat([posemb[:, :old_cls_idx], posemb[:, old_cls_idx+num_tokens:]], dim=1)
        else:
            # 标准 ViT 结构 [CLS, Grid]
            posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[:, num_tokens:]
            
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb
    
    gs_old = int(math.sqrt(len(posemb_grid[0])))
    
    if ntok_new != len(posemb_grid[0]):
        if logger:
            debug_logger.debug(f'Position embedding grid resize from {gs_old}x{gs_old} to {gs_new[0]}x{gs_new[1]}')
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).flatten(1, 2)
        
        # 2. 组装输出权重 (Re-assemble Output)
        if mid_cls:
            # === 关键修复：将 CLS 放回新序列的中间 ===
            new_cls_idx = posemb_grid.shape[1] // 2
            posemb = torch.cat([
                posemb_grid[:, :new_cls_idx], 
                posemb_tok, 
                posemb_grid[:, new_cls_idx:]
            ], dim=1)
        else:
            posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
            
    return posemb


class DisentangleModule(nn.Module):
    """
    特征分离模块的基类（保留用于向后兼容）
    实际使用中建议直接使用 GS3Module
    """
    def __init__(self, dim):
        """
        简化的特征分离模块（消融实验版本），移除复杂的注意力机制。

        Args:
            dim (int): 输入特征的维度（每个 token 的维度）。
        """
        super().__init__()
        # 简化为基本的线性变换
        self.id_linear = nn.Linear(dim, dim)
        self.cloth_linear = nn.Linear(dim, dim)
        
        # 保留门控机制以维持接口一致性
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        """
        简化的前向传播，移除注意力机制，使用基本的线性变换和池化。

        Args:
            x (torch.Tensor): 输入特征，形状为 [batch_size, seq_len, dim]。
            return_attention (bool): 是否返回注意力权重（用于可视化）。

        Returns:
            tuple: 如果 return_attention=False，返回 (id_feat, cloth_feat, gate)
                  如果 return_attention=True，返回 (id_feat, cloth_feat, gate, id_attn_map, cloth_attn_map)
        """
        batch_size, seq_len, dim = x.size()

        # 简化的身份特征处理：线性变换 + 全局平均池化
        id_feat = self.id_linear(x)  # [batch_size, seq_len, dim]
        
        # 计算身份分支的注意力权重（基于特征幅值）
        id_attn_map = None
        if return_attention:
            # 使用 L2 范数作为注意力权重的近似
            id_attn_map = torch.norm(id_feat, p=2, dim=-1)  # [batch_size, seq_len]
            id_attn_map = torch.softmax(id_attn_map, dim=-1)  # 归一化
        
        id_feat = id_feat.mean(dim=1)  # [batch_size, dim]

        # 简化的服装特征处理：线性变换 + 全局平均池化  
        cloth_feat = self.cloth_linear(x)  # [batch_size, seq_len, dim]
        
        # 计算服装分支的注意力权重（基于特征幅值）
        cloth_attn_map = None
        if return_attention:
            cloth_attn_map = torch.norm(cloth_feat, p=2, dim=-1)  # [batch_size, seq_len]
            cloth_attn_map = torch.softmax(cloth_attn_map, dim=-1)  # 归一化
        
        cloth_feat = cloth_feat.mean(dim=1)  # [batch_size, dim]

        # 保持门控机制以维持原有接口
        gate = self.gate(torch.cat([id_feat, cloth_feat], dim=-1))  # [batch_size, dim]
        id_feat = gate * id_feat
        cloth_feat = (1 - gate) * cloth_feat
        
        if return_attention:
            return id_feat, cloth_feat, gate, id_attn_map, cloth_attn_map
        else:
            return id_feat, cloth_feat, gate


class Model(nn.Module):
    def __init__(self, net_config, logger=None):
        """
        文本-图像行人重识别模型（消融实验版本），移除了复杂的解纠缠模块。

        Args:
            net_config (dict): 模型配置字典，包含BERT路径、ViT路径、融合模块配置等。
            logger: TrainingMonitor实例，用于记录日志
        """
        super().__init__()
        self.net_config = net_config
        self.logger = logger
        
        # === Upgrade: Switch to CLIP Text Encoder ===
        clip_base_path = Path(net_config.get('clip_pretrained', 'pretrained/openai/clip-vit-base-patch16'))
        vit_base_path = Path(net_config.get('vit_pretrained', 'pretrained/vit-base-patch16-224'))
        fusion_config = net_config.get('fusion', {})
        num_classes = net_config.get('num_classes', 8000)

        # 验证预训练模型路径
        if not clip_base_path.exists():
            # Fallback to searching in pretrained folder if exact path not found
            fallback = list(Path("pretrained").glob("**/clip-vit-base-patch16"))
            if fallback:
                clip_base_path = fallback[0]
                if self.logger:
                    self.debug_logger.warning(f"Exact CLIP path not found, using fallback: {clip_base_path}")
            else:
                 # Last resort: try checking parent directories if relative path issue
                if (Path.cwd() / "pretrained/clip-vit-base-patch16").exists():
                    clip_base_path = Path.cwd() / "pretrained/clip-vit-base-patch16"

        if not clip_base_path.exists() or not vit_base_path.exists():
             # If strictly not found, still raise error
            if not clip_base_path.exists():
                 raise FileNotFoundError(f"CLIP model path not found: {clip_base_path}")
            if not vit_base_path.exists():
                 raise FileNotFoundError(f"ViT model path not found: {vit_base_path}")

        # 初始化文本编码器 (CLIP)
        if self.logger:
            self.debug_logger.info(f"Loading CLIP Text Encoder from: {clip_base_path}")
        self.tokenizer = CLIPTokenizer.from_pretrained(str(clip_base_path))
        
        # 初始化 CLIPTextModel (使用 safetensors 加载以规避安全检查)
        self.text_encoder = CLIPTextModel.from_pretrained(str(clip_base_path))
        
        # 尝试加载 safetensors 权重 (如果有)
        safetensors_path = clip_base_path / "model.safetensors"
        if safetensors_path.exists():
            if self.logger:
                self.debug_logger.info(f"Loading CLIP weights from safetensors: {safetensors_path}")
            state_dict = load_file(str(safetensors_path))
            
            # CLIPModel 包含 text_model. 前缀，我们需要剥离它以适配 CLIPTextModel
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("text_model."):
                    new_key = k[len("text_model."):]  # remove "text_model." prefix
                    new_state_dict[new_key] = v
            
            if new_state_dict:
                missing, unexpected = self.text_encoder.load_state_dict(new_state_dict, strict=False)
                if missing and self.logger:
                    self.debug_logger.info(f"Loaded CLIP with missing keys (expected for visual part): {len(missing)}")
            else:
                if self.logger:
                    self.debug_logger.warning("No 'text_model.' keys found in safetensors. Assuming pure TextModel format.")
        
        self.clip_dim = self.text_encoder.config.hidden_size # 通常是 512
        self.text_width = 768  # 系统目标维度 (适配 G-S3/Vim)

        # === 维度适配层 (Adapter) ===
        # 将 CLIP 的 512 维映射到系统的 768 维
        # 使用 Linear -> LayerNorm -> GELU 增强表达能力
        if self.clip_dim != self.text_width:
            self.text_proj = nn.Sequential(
                nn.Linear(self.clip_dim, self.text_width),
                nn.LayerNorm(self.text_width),
                nn.GELU()
            )
            # 初始化
            nn.init.xavier_uniform_(self.text_proj[0].weight)
            nn.init.zeros_(self.text_proj[0].bias)
            if self.logger:
                self.debug_logger.info(f"Added CLIP Adapter: {self.clip_dim} -> {self.text_width} (Linear+LN+GELU)")
        else:
            self.text_proj = nn.Identity()

        # 初始化图像编码器
        self.vision_backbone_type = net_config.get('vision_backbone', 'vit')
        if self.vision_backbone_type == 'vim':
            # Vision Mamba (Vim-S)
            vim_pretrained_path = net_config.get('vim_pretrained', 'pretrained/Vision Mamba/vim_s_midclstok.pth')
            # 获取图像尺寸，默认为 224x224
            img_size = net_config.get('img_size', (224, 224))
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            
            # 使用配置的尺寸初始化 VisionMamba
            self.visual_encoder = VisionMamba(img_size=img_size, patch_size=16, embed_dim=384, depth=24)
            
            # 加载 Vim 预训练权重
            if Path(vim_pretrained_path).exists():
                # weights_only=False 以支持加载包含 argparse.Namespace 的 checkpoint
                checkpoint = torch.load(vim_pretrained_path, map_location='cpu', weights_only=False)
                # 提取模型权重 (处理 checkpoint 包含 'model' 键的情况)
                state_dict = checkpoint.get('model', checkpoint)
                
                # === 关键修复：位置编码插值 ===
                if 'pos_embed' in state_dict:
                    # 计算新的 Grid 尺寸
                    # img_size 是 (H, W)，patch_size 是 16
                    grid_h = img_size[0] // 16
                    grid_w = img_size[1] // 16
                    
                    state_dict['pos_embed'] = resize_pos_embed(
                        state_dict['pos_embed'], 
                        self.visual_encoder.pos_embed, 
                        num_tokens=1, 
                        gs_new=(grid_h, grid_w),
                        logger=self.logger
                    )
                
                # 加载权重 (非严格模式，允许部分不匹配，如头部)
                missing, unexpected = self.visual_encoder.load_state_dict(state_dict, strict=False)
                if self.logger:
                    self.debug_logger.info(f"Loaded Vim backbone from {vim_pretrained_path}")
                    if missing:
                        self.debug_logger.warning(f"Missing keys in Vim: {missing}")
            else:
                if self.logger:
                    self.debug_logger.warning(f"Vim pretrained path not found: {vim_pretrained_path}. Using random init.")
            
            # 投影层：将 Vim 的 384 维映射到 768 维以适配后续模块
            self.visual_proj = nn.Linear(384, self.text_width)
            if self.logger:
                self.debug_logger.info(f"Using Vision Mamba (Vim-S) backbone with projection (384->768), img_size={img_size}")
            
        else:
            # ViT-Base (默认)
            self.visual_encoder = ViTModel.from_pretrained(str(vit_base_path), weights_only=False)
            self.visual_proj = nn.Identity() # ViT 输出已经是 768，无需投影
            if self.logger:
                self.debug_logger.info("Using ViT-Base backbone")

        # 初始化 G-S3 特征分离模块（对称解耦版本）
        disentangle_type = net_config.get('disentangle_type', 'gs3')
        if disentangle_type == 'gs3':
            gs3_config = net_config.get('gs3', {})
            self.disentangle = GS3Module(
                dim=self.text_width,
                num_heads=gs3_config.get('num_heads', 8),
                d_state=gs3_config.get('d_state', 16),
                d_conv=gs3_config.get('d_conv', 4),
                dropout=gs3_config.get('dropout', 0.1),
                logger=self.logger
            )
            if self.logger:
                self.debug_logger.info("Using Symmetric G-S3 (对称解耦) disentangle module")
        else:
            self.disentangle = DisentangleModule(dim=self.text_width)
            if self.logger:
                self.debug_logger.info("Using simplified disentangle module")
        
        # === 新增：CLIP语义引导模块 ===
        # 使用已加载的CLIP Text Encoder和Tokenizer
        self.semantic_guidance = SemanticGuidedDecoupling(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            dim=self.text_width,
            logger=self.logger
        )
        if self.logger:
            self.debug_logger.info("✅ Initialized CLIP Semantic Guidance for ID/Attr decoupling")

        # ================================================================
        # === 方案C (优化版)：BNNeck 监督 - 使用 BNNeck 替代深层残差分类器 ===
        # ================================================================
        
        # === 分支1：专用于分类的id特征处理 ===
        # 使用 BNNeck (BatchNorm1d) 规范化特征分布
        # 这一步将特征拉向超球面，有利于 CrossEntropy 收敛，同时不破坏流形结构
        self.id_bn = nn.BatchNorm1d(self.text_width)
        # 初始化 BNNeck：weight=1, bias=0 (标准做法)
        nn.init.constant_(self.id_bn.weight, 1.0)
        nn.init.constant_(self.id_bn.bias, 0.0)
        
        if self.logger:
            self.debug_logger.info("Using BNNeck (BatchNorm1d) for classification branch")

        # 身份分类器：从 text_width (768) 直接映射到类别数
        # bias=False 是 ReID 常见 Trick，让模型关注向量角度而非模长
        self.id_classifier = nn.Linear(self.text_width, num_classes, bias=False)
        # 初始化分类器权重
        nn.init.normal_(self.id_classifier.weight, std=0.001)
        
        # === 分支2：专用于检索的id特征处理（保持原有设计）===
        # 共享MLP：用于降维
        # === 分支2：专用于检索的id特征处理（保持原有设计）===
        # 共享MLP：用于降维
        self.shared_mlp = nn.Linear(self.text_width, 512)

        # 图像特征MLP：多层映射
        self.image_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )

        # 文本特征MLP：多层映射
        self.text_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        
        # === 分支3：cloth特征处理（共享检索分支的MLP）===
        # cloth_embeds也使用shared_mlp和image_mlp进行投影
        # 这样可以确保cloth特征和id特征在同一空间中对比
        
        if self.logger:
            self.debug_logger.info("=" * 60)
            self.debug_logger.info("Branch Decoupling Architecture (Optimized):")
            self.debug_logger.info(f"  - Classification Branch: {self.text_width} → BNNeck(768) → {num_classes}")
            self.debug_logger.info(f"  - Retrieval Branch: {self.text_width} → 512 → 256")
            self.debug_logger.info(f"  - Total Classifier Params: ~{self._count_classifier_params() / 1e6:.2f}M")
            self.debug_logger.info("=" * 60)

        # 修改为 3 层文本自注意力模块
        self.text_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.text_width, num_heads=4, dropout=0.1) for _ in range(3)
        ])
        self.text_attn_norm_layers = nn.ModuleList([
            nn.LayerNorm(self.text_width) for _ in range(3)
        ])

        # 初始化融合模块
        self.fusion = get_fusion_module(fusion_config) if fusion_config else None
        self.feat_dim = fusion_config.get("output_dim", 256) if fusion_config else 256

        # 初始化可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)

        # 日志记录初始化信息
        if self.logger:
            self.debug_logger.info(
                f"Initialized model with scale: {self.scale.item():.4f}, fusion: {fusion_config.get('type', 'None')}")

        # 文本分词结果缓存
        self.text_cache = {}
    
    def _count_classifier_params(self):
        """计算分类分支的参数量"""
        params = sum(p.numel() for p in self.id_bn.parameters())
        params += sum(p.numel() for p in self.id_classifier.parameters())
        return params

    def encode_image(self, image):
        """
        编码图像，提取图像特征并进行标准化，使用 ViT/Vim 整个序列。

        Args:
            image (torch.Tensor): 输入图像，形状为 [batch_size, channels, height, width] 或更高维。

        Returns:
            torch.Tensor: 标准化后的图像嵌入，形状为 [batch_size, 256]。
        """
        if image is None:
            return None
        device = next(self.parameters()).device
        if image.dim() == 5:
            image = image.squeeze(-1)
        image = image.to(device)
        
        # 获取图像特征
        if self.vision_backbone_type == 'vim':
            # Vim 返回 [batch_size, seq_len, 384]
            image_embeds_raw = self.visual_encoder(image)
        else:
            # ViT 返回 BaseModelOutput，取 last_hidden_state [batch_size, seq_len, 768]
            image_outputs = self.visual_encoder(image)
            image_embeds_raw = image_outputs.last_hidden_state
            
        # 投影到统一维度 (如果是 Vim: 384->768; 如果是 ViT: Identity)
        image_embeds_raw = self.visual_proj(image_embeds_raw)
        
        # 后续处理保持不变 (解耦 -> 检索MLP -> 归一化)
        id_embeds, _, _ = self.disentangle(image_embeds_raw)  # [batch_size, hidden_size]
        image_embeds = self.shared_mlp(id_embeds)
        image_embeds = self.image_mlp(image_embeds)
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1, eps=1e-8)
        return image_embeds

    def encode_text(self, instruction):
        """
        编码文本，提取文本特征并进行标准化。
        Adapted for CLIP:
        1. Max length 77
        2. Projection 512 -> 768
        """
        if instruction is None:
            return None
        device = next(self.parameters()).device
        if isinstance(instruction, list):
            texts = instruction
        else:
            texts = [instruction]

        # 检查缓存以复用分词结果
        cache_key = tuple(texts)
        if cache_key in self.text_cache:
            tokenized = self.text_cache[cache_key]
        else:
            # CLIP Limit is 77
            tokenized = self.tokenizer(
                texts,
                padding='max_length',
                max_length=77,  # CLIP specific
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )
            self.text_cache[cache_key] = tokenized

        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        # CLIP编码
        # CLIP output: last_hidden_state=[B, 77, 512], pooler_output=[B, 512]
        # 我们使用 last_hidden_state 以保留序列信息用于后续 Attention
        text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state 
        
        # === 维度适配: 512 -> 768 ===
        text_embeds = self.text_proj(text_embeds) # [B, 77, 768]

        # 3 层自注意力处理
        # 注意: CLIP attention mask 是 1 (attend), 0 (ignore)
        # nn.MultiheadAttention key_padding_mask 需要 True (ignore), False (attend)
        # 所以使用 ~attention_mask.bool()
        
        text_embeds = text_embeds.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        for attn, norm in zip(self.text_attn_layers, self.text_attn_norm_layers):
            attn_output, _ = attn(
                query=text_embeds,
                key=text_embeds,
                value=text_embeds,
                key_padding_mask=~attention_mask.bool()
            )
            text_embeds = attn_output + text_embeds  # 残差连接
            text_embeds = norm(text_embeds)
        text_embeds = text_embeds.transpose(0, 1)  # [batch_size, seq_len, hidden_size]

        # 均值池化，结合 attention_mask 忽略填充 token
        attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        text_embeds = torch.sum(text_embeds * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        # 形状: [batch_size, hidden_size]

        # 降维和标准化
        text_embeds = self.shared_mlp(text_embeds)
        text_embeds = self.text_mlp(text_embeds)
        # 使用更稳定的归一化，添加eps避免除零
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1, eps=1e-8)

        if not isinstance(instruction, list):
            text_embeds = text_embeds.squeeze(0)
        return text_embeds

    def forward(self, image=None, cloth_instruction=None, id_instruction=None, return_attention=False):
        """
        前向传播，处理图像和文本输入，输出多模态特征和分类结果。
        
        === 重构后的流程（分支解耦）===
        1. ViT编码 → image_embeds [B, 197, 768]
        2. G-S3解耦 → id_embeds, cloth_embeds [B, 768]
        3. 分支1（分类）：id_embeds → id_for_classification → id_logits
        4. 分支2（检索）：id_embeds → shared_mlp → image_mlp → image_embeds
        5. 分支3（cloth）：cloth_embeds → shared_mlp → image_mlp → cloth_image_embeds

        Args:
            image (torch.Tensor, optional): 输入图像，形状为 [batch_size, channels, height, width]。
            cloth_instruction (str or list, optional): 服装描述文本。
            id_instruction (str or list, optional): 身份描述文本。
            return_attention (bool): 是否返回注意力图（用于可视化）。

        Returns:
            tuple: (image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                    cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate, gate_weights,
                    id_cls_features)  # 新增：分类分支的中间特征
                   或包含注意力图的扩展元组
        """
        device = next(self.parameters()).device
        id_logits, id_embeds, cloth_embeds, gate = None, None, None, None
        id_attn_map, cloth_attn_map = None, None
        id_cls_features = None  # 新增：分类分支的中间特征
        
        if image is not None:
            if image.dim() == 5:
                image = image.squeeze(-1)
            image = image.to(device)
            
            # === 获取图像原始特征 ===
            if self.vision_backbone_type == 'vim':
                image_embeds_raw = self.visual_encoder(image) # [B, 197, 384]
            else:
                image_outputs = self.visual_encoder(image)
                image_embeds_raw = image_outputs.last_hidden_state  # [B, 197, 768]
            
            # 检查编码器输出NaN
            if torch.isnan(image_embeds_raw).any():
                if self.logger:
                    self.debug_logger.error("⚠️  CRITICAL: Visual encoder output contains NaN!")
                # 返回零张量避免崩溃
                image_embeds_raw = torch.zeros_like(image_embeds_raw)
            
            # === 维度对齐 (Vim 384 -> 768, ViT 768 -> 768) ===
            image_embeds_raw = self.visual_proj(image_embeds_raw)
            
            # 检查投影后NaN
            if torch.isnan(image_embeds_raw).any():
                if self.logger:
                    self.debug_logger.error("⚠️  CRITICAL: Visual projection output contains NaN!")
                image_embeds_raw = torch.zeros_like(image_embeds_raw)
            
            # ============================================================
            # 步骤1：对称G-S3解耦，得到id和attr特征
            # ============================================================
            if return_attention:
                id_embeds, cloth_embeds, gate_stats, original_feat, id_attn_map, cloth_attn_map = self.disentangle(
                    image_embeds_raw, return_attention=True)
            else:
                id_embeds, cloth_embeds, gate_stats, original_feat = self.disentangle(
                    image_embeds_raw, return_attention=False)
            
            # NaN检查
            if torch.isnan(id_embeds).any():
                if self.logger:
                    self.debug_logger.error("⚠️  CRITICAL: id_embeds contains NaN after disentangle!")
                id_embeds = torch.zeros_like(id_embeds)
            if torch.isnan(cloth_embeds).any():
                if self.logger:
                    self.debug_logger.error("⚠️  CRITICAL: cloth_embeds contains NaN after disentangle!")
                cloth_embeds = torch.zeros_like(cloth_embeds)
            
            # gate_stats现在是一个dict，记录到日志
            if self.logger and hasattr(self, '_log_counter_gate'):
                self._log_counter_gate = getattr(self, '_log_counter_gate', 0) + 1
                if self._log_counter_gate % 200 == 0:
                    self.debug_logger.debug(
                        f"Gate stats: ID[mean={gate_stats['gate_id_mean']:.4f}, std={gate_stats['gate_id_std']:.4f}], "
                        f"Attr[mean={gate_stats['gate_attr_mean']:.4f}, std={gate_stats['gate_attr_std']:.4f}], "
                        f"Diversity={gate_stats['diversity']:.4f}"
                    )
            
            # 存储中间特征用于调试（仅在 debug 模式下）
            if hasattr(self, '_debug_mode') and self._debug_mode:
                self._debug_features = {
                    'image_embeds_raw': image_embeds_raw,
                    'id_embeds': id_embeds,
                    'cloth_embeds': cloth_embeds,
                    'gate_stats': gate_stats
                }
                if hasattr(self.disentangle, '_debug_info'):
                    self._debug_features.update(self.disentangle._debug_info)
            
            # ============================================================
            # 步骤2：分支1 - 分类分支（BNNeck 隐式监督）
            # ============================================================
            # 1. 通过 BNNeck 进行特征规范化
            # id_embeds [B, 768] → id_cls_features [B, 768]
            id_cls_features = self.id_bn(id_embeds)
            
            # 2. 计算分类 Logits
            # id_cls_features [B, 768] → id_logits [B, num_classes]
            id_logits = self.id_classifier(id_cls_features)
            
            # 注意：检索分支依然使用原始 id_embeds，这使得 id_embeds 同时受到 
            # BNNeck(分类) 和 MLP(检索) 的双重梯度约束，促进特征鲁棒性
            
            # ============================================================
            # 步骤3：分支2 - 检索分支（用于info_nce）
            # ============================================================
            # id_embeds [B, 768] → shared_mlp [B, 512] → image_mlp [B, 256]
            image_embeds = self.shared_mlp(id_embeds)
            image_embeds = self.image_mlp(image_embeds)
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1, eps=1e-8)
            
            # ============================================================
            # 步骤4：分支3 - cloth检索分支（用于cloth_semantic）
            # ============================================================
            # cloth_embeds [B, 768] → shared_mlp [B, 512] → image_mlp [B, 256]
            cloth_image_embeds = self.shared_mlp(cloth_embeds)
            cloth_image_embeds = self.image_mlp(cloth_image_embeds)
            cloth_image_embeds = torch.nn.functional.normalize(cloth_image_embeds, dim=-1, eps=1e-8)
        else:
            image_embeds = None
            cloth_image_embeds = None
            gate_stats = None

        # ============================================================
        # 步骤5：文本编码和融合
        # ============================================================
        cloth_text_embeds = self.encode_text(cloth_instruction)
        id_text_embeds = self.encode_text(id_instruction)
        
        fused_embeds, gate_weights = None, None
        if self.fusion and image_embeds is not None and id_text_embeds is not None:
            fused_embeds, gate_weights = self.fusion(image_embeds, id_text_embeds)
            fused_embeds = self.scale * torch.nn.functional.normalize(fused_embeds, dim=-1, eps=1e-8)
        else:
            # Fusion模块未激活时，使用image_embeds作为fallback
            # 确保fused_embeds始终有值参与损失计算
            if image_embeds is not None:
                fused_embeds = image_embeds
        
        # ============================================================
        # 返回值（新增：original_feat用于重构监督）
        # ============================================================
        # 注意：gate_stats是dict，包含门控统计信息
        base_outputs = (image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                       cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate_stats, gate_weights,
                       id_cls_features, original_feat)  # 新增original_feat
        
        if return_attention:
            return base_outputs + (id_attn_map, cloth_attn_map)
        else:
            return base_outputs

    def load_param(self, trained_path):
        """
        加载预训练模型参数。

        Args:
            trained_path (str): 预训练模型文件路径。

        Returns:
            T2IReIDModel: 加载参数后的模型。
        """
        trained_path = Path(trained_path)
        checkpoint = torch.load(trained_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        self = copy_state_dict(state_dict, self)
        if self.logger:
            self.debug_logger.info(f"Loaded checkpoint from {trained_path}, scale: {self.scale.item():.4f}")
        return self