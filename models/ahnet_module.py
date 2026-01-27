"""
AH-Net Module (Optimized)
实现不对称异构网络的核心交互逻辑
- 升级: 静态 Query -> 动态实例感知 Query
- 升级: 单头 Attention -> 多头 Attention (8 Heads)
- 新增: Query 正交性正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ahnet_streams import IDStructureStream, AttributeTextureStream

class DynamicQueryGenerator(nn.Module):
    """
    动态 Query 生成器
    将特征图压缩并映射为 Query 向量，赋予模型"实例感知"能力。
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim // 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)
        )
        
        # 初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """🔥 更安全的权重初始化"""
        if isinstance(m, nn.Linear):
            # 使用更小的标准差，防止NaN梯度
            nn.init.xavier_normal_(m.weight, gain=0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, D, H, W]
        Returns:
            query: [B, 1, D]
        """
        B, D, H, W = x.shape
        x_flat = self.pool(x).flatten(1) # [B, D]
        query = self.mlp(x_flat)         # [B, D]
        return query.unsqueeze(1)        # [B, 1, D]


class MultiHeadAttention2D(nn.Module):
    """
    针对 2D 特征图优化的多头注意力模块
    🔥 改进：更安全的权重初始化
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, 1) # Use 1x1 Conv for spatial features
        self.v_proj = nn.Conv2d(dim, dim, 1)
        
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
        # 🔥 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """🔥 更安全的权重初始化"""
        # Q, K, V投影：使用更小的标准差
        for m in [self.q_proj, self.out_proj]:
            nn.init.xavier_normal_(m.weight, gain=0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        # K, V的1x1卷积
        for m in [self.k_proj, self.v_proj]:
            nn.init.xavier_normal_(m.weight, gain=0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, query, feature_map):
        """
        Args:
            query: [B, 1, D] (Dynamic Query)
            feature_map: [B, D, H, W] (Key/Value Source)
        Returns:
            context: [B, D]
            attn_map: [B, 1, H, W] (Averaged over heads for visualization)
        """
        B, _, D = query.shape
        _, _, H, W = feature_map.shape
        
        # 1. Projections
        # Q: [B, 1, D] -> [B, 1, Heads, Dim_Head] -> [B, Heads, 1, Dim_Head]
        q = self.q_proj(query).view(B, 1, self.num_heads, -1).permute(0, 2, 1, 3)
        
        # K, V: [B, D, H, W] -> [B, Heads, Dim_Head, H*W]
        k = self.k_proj(feature_map).flatten(2).view(B, self.num_heads, -1, H*W) # [B, H, D_h, N]
        v = self.v_proj(feature_map).flatten(2).view(B, self.num_heads, -1, H*W) # [B, H, D_h, N]
        
        # 2. Attention
        # Scores: Q * K^T -> [B, Heads, 1, N]
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn) # [B, Heads, 1, N]
        
        # 3. Context
        # Context: Attn * V^T -> [B, Heads, 1, Dim_Head] -> [B, 1, D]
        context = (attn @ v.transpose(-1, -2)).permute(0, 2, 1, 3).reshape(B, 1, D)
        context = self.out_proj(context)
        context = self.norm(context + query) # Residual + Norm
        context = context.squeeze(1) # [B, D]
        
        # 4. Attention Map for Visualization / Loss
        # Reshape [B, Heads, 1, H*W] -> [B, Heads, H, W]
        attn_map_heads = attn.view(B, self.num_heads, H, W)
        
        # Average over heads for downstream "Spatial Conflict" calculation
        # Or keep heads? AH-Net original logic uses simple overlap. 
        # Mean is a safe proxy for "Global Attention Intensity".
        attn_map_avg = attn_map_heads.mean(dim=1, keepdim=True) # [B, 1, H, W]
        
        return context, attn_map_avg


class AHNetModule(nn.Module):
    """
    AH-Net: Asymmetric Heterogeneous Network Module (Extreme Performance Ver.)
    
    架构升级：
    1. 输入处理: Seq -> Grid
    2. 双流分支: ID Stream (Mamba) & Attr Stream (CNN)
    3. 交互机制: 
       - Dynamic Query Generation (Instance Aware)
       - Multi-Head Attention (High Capacity)
    4. 互斥解耦: Conflict Score + Orthogonality Regularization
    """
    def __init__(self, dim=384, img_size=(384, 128), patch_size=16, 
                 d_state=16, d_conv=4, expand=2, logger=None):
        super().__init__()
        self.dim = dim
        self.logger = logger
        
        # 计算网格尺寸
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        
        if logger:
            logger.debug_logger.info(f"🚀 AH-Net (Extreme): Grid=({self.grid_h}, {self.grid_w}), Dim={dim}, Heads=8")
        
        # === 1. 不对称双流 ===
        self.id_stream = IDStructureStream(
            dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, logger=logger
        )
        self.attr_stream = AttributeTextureStream(
            dim=dim, grid_size=(self.grid_h, self.grid_w), logger=logger
        )
        
        # === 2. 动态查询生成器 (Dynamic Query) ===
        self.id_query_gen = DynamicQueryGenerator(dim)
        self.attr_query_gen = DynamicQueryGenerator(dim)
        
        # === 3. 多头注意力 (Multi-Head Attention) ===
        self.id_attn = MultiHeadAttention2D(dim, num_heads=8)
        self.attr_attn = MultiHeadAttention2D(dim, num_heads=8)
        
        # === 4. 特征解码器 (用于重构 Loss) ===
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        
        # 🔥 初始化权重，防止NaN梯度
        self._init_weights()

    def _compute_conflict_score(self, map_id, map_attr):
        """
        计算冲突分数。
        输入为已平均的多头注意力图 [B, 1, H, W]
        """
        overlap = map_id * map_attr  # [B, 1, H, W]
        conflict = overlap.sum(dim=(2, 3))  # [B, 1]
        pixel_count = map_id.shape[2] * map_id.shape[3]
        conflict_score = conflict.squeeze(1) / pixel_count  # [B]
        return conflict_score

    def forward(self, x_grid, return_attention=False):
        """
        Args:
            x_grid: [B, D, H, W] 输入特征网格
        Returns:
            v_id: [B, D]
            v_attr: [B, D]
            aux_info: dict
        """
        # 🔥 添加输入验证
        assert x_grid.dim() == 4, f"Expected 4D tensor [B, D, H, W], got {x_grid.dim()}D"
        B, D, H, W = x_grid.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"
        
        # === 1. 双流处理 ===
        f_id_map = self.id_stream(x_grid) # [B, D, H/2, W/2]
        f_attr_map = self.attr_stream(x_grid) # [B, D, H, W]
        
        # === 2. 动态查询生成 ===
        # 根据各自流的特征生成"想看什么"的 Query
        q_id = self.id_query_gen(f_id_map)     # [B, 1, D]
        q_attr = self.attr_query_gen(f_attr_map) # [B, 1, D]
        
        # === 3. 多头注意力交互 ===
        v_id, map_id = self.id_attn(q_id, f_id_map)
        v_attr, map_attr = self.attr_attn(q_attr, f_attr_map)
        
        # === 4. 后处理与互斥 ===
        # 上采样 ID Map 使得尺寸匹配
        map_id_up = F.interpolate(map_id, size=(H, W), mode='bilinear', align_corners=False)
        
        # 计算空间冲突分数
        conflict_score = self._compute_conflict_score(map_id_up, map_attr)
        
        # 计算 Query 正交性 (用于 Loss 惩罚)
        # Cosine Similarity between Q_id and Q_attr
        q_id_norm = F.normalize(q_id.squeeze(1), p=2, dim=1)
        q_attr_norm = F.normalize(q_attr.squeeze(1), p=2, dim=1)
        ortho_reg = (q_id_norm * q_attr_norm).sum(dim=1).abs().mean()

        # === 5. 重构 ===
        # 🔥 修复 Bug #4: 移除v_id的detach(),让重构损失同时优化ID和Attr分支
        recon_input = v_id + v_attr
        recon_feat = self.decoder(recon_input)
        original_global = x_grid.mean(dim=(2, 3))

        # 🔥 调试日志
        if self.logger and hasattr(self, '_log_counter'):
            self._log_counter = getattr(self, '_log_counter', 0) + 1
            if self._log_counter % 200 == 0:
                self.logger.debug_logger.debug(
                    f"[AH-Net Extreme] Conflict: {conflict_score.mean():.4f} | Ortho Reg: {ortho_reg.item():.4f}"
                )

        aux_info = {
            'map_id': map_id_up,
            'map_attr': map_attr,
            'conflict_score': conflict_score,
            'recon_feat': recon_feat,
            'target_feat': original_global,
            'ortho_reg': ortho_reg, # 新增：正交正则项
            'v_id': v_id,
            'v_attr': v_attr
        }
        
        return v_id, v_attr, aux_info
    
    def _init_weights(self):
        """🔥 改进的权重初始化，防止NaN梯度"""
        # 初始化解码器
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

# Alias
FSHDModule = AHNetModule