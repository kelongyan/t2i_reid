# models/ahnet_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ahnet_streams import IDStructureStream, AttributeTextureStream

class DynamicQueryGenerator(nn.Module):
    # 动态查询生成器：通过自适应池化和 MLP 将特征图映射为 Query 向量，实现实例感知的特征提取
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        # 初始化权重：使用较小的 gain 防止梯度爆炸
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, D, H, W] -> query: [B, 1, D]
        B, D, H, W = x.shape
        x_flat = self.pool(x).flatten(1)
        query = self.mlp(x_flat)
        return query.unsqueeze(1)


class MultiHeadAttention2D(nn.Module):
    # 针对 2D 特征图优化的多头注意力模块，用于 Query 与特征图之间的交互
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, 1) # 使用 1x1 卷积处理空间特征
        self.v_proj = nn.Conv2d(dim, dim, 1)
        
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # 权重初始化
        for m in [self.q_proj, self.out_proj]:
            nn.init.xavier_normal_(m.weight, gain=0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        for m in [self.k_proj, self.v_proj]:
            nn.init.xavier_normal_(m.weight, gain=0.05)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, query, feature_map):
        # query: [B, 1, D], feature_map: [B, D, H, W]
        B, _, D = query.shape
        _, _, H, W = feature_map.shape
        
        # 1. 投影变换
        q = self.q_proj(query).view(B, 1, self.num_heads, -1).permute(0, 2, 1, 3)
        k = self.k_proj(feature_map).flatten(2).view(B, self.num_heads, -1, H*W)
        v = self.v_proj(feature_map).flatten(2).view(B, self.num_heads, -1, H*W)
        
        # 2. 计算注意力分数
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 3. 计算上下文向量
        context = (attn @ v.transpose(-1, -2)).permute(0, 2, 1, 3).reshape(B, 1, D)
        context = self.out_proj(context)
        context = self.norm(context + query) # 残差连接与归一化
        context = context.squeeze(1)
        
        # 4. 生成注意力图用于可视化或损失计算
        attn_map_heads = attn.view(B, self.num_heads, H, W)
        attn_map_avg = attn_map_heads.mean(dim=1, keepdim=True)
        
        return context, attn_map_avg


class AHNetModule(nn.Module):
    # AH-Net: 不对称异构网络模块，通过双流结构（ID 结构流与属性纹理流）实现特征的精准分离
    def __init__(self, dim=384, img_size=(384, 128), patch_size=16, 
                 d_state=16, d_conv=4, expand=2, logger=None):
        super().__init__()
        self.dim = dim
        self.logger = logger
        
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        
        if logger:
            logger.debug_logger.info(f"🚀 AH-Net (Extreme): Grid=({self.grid_h}, {self.grid_w}), Dim={dim}, Heads=8")
        
        # 1. 不对称双流分支
        self.id_stream = IDStructureStream(
            dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, logger=logger
        )
        self.attr_stream = AttributeTextureStream(
            dim=dim, grid_size=(self.grid_h, self.grid_w), logger=logger
        )
        
        # 2. 动态查询生成
        self.id_query_gen = DynamicQueryGenerator(dim)
        self.attr_query_gen = DynamicQueryGenerator(dim)
        
        # 3. 多头注意力交互
        self.id_attn = MultiHeadAttention2D(dim, num_heads=8)
        self.attr_attn = MultiHeadAttention2D(dim, num_heads=8)
        
        self._init_weights()

    def _compute_conflict_score(self, map_id, map_attr):
        # 计算空间冲突分数：衡量 ID 注意力图与属性注意力图的重叠程度
        overlap = map_id * map_attr
        conflict = overlap.sum(dim=(2, 3))
        pixel_count = map_id.shape[2] * map_id.shape[3]
        conflict_score = conflict.squeeze(1) / pixel_count
        return conflict_score

    def forward(self, x_grid, return_attention=False):
        # x_grid: 输入的 4D 特征网格 [B, D, H, W]
        assert x_grid.dim() == 4, f"Expected 4D tensor [B, D, H, W], got {x_grid.dim()}D"
        B, D, H, W = x_grid.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"
        
        # 1. 双流特征提取
        f_id_map = self.id_stream(x_grid)
        f_attr_map = self.attr_stream(x_grid)
        
        # 2. 生成实例感知的 Query
        q_id = self.id_query_gen(f_id_map)
        q_attr = self.attr_query_gen(f_attr_map)
        
        # 3. 注意力交互，获取分离后的特征和权重图
        v_id, map_id = self.id_attn(q_id, f_id_map)
        v_attr, map_attr = self.attr_attn(q_attr, f_attr_map)
        
        # 4. 空间冲突与正交性计算
        map_id_up = F.interpolate(map_id, size=(H, W), mode='bilinear', align_corners=False)
        conflict_score = self._compute_conflict_score(map_id_up, map_attr)
        
        # 计算 Query 的正交正则化项
        q_id_norm = F.normalize(q_id.squeeze(1), p=2, dim=1)
        q_attr_norm = F.normalize(q_attr.squeeze(1), p=2, dim=1)
        ortho_reg = (q_id_norm * q_attr_norm).sum(dim=1).abs().mean()

        # 5. 特征重构：验证解耦特征是否完整保留了原始信息
        original_global = x_grid.mean(dim=(2, 3))

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
            'target_feat': original_global,
            'ortho_reg': ortho_reg,
            'v_id': v_id,
            'v_attr': v_attr
        }
        
        return v_id, v_attr, aux_info
    
    def _init_weights(self):
        # 解码器权重初始化
        pass


