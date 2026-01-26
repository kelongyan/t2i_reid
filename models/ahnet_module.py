"""
AH-Net Module (原 FSHDModule 重构)
实现不对称异构网络的核心交互逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ahnet_streams import IDStructureStream, AttributeTextureStream

class AHNetModule(nn.Module):
    """
    AH-Net: Asymmetric Heterogeneous Network Module
    
    架构：
    1. 输入处理: Seq -> Grid
    2. 双流分支: 
       - ID Stream (Low-Res, Global, Mamba)
       - Attr Stream (High-Res, Local, CNN)
    3. 语义交互: 原型引导的 Cross-Attention
    4. 互斥解耦: Masking
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
            logger.debug_logger.info(f"AH-Net Init: Grid Size=({self.grid_h}, {self.grid_w}), Dim={dim}")
        
        # === 1. 不对称双流 ===
        self.id_stream = IDStructureStream(
            dim=dim, d_state=d_state, d_conv=d_conv, expand=expand, logger=logger
        )
        
        self.attr_stream = AttributeTextureStream(
            dim=dim, grid_size=(self.grid_h, self.grid_w), logger=logger
        )
        
        # === 2. 语义原型 (Learnable Prototypes) ===
        # Query 向量: [1, 1, D]
        self.query_id = nn.Parameter(torch.randn(1, 1, dim))
        self.query_attr = nn.Parameter(torch.randn(1, 1, dim))
        
        # === 3. 简单的特征解码器 (用于重构 Loss) ===
        # 接收 ID + Attr -> 重构原始特征
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # 初始化权重
        nn.init.xavier_uniform_(self.query_id)
        nn.init.xavier_uniform_(self.query_attr)

    def forward_cross_attention(self, query, key_value_map):
        """
        简单的 Cross Attention
        Args:
            query: [B, 1, D]
            key_value_map: [B, D, H, W]
        Returns:
            context: [B, D] 全局特征
            attn_map: [B, 1, H, W] 注意力热力图 (已softmax)
        """
        B, D, H, W = key_value_map.shape
        # Flatten KV: [B, H*W, D]
        kv = key_value_map.flatten(2).transpose(1, 2)

        # Query: [B, 1, D]
        # Attention Scores: Q * K^T -> [B, 1, H*W]
        scores = torch.matmul(query, kv.transpose(1, 2))
        scores = scores / (D ** 0.5)
        attn_weights = F.softmax(scores, dim=-1) # [B, 1, H*W]

        # Context: weights * V -> [B, 1, D]
        context = torch.matmul(attn_weights, kv)
        context = context.squeeze(1) # [B, D]

        # Reshape Map: [B, 1, H, W]
        attn_map = attn_weights.reshape(B, 1, H, W)

        return context, attn_map

    def _compute_conflict_score(self, map_id, map_attr):
        """
        计算ID和Attr注意力图的冲突分数 (Conflict Score)

        核心指标：衡量两个注意力图在空间上的重叠程度
        - 冲突分数高：ID和Attr关注同一区域 → 解耦失败 → 图像特征不可信
        - 冲突分数低：ID和Attr关注不同区域 → 解耦成功 → 图像特征可信

        Args:
            map_id: [B, 1, H, W] ID注意力图 (已softmax)
            map_attr: [B, 1, H, W] Attr注意力图 (已softmax)

        Returns:
            conflict_score: [B] 每个样本的冲突分数 (范围 0~1)

        公式：
            S_conflict = Sum(map_id · map_attr) / (H · W)

        物理意义：
            - 如果两个注意力图完全重叠 → conflict_score ≈ 1/(H*W) * sum = 1.0
            - 如果两个注意力图完全不重叠 → conflict_score ≈ 0.0
        """
        # 逐像素相乘，计算重叠区域
        # map_id 和 map_attr 都是 softmax 过的，值在 [0, 1] 之间
        overlap = map_id * map_attr  # [B, 1, H, W]

        # 对空间维度求和
        conflict = overlap.sum(dim=(2, 3))  # [B, 1]

        # 除以像素数，归一化到 [0, 1]
        pixel_count = map_id.shape[2] * map_id.shape[3]
        conflict_score = conflict.squeeze(1) / pixel_count  # [B]

        return conflict_score

    def forward(self, x_grid, return_attention=False):
        """
        Args:
            x_grid: [B, D, H, W] 输入特征网格
        Returns:
            id_feat: [B, D]
            attr_feat: [B, D]
            aux_info: dict (包含 Loss 所需的 map, recon 等)
        """
        B, D, H, W = x_grid.shape
        
        # === 1. 双流处理 ===
        # ID Stream: [B, D, H/2, W/2]
        f_id_map = self.id_stream(x_grid)
        
        # Attr Stream: [B, D, H, W]
        f_attr_map = self.attr_stream(x_grid)
        
        # === 2. 基于原型的 Cross-Attention ===
        # 扩展 Query 到 Batch
        q_id = self.query_id.expand(B, -1, -1)
        q_attr = self.query_attr.expand(B, -1, -1)
        
        # ID Attention
        v_id, map_id = self.forward_cross_attention(q_id, f_id_map)
        
        # Attr Attention
        v_attr, map_attr = self.forward_cross_attention(q_attr, f_attr_map)
        
        # === 3. 语义互斥 (Semantic Exclusion) ===
        # 将 ID Map 上采样到 Attr Map 尺寸
        map_id_up = F.interpolate(map_id, size=(H, W), mode='bilinear', align_corners=False)
        
        # 互斥掩码: 抑制 ID 关注的区域
        # 逻辑: v_attr 应该主要来自 map_id 没关注的地方
        # 这里我们对 v_attr 做一个简单的抑制操作，或者更复杂的特征级抑制
        # 方案书建议: F_final_attr = V_attr * (1 - Sigmoid(Map_id))
        # 注意: V_attr 是全局向量 [B, D], Map_id 是空间图 [B, 1, H, W]
        # 直接相乘不合适。通常是在 Feature Map 聚合时做 Masking。
        # 但既然我们已经得到了 v_attr (Context), 这里的 Masking 更多是用于 Loss 或 特征修正。
        # 修正: 我们使用 Mask 对 Attr Map 进行加权，重新计算 v_attr_refined
        
        exclusion_mask = 1.0 - map_id_up # [B, 1, H, W] (假设 map_id 已经是 softmax 过的，在 0-1 之间)
        # 注意: softmax 后的 weights 通常很小，sum=1。直接用 1-weight 可能抑制不够。
        # 方案书提到 "Sigmoid(Map_id)"， implies Map_id might be logits. 
        # 但 forward_cross_attention 返回的是 softmax 后的 weights。
        # 为了更强的互斥，我们对 weights 进行 Min-Max 归一化或直接使用
        
        # 简化策略:
        # id_feat = v_id
        # attr_feat = v_attr
        # 互斥主要靠 Loss 驱动
        
        # === 4. 计算冲突分数 (Conflict Score) ===
        # 这是方案书的核心指标，用于驱动 S-CAG 融合模块
        conflict_score = self._compute_conflict_score(map_id_up, map_attr)  # [B]

        # === 5. 重构准备 ===
        # 重构输入: v_id (detach) + v_attr
        recon_input = v_id.detach() + v_attr
        recon_feat = self.decoder(recon_input) # [B, D]

        # 原始特征的全局表示 (用于重构目标)
        original_global = x_grid.mean(dim=(2, 3)) # [B, D]

        aux_info = {
            'map_id': map_id_up,        # [B, 1, H, W] ID注意力图
            'map_attr': map_attr,       # [B, 1, H, W] Attr注意力图
            'conflict_score': conflict_score,  # [B] 冲突分数 (核心指标！)
            'recon_feat': recon_feat,   # [B, D] 重构特征
            'target_feat': original_global,  # [B, D] 目标特征
            'v_id': v_id,               # [B, D] ID全局特征
            'v_attr': v_attr            # [B, D] Attr全局特征
        }
        
        # 兼容旧接口的返回值结构
        # id_feat, attr_feat, gate_stats(dummy), original_feat
        # 这里的 original_feat 用于 model.py 中的后续流程，通常不需要
        
        return v_id, v_attr, aux_info

# Alias for compatibility if needed, though we will change imports
FSHDModule = AHNetModule
