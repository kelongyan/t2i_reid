# models/gs3_module.py
"""
G-S3 (Geometry-Guided Selective State Space) Module - Symmetric Decoupling Version
实现对称式双分支解耦：基于 F_input ≈ F_id + F_attr 的设计理念

核心改进：
1. SymmetricBranchAttention: 对称双分支注意力（替代原OPA的单向抑制）
2. Dual Mamba Streams: ID流和Attribute流并行处理
3. Reconstruction Constraint: 保证信息完整性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class SymmetricBranchAttention(nn.Module):
    """
    对称双分支注意力模块 (Symmetric Branch Attention)
    
    核心思想（对称设计）：
    - 不再单向抑制，而是让两个分支同时学习不同的注意力模式
    - ID分支：关注结构性、全局性特征（体态、姿势）
    - Attr分支：关注局部性、纹理性特征（颜色、配饰）
    - 通过Soft Orthogonal Mask实现隐式分离
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.1, logger=None):
        """
        Args:
            dim (int): 输入特征维度
            num_heads (int): 多头注意力的头数
            qkv_bias (bool): QKV投影是否使用偏置
            dropout (float): Dropout比率
            logger: TrainingMonitor实例
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logger = logger
        
        # === 对称设计：两个分支独立的QKV投影 ===
        # ID分支：保留完整的多头注意力
        self.qkv_id = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Attr分支：同样完整的多头注意力
        self.qkv_attr = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # === 输出投影（对称） ===
        self.proj_id = nn.Linear(dim, dim)
        self.proj_attr = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # LayerNorm for stabilization
        self.norm_id = nn.LayerNorm(dim)
        self.norm_attr = nn.LayerNorm(dim)
        
        # === 对称设计的关键：Soft Orthogonal Mask ===
        # 使用可学习的温度参数控制正交化强度
        self.ortho_temperature = nn.Parameter(torch.tensor(1.0))
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，防止梯度爆炸"""
        for m in [self.qkv_id, self.qkv_attr, self.proj_id, self.proj_attr]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def compute_orthogonal_mask(self, attn_id, attn_attr):
        """
        计算软正交掩码：让两个分支的注意力模式互斥
        
        Args:
            attn_id: ID分支的注意力权重 [B, num_heads, N, N]
            attn_attr: Attr分支的注意力权重 [B, num_heads, N, N]
        
        Returns:
            mask_id, mask_attr: 软掩码，用于抑制重叠区域
        """
        # 计算注意力的空间重叠度
        # 使用KL散度衡量两个注意力分布的相似性
        # 注意：在sequence维度（最后一维）上计算KL
        
        # 为数值稳定性添加epsilon
        eps = 1e-8
        attn_id_safe = attn_id + eps
        attn_attr_safe = attn_attr + eps
        
        # 计算互信息（简化版：逐元素相乘后求和）
        overlap = (attn_id_safe * attn_attr_safe).sum(dim=-1, keepdim=True)  # [B, num_heads, N, 1]
        
        # 生成软掩码：overlap高的地方，对应分支的注意力应该降低
        # 使用sigmoid + temperature控制抑制强度
        temp = torch.clamp(self.ortho_temperature, min=0.1, max=5.0)
        mask_id = torch.sigmoid(-overlap * temp)      # overlap高 -> mask小（抑制）
        mask_attr = torch.sigmoid(-overlap * temp)
        
        return mask_id, mask_attr
    
    def forward(self, x, return_attention=False):
        """
        对称双分支前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            return_attention: 是否返回注意力图
            
        Returns:
            id_features, attr_features, (可选)attention_maps
        """
        B, N, C = x.shape
        
        # === 步骤 1: 生成 QKV（对称） ===
        # ID分支
        qkv_id = self.qkv_id(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv_id = qkv_id.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q_id, k_id, v_id = qkv_id[0], qkv_id[1], qkv_id[2]
        
        # Attr分支（对称）
        qkv_attr = self.qkv_attr(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv_attr = qkv_attr.permute(2, 0, 3, 1, 4)
        q_attr, k_attr, v_attr = qkv_attr[0], qkv_attr[1], qkv_attr[2]
        
        # === 步骤 2: 计算注意力（无正交投影，让模型自由学习）===
        # ID分支注意力
        attn_id = (q_id @ k_id.transpose(-2, -1)) * self.scale
        attn_id = torch.clamp(attn_id, min=-20.0, max=20.0)
        attn_id = attn_id.softmax(dim=-1)
        
        # Attr分支注意力
        attn_attr = (q_attr @ k_attr.transpose(-2, -1)) * self.scale
        attn_attr = torch.clamp(attn_attr, min=-20.0, max=20.0)
        attn_attr = attn_attr.softmax(dim=-1)
        
        # === 步骤 3: 应用软正交掩码（隐式引导分离）===
        mask_id, mask_attr = self.compute_orthogonal_mask(attn_id, attn_attr)
        
        # 应用掩码：让注意力模式互斥
        attn_id_masked = attn_id * mask_id
        attn_attr_masked = attn_attr * mask_attr
        
        # 重新归一化（保持概率分布性质）
        attn_id_masked = attn_id_masked / (attn_id_masked.sum(dim=-1, keepdim=True) + 1e-8)
        attn_attr_masked = attn_attr_masked / (attn_attr_masked.sum(dim=-1, keepdim=True) + 1e-8)
        
        # NaN检查
        if torch.isnan(attn_id_masked).any():
            attn_id_masked = torch.where(torch.isnan(attn_id_masked), torch.zeros_like(attn_id_masked), attn_id_masked)
        if torch.isnan(attn_attr_masked).any():
            attn_attr_masked = torch.where(torch.isnan(attn_attr_masked), torch.zeros_like(attn_attr_masked), attn_attr_masked)
        
        # Dropout
        attn_id_masked = self.attn_dropout(attn_id_masked)
        attn_attr_masked = self.attn_dropout(attn_attr_masked)
        
        # === 步骤 4: 应用注意力到Value ===
        out_id = (attn_id_masked @ v_id).transpose(1, 2).reshape(B, N, C)
        out_attr = (attn_attr_masked @ v_attr).transpose(1, 2).reshape(B, N, C)
        
        # === 步骤 5: 输出投影 ===
        id_features = self.proj_dropout(self.proj_id(out_id))
        attr_features = self.proj_dropout(self.proj_attr(out_attr))
        
        # LayerNorm
        id_features = self.norm_id(id_features)
        attr_features = self.norm_attr(attr_features)
        
        if return_attention:
            # 返回注意力图用于可视化
            attn_map_id = attn_id_masked.mean(dim=1)  # [B, N, N]
            attn_map_attr = attn_attr_masked.mean(dim=1)
            return id_features, attr_features, attn_map_id, attn_map_attr
        else:
            return id_features, attr_features


class DualMambaStream(nn.Module):
    """
    双Mamba流：对称式状态空间建模
    
    核心思想：
    - ID Stream: 使用标准Mamba捕捉长程依赖（体态、姿势的时序关系）
    - Attr Stream: 使用轻量Mamba或简化版SSM（局部纹理、颜色不需要复杂建模）
    """
    
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, logger=None):
        """
        Args:
            dim (int): 特征维度
            d_state (int): Mamba状态维度
            d_conv (int): Mamba卷积核大小
            expand (int): Mamba扩展因子
            logger: TrainingMonitor实例
        """
        super().__init__()
        self.logger = logger
        
        # === ID Stream: 标准Mamba（保留复杂度）===
        self.mamba_id = Mamba(
            d_model=dim,
            d_state=d_state,      # 较大的状态维度，捕捉长程依赖
            d_conv=d_conv,
            expand=expand
        )
        
        # === Attr Stream: 轻量Mamba（降低复杂度）===
        # 属性特征通常是局部的，不需要太复杂的SSM
        self.mamba_attr = Mamba(
            d_model=dim,
            d_state=max(d_state // 2, 8),  # 状态维度减半
            d_conv=d_conv,
            expand=max(expand // 2, 1)     # 扩展因子减半
        )
        
        # Pre-Norm
        self.norm_id = nn.LayerNorm(dim)
        self.norm_attr = nn.LayerNorm(dim)
        
    def forward(self, id_seq, attr_seq):
        """
        双流并行处理
        
        Args:
            id_seq: ID分支的序列特征 [B, N, dim]
            attr_seq: Attr分支的序列特征 [B, N, dim]
            
        Returns:
            id_filtered, attr_filtered: 过滤后的特征
        """
        # NaN检查
        if torch.isnan(id_seq).any() or torch.isnan(attr_seq).any():
            if self.logger:
                self.logger.debug_logger.warning("⚠️  DualMambaStream: Input contains NaN")
            return id_seq, attr_seq
        
        # === ID Stream 处理 ===
        id_normed = self.norm_id(id_seq)
        id_mamba_out = self.mamba_id(id_normed)
        
        if torch.isnan(id_mamba_out).any():
            if self.logger:
                self.logger.debug_logger.warning("⚠️  DualMambaStream: ID Mamba output contains NaN")
            id_mamba_out = id_seq
        
        id_filtered = id_seq + 0.1 * id_mamba_out  # 残差连接，初始scale较小
        
        # === Attr Stream 处理 ===
        attr_normed = self.norm_attr(attr_seq)
        attr_mamba_out = self.mamba_attr(attr_normed)
        
        if torch.isnan(attr_mamba_out).any():
            if self.logger:
                self.logger.debug_logger.warning("⚠️  DualMambaStream: Attr Mamba output contains NaN")
            attr_mamba_out = attr_seq
        
        attr_filtered = attr_seq + 0.1 * attr_mamba_out  # 残差连接
        
        return id_filtered, attr_filtered


class SymmetricGS3Module(nn.Module):
    """
    对称式G-S3模块 (Symmetric G-S3)
    
    架构设计：F_input ≈ F_id + F_attr
    
    核心组件：
    1. SymmetricBranchAttention: 双分支对称注意力
    2. DualMambaStream: 双Mamba流并行处理
    3. Symmetric Gating: 对称门控（两个分支同等重要）
    """
    
    def __init__(self, dim, num_heads=8, d_state=16, d_conv=4, dropout=0.1, logger=None):
        """
        Args:
            dim (int): 特征维度
            num_heads (int): 注意力头数
            d_state (int): Mamba状态维度
            d_conv (int): Mamba卷积核大小
            dropout (float): Dropout比率
            logger: TrainingMonitor实例
        """
        super().__init__()
        self.logger = logger
        
        # 阶段 1: 对称双分支注意力
        self.symmetric_attn = SymmetricBranchAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            logger=logger
        )
        
        # 阶段 2: 双Mamba流
        self.dual_mamba = DualMambaStream(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            logger=logger
        )
        
        # === 对称门控机制 (Symmetric Gating) ===
        # 关键改进：两个分支使用独立的门控，不再强制互补
        self.gate_id = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
        self.gate_attr = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
        # 全局池化
        self.pool_id = nn.AdaptiveAvgPool1d(1)
        self.pool_attr = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x, return_attention=False):
        """
        对称式前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            return_attention: 是否返回注意力图
            
        Returns:
            如果 return_attention=False:
                (id_feat, attr_feat, gate_stats, original_feat)
            如果 return_attention=True:
                (id_feat, attr_feat, gate_stats, original_feat, id_attn_map, attr_attn_map)
                
            注意：新增 original_feat 用于重构损失监督
        """
        batch_size, seq_len, dim = x.size()
        
        # 保存原始特征用于重构监督
        original_feat = self.pool_id(x.transpose(1, 2)).squeeze(-1)  # [B, dim]
        
        # === 阶段 1: 对称双分支注意力 ===
        if return_attention:
            id_seq, attr_seq, id_attn_map, attr_attn_map = \
                self.symmetric_attn(x, return_attention=True)
        else:
            id_seq, attr_seq = self.symmetric_attn(x, return_attention=False)
            id_attn_map, attr_attn_map = None, None
        
        # === 阶段 2: 双Mamba流处理 ===
        id_seq_filtered, attr_seq_filtered = self.dual_mamba(id_seq, attr_seq)
        
        # === 阶段 3: 全局池化 ===
        id_feat = self.pool_id(id_seq_filtered.transpose(1, 2)).squeeze(-1)     # [B, dim]
        attr_feat = self.pool_attr(attr_seq_filtered.transpose(1, 2)).squeeze(-1)  # [B, dim]
        
        # === 阶段 4: 对称门控 ===
        concat_feat = torch.cat([id_feat, attr_feat], dim=-1)  # [B, dim*2]
        
        gate_id = self.gate_id(concat_feat)      # [B, dim]
        gate_attr = self.gate_attr(concat_feat)  # [B, dim]
        
        # 防止门控塌缩：限制范围在 [0.2, 0.8]（扩大下界）
        gate_id = torch.clamp(gate_id, min=0.2, max=0.8)
        gate_attr = torch.clamp(gate_attr, min=0.2, max=0.8)
        
        # === 关键改进：对称应用门控 ===
        # 两个分支都应用门控，避免一个分支"摸鱼"
        id_feat_gated = gate_id * id_feat
        attr_feat_gated = gate_attr * attr_feat
        
        # === 门控统计信息 ===
        gate_stats = {
            'gate_id_mean': gate_id.mean().item(),
            'gate_id_std': gate_id.std().item(),
            'gate_attr_mean': gate_attr.mean().item(),
            'gate_attr_std': gate_attr.std().item(),
            'diversity': torch.abs(gate_id - gate_attr).mean().item()
        }
        
        # === 调试信息 ===
        if hasattr(self, '_debug_mode') and self._debug_mode:
            self._debug_info = {
                'id_seq': id_seq,
                'attr_seq': attr_seq,
                'id_seq_filtered': id_seq_filtered,
                'attr_seq_filtered': attr_seq_filtered,
                'gate_id': gate_id,
                'gate_attr': gate_attr,
                'gate_stats': gate_stats
            }
        
        if return_attention:
            return id_feat_gated, attr_feat_gated, gate_stats, original_feat, id_attn_map, attr_attn_map
        else:
            return id_feat_gated, attr_feat_gated, gate_stats, original_feat


# === 兼容性别名：保留旧名称 GS3Module ===
# 自动使用新的对称式设计
GS3Module = SymmetricGS3Module
