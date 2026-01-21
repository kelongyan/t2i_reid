# models/hybrid_stream.py
"""
异构双流模块 - ID Stream (Mamba) + Attr Stream (Multi-scale CNN)
体现频域-空域联合建模的核心差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MultiScaleCNN(nn.Module):
    """
    多尺度CNN模块 - 专门用于属性特征提取
    使用并行的不同感受野卷积捕捉局部纹理
    """
    def __init__(self, dim=768, num_scales=3):
        """
        Args:
            dim: 特征维度
            num_scales: 尺度数量（默认3：小、中、大感受野）
        """
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        # 多尺度卷积分支（并行）
        self.conv_branches = nn.ModuleList()
        
        # 尺度1：3x3卷积（小感受野，捕捉细粒度纹理）
        self.conv_branches.append(nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),  # Depthwise
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),  # Pointwise
            nn.BatchNorm1d(dim)
        ))
        
        # 尺度2：5x5空洞卷积（中感受野，捕捉局部图案）
        self.conv_branches.append(nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=5, padding=4, dilation=2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        ))
        
        # 尺度3：7x7空洞卷积（大感受野，捕捉衣物整体）
        self.conv_branches.append(nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=9, dilation=3, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.BatchNorm1d(dim)
        ))
        
        # 多尺度融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(dim * num_scales, dim, kernel_size=1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )
        
        # 残差连接的权重
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D]
        Returns:
            out: [B, N, D]
        """
        B, N, D = x.shape
        
        # Transpose for Conv1d: [B, N, D] → [B, D, N]
        x_transposed = x.transpose(1, 2)
        
        # 并行多尺度卷积
        multi_scale_features = []
        for conv_branch in self.conv_branches:
            feat = conv_branch(x_transposed)  # [B, D, N]
            multi_scale_features.append(feat)
        
        # 拼接多尺度特征
        concat_features = torch.cat(multi_scale_features, dim=1)  # [B, D*num_scales, N]
        
        # 融合
        fused_features = self.fusion(concat_features)  # [B, D, N]
        
        # Transpose back
        out = fused_features.transpose(1, 2)  # [B, N, D]
        
        # 残差连接（可学习权重）
        residual_weight = torch.sigmoid(self.residual_weight)
        out = x + residual_weight * out
        
        return out


class HybridDualStream(nn.Module):
    """
    异构双流架构
    - ID Stream: Mamba SSM（全局长程依赖建模）
    - Attr Stream: Multi-scale CNN（局部纹理提取）
    """
    def __init__(self, dim=768, d_state=16, d_conv=4, expand=2, use_multi_scale_cnn=True, logger=None):
        """
        Args:
            dim: 特征维度
            d_state: Mamba状态维度
            d_conv: Mamba卷积核大小
            expand: Mamba扩展因子
            use_multi_scale_cnn: 是否使用多尺度CNN（False则使用轻量Mamba）
            logger: 日志记录器
        """
        super().__init__()
        self.use_multi_scale_cnn = use_multi_scale_cnn
        self.logger = logger
        
        # === ID Stream: 标准Mamba（保留完整能力）===
        self.mamba_id = Mamba(
            d_model=dim,
            d_state=d_state,      # 较大的状态空间
            d_conv=d_conv,
            expand=expand
        )
        
        # === Attr Stream: 根据配置选择异构结构 ===
        if use_multi_scale_cnn:
            # 方案A：多尺度CNN（激进方案）
            self.attr_processor = MultiScaleCNN(dim=dim, num_scales=3)
            if logger:
                logger.debug_logger.info("Using Multi-scale CNN for Attr Stream")
        else:
            # 方案B：轻量Mamba（保守方案）
            self.attr_processor = Mamba(
                d_model=dim,
                d_state=max(d_state // 2, 8),
                d_conv=d_conv,
                expand=max(expand // 2, 1)
            )
            if logger:
                logger.debug_logger.info("Using Lightweight Mamba for Attr Stream")
        
        # LayerNorm
        self.norm_id = nn.LayerNorm(dim)
        self.norm_attr = nn.LayerNorm(dim)
        
        # 可学习的残差权重
        self.id_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.attr_residual_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, id_seq, attr_seq):
        """
        双流并行处理
        Args:
            id_seq: ID分支输入（低频增强） [B, N, D]
            attr_seq: Attr分支输入（高频增强） [B, N, D]
        Returns:
            id_filtered, attr_filtered
        """
        # NaN检查
        if torch.isnan(id_seq).any() or torch.isnan(attr_seq).any():
            if self.logger:
                self.logger.debug_logger.warning("⚠️  HybridDualStream: Input contains NaN")
            return id_seq, attr_seq
        
        # === ID Stream: Mamba处理 ===
        id_normed = self.norm_id(id_seq)
        id_mamba_out = self.mamba_id(id_normed)
        
        if torch.isnan(id_mamba_out).any():
            if self.logger:
                self.logger.debug_logger.warning("⚠️  HybridDualStream: ID Mamba output contains NaN")
            id_mamba_out = id_seq
        
        # 残差连接（可学习缩放）
        id_scale = torch.sigmoid(self.id_residual_scale)
        id_filtered = id_seq + id_scale * id_mamba_out
        
        # === Attr Stream: CNN/Mamba处理 ===
        attr_normed = self.norm_attr(attr_seq)
        
        if self.use_multi_scale_cnn and isinstance(self.attr_processor, MultiScaleCNN):
            # CNN不需要残差（内部已有）
            attr_filtered = self.attr_processor(attr_normed)
        else:
            # Mamba需要残差
            attr_out = self.attr_processor(attr_normed)
            if torch.isnan(attr_out).any():
                if self.logger:
                    self.logger.debug_logger.warning("⚠️  HybridDualStream: Attr output contains NaN")
                attr_out = attr_seq
            attr_scale = torch.sigmoid(self.attr_residual_scale)
            attr_filtered = attr_seq + attr_scale * attr_out
        
        return id_filtered, attr_filtered


class FrequencyGuidedAttention(nn.Module):
    """
    频域引导的双分支注意力
    在原有Symmetric Attention基础上，增加频域先验引导
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # === 对称双分支QKV投影 ===
        self.qkv_id = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_attr = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # === 频域引导投影 ===
        self.freq_guide_id = nn.Linear(dim, dim)    # 低频 → ID引导
        self.freq_guide_attr = nn.Linear(dim, dim)  # 高频 → Attr引导
        
        # 输出投影
        self.proj_id = nn.Linear(dim, dim)
        self.proj_attr = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # LayerNorm
        self.norm_id = nn.LayerNorm(dim)
        self.norm_attr = nn.LayerNorm(dim)
        
        # 软正交温度参数
        self.ortho_temperature = nn.Parameter(torch.tensor(1.0))
        
        # 频域引导强度（可学习）
        self.freq_guide_strength = nn.Parameter(torch.tensor(0.3))
    
    def compute_orthogonal_mask(self, attn_id, attn_attr):
        """计算软正交掩码"""
        eps = 1e-8
        attn_id_safe = attn_id + eps
        attn_attr_safe = attn_attr + eps
        
        overlap = (attn_id_safe * attn_attr_safe).sum(dim=-1, keepdim=True)
        temp = torch.clamp(self.ortho_temperature, min=0.1, max=5.0)
        mask_id = torch.sigmoid(-overlap * temp)
        mask_attr = torch.sigmoid(-overlap * temp)
        
        return mask_id, mask_attr
    
    def forward(self, x, low_freq_feat=None, high_freq_feat=None, return_attention=False):
        """
        频域引导的对称注意力
        Args:
            x: 输入特征 [B, N, D]
            low_freq_feat: 低频特征（可选） [B, N, D]
            high_freq_feat: 高频特征（可选） [B, N, D]
            return_attention: 是否返回注意力图
        """
        B, N, C = x.shape
        
        # === QKV生成 ===
        qkv_id = self.qkv_id(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv_id = qkv_id.permute(2, 0, 3, 1, 4)
        q_id, k_id, v_id = qkv_id[0], qkv_id[1], qkv_id[2]
        
        qkv_attr = self.qkv_attr(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv_attr = qkv_attr.permute(2, 0, 3, 1, 4)
        q_attr, k_attr, v_attr = qkv_attr[0], qkv_attr[1], qkv_attr[2]
        
        # === 频域引导（可选）===
        if low_freq_feat is not None and high_freq_feat is not None:
            freq_strength = torch.sigmoid(self.freq_guide_strength)
            
            # 低频引导ID分支
            low_freq_guide = self.freq_guide_id(low_freq_feat)  # [B, N, D]
            low_freq_guide = low_freq_guide.reshape(B, N, self.num_heads, self.head_dim)
            low_freq_guide = low_freq_guide.permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
            v_id = v_id + freq_strength * low_freq_guide
            
            # 高频引导Attr分支
            high_freq_guide = self.freq_guide_attr(high_freq_feat)
            high_freq_guide = high_freq_guide.reshape(B, N, self.num_heads, self.head_dim)
            high_freq_guide = high_freq_guide.permute(0, 2, 1, 3)
            v_attr = v_attr + freq_strength * high_freq_guide
        
        # === 计算注意力 ===
        attn_id = (q_id @ k_id.transpose(-2, -1)) * self.scale
        attn_id = torch.clamp(attn_id, min=-20.0, max=20.0)
        attn_id = attn_id.softmax(dim=-1)
        
        attn_attr = (q_attr @ k_attr.transpose(-2, -1)) * self.scale
        attn_attr = torch.clamp(attn_attr, min=-20.0, max=20.0)
        attn_attr = attn_attr.softmax(dim=-1)
        
        # === 软正交掩码 ===
        mask_id, mask_attr = self.compute_orthogonal_mask(attn_id, attn_attr)
        
        attn_id_masked = attn_id * mask_id
        attn_attr_masked = attn_attr * mask_attr
        
        # 重新归一化
        attn_id_masked = attn_id_masked / (attn_id_masked.sum(dim=-1, keepdim=True) + 1e-8)
        attn_attr_masked = attn_attr_masked / (attn_attr_masked.sum(dim=-1, keepdim=True) + 1e-8)
        
        # NaN检查
        if torch.isnan(attn_id_masked).any():
            attn_id_masked = torch.where(torch.isnan(attn_id_masked), 
                                        torch.zeros_like(attn_id_masked), attn_id_masked)
        if torch.isnan(attn_attr_masked).any():
            attn_attr_masked = torch.where(torch.isnan(attn_attr_masked), 
                                          torch.zeros_like(attn_attr_masked), attn_attr_masked)
        
        attn_id_masked = self.attn_dropout(attn_id_masked)
        attn_attr_masked = self.attn_dropout(attn_attr_masked)
        
        # === 应用注意力 ===
        out_id = (attn_id_masked @ v_id).transpose(1, 2).reshape(B, N, C)
        out_attr = (attn_attr_masked @ v_attr).transpose(1, 2).reshape(B, N, C)
        
        # === 输出投影 ===
        id_features = self.proj_dropout(self.proj_id(out_id))
        attr_features = self.proj_dropout(self.proj_attr(out_attr))
        
        id_features = self.norm_id(id_features)
        attr_features = self.norm_attr(attr_features)
        
        if return_attention:
            attn_map_id = attn_id_masked.mean(dim=1)
            attn_map_attr = attn_attr_masked.mean(dim=1)
            return id_features, attr_features, attn_map_id, attn_map_attr
        else:
            return id_features, attr_features
