"""
AH-Net 异构双流核心组件 (Extreme Performance Ver.)
包含：
1. IDStructureStream: PatchMerging (Learnable Downsample) + Mamba
2. AttributeTextureStream: CNN + ECA (Channel Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class PatchMerging(nn.Module):
    """
    可学习的下采样模块 (替代 AvgPool)
    类似于 Swin Transformer 的 Patch Merging，但这里简化为 Strided Conv
    """
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.conv = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv(x) # [B, C, H/2, W/2]
        # LayerNorm expects [B, L, C] or [B, C, H, W] if handled carefully. 
        # Pytorch LayerNorm default is over last dim.
        # Let's permute for norm then back.
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) # [B, N, C]
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class EfficientChannelAttention(nn.Module):
    """
    ECA-Net: Efficient Channel Attention
    无需降维的自适应通道注意力，增强纹理特征的选择性。
    """
    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x) # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2) # [B, 1, C]
        y = self.conv(y) # [B, 1, C]
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1)) # [B, C, 1, 1]
        return x * y.expand_as(x)

class IDStructureStream(nn.Module):
    """
    ID 结构流 (Extreme Ver.)
    升级：AvgPool -> PatchMerging (保留更多结构信息)
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, logger=None):
        super().__init__()
        self.logger = logger
        
        # 1. 空间降采样: Patch Merging (Learnable)
        self.downsample = PatchMerging(dim)
        
        # 2. 全局序列建模: Mamba
        if Mamba is None:
            raise ImportError("Mamba module not found. IDStructureStream requires mamba-ssm.")
            
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, D, H, W] 输入特征图
        Returns:
            out: [B, D, H/2, W/2] ID结构特征
        """
        # 1. 降采样
        x_down = self.downsample(x) # [B, D, h, w]
        B, D, h, w = x_down.shape
        
        # 2. 序列化
        x_seq = x_down.flatten(2).transpose(1, 2)
        
        # 3. Mamba 建模
        dtype_in = x_seq.dtype
        x_seq = x_seq.float()
        x_seq = self.norm(x_seq)
        x_mamba = self.mamba(x_seq)
        x_mamba = x_mamba.to(dtype_in)
        
        # 4. 还原
        out = x_mamba.transpose(1, 2).reshape(B, D, h, w)
        
        return out


class AttributeTextureStream(nn.Module):
    """
    属性纹理流 (Extreme Ver.)
    升级：Bottleneck -> Bottleneck + ECA (关注关键纹理通道)
    """
    def __init__(self, dim, grid_size, logger=None):
        super().__init__()
        self.logger = logger
        
        # 1. PE
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, grid_size[0], grid_size[1]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 2. Bottleneck CNN (2层堆叠)
        hidden_dim = dim // 2
        
        self.layers = nn.ModuleList([
            self._make_bottleneck(dim, hidden_dim),
            self._make_bottleneck(dim, hidden_dim)
        ])
        
    def _make_bottleneck(self, dim, hidden_dim):
        return nn.Sequential(
            # 降维
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 局部特征 (3x3)
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 升维
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            # 🔥 注入 ECA Attention
            EfficientChannelAttention(kernel_size=3)
        )
        
    def forward(self, x):
        out = x + self.pos_embed
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual
        return out