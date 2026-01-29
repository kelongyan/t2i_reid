# models/ahnet_streams.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 Mamba SSM 库
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class PatchMerging(nn.Module):
    # 可学习的下采样模块：通过跨步卷积和归一化替代传统的池化操作，保留更多空间结构信息
    def __init__(self, dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.conv = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, C, H/2, W/2]
        x = self.conv(x)
        B, C, H, W = x.shape
        # 对空间维度进行平坦化以适应 LayerNorm，然后再还原
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

class EfficientChannelAttention(nn.Module):
    # ECA-Net: 高效通道注意力模块。利用 1D 卷积实现局部跨通道交互，增强对关键纹理特征的选择性
    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)

class IDStructureStream(nn.Module):
    # ID 结构流：专注于提取身份相关的结构信息，使用 Mamba 进行全局序列建模
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, logger=None):
        super().__init__()
        self.logger = logger
        
        # 1. 可学习的下采样
        self.downsample = PatchMerging(dim)
        
        # 2. Mamba 序列建模
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
        # 1. 空间降采样
        x_down = self.downsample(x)
        B, D, h, w = x_down.shape
        
        # 2. 转换为序列形式
        x_seq = x_down.flatten(2).transpose(1, 2)
        
        # 3. Mamba 建模：处理全局结构关联
        dtype_in = x_seq.dtype
        x_seq = x_seq.float()
        x_seq = self.norm(x_seq)
        x_mamba = self.mamba(x_seq)
        x_mamba = x_mamba.to(dtype_in)
        
        # 4. 还原为特征图
        out = x_mamba.transpose(1, 2).reshape(B, D, h, w)
        
        return out


class AttributeTextureStream(nn.Module):
    # 属性纹理流：专注于提取服装纹理和颜色信息，采用带有通道注意力的 CNN 瓶颈结构
    def __init__(self, dim, grid_size, logger=None):
        super().__init__()
        self.logger = logger
        
        # 1. 空间位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, grid_size[0], grid_size[1]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 2. 堆叠的瓶颈卷积层
        hidden_dim = dim // 2
        self.layers = nn.ModuleList([
            self._make_bottleneck(dim, hidden_dim),
            self._make_bottleneck(dim, hidden_dim)
        ])
        
    def _make_bottleneck(self, dim, hidden_dim):
        # 瓶颈结构：1x1 降维 -> 3x3 局部特征提取 -> 1x1 升维 -> ECA 通道注意力
        return nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            EfficientChannelAttention(kernel_size=3)
        )
        
    def forward(self, x):
        # 注入位置信息并进行残差连接处理
        out = x + self.pos_embed
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual
        return out
