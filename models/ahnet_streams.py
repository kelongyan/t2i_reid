"""
AH-Net 异构双流核心组件
包含：
1. IDStructureStream: 基于 Mamba 的低分辨率全局结构流
2. AttributeTextureStream: 基于 CNN 的高分辨率局部纹理流
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class IDStructureStream(nn.Module):
    """
    ID 结构流 (The Coarse-Grained ID Stream)
    设计目标：物理性抹除纹理细节，强迫模型学习长程身体结构。
    
    流程：
    1. AvgPool2d (降采样) -> 模糊细节，保留结构
    2. Flatten -> Mamba -> Reshape -> 全局长程建模
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, logger=None):
        super().__init__()
        self.logger = logger
        
        # 1. 空间降采样: AvgPool2d 2x2
        # 输入: [B, D, H, W] -> 输出: [B, D, H/2, W/2]
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        
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
        
        # 2. 序列化 [B, D, h, w] -> [B, h*w, D]
        x_seq = x_down.flatten(2).transpose(1, 2)
        
        # 3. Mamba 建模 (需确保 float32 稳定性)
        dtype_in = x_seq.dtype
        x_seq = x_seq.float()
        x_seq = self.norm(x_seq)
        x_mamba = self.mamba(x_seq)
        x_mamba = x_mamba.to(dtype_in)
        
        # 4. 还原为 2D [B, h*w, D] -> [B, D, h, w]
        out = x_mamba.transpose(1, 2).reshape(B, D, h, w)
        
        return out


class AttributeTextureStream(nn.Module):
    """
    属性纹理流 (The Fine-Grained Attribute Stream)
    设计目标：限制感受野，切断全局联系，强迫模型关注局部斑块。
    
    流程：
    1. 注入可学习位置编码 (PE)
    2. Stacked Bottleneck CNNs (1x1 -> 3x3 -> 1x1)
    """
    def __init__(self, dim, grid_size, logger=None):
        """
        Args:
            dim: 特征维度
            grid_size: (H, W) 输入网格尺寸
        """
        super().__init__()
        self.logger = logger
        
        # 1. 可学习的位置编码 [1, D, H, W]
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, grid_size[0], grid_size[1]))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 2. Bottleneck CNN (2层堆叠)
        # 限制感受野: 3x3 Conv, padding=1, 仅看局部
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
            # 局部特征提取 (3x3)
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # 升维
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, D, H, W]
        Returns:
            out: [B, D, H, W]
        """
        # 1. 注入 PE
        out = x + self.pos_embed
        
        # 2. CNN 处理 (Residual)
        for layer in self.layers:
            residual = out
            out = layer(out)
            out = out + residual # 残差连接
            
        return out