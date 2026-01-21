# models/frequency_module.py
"""
频域分解模块 - FSHD-Net核心组件
支持DCT和Wavelet两种频域变换，可学习的频域掩码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableFrequencyMask(nn.Module):
    """
    可学习的频域掩码
    使用高斯分布初始化，通过反向传播优化频域关注区域
    """
    def __init__(self, freq_size, init_radius_ratio=0.5, mask_type='low'):
        """
        Args:
            freq_size: 频域尺寸 (H, W)
            init_radius_ratio: 初始半径比例
            mask_type: 'low' or 'high'
        """
        super().__init__()
        self.freq_size = freq_size
        self.mask_type = mask_type
        
        # 可学习的高斯中心和半径
        h, w = freq_size
        self.center_h = nn.Parameter(torch.tensor(h / 2.0))
        self.center_w = nn.Parameter(torch.tensor(w / 2.0))
        self.radius = nn.Parameter(torch.tensor(min(h, w) * init_radius_ratio / 2.0))
        
        # 可学习的锐度参数（控制边界的平滑程度）
        self.sharpness = nn.Parameter(torch.tensor(2.0))
    
    def forward(self):
        """
        生成频域掩码
        Returns:
            mask: [1, H, W]
        """
        h, w = self.freq_size
        device = self.center_h.device
        
        # 生成坐标网格
        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        
        # 计算到中心的距离
        dist = torch.sqrt((y_grid - self.center_h)**2 + (x_grid - self.center_w)**2)
        
        # 生成高斯掩码（使用可学习的锐度）
        sharpness_clamped = torch.clamp(self.sharpness, min=0.5, max=10.0)
        radius_clamped = torch.clamp(self.radius, min=1.0, max=min(h, w)/2.0)
        
        if self.mask_type == 'low':
            # 低通：中心为1，边缘为0
            mask = torch.exp(-(dist / radius_clamped)**sharpness_clamped)
        else:
            # 高通：中心为0，边缘为1
            mask = 1.0 - torch.exp(-(dist / radius_clamped)**sharpness_clamped)
        
        return mask.unsqueeze(0)  # [1, H, W]


class DCTFrequencySplitter(nn.Module):
    """
    基于DCT的频域分解模块
    使用2D DCT分解特征图，通过可学习掩码分离低频和高频
    """
    def __init__(self, dim=768, img_size=(14, 14), init_radius_ratio=0.5):
        """
        Args:
            dim: 特征维度
            img_size: Patch grid尺寸
            init_radius_ratio: 初始频域半径比例
        """
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        
        # 可学习的频域掩码
        self.low_freq_mask = LearnableFrequencyMask(img_size, init_radius_ratio, 'low')
        self.high_freq_mask = LearnableFrequencyMask(img_size, init_radius_ratio, 'high')
        
        # 频域融合权重（自适应控制频域信息的强度）
        self.alpha_low = nn.Parameter(torch.tensor(0.3))
        self.alpha_high = nn.Parameter(torch.tensor(0.3))
        
        # 通道注意力（不同通道可能需要不同的频域处理）
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 2),  # 输出低频和高频的通道权重
            nn.Sigmoid()
        )
    
    def dct2d(self, x):
        """
        2D DCT变换（使用FFT实现，更高效）
        Args:
            x: [B, C, H, W]
        Returns:
            dct_coeff: [B, C, H, W]
        """
        # 使用FFT实现DCT（更快）
        # [Fix] Force float32 for FFT to avoid cuFFT half-precision power-of-2 limitation
        x_fft = torch.fft.fft2(x.float(), dim=(-2, -1))
        return x_fft
    
    def idct2d(self, dct_coeff, target_dtype=None):
        """
        2D IDCT变换
        Args:
            dct_coeff: [B, C, H, W]
            target_dtype: 目标数据类型（如float16），如果为None则返回默认float32
        Returns:
            x: [B, C, H, W]
        """
        x = torch.fft.ifft2(dct_coeff, dim=(-2, -1))
        # [Fix] Cast back to original dtype (e.g., float16) if specified
        x_real = x.real
        if target_dtype is not None:
            return x_real.to(target_dtype)
        return x_real
    
    def forward(self, x, cls_token_idx=None):
        """
        频域分解
        Args:
            x: [B, N, D] 输入序列（包含CLS token）
            cls_token_idx: CLS token位置（Vim: N//2, ViT: 0）
        Returns:
            low_freq_seq: 低频特征序列
            high_freq_seq: 高频特征序列
            freq_info: 频域信息字典
        """
        B, N, D = x.shape
        H, W = self.img_size
        
        # 1. 提取CLS token（不参与频域变换）
        if cls_token_idx is not None:
            cls_token = x[:, cls_token_idx:cls_token_idx+1, :]  # [B, 1, D]
            # 移除CLS token
            if cls_token_idx == 0:
                patches = x[:, 1:, :]
            elif cls_token_idx == N - 1:
                patches = x[:, :-1, :]
            else:
                patches = torch.cat([x[:, :cls_token_idx, :], 
                                    x[:, cls_token_idx+1:, :]], dim=1)
        else:
            cls_token = None
            patches = x
        
        # 验证尺寸
        if patches.size(1) != H * W:
            raise ValueError(f"Expected {H*W} patches, got {patches.size(1)}")
        
        # 2. Reshape to 2D: [B, H*W, D] → [B, D, H, W]
        patches_2d = patches.transpose(1, 2).reshape(B, D, H, W)
        
        # 3. DCT变换
        freq_coeff = self.dct2d(patches_2d)  # Complex tensor
        freq_real = freq_coeff.real
        freq_imag = freq_coeff.imag
        
        # 4. 生成可学习的频域掩码
        low_mask = self.low_freq_mask()   # [1, H, W]
        high_mask = self.high_freq_mask()  # [1, H, W]
        
        # 5. 应用掩码分离频域
        low_freq_real = freq_real * low_mask
        low_freq_imag = freq_imag * low_mask
        high_freq_real = freq_real * high_mask
        high_freq_imag = freq_imag * high_mask
        
        # 6. IDCT变换回空域
        low_freq_complex = torch.complex(low_freq_real, low_freq_imag)
        high_freq_complex = torch.complex(high_freq_real, high_freq_imag)
        
        # [Fix] Pass x.dtype to ensure output matches input precision (e.g., Half)
        low_freq_spatial = self.idct2d(low_freq_complex, target_dtype=x.dtype)   # [B, D, H, W]
        high_freq_spatial = self.idct2d(high_freq_complex, target_dtype=x.dtype)  # [B, D, H, W]
        
        # 7. Reshape back to sequence: [B, D, H, W] → [B, H*W, D]
        low_freq_patches = low_freq_spatial.reshape(B, D, -1).transpose(1, 2)
        high_freq_patches = high_freq_spatial.reshape(B, D, -1).transpose(1, 2)
        
        # 8. 通道自适应门控
        # 使用全局平均池化的特征计算通道权重
        global_feat = patches.mean(dim=1)  # [B, D]
        channel_weights = self.channel_gate(global_feat)  # [B, 2]
        weight_low = channel_weights[:, 0:1].unsqueeze(1)  # [B, 1, 1]
        weight_high = channel_weights[:, 1:2].unsqueeze(1)  # [B, 1, 1]
        
        # 9. 加权融合（保留原始信息 + 频域增强）
        alpha_low = torch.sigmoid(self.alpha_low)
        alpha_high = torch.sigmoid(self.alpha_high)
        
        low_freq_patches = low_freq_patches * weight_low * alpha_low
        high_freq_patches = high_freq_patches * weight_high * alpha_high
        
        # 10. 重新插入CLS token
        if cls_token is not None:
            if cls_token_idx == 0:
                low_freq_seq = torch.cat([cls_token, low_freq_patches], dim=1)
                high_freq_seq = torch.cat([cls_token, high_freq_patches], dim=1)
            elif cls_token_idx == N - 1:
                low_freq_seq = torch.cat([low_freq_patches, cls_token], dim=1)
                high_freq_seq = torch.cat([high_freq_patches, cls_token], dim=1)
            else:
                low_freq_seq = torch.cat([
                    low_freq_patches[:, :cls_token_idx, :],
                    cls_token,
                    low_freq_patches[:, cls_token_idx:, :]
                ], dim=1)
                high_freq_seq = torch.cat([
                    high_freq_patches[:, :cls_token_idx, :],
                    cls_token,
                    high_freq_patches[:, cls_token_idx:, :]
                ], dim=1)
        else:
            low_freq_seq = low_freq_patches
            high_freq_seq = high_freq_patches
        
        # 11. 返回频域信息（用于可视化和损失计算）
        freq_info = {
            'low_freq': low_freq_seq.detach(),
            'high_freq': high_freq_seq.detach(),
            'low_mask': low_mask.detach(),
            'high_mask': high_mask.detach(),
            'alpha_low': alpha_low.item(),
            'alpha_high': alpha_high.item(),
            'channel_weights': channel_weights.detach(),
            'freq_magnitude': torch.abs(freq_coeff).mean(dim=(1, 2, 3)).detach()  # [B]
        }
        
        return low_freq_seq, high_freq_seq, freq_info


class WaveletFrequencySplitter(nn.Module):
    """
    基于小波变换的轻量级频域分解
    计算效率高，边界效应小
    """
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim
        
        # 可学习的小波滤波器（Depthwise Conv1d）
        self.low_pass_filter = nn.Conv1d(
            dim, dim, kernel_size=5, padding=2, groups=dim, bias=False
        )
        self.high_pass_filter = nn.Conv1d(
            dim, dim, kernel_size=5, padding=2, groups=dim, bias=False
        )
        
        # 初始化为Haar小波
        self._init_wavelet_filters()
        
        # 自适应融合权重
        self.alpha_low = nn.Parameter(torch.tensor(0.2))
        self.alpha_high = nn.Parameter(torch.tensor(0.2))
    
    def _init_wavelet_filters(self):
        """初始化小波滤波器权重"""
        # Low-pass: 平滑滤波器 [1, 2, 4, 2, 1] / 10
        low_kernel = torch.tensor([1, 2, 4, 2, 1], dtype=torch.float32) / 10.0
        # High-pass: 边缘检测 [-1, -2, 0, 2, 1] / 6
        high_kernel = torch.tensor([-1, -2, 0, 2, 1], dtype=torch.float32) / 6.0
        
        for i in range(self.dim):
            self.low_pass_filter.weight.data[i, 0, :] = low_kernel
            self.high_pass_filter.weight.data[i, 0, :] = high_kernel
    
    def forward(self, x, cls_token_idx=None):
        """
        小波分解
        Args:
            x: [B, N, D]
            cls_token_idx: CLS token位置
        Returns:
            low_freq_seq, high_freq_seq, freq_info
        """
        B, N, D = x.shape
        
        # 提取CLS token
        if cls_token_idx is not None:
            cls_token = x[:, cls_token_idx:cls_token_idx+1, :]
            if cls_token_idx == 0:
                patches = x[:, 1:, :]
            elif cls_token_idx == N - 1:
                patches = x[:, :-1, :]
            else:
                patches = torch.cat([x[:, :cls_token_idx, :], 
                                    x[:, cls_token_idx+1:, :]], dim=1)
        else:
            cls_token = None
            patches = x
        
        # Transpose for Conv1d: [B, N, D] → [B, D, N]
        patches_transposed = patches.transpose(1, 2)
        
        # 应用小波滤波
        low_freq = self.low_pass_filter(patches_transposed)
        high_freq = self.high_pass_filter(patches_transposed)
        
        # Transpose back: [B, D, N] → [B, N, D]
        low_freq_patches = low_freq.transpose(1, 2)
        high_freq_patches = high_freq.transpose(1, 2)
        
        # 加权
        alpha_low = torch.sigmoid(self.alpha_low)
        alpha_high = torch.sigmoid(self.alpha_high)
        
        low_freq_patches = low_freq_patches * alpha_low
        high_freq_patches = high_freq_patches * alpha_high
        
        # 重新插入CLS token
        if cls_token is not None:
            if cls_token_idx == 0:
                low_freq_seq = torch.cat([cls_token, low_freq_patches], dim=1)
                high_freq_seq = torch.cat([cls_token, high_freq_patches], dim=1)
            elif cls_token_idx == N - 1:
                low_freq_seq = torch.cat([low_freq_patches, cls_token], dim=1)
                high_freq_seq = torch.cat([high_freq_patches, cls_token], dim=1)
            else:
                low_freq_seq = torch.cat([
                    low_freq_patches[:, :cls_token_idx, :],
                    cls_token,
                    low_freq_patches[:, cls_token_idx:, :]
                ], dim=1)
                high_freq_seq = torch.cat([
                    high_freq_patches[:, :cls_token_idx, :],
                    cls_token,
                    high_freq_patches[:, cls_token_idx:, :]
                ], dim=1)
        else:
            low_freq_seq = low_freq_patches
            high_freq_seq = high_freq_patches
        
        freq_info = {
            'low_freq': low_freq_seq.detach(),
            'high_freq': high_freq_seq.detach(),
            'alpha_low': alpha_low.item(),
            'alpha_high': alpha_high.item()
        }
        
        return low_freq_seq, high_freq_seq, freq_info


def get_frequency_splitter(freq_type='dct', **kwargs):
    """
    工厂函数：根据类型创建频域分解器
    """
    if freq_type == 'dct':
        return DCTFrequencySplitter(**kwargs)
    elif freq_type == 'wavelet':
        return WaveletFrequencySplitter(**kwargs)
    else:
        raise ValueError(f"Unknown frequency type: {freq_type}")
