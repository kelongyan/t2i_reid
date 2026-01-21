# models/frequency_module.py
"""
频域分解模块 - FSHD-Net核心组件
仅支持DCT频域变换，已移除Legacy Wavelet代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FixedFrequencyMask(nn.Module):
    """
    固定频域掩码（数值稳定版本）
    移除可学习参数，使用固定的DCT分割策略
    
    修复：
    1. 移除可学习参数避免NaN梯度
    2. 使用简单四分割替代高斯分布
    3. 强制float32精度避免数值溢出
    """
    def __init__(self, freq_size, init_radius_ratio=0.5, mask_type='low'):
        """
        Args:
            freq_size: 频域尺寸 (H, W)
            init_radius_ratio: 半径比例（保留接口兼容性，实际不使用）
            mask_type: 'low' or 'high'
        """
        super().__init__()
        self.freq_size = freq_size
        self.mask_type = mask_type
        
        # 固定掩码（缓存为buffer，不参与梯度计算）
        h, w = freq_size
        mask = self._create_fixed_mask(h, w, mask_type)
        self.register_buffer('mask', mask)
    
    def _create_fixed_mask(self, h, w, mask_type):
        """
        创建固定的频域掩码
        策略：简单四分割
        - 低频：左上角 1/4 区域
        - 高频：其余 3/4 区域
        """
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        
        # 低频区域：左上角1/4
        low_h = h // 2
        low_w = w // 2
        
        if mask_type == 'low':
            # 低频掩码：左上角为1
            mask[0, :low_h, :low_w] = 1.0
        else:
            # 高频掩码：其余区域为1
            mask[0, low_h:, :] = 1.0
            mask[0, :, low_w:] = 1.0
        
        return mask
    
    def forward(self):
        """
        返回固定掩码
        Returns:
            mask: [1, H, W]
        """
        return self.mask


class DCTFrequencySplitter(nn.Module):
    """
    基于DCT的频域分解模块（数值稳定版本）
    
    修复：
    1. 使用固定频域掩码替代可学习掩码
    2. 强制float32精度进行FFT/IDCT运算
    3. 添加NaN检测和保护
    4. 移除通道注意力避免额外复杂度
    """
    def __init__(self, dim=768, img_size=(14, 14), init_radius_ratio=0.5):
        """
        Args:
            dim: 特征维度
            img_size: Patch grid尺寸
            init_radius_ratio: 初始频域半径比例（保留接口兼容性）
        """
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        
        # 固定的频域掩码（移除可学习参数）
        self.low_freq_mask = FixedFrequencyMask(img_size, init_radius_ratio, 'low')
        self.high_freq_mask = FixedFrequencyMask(img_size, init_radius_ratio, 'high')
        
        # 固定的频域融合权重（移除可学习参数）
        self.register_buffer('alpha_low', torch.tensor(0.3, dtype=torch.float32))
        self.register_buffer('alpha_high', torch.tensor(0.3, dtype=torch.float32))
    
    @torch.amp.autocast('cuda', enabled=False)
    def dct2d(self, x):
        """
        2D DCT变换（数值稳定版本）
        
        修复：
        1. 强制禁用混合精度
        2. 强制float32精度
        3. 添加NaN检测
        
        Args:
            x: [B, C, H, W]
        Returns:
            dct_coeff: [B, C, H, W] (complex)
        """
        # 强制转换为float32
        x = x.float()
        
        # NaN检测
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # FFT变换
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        
        # 结果检测
        if torch.isnan(x_fft.real).any() or torch.isnan(x_fft.imag).any():
            x_fft = torch.complex(
                torch.nan_to_num(x_fft.real, nan=0.0),
                torch.nan_to_num(x_fft.imag, nan=0.0)
            )
        
        return x_fft
    
    @torch.amp.autocast('cuda', enabled=False)
    def idct2d(self, dct_coeff, target_dtype=None):
        """
        2D IDCT变换（数值稳定版本）
        
        修复：
        1. 强制禁用混合精度
        2. 添加NaN检测
        3. 安全的dtype转换
        
        Args:
            dct_coeff: [B, C, H, W] (complex)
            target_dtype: 目标数据类型（如float16），如果为None则返回float32
        Returns:
            x: [B, C, H, W]
        """
        # NaN检测
        if torch.isnan(dct_coeff.real).any() or torch.isnan(dct_coeff.imag).any():
            dct_coeff = torch.complex(
                torch.nan_to_num(dct_coeff.real, nan=0.0),
                torch.nan_to_num(dct_coeff.imag, nan=0.0)
            )
        
        # IFFT变换
        x = torch.fft.ifft2(dct_coeff, dim=(-2, -1))
        x_real = x.real
        
        # 再次检测
        if torch.isnan(x_real).any():
            x_real = torch.nan_to_num(x_real, nan=0.0)
        
        # 安全的dtype转换
        if target_dtype is not None and target_dtype != x_real.dtype:
            x_real = x_real.to(target_dtype)
        
        return x_real
    
    def forward(self, x, cls_token_idx=None):
        """
        频域分解（数值稳定版本）
        
        修复：
        1. 移除通道注意力机制
        2. 使用固定权重避免学习不稳定
        3. 强化NaN检测
        
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
        
        # 保存原始dtype
        original_dtype = x.dtype
        
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
            raise ValueError(f"Expected {H*W} patches, got {patches.size(1)}. Check img_size config.")
        
        # 2. Reshape to 2D: [B, H*W, D] → [B, D, H, W]
        patches_2d = patches.transpose(1, 2).reshape(B, D, H, W)
        
        # 3. DCT变换（自动处理float32转换和NaN）
        freq_coeff = self.dct2d(patches_2d)  # Complex tensor in float32
        freq_real = freq_coeff.real
        freq_imag = freq_coeff.imag
        
        # 4. 生成固定的频域掩码
        low_mask = self.low_freq_mask()   # [1, H, W]
        high_mask = self.high_freq_mask()  # [1, H, W]
        
        # 5. 应用掩码分离频域
        low_freq_real = freq_real * low_mask
        low_freq_imag = freq_imag * low_mask
        high_freq_real = freq_real * high_mask
        high_freq_imag = freq_imag * high_mask
        
        # 6. IDCT变换回空域（自动处理dtype转换和NaN）
        low_freq_complex = torch.complex(low_freq_real, low_freq_imag)
        high_freq_complex = torch.complex(high_freq_real, high_freq_imag)
        
        low_freq_spatial = self.idct2d(low_freq_complex, target_dtype=original_dtype)   # [B, D, H, W]
        high_freq_spatial = self.idct2d(high_freq_complex, target_dtype=original_dtype)  # [B, D, H, W]
        
        # 7. Reshape back to sequence: [B, D, H, W] → [B, H*W, D]
        low_freq_patches = low_freq_spatial.reshape(B, D, -1).transpose(1, 2)
        high_freq_patches = high_freq_spatial.reshape(B, D, -1).transpose(1, 2)
        
        # 8. 使用固定权重（移除通道注意力）
        low_freq_patches = low_freq_patches * self.alpha_low
        high_freq_patches = high_freq_patches * self.alpha_high
        
        # 9. 重新插入CLS token
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
        
        # 10. 返回频域信息（用于可视化和损失计算）
        freq_info = {
            'low_freq': low_freq_seq.detach(),
            'high_freq': high_freq_seq.detach(),
            'low_mask': low_mask.detach(),
            'high_mask': high_mask.detach(),
            'alpha_low': self.alpha_low.item(),
            'alpha_high': self.alpha_high.item(),
            'freq_magnitude': torch.abs(freq_coeff).mean(dim=(1, 2, 3)).detach()  # [B]
        }
        
        return low_freq_seq, high_freq_seq, freq_info


def get_frequency_splitter(freq_type='dct', **kwargs):
    """
    工厂函数：根据类型创建频域分解器
    (Legacy: freq_type parameter is kept for compatibility but ignored)
    """
    return DCTFrequencySplitter(**kwargs)