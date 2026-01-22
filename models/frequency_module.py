# models/frequency_module.py
"""
频域分解模块 - FSHD-Net核心组件
使用真正的DCT（离散余弦变换）进行频域分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


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
        真正的2D DCT变换（Type-II）
        
        实现：
        1. 使用PyTorch原生实现（高效GPU加速）
        2. 强制float32精度避免数值问题
        3. 正交归一化保持能量守恒
        4. 添加NaN检测保护
        
        Args:
            x: [B, C, H, W] 输入特征
        Returns:
            dct_coeff: [B, C, H, W] DCT系数（实数）
        """
        # 强制转换为float32
        x = x.float()
        
        # NaN检测
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        B, C, H, W = x.shape
        
        # 方法1: 使用矩阵乘法实现DCT（适用于任何PyTorch版本）
        # 构建DCT基矩阵
        if not hasattr(self, '_dct_matrix_h') or self._dct_matrix_h.shape[0] != H:
            self._dct_matrix_h = self._get_dct_matrix(H).to(x.device)
        if not hasattr(self, '_dct_matrix_w') or self._dct_matrix_w.shape[0] != W:
            self._dct_matrix_w = self._get_dct_matrix(W).to(x.device)
        
        # 2D DCT = DCT_H @ X @ DCT_W^T
        # [B, C, H, W] -> [B*C, H, W]
        x_reshape = x.reshape(B * C, H, W)
        
        # 沿H维度DCT: [B*C, H, W] @ [W, W]^T = [B*C, H, W]
        dct_h = torch.matmul(self._dct_matrix_h, x_reshape)
        
        # 沿W维度DCT: [B*C, H, W] @ [W, W]^T
        # 需要转置: [B*C, H, W] -> [B*C, W, H]
        dct_h_t = dct_h.transpose(-2, -1)  # [B*C, W, H]
        dct_hw = torch.matmul(self._dct_matrix_w, dct_h_t)  # [B*C, W, H]
        dct_hw = dct_hw.transpose(-2, -1)  # [B*C, H, W]
        
        # Reshape回来: [B*C, H, W] -> [B, C, H, W]
        dct_coeff = dct_hw.reshape(B, C, H, W)
        
        # 结果检测
        if torch.isnan(dct_coeff).any():
            dct_coeff = torch.nan_to_num(dct_coeff, nan=0.0)
        
        return dct_coeff
    
    def _get_dct_matrix(self, N):
        """
        构建1D DCT-II正交基矩阵
        
        DCT-II公式:
        X_k = sqrt(2/N) * alpha(k) * sum_{n=0}^{N-1} x_n * cos(pi*k*(2n+1)/(2N))
        alpha(0) = 1/sqrt(2), alpha(k) = 1 for k>0
        
        Args:
            N: 序列长度
        Returns:
            DCT_matrix: [N, N] 正交DCT基矩阵
        """
        dct_matrix = torch.zeros(N, N, dtype=torch.float32)
        
        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct_matrix[k, n] = math.sqrt(1.0 / N) * math.cos(math.pi * k * (2*n + 1) / (2*N))
                else:
                    dct_matrix[k, n] = math.sqrt(2.0 / N) * math.cos(math.pi * k * (2*n + 1) / (2*N))
        
        return dct_matrix
    
    @torch.amp.autocast('cuda', enabled=False)
    def idct2d(self, dct_coeff, target_dtype=None):
        """
        真正的2D IDCT变换（Type-III，DCT-II的逆）
        
        实现：
        1. 使用DCT矩阵的转置（正交矩阵）
        2. 强制float32精度
        3. 添加NaN检测
        
        Args:
            dct_coeff: [B, C, H, W] DCT系数（实数）
            target_dtype: 目标数据类型（如float16）
        Returns:
            x: [B, C, H, W] 重构信号
        """
        # NaN检测
        if torch.isnan(dct_coeff).any():
            dct_coeff = torch.nan_to_num(dct_coeff, nan=0.0)
        
        B, C, H, W = dct_coeff.shape
        
        # 获取DCT基矩阵（如果已缓存则直接使用）
        if not hasattr(self, '_dct_matrix_h') or self._dct_matrix_h.shape[0] != H:
            self._dct_matrix_h = self._get_dct_matrix(H).to(dct_coeff.device)
        if not hasattr(self, '_dct_matrix_w') or self._dct_matrix_w.shape[0] != W:
            self._dct_matrix_w = self._get_dct_matrix(W).to(dct_coeff.device)
        
        # 2D IDCT = DCT_H^T @ X @ DCT_W
        # [B, C, H, W] -> [B*C, H, W]
        dct_reshape = dct_coeff.reshape(B * C, H, W)
        
        # 沿H维度IDCT: [H, H]^T @ [B*C, H, W]
        idct_h = torch.matmul(self._dct_matrix_h.T, dct_reshape)
        
        # 沿W维度IDCT
        idct_h_t = idct_h.transpose(-2, -1)  # [B*C, W, H]
        idct_hw = torch.matmul(self._dct_matrix_w.T, idct_h_t)  # [B*C, W, H]
        idct_hw = idct_hw.transpose(-2, -1)  # [B*C, H, W]
        
        # Reshape回来
        x_real = idct_hw.reshape(B, C, H, W)
        
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
        
        # 3. 真正的DCT变换（返回实数，无虚部）
        freq_coeff = self.dct2d(patches_2d)  # [B, D, H, W] 实数tensor
        
        # === 新增：计算频域能量分布 (OFC-Gate 输入) ===
        # 使用Log-Scale能量聚合，平衡低频和高频的贡献
        # V_energy = AvgPool(log(1 + |C_dct|))
        v_energy = torch.mean(torch.log1p(torch.abs(freq_coeff)), dim=(2, 3)) # [B, D]
        
        # 4. 生成固定的频域掩码
        low_mask = self.low_freq_mask()   # [1, H, W]
        high_mask = self.high_freq_mask()  # [1, H, W]
        
        # 5. 应用掩码分离频域（DCT无虚部，只有实部）
        low_freq_coeff = freq_coeff * low_mask   # [B, D, H, W]
        high_freq_coeff = freq_coeff * high_mask  # [B, D, H, W]
        
        # 6. IDCT变换回空域（自动处理dtype转换和NaN）
        
        # 6. IDCT逆变换（DCT无需重组复数）
        low_freq_spatial = self.idct2d(low_freq_coeff, target_dtype=original_dtype)   # [B, D, H, W]
        high_freq_spatial = self.idct2d(high_freq_coeff, target_dtype=original_dtype)  # [B, D, H, W]
        
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
            'freq_magnitude': torch.abs(freq_coeff).mean(dim=(1, 2, 3)).detach(),  # [B] DCT系数幅度
            'freq_coeff': freq_coeff.detach(),  # 保存DCT系数用于可视化
            'v_energy': v_energy  # [B, D] 频域通道能量 (OFC-Gate使用, 梯度需要保留)
        }
        
        return low_freq_seq, high_freq_seq, freq_info


def get_frequency_splitter(freq_type='dct', **kwargs):
    """
    工厂函数：根据类型创建频域分解器
    (Legacy: freq_type parameter is kept for compatibility but ignored)
    """
    return DCTFrequencySplitter(**kwargs)