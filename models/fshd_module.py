# models/fshd_module.py
"""
FSHD-Net完整模块
整合频域分解、异构双流、频域引导注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .frequency_module import get_frequency_splitter
from .hybrid_stream import HybridDualStream, FrequencyGuidedAttention


class OFC_Gate(nn.Module):
    """
    OFC-Gate: Orthogonal Frequency-Channel Gating
    图像端门控机制改进方案
    
    1. 物理通道筛选 (Push): 利用DCT频域能量鉴别通道物理特性
    2. 正交解耦抑制 (Pull): 强制ID与属性特征正交，抑制纠缠
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        
        # === A. 频域物理通道注意力 (Physics Branch) ===
        # Input: v_energy [B, D] -> m_freq [B, D]
        # 瓶颈结构 MLP (r=16)
        self.phy_mlp = nn.Sequential(
            nn.Linear(dim, dim // 16),
            nn.ReLU(),
            nn.Linear(dim // 16, dim),
            nn.Sigmoid()
        )
        
        # 针对ID和Attr的不同频率偏好投影
        self.phy_proj_id = nn.Linear(dim, dim)
        self.phy_proj_attr = nn.Linear(dim, dim)
        
        # === B. 语义自适应门控 (Semantic Branch) ===
        # 轻量化自适应门控
        self.sem_gate_id = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
        self.sem_gate_attr = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
    def forward(self, id_feat, attr_feat, v_energy):
        """
        Args:
            id_feat: [B, D] ID特征
            attr_feat: [B, D] 属性特征
            v_energy: [B, D] 频域通道能量 (来自DCT)
        """
        # === 1. 物理通道感知 (Physics-Aware) ===
        # 生成基础频率掩码
        m_freq = self.phy_mlp(v_energy) # [B, D]
        
        # 映射到各自的分支 (ID通常偏好低频，Attr偏好高频，让模型自己学)
        a_id_phy = torch.sigmoid(self.phy_proj_id(m_freq))
        a_attr_phy = torch.sigmoid(self.phy_proj_attr(m_freq))
        
        # === 2. 动态正交抑制 (Dynamic Orthogonal Suppression) ===
        # 计算余弦相似度
        # 关键策略：对ID特征阻断梯度，防止ID被"带偏"，只允许调整Attr以远离ID
        id_norm = F.normalize(id_feat.detach(), dim=-1, eps=1e-8)
        attr_norm = F.normalize(attr_feat, dim=-1, eps=1e-8)
        
        # Sim: [B, 1]
        sim = (id_norm * attr_norm).sum(dim=-1, keepdim=True)
        
        # 正交抑制因子 W = 1 - S^2
        # 当Sim -> 0 (正交)时，W -> 1 (保留)
        # 当Sim -> 1 (纠缠)时，W -> 0 (抑制)
        # 添加微小epsilon防止完全关死
        w_ortho = 1.0 - sim.pow(2) + 1e-6 
        
        # === 3. 语义自适应 (Semantic Self-Gating) ===
        a_id_sem = self.sem_gate_id(id_feat)
        a_attr_sem = self.sem_gate_attr(attr_feat)
        
        # === 4. 最终融合 ===
        # ID门控: 语义 * 物理
        g_id = a_id_sem * a_id_phy
        
        # Attr门控: 语义 * 物理 * 正交抑制
        g_attr = a_attr_sem * a_attr_phy * w_ortho
        
        # 约束范围 (保持数值稳定性)
        g_id = torch.clamp(g_id, min=0.05, max=0.95)
        g_attr = torch.clamp(g_attr, min=0.05, max=0.95)
        
        # 特征加权
        id_out = id_feat * g_id
        attr_out = attr_feat * g_attr
        
        return id_out, attr_out, g_id, g_attr


class FSHDModule(nn.Module):
    """
    Frequency-Spatial Hybrid Decoupling Module
    
    完整流程：
    1. 频域分解：DCT → 低频/高频特征
    2. 频域引导注意力：低频→ID分支，高频→Attr分支
    3. 异构双流建模：Mamba(ID) + Multi-scale CNN(Attr)
    4. 对称门控：OFC-Gate (物理感知 + 正交抑制)
    """
    
    def __init__(self, dim=768, num_heads=8, d_state=16, d_conv=4, dropout=0.1,
                 img_size=(14, 14), use_multi_scale_cnn=True, logger=None):
        """
        Args:
            dim: 特征维度
            num_heads: 注意力头数
            d_state: Mamba状态维度
            d_conv: Mamba卷积核大小
            dropout: Dropout比率
            img_size: 图像patch grid尺寸
            use_multi_scale_cnn: 是否使用多尺度CNN
            logger: 日志记录器
        """
        super().__init__()
        self.logger = logger
        
        # === 阶段1：频域分解 (固定为DCT) ===
        self.freq_splitter = get_frequency_splitter(
            dim=dim,
            img_size=img_size
        )
        if logger:
            logger.debug_logger.info(f"✅ FSHD: Using DCT frequency splitter")
        
        # === 阶段2：频域引导注意力 ===
        self.freq_guided_attn = FrequencyGuidedAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        if logger:
            logger.debug_logger.info("✅ FSHD: Frequency-guided attention initialized")
        
        # === 阶段3：异构双流 ===
        self.hybrid_stream = HybridDualStream(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=2,
            use_multi_scale_cnn=use_multi_scale_cnn,
            logger=logger
        )
        if logger:
            stream_type = "Multi-scale CNN" if use_multi_scale_cnn else "Lightweight Mamba"
            logger.debug_logger.info(f"✅ FSHD: Hybrid stream with {stream_type}")
        
        # === 阶段4：对称门控 (升级为 OFC-Gate) ===
        self.ofc_gate = OFC_Gate(dim, dropout)
        
        # 全局池化
        self.pool_id = nn.AdaptiveAvgPool1d(1)
        self.pool_attr = nn.AdaptiveAvgPool1d(1)
        
        if logger:
            logger.debug_logger.info("=" * 60)
            logger.debug_logger.info("FSHD-Net Architecture Summary:")
            logger.debug_logger.info(f"  1. Frequency Decomposition: DCT")
            logger.debug_logger.info(f"  2. Attention: Frequency-Guided + Soft Orthogonal")
            logger.debug_logger.info(f"  3. Dual Stream: Mamba(ID) + {'Multi-CNN' if use_multi_scale_cnn else 'Light-Mamba'}(Attr)")
            logger.debug_logger.info(f"  4. Gating: OFC-Gate (Physics-Aware + Ortho-Suppression)")
            logger.debug_logger.info("=" * 60)
    
    def forward(self, x, return_attention=False, return_freq_info=False):
        """
        前向传播
        Args:
            x: 输入特征 [B, N, D]
            return_attention: 是否返回注意力图
            return_freq_info: 是否返回频域信息
        Returns:
            如果 return_attention=False and return_freq_info=False:
                (id_feat, attr_feat, gate_stats, original_feat)
            如果 return_attention=True:
                + (id_attn_map, attr_attn_map)
            如果 return_freq_info=True:
                + (freq_info,)
        """
        B, N, D = x.shape
        
        # 保存原始特征（用于重构损失）
        original_feat = self.pool_id(x.transpose(1, 2)).squeeze(-1)  # [B, D]
        
        # === 阶段1：频域分解 ===
        # 确定CLS token位置（根据序列长度判断是Vim还是ViT）
        # Vim: mid-token (N//2), ViT: first token (0)
        # 这里我们假设如果N是奇数，则为Vim（mid-token）
        cls_token_idx = N // 2 if N % 2 == 1 else 0
        
        low_freq_seq, high_freq_seq, freq_info = self.freq_splitter(x, cls_token_idx=cls_token_idx)
        
        # 原始特征 + 频域增强（加权融合）
        # 这样保证即使频域分解失败，也不会破坏原有特征
        x_enhanced = x + low_freq_seq + high_freq_seq
        
        # === 阶段2：频域引导注意力 ===
        if return_attention:
            id_seq, attr_seq, id_attn_map, attr_attn_map = self.freq_guided_attn(
                x_enhanced, 
                low_freq_feat=low_freq_seq,
                high_freq_feat=high_freq_seq,
                return_attention=True
            )
        else:
            id_seq, attr_seq = self.freq_guided_attn(
                x_enhanced,
                low_freq_feat=low_freq_seq,
                high_freq_feat=high_freq_seq,
                return_attention=False
            )
            id_attn_map, attr_attn_map = None, None
        
        # === 阶段3：异构双流建模 ===
        # ID流输入：原始id_seq + 低频增强
        # Attr流输入：原始attr_seq + 高频增强
        id_seq_filtered, attr_seq_filtered = self.hybrid_stream(id_seq, attr_seq)
        
        # === 阶段4：全局池化 ===
        id_feat = self.pool_id(id_seq_filtered.transpose(1, 2)).squeeze(-1)      # [B, D]
        attr_feat = self.pool_attr(attr_seq_filtered.transpose(1, 2)).squeeze(-1) # [B, D]
        
        # === 阶段5：OFC-Gate 门控 (升级) ===
        # 获取频域通道能量 (确保存在)
        v_energy = freq_info.get('v_energy', torch.zeros_like(id_feat))
        
        # 应用 OFC-Gate
        id_feat_gated, attr_feat_gated, gate_id, gate_attr = self.ofc_gate(id_feat, attr_feat, v_energy)
        
        # === 门控统计信息 ===
        # 计算频域能量比率 (r_E) 用于 SAMG (保留)
        energy_high = high_freq_seq.norm(p=2, dim=-1).mean(dim=1) # [B]
        energy_total = x.norm(p=2, dim=-1).mean(dim=1) + 1e-8      # [B]
        r_E = energy_high / energy_total                           # [B]

        gate_stats = {
            'gate_id_mean': gate_id.mean().item(),
            'gate_id_std': gate_id.std().item(),
            'gate_attr_mean': gate_attr.mean().item(),
            'gate_attr_std': gate_attr.std().item(),
            'diversity': torch.abs(gate_id - gate_attr).mean().item(),
            'freq_type': 'dct',
            'low_freq_energy': freq_info.get('freq_magnitude', torch.tensor(0.0)).mean().item() if 'freq_magnitude' in freq_info else 0.0,
            'energy_ratio': r_E  # [B] Tensor, not item, passed to fusion
        }
        
        # === 构建返回值 ===
        base_outputs = (id_feat_gated, attr_feat_gated, gate_stats, original_feat)
        
        if return_attention:
            base_outputs = base_outputs + (id_attn_map, attr_attn_map)
        
        if return_freq_info:
            # 增强freq_info，添加更多可视化信息
            freq_info['id_seq'] = id_seq.detach()
            freq_info['attr_seq'] = attr_seq.detach()
            freq_info['id_feat'] = id_feat_gated.detach()
            freq_info['attr_feat'] = attr_feat_gated.detach()
            base_outputs = base_outputs + (freq_info,)
        
        return base_outputs


# 兼容性别名
GS3Module = FSHDModule
SymmetricGS3Module = FSHDModule