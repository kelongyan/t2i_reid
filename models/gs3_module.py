# models/gs3_module.py
"""
G-S3 (Geometry-Guided Selective State Space) Module
实现基于几何引导的选择性状态空间模型用于身份-服装解耦

核心组件：
1. OPA (Orthogonal Projection Attention): 正交投影注意力
2. Content-Aware Mamba Filter: 内容感知的 Mamba 过滤器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class OrthogonalProjectionAttention(nn.Module):
    """
    正交投影注意力模块 (OPA)
    
    核心思想：
    - 在注意力机制内部，强制身份分支的查询向量与服装语义正交
    - 通过 Gram-Schmidt 正交化避免身份特征"偷看"服装信息
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.1):
        """
        Args:
            dim (int): 输入特征维度
            num_heads (int): 多头注意力的头数
            qkv_bias (bool): QKV投影是否使用偏置
            dropout (float): Dropout比率
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 身份分支的 QKV 投影
        self.qkv_id = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 服装分支的 QKV 投影
        self.qkv_cloth = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 输出投影
        self.proj_id = nn.Linear(dim, dim)
        self.proj_cloth = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # LayerNorm for stabilization
        self.norm_id = nn.LayerNorm(dim)
        self.norm_cloth = nn.LayerNorm(dim)
        
        # 初始化权重：使用Xavier初始化降低初始梯度
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，防止梯度爆炸"""
        for m in [self.qkv_id, self.qkv_cloth, self.proj_id, self.proj_cloth]:
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def orthogonal_projection(self, q_id, q_cloth):
        """
        执行正交投影: Q_id_perp = Q_id - proj_{Q_cloth}(Q_id)
        
        数学公式：
        Q_id_perp = Q_id - <Q_id, Q_cloth> / ||Q_cloth||^2 * Q_cloth
        
        Args:
            q_id: 身份查询向量 [B, num_heads, N, head_dim]
            q_cloth: 服装查询向量 [B, num_heads, N, head_dim]
            
        Returns:
            q_id_ortho: 正交化后的身份查询向量
            saliency_score: 服装显著性分数 (用于后续 Mamba Filter)
        """
        # 计算点积 <Q_id, Q_cloth>
        dot_product = (q_id * q_cloth).sum(dim=-1, keepdim=True)  # [B, num_heads, N, 1]
        
        # 计算 ||Q_cloth||^2，增大eps防止数值不稳定
        norm_sq = (q_cloth * q_cloth).sum(dim=-1, keepdim=True) + 1e-8  # 防止除零
        
        # 计算投影系数并进行梯度裁剪
        projection_coeff = dot_product / norm_sq
        projection_coeff = torch.clamp(projection_coeff, min=-5.0, max=5.0)  # 放宽裁剪范围
        
        # 计算投影分量
        projection = projection_coeff * q_cloth  # [B, num_heads, N, head_dim]
        
        # 执行正交化：减去投影分量，降低正交化强度避免信息丢失
        q_id_ortho = q_id - 0.3 * projection  # 进一步降低正交化强度（从0.5到0.3）
        
        # 计算服装显著性分数 (归一化的点积，范围 [0, 1])
        # 点积越大，说明当前 token 包含越多服装信息
        saliency_score = torch.abs(dot_product) / (torch.sqrt(norm_sq) + 1e-8)
        saliency_score = torch.clamp(saliency_score.mean(dim=1), min=0.0, max=1.0)  # 对所有头取平均 [B, N, 1]
        
        return q_id_ortho, saliency_score
    
    def forward(self, x, return_attention=False):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            return_attention: 是否返回注意力图
            
        Returns:
            id_features: 身份特征 [batch_size, seq_len, dim]
            cloth_features: 服装特征 [batch_size, seq_len, dim]
            saliency_score: 服装显著性分数 [batch_size, seq_len, 1]
            (可选) attn_weights_id, attn_weights_cloth: 注意力权重
        """
        B, N, C = x.shape
        
        # === 步骤 1: 生成 QKV ===
        # 身份分支
        qkv_id = self.qkv_id(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv_id = qkv_id.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q_id, k_id, v_id = qkv_id[0], qkv_id[1], qkv_id[2]
        
        # 服装分支
        qkv_cloth = self.qkv_cloth(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv_cloth = qkv_cloth.permute(2, 0, 3, 1, 4)
        q_cloth, k_cloth, v_cloth = qkv_cloth[0], qkv_cloth[1], qkv_cloth[2]
        
        # === 步骤 2: 正交投影 (核心创新) ===
        q_id_ortho, saliency_score = self.orthogonal_projection(q_id, q_cloth)
        
        # === 步骤 3: 计算注意力 ===
        # 身份分支：使用正交化后的查询
        attn_id = (q_id_ortho @ k_id.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        # 裁剪防止softmax溢出
        attn_id = torch.clamp(attn_id, min=-50.0, max=50.0)
        attn_id = attn_id.softmax(dim=-1)
        attn_id = self.attn_dropout(attn_id)
        out_id = (attn_id @ v_id).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # 服装分支：正常计算
        attn_cloth = (q_cloth @ k_cloth.transpose(-2, -1)) * self.scale
        attn_cloth = torch.clamp(attn_cloth, min=-50.0, max=50.0)
        attn_cloth = attn_cloth.softmax(dim=-1)
        attn_cloth = self.attn_dropout(attn_cloth)
        out_cloth = (attn_cloth @ v_cloth).transpose(1, 2).reshape(B, N, C)
        
        # === 步骤 4: 输出投影 ===
        id_features = self.proj_dropout(self.proj_id(out_id))
        cloth_features = self.proj_dropout(self.proj_cloth(out_cloth))
        
        # LayerNorm
        id_features = self.norm_id(id_features)
        cloth_features = self.norm_cloth(cloth_features)
        
        if return_attention:
            # 计算注意力图用于可视化 (取所有头的平均)
            attn_map_id = attn_id.mean(dim=1)  # [B, N, N]
            attn_map_cloth = attn_cloth.mean(dim=1)
            return id_features, cloth_features, saliency_score, attn_map_id, attn_map_cloth
        else:
            return id_features, cloth_features, saliency_score


class ContentAwareMambaFilter(nn.Module):
    """
    内容感知的 Mamba 过滤器
    
    核心思想：
    - 利用 OPA 计算的服装显著性分数，动态调节 Mamba 的输入门控
    - 当显著性分数高时（包含服装信息），抑制信息流入 Mamba 状态
    """
    
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        """
        Args:
            dim (int): 特征维度
            d_state (int): Mamba 状态维度
            d_conv (int): Mamba 卷积核大小
            expand (int): Mamba 扩展因子
        """
        super().__init__()
        
        # Mamba SSM 核心
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # 显著性评分网络：将原始分数映射为门控信号
        self.saliency_mlp = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
        
        # LayerNorm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, saliency_score):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            saliency_score: 服装显著性分数 [batch_size, seq_len, 1]
            
        Returns:
            filtered_features: 过滤后的特征 [batch_size, seq_len, dim]
        """
        # === 步骤 1: 计算门控信号 ===
        gate_signal = self.saliency_mlp(saliency_score)  # [B, N, 1]
        
        # === 步骤 2: 输入级抑制（降低抑制强度）===
        # 门控逻辑：(1 - 0.2*gate_signal) 表示"保留比例"，进一步降低抑制强度
        # 当 gate_signal ≈ 1（服装显著性高）时，输入被轻微抑制
        # 当 gate_signal ≈ 0（身份显著性高）时，输入正常通过
        x_gated = x * (1 - 0.2 * gate_signal)  # [B, N, dim]（从0.3降到0.2）
        
        # === 步骤 3: Mamba 状态空间建模 ===
        # Mamba 会对序列进行时序建模，门控后的输入确保服装信息不会污染隐状态
        x_filtered = self.mamba(x_gated)
        
        # === 步骤 4: 残差连接 + LayerNorm（进一步降低Mamba输出权重）===
        out = self.norm(x + 0.3 * x_filtered)  # 进一步降低Mamba输出的影响（从0.5到0.3）
        
        return out


class GS3Module(nn.Module):
    """
    G-S3 (Geometry-Guided Selective State Space) 主模块
    
    将 OPA 和 Content-Aware Mamba Filter 整合为完整的解耦模块
    可直接替换原有的 DisentangleModule
    """
    
    def __init__(self, dim, num_heads=8, d_state=16, d_conv=4, dropout=0.1):
        """
        Args:
            dim (int): 特征维度
            num_heads (int): OPA 的注意力头数
            d_state (int): Mamba 状态维度
            d_conv (int): Mamba 卷积核大小
            dropout (float): Dropout 比率
        """
        super().__init__()
        
        # 阶段 1: 正交投影注意力
        self.opa = OrthogonalProjectionAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # 阶段 2: 内容感知 Mamba 过滤器
        self.mamba_filter = ContentAwareMambaFilter(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv
        )
        
        # 门控机制 (保持与原接口兼容)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
        # 全局池化方式
        self.pool_id = nn.AdaptiveAvgPool1d(1)
        self.pool_cloth = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x, return_attention=False):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            return_attention: 是否返回注意力图
            
        Returns:
            如果 return_attention=False:
                (id_feat, cloth_feat, gate)
            如果 return_attention=True:
                (id_feat, cloth_feat, gate, id_attn_map, cloth_attn_map)
        """
        batch_size, seq_len, dim = x.size()
        
        # === 阶段 1: OPA 正交投影注意力 ===
        if return_attention:
            id_seq, cloth_seq, saliency_score, id_attn_map, cloth_attn_map = \
                self.opa(x, return_attention=True)
        else:
            id_seq, cloth_seq, saliency_score = self.opa(x, return_attention=False)
            id_attn_map, cloth_attn_map = None, None
        
        # === 阶段 2: Mamba 过滤器净化身份特征 ===
        id_seq_filtered = self.mamba_filter(id_seq, saliency_score)
        
        # === 阶段 3: 全局池化 ===
        # 对序列维度进行池化得到全局特征
        id_feat = self.pool_id(id_seq_filtered.transpose(1, 2)).squeeze(-1)  # [B, dim]
        cloth_feat = self.pool_cloth(cloth_seq.transpose(1, 2)).squeeze(-1)  # [B, dim]
        
        # === 阶段 4: 门控平衡 ===
        gate = self.gate(torch.cat([id_feat, cloth_feat], dim=-1))  # [B, 1]
        gate_value = gate  # 保存gate原始值用于损失计算
        gate_expanded = gate.expand(-1, dim)  # [B, dim]
        
        # 应用门控
        id_feat_gated = gate_expanded * id_feat
        cloth_feat_gated = (1 - gate_expanded) * cloth_feat
        
        # === 调试信息（不要在训练中频繁调用）===
        if hasattr(self, '_debug_mode') and self._debug_mode:
            self._debug_info = {
                'id_seq': id_seq,
                'cloth_seq': cloth_seq,
                'saliency_score': saliency_score,
                'id_seq_filtered': id_seq_filtered,
                'gate': gate_value
            }
        
        if return_attention:
            return id_feat_gated, cloth_feat_gated, gate_value, id_attn_map, cloth_attn_map
        else:
            return id_feat_gated, cloth_feat_gated, gate_value
