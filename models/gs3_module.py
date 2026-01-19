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
        
        # === 改进：可学习的正交化缩放参数 ===
        # 初始化为 0.1，允许模型自适应调整正交化强度
        self.orth_scale = nn.Parameter(torch.tensor(0.1))
        
        # 初始化权重：使用Xavier初始化降低初始梯度
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重，防止梯度爆炸"""
        for m in [self.qkv_id, self.qkv_cloth, self.proj_id, self.proj_cloth]:
            nn.init.xavier_uniform_(m.weight, gain=1.0)  # 使用标准gain，避免初始化过小
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def orthogonal_projection(self, q_id, q_cloth):
        """
        执行正交投影: Q_id_perp = Q_id - proj_{Q_cloth}(Q_id)
        
        数学公式：
        Q_id_perp = Q_id - scale * (<Q_id, Q_cloth> / ||Q_cloth||^2 * Q_cloth)
        """
        # 计算点积 <Q_id, Q_cloth>
        dot_product = (q_id * q_cloth).sum(dim=-1, keepdim=True)  # [B, num_heads, N, 1]
        
        # 计算 ||Q_cloth||^2，使用更大的 eps 防止数值不稳定
        norm_sq = (q_cloth * q_cloth).sum(dim=-1, keepdim=True) + 1e-5
        
        # 计算投影系数并进行更严格的梯度裁剪
        projection_coeff = dot_product / norm_sq
        projection_coeff = torch.clamp(projection_coeff, min=-1.0, max=1.0)
        
        # 计算投影分量
        projection = projection_coeff * q_cloth  # [B, num_heads, N, head_dim]
        
        # === 改进：使用可学习参数进行自适应正交化 ===
        # 限制 scale 范围在 [0.0, 1.0]，防止反向缩放或过度矫正
        scale = torch.clamp(self.orth_scale, 0.0, 1.0)
        q_id_ortho = q_id - scale * projection
        
        # 计算服装显著性分数 (归一化的点积)
        norm_sqrt = torch.sqrt(norm_sq)
        saliency_score = torch.abs(dot_product) / norm_sqrt
        saliency_score = torch.clamp(saliency_score.mean(dim=1), min=0.0, max=1.0)  # 对所有头取平均
        
        return q_id_ortho, saliency_score
    
    def forward(self, x, return_attention=False):
        # ... (后续代码保持不变，通过 _init_weights 等 helper function 复用) ...
        # 注意：为了replace调用的简洁性，我们只需替换 OrthogonalProjectionAttention 类的定义
        # 下面的 forward 代码会随着类定义的替换一并替换
        """
        前向传播
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
        # 更严格的裁剪防止softmax溢出导致NaN
        attn_id = torch.clamp(attn_id, min=-20.0, max=20.0)
        attn_id = attn_id.softmax(dim=-1)
        # 检查NaN
        if torch.isnan(attn_id).any():
            attn_id = torch.where(torch.isnan(attn_id), torch.zeros_like(attn_id), attn_id)
        attn_id = self.attn_dropout(attn_id)
        out_id = (attn_id @ v_id).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        
        # 服装分支：正常计算
        attn_cloth = (q_cloth @ k_cloth.transpose(-2, -1)) * self.scale
        attn_cloth = torch.clamp(attn_cloth, min=-20.0, max=20.0)
        attn_cloth = attn_cloth.softmax(dim=-1)
        # 检查NaN
        if torch.isnan(attn_cloth).any():
            attn_cloth = torch.where(torch.isnan(attn_cloth), torch.zeros_like(attn_cloth), attn_cloth)
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
    内容感知的 Mamba 过滤器 (FiLM 增强版)
    
    核心思想：
    - 利用 OPA 计算的服装显著性分数，通过 FiLM (Feature-wise Linear Modulation) 机制
    - 动态调制特征分布，引导 Mamba 关注或抑制特定区域
    """
    
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, logger=None):
        """
        Args:
            dim (int): 特征维度
            d_state (int): Mamba 状态维度
            d_conv (int): Mamba 卷积核大小
            expand (int): Mamba 扩展因子
            logger: TrainingMonitor实例
        """
        super().__init__()
        self.logger = logger
        
        # Mamba SSM 核心
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # === 改进：FiLM 调制层 ===
        # 将显著性分数 (scalar) 映射为 affine 参数 gamma (scale) 和 beta (shift)
        # 输入: [B, N, 1] -> 输出: [B, N, 2*dim]
        self.saliency_proj = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim * 2)  # chunk(2) -> gamma, beta
        )
        
        # 初始化 FiLM 层
        # gamma 初始化为 0 (scale=1)，beta 初始化为 0 (shift=0)，保证初始状态为恒等变换
        nn.init.constant_(self.saliency_proj[-1].weight, 0)
        nn.init.constant_(self.saliency_proj[-1].bias, 0)
        
        # Pre-Norm
        self.norm_pre = nn.LayerNorm(dim)
        
    def forward(self, x, saliency_score):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, dim]
            saliency_score: 服装显著性分数 [batch_size, seq_len, 1]
            
        Returns:
            filtered_features: 过滤后的特征 [batch_size, seq_len, dim]
        """
        # 检查输入NaN
        if torch.isnan(x).any() or torch.isnan(saliency_score).any():
            if self.logger:
                self.logger.debug_logger.warning("⚠️  ContentAwareMambaFilter: Input contains NaN, returning original x")
            return x
        
        # === 步骤 1: 生成 FiLM 调制参数 ===
        # [B, N, 1] -> [B, N, 2*dim]
        affine_params = self.saliency_proj(saliency_score)
        gamma, beta = affine_params.chunk(2, dim=-1)  # gamma: [B, N, dim], beta: [B, N, dim]
        
        # 限制 gamma 范围，防止过度缩放导致梯度爆炸
        gamma = torch.tanh(gamma)  # 限制在 [-1, 1] 之间，对应 scale 范围 [0, 2]
        
        # === 步骤 2: 应用 FiLM 调制 ===
        # x_modulated = x * (1 + gamma) + beta
        # 这种方式比硬门控 (x * (1-gate)) 更稳定，保留了梯度流
        x_modulated = x * (1 + gamma) + beta
        
        # === 步骤 3: Mamba 状态空间建模 ===
        # Mamba 会对调制后的序列进行时序建模
        x_mamba = self.mamba(x_modulated)
        
        # 检查Mamba输出NaN
        if torch.isnan(x_mamba).any():
            if self.logger:
                self.logger.debug_logger.warning("⚠️  ContentAwareMambaFilter: Mamba output contains NaN, using identity")
            x_mamba = x
        
        # === 步骤 4: 残差连接 + LayerNorm ===
        # 残差连接到原始 x，Mamba 分支作为一个修正项
        # 初始时 scale 较小，让模型慢慢学习
        out = self.norm_pre(x + 0.1 * x_mamba)
        
        # 最终NaN检查
        if torch.isnan(out).any():
            if self.logger:
                self.logger.debug_logger.warning("⚠️  ContentAwareMambaFilter: Output contains NaN, returning original x")
            return x
        
        return out


class GS3Module(nn.Module):
    """
    G-S3 (Geometry-Guided Selective State Space) 主模块
    
    将 OPA 和 Content-Aware Mamba Filter 整合为完整的解耦模块
    可直接替换原有的 DisentangleModule
    """
    
    def __init__(self, dim, num_heads=8, d_state=16, d_conv=4, dropout=0.1, logger=None):
        """
        Args:
            dim (int): 特征维度
            num_heads (int): OPA 的注意力头数
            d_state (int): Mamba 状态维度
            d_conv (int): Mamba 卷积核大小
            dropout (float): Dropout 比率
            logger: TrainingMonitor实例
        """
        super().__init__()
        self.logger = logger
        
        # 阶段 1: 正交投影注意力
        self.opa = OrthogonalProjectionAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            logger=logger
        )
        
        # 阶段 2: 内容感知 Mamba 过滤器
        self.mamba_filter = ContentAwareMambaFilter(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            logger=logger
        )
        
        # === 软门控机制 (Soft Gating) - 独立调节 ===
        # 改进：使用两个独立的门控网络，避免强制互补
        self.gate_id = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()  # 输出 [0, 1]
        )
        
        self.gate_cloth = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()  # 输出 [0, 1]
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
                (id_feat, cloth_feat, gate_stats)
            如果 return_attention=True:
                (id_feat, cloth_feat, gate_stats, id_attn_map, cloth_attn_map)
                
            gate_stats: dict包含gate_id和gate_cloth的统计信息
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
        
        # === 阶段 4: 软门控 (Soft Gating) ===
        # 拼接特征用于门控预测
        concat_feat = torch.cat([id_feat, cloth_feat], dim=-1)  # [B, dim*2]
        
        # 独立预测两个门控权重
        gate_id = self.gate_id(concat_feat)      # [B, dim]
        gate_cloth = self.gate_cloth(concat_feat)  # [B, dim]
        
        # === 关键修复：防止门控塌缩 ===
        # 限制门控范围在 [0.05, 0.95]，保证始终有梯度流过
        gate_id = torch.clamp(gate_id, min=0.05, max=0.95)
        gate_cloth = torch.clamp(gate_cloth, min=0.05, max=0.95)
        
        # 应用门控
        # ID特征保持门控，因为我们需要它去除衣服信息
        id_feat_gated = gate_id * id_feat        # [B, dim]
        
        # [架构松绑] 服装特征不再应用门控！
        # 原因：防止模型通过关闭门控来制造特征坍缩 (Zero Feature Collapse)
        # 我们希望服装特征始终存在，并由 semantic loss 赋予语义
        cloth_feat_gated = cloth_feat            # [B, dim] (直通)
        
        # === 门控统计信息（用于监控和调试）===
        gate_stats = {
            'gate_id_mean': gate_id.mean().item(),
            'gate_id_std': gate_id.std().item(),
            'gate_cloth_mean': gate_cloth.mean().item(),
            'gate_cloth_std': gate_cloth.std().item(),
            'diversity': torch.abs(gate_id - gate_cloth).mean().item()
        }
        
        # === 调试信息（不要在训练中频繁调用）===
        if hasattr(self, '_debug_mode') and self._debug_mode:
            self._debug_info = {
                'id_seq': id_seq,
                'cloth_seq': cloth_seq,
                'saliency_score': saliency_score,
                'id_seq_filtered': id_seq_filtered,
                'gate_id': gate_id,
                'gate_cloth': gate_cloth,
                'gate_stats': gate_stats
            }
        
        if return_attention:
            return id_feat_gated, cloth_feat_gated, gate_stats, id_attn_map, cloth_attn_map
        else:
            return id_feat_gated, cloth_feat_gated, gate_stats
