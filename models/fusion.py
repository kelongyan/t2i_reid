import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class EnhancedMambaFusion(nn.Module):
    """优化后的 Mamba SSM 融合模块，门控机制前置，用于高效整合图像和文本特征。"""
    
    def __init__(self, dim, d_state=16, d_conv=4, num_layers=3, output_dim=256, dropout=0.1, logger=None):
        super().__init__()
        self.logger = logger
        # 模态对齐层
        self.image_align = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        self.text_align = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
        
        # 前置门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # 多层 Mamba SSM
        if Mamba is None:
             raise ImportError("Mamba module not found")

        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=dim * 2,
                d_state=d_state,
                d_conv=d_conv,
                expand=2
            ) for _ in range(num_layers)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(dim * 2) for _ in range(num_layers)])
        
        # 输出投影
        self.fc = nn.Linear(dim * 2, output_dim)
        self.norm_final = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, text_features):
        if torch.isnan(image_features).any() or torch.isnan(text_features).any():
            if self.logger:
                self.logger.debug_logger.warning("⚠️  EnhancedMambaFusion: Input contains NaN, returning zeros")
            batch_size = image_features.size(0)
            device = image_features.device
            return torch.zeros(batch_size, self.fc.out_features, device=device), \
                   torch.ones(batch_size, 2, device=device) * 0.5
        
        image_features = self.image_align(image_features)
        text_features = self.text_align(text_features)
        
        if torch.isnan(image_features).any() or torch.isnan(text_features).any():
            image_features = torch.where(torch.isnan(image_features), torch.zeros_like(image_features), image_features)
            text_features = torch.where(torch.isnan(text_features), torch.zeros_like(text_features), text_features)
        
        concat_features = torch.cat([image_features, text_features], dim=-1)
        gate_weights = self.gate(concat_features)
        
        if torch.isnan(gate_weights).any():
            gate_weights = torch.ones_like(gate_weights) * 0.5
        
        image_weight, text_weight = gate_weights[:, 0:1], gate_weights[:, 1:2]
        
        weighted_image = image_weight * image_features
        weighted_text = text_weight * text_features
        weighted_features = torch.cat([weighted_image, weighted_text], dim=-1)
        weighted_features = weighted_features.unsqueeze(1)
        
        mamba_output = weighted_features
        for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
            residual = mamba_output
            mamba_output = mamba(mamba_output)
            
            if torch.isnan(mamba_output).any():
                mamba_output = residual
            
            mamba_output = norm(mamba_output + residual)
        
        mamba_output = mamba_output.squeeze(1)
        
        fused_features = self.fc(mamba_output)
        fused_features = self.dropout(fused_features)
        fused_features = self.norm_final(fused_features)
        
        if torch.isnan(fused_features).any():
            fused_features = torch.zeros_like(fused_features)
        
        return fused_features, gate_weights

class SCAG_Gate(nn.Module):
    """
    语义置信度感知门控 (Semantic-Confidence Aware Gating)

    核心创新：
    - 不再依赖频谱能量比 (energy_ratio)
    - 使用冲突分数 (conflict_score) 来衡量解耦质量
    - conflict_score 高 → 解耦失败 → 降低图像权重 → "弃图保文"
    - conflict_score 低 → 解耦成功 → 提升图像权重 → "图文并重"

    这是方案书的核心设计，相比 SAMG 实现了从"物理统计"到"语义质量"的跨越。
    """
    def __init__(self, dim):
        super().__init__()

        # Confidence MLP: 将冲突分数转换为置信度权重
        # conflict_score ∈ [0, 1] → confidence ∈ [0, 1]
        # 冲突分数高 → 置信度低 (解耦失败)
        # 冲突分数低 → 置信度高 (解耦成功)
        self.confidence_mlp = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()  # 输出 0~1 的置信度权重
        )

        # 互查询分支 (保留原 SAMG 的逻辑，用于特征交互)
        self.mutual_proj = nn.Linear(dim * 2, dim * 2)

    def forward(self, id_feat, attr_feat, conflict_score):
        """
        Args:
            id_feat: [B, D] ID 特征
            attr_feat: [B, D] Attr 特征
            conflict_score: [B] 冲突分数 (来自 AH-Net Module)

        Returns:
            id_out: [B, D] 门控后的 ID 特征
            attr_out: [B, D] 门控后的 Attr 特征
            confidence_weight: [B, D] 置信度权重 (用于日志记录)
        """
        if conflict_score.dim() == 1:
            conflict_score = conflict_score.unsqueeze(1)  # [B, 1]

        # 1. 计算置信度权重
        # conflict_score 高 → confidence 低 → 降低图像权重
        # conflict_score 低 → confidence 高 → 提升图像权重
        confidence_weight = self.confidence_mlp(conflict_score)  # [B, D]

        # 2. 互查询门控 (保留原 SAMG 的特征交互逻辑)
        joint_feat = torch.cat([id_feat, attr_feat], dim=-1)  # [B, 2D]
        raw_gates = self.mutual_proj(joint_feat)  # [B, 2D]
        raw_gates = raw_gates.view(-1, 2, id_feat.shape[-1])  # [B, 2, D]
        gates = F.softmax(raw_gates, dim=1)  # [B, 2, D]

        g_id = gates[:, 0, :]     # [B, D]
        g_attr = gates[:, 1, :]   # [B, D]

        # 3. 应用置信度调节
        # 这是核心创新点：根据解耦质量动态调整图像特征权重
        id_out = id_feat * g_id * confidence_weight
        attr_out = attr_feat * g_attr * confidence_weight

        return id_out, attr_out, confidence_weight

class RCSM_Fusion(nn.Module):
    """
    残差交叉扫描 Mamba 融合 (Residual Cross-Scan Mamba Fusion)
    """
    def __init__(self, dim, output_dim=256, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        if Mamba is None:
             raise ImportError("Mamba module not found")
             
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=2)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=2)
        
        self.dropout = nn.Dropout(dropout)
        self.injection_proj = nn.Linear(dim, dim)
        
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.out_proj = nn.Linear(dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, img_id, img_attr, txt_id, txt_attr):
        B, D = img_id.shape
        
        # Interleaved Construction: [Img_ID, Img_Attr, Txt_ID, Txt_Attr]
        s_inter = torch.stack([img_id, img_attr, txt_id, txt_attr], dim=1) # [B, 4, D] 
        
        x_norm = self.ln(s_inter)
        
        s_fwd = self.mamba_fwd(x_norm)
        s_rev = x_norm.flip(dims=[1])
        s_bwd = self.mamba_bwd(s_rev).flip(dims=[1])
        
        s_mamba = s_inter + self.dropout(s_fwd + s_bwd)
        
        s_shortcut = self.injection_proj(s_inter)
        s_fused = F.layer_norm(s_mamba + s_shortcut, (D,))
        
        s_out = self.bottleneck(s_fused)
        fused_feat = s_out.mean(dim=1)
        
        out = self.out_proj(fused_feat)
        out = self.out_norm(out)
        
        return out

class ScagRcsmFusion(nn.Module):
    """
    S-CAG + RCSM 融合模块

    语义置信度感知门控 (S-CAG) + 残差交叉扫描 Mamba (RCSM)
    这是方案书的完整实现，替代了原有的 SAMG+RCSM 架构。

    核心优势：
    1. 基于语义质量 (conflict_score) 而非物理统计 (energy_ratio)
    2. 具备"自知之明"，知道何时该信图片，何时该信文字
    """
    def __init__(self, dim=768, output_dim=256, **kwargs):
        super().__init__()
        self.scag = SCAG_Gate(dim)
        self.rcsm = RCSM_Fusion(dim, output_dim=output_dim, **kwargs)

    def forward(self, img_id, img_attr, txt_id, txt_attr, conflict_score):
        """
        Args:
            conflict_score: [B] 冲突分数 (来自 AH-Net Module)
                              替代了旧的 energy_ratio 参数
        """
        # 使用 S-CAG 门控，基于冲突分数调节图像特征
        img_id_gated, img_attr_gated, confidence_weight = self.scag(
            img_id, img_attr, conflict_score
        )

        # RCSM 融合
        fused_embeds = self.rcsm(img_id_gated, img_attr_gated, txt_id, txt_attr)

        return fused_embeds, confidence_weight

def get_fusion_module(config):
    """
    动态创建融合模块。

    支持的类型：
    - 'scag_rcsm': S-CAG + RCSM (方案书推荐，最新)
    - 'enhanced_mamba': Enhanced Mamba Fusion (备用)
    """
    fusion_type = config.get("type")

    if fusion_type == "scag_rcsm":
        # 方案书的完整实现
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'd_state', 'd_conv', 'output_dim', 'dropout']}
        return ScagRcsmFusion(**valid_params)

    elif fusion_type == "samg_rcsm":
        # 兼容旧命名，内部使用 S-CAG
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'd_state', 'd_conv', 'output_dim', 'dropout']}
        return ScagRcsmFusion(**valid_params)

    elif fusion_type == "enhanced_mamba":
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'd_state', 'd_conv', 'num_layers', 'output_dim', 'dropout']}
        return EnhancedMambaFusion(**valid_params)

    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Supported: ['scag_rcsm', 'samg_rcsm', 'enhanced_mamba']")
