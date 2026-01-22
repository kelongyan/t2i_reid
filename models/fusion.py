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

class SAMG_Gate(nn.Module):
    """
    频谱感知互斥门控 (Spectrum-Aware Mutual Gating)
    """
    def __init__(self, dim):
        super().__init__()
        
        # 1. 物理先验分支
        self.energy_mlp = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
        # 2. 互查询分支
        self.mutual_proj = nn.Linear(dim * 2, dim * 2)
        
    def forward(self, id_feat, attr_feat, energy_ratio):
        if energy_ratio.dim() == 1:
            energy_ratio = energy_ratio.unsqueeze(1)
            
        w_energy = self.energy_mlp(energy_ratio)
        
        joint_feat = torch.cat([id_feat, attr_feat], dim=-1)
        raw_gates = self.mutual_proj(joint_feat)
        
        raw_gates = raw_gates.view(-1, 2, id_feat.shape[-1])
        gates = F.softmax(raw_gates, dim=1)
        
        g_id = gates[:, 0, :]
        g_attr = gates[:, 1, :]
        
        id_out = id_feat * g_id
        attr_out = attr_feat * (g_attr * w_energy)
        
        return id_out, attr_out, w_energy

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

class SamgRcsmFusion(nn.Module):
    """
    Wrapper for SAMG + RCSM (Previously FusionV2)
    """
    def __init__(self, dim=768, output_dim=256, **kwargs):
        super().__init__()
        self.samg = SAMG_Gate(dim)
        self.rcsm = RCSM_Fusion(dim, output_dim=output_dim, **kwargs)
        
    def forward(self, img_id, img_attr, txt_id, txt_attr, energy_ratio):
        img_id_gated, img_attr_gated, w_e = self.samg(img_id, img_attr, energy_ratio)
        fused_embeds = self.rcsm(img_id_gated, img_attr_gated, txt_id, txt_attr)
        return fused_embeds, w_e

def get_fusion_module(config):
    """
    动态创建融合模块。
    """
    fusion_type = config.get("type")
    
    if fusion_type == "samg_rcsm":
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'd_state', 'd_conv', 'output_dim', 'dropout']}
        return SamgRcsmFusion(**valid_params)
        
    elif fusion_type == "enhanced_mamba":
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'd_state', 'd_conv', 'num_layers', 'output_dim', 'dropout']}
        return EnhancedMambaFusion(**valid_params)
        
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
