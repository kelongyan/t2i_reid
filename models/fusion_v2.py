import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class SAMG_Gate(nn.Module):
    """
    频谱感知互斥门控 (Spectrum-Aware Mutual Gating)
    
    Inputs:
        id_feat: [B, D]
        attr_feat: [B, D]
        energy_ratio: [B, 1] (or [B]) - Derived from Frequency Domain
    """
    def __init__(self, dim):
        super().__init__()
        
        # 1. 物理先验分支 (Physics-Prior Branch)
        # energy_ratio -> MLP -> energy_gate
        self.energy_mlp = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )
        
        # 2. 互查询分支 (Mutual-Query Branch)
        # [id; attr] -> Linear -> Softmax -> [g_id, g_attr]
        self.mutual_proj = nn.Linear(dim * 2, dim * 2)
        
    def forward(self, id_feat, attr_feat, energy_ratio):
        # energy_ratio shape check
        if energy_ratio.dim() == 1:
            energy_ratio = energy_ratio.unsqueeze(1) # [B, 1]
            
        # Physics Gate
        w_energy = self.energy_mlp(energy_ratio) # [B, D]
        
        # Mutual Gate
        joint_feat = torch.cat([id_feat, attr_feat], dim=-1) # [B, 2D]
        raw_gates = self.mutual_proj(joint_feat) # [B, 2D]
        
        # Reshape for Softmax over the two branches per channel?
        # "Use Softmax ... force competition"
        # We want gate for ID and gate for Attr to sum to 1 per channel?
        # Or distinct gates? The plan says: [g_id, g_attr] = Softmax(...)
        # Usually Softmax is over dimension.
        raw_gates = raw_gates.view(-1, 2, id_feat.shape[-1]) # [B, 2, D]
        gates = F.softmax(raw_gates, dim=1) # [B, 2, D]
        
        g_id = gates[:, 0, :]
        g_attr = gates[:, 1, :]
        
        # Final Gating
        # ID: Robust, just semantic gating
        id_out = id_feat * g_id
        
        # Attr: Needs semantic gating AND physical clarity (energy)
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
        
        # LayerNorm before Mamba
        self.ln = nn.LayerNorm(dim)
        
        # Bi-Directional Mamba
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=2)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=2)
        
        self.dropout = nn.Dropout(dropout)
        
        # Global Input Injection Projection
        # Input to injection is Concat(S_id, S_attr) -> 4 * D?
        # S_inter has length 4.
        # Injection adds to the SEQUENCE.
        # So we project [B, 4, D] to [B, 4, D]? Or simple linear.
        # Plan: S_shortcut = Linear(Concat(S_id, S_attr))
        # If we interpret S_id as [Img_ID, Txt_ID], S_attr as [Img_Attr, Txt_Attr]
        # Concat is size 4 in sequence dim.
        # We effectively want a residual on the sequence.
        self.injection_proj = nn.Linear(dim, dim) # Applied per token
        
        # Bottleneck Projection
        self.bottleneck = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Final Output Projection
        self.out_proj = nn.Linear(dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, img_id, img_attr, txt_id, txt_attr):
        """
        Args:
            img_id, img_attr, txt_id, txt_attr: [B, D]
        """
        B, D = img_id.shape
        
        # 1. Interleaved Construction
        # Sequence: [Img_ID, Img_Attr, Txt_ID, Txt_Attr]
        # Wait, Plan says: [id_1, attr_1, id_2, attr_2]
        # Pair 1 (Image): Img_ID, Img_Attr
        # Pair 2 (Text): Txt_ID, Txt_Attr
        s_inter = torch.stack([img_id, img_attr, txt_id, txt_attr], dim=1) # [B, 4, D]
        
        # 2. Bi-Directional Mamba
        x_norm = self.ln(s_inter)
        
        # Forward
        s_fwd = self.mamba_fwd(x_norm)
        
        # Backward (Flip -> Mamba -> Flip)
        s_rev = x_norm.flip(dims=[1])
        s_bwd = self.mamba_bwd(s_rev).flip(dims=[1])
        
        # Residual
        s_mamba = s_inter + self.dropout(s_fwd + s_bwd)
        
        # 3. Global Input Injection
        # We re-inject the original features
        s_shortcut = self.injection_proj(s_inter)
        s_fused = F.layer_norm(s_mamba + s_shortcut, (D,))
        
        # 4. Bottleneck
        s_out = self.bottleneck(s_fused) # [B, 4, D]
        
        # 5. Global Pooling & Projection
        # Average over the sequence (fusion of all parts)
        fused_feat = s_out.mean(dim=1) # [B, D]
        
        out = self.out_proj(fused_feat)
        out = self.out_norm(out)
        
        return out

class FusionV2(nn.Module):
    """
    Wrapper for SAMG + RCSM
    """
    def __init__(self, dim=768, output_dim=256, **kwargs):
        super().__init__()
        self.samg = SAMG_Gate(dim)
        self.rcsm = RCSM_Fusion(dim, output_dim=output_dim, **kwargs)
        
    def forward(self, img_id, img_attr, txt_id, txt_attr, energy_ratio):
        # 1. Gating (Only on Image features as per plan logic "Physics-Prior")
        # Text features are semantically generated, assume consistent?
        # Or do we gate text too? Plan implies SAMG is for "Physical Clarity".
        # So we gate Image.
        img_id_gated, img_attr_gated, w_e = self.samg(img_id, img_attr, energy_ratio)
        
        # 2. Fusion
        fused_embeds = self.rcsm(img_id_gated, img_attr_gated, txt_id, txt_attr)
        
        return fused_embeds, w_e
