# models/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class EnhancedMambaFusion(nn.Module):
    # 优化后的 Mamba 融合模块：通过门控机制整合图像和文本特征，利用 SSM 实现跨模态信息交互
    def __init__(self, dim, d_state=16, d_conv=4, num_layers=3, output_dim=256, dropout=0.1, logger=None):
        super().__init__()
        self.logger = logger
        
        # 模态对齐层
        self.image_align = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim)
        )
        self.text_align = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim),
            nn.Linear(dim, dim), nn.ReLU(), nn.LayerNorm(dim)
        )
        
        # 前置门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim // 2, 2), nn.Softmax(dim=-1)
        )
        
        if Mamba is None:
             raise ImportError("Mamba module not found")

        self.mamba_layers = nn.ModuleList([
            Mamba(d_model=dim * 2, d_state=d_state, d_conv=d_conv, expand=2) 
            for _ in range(num_layers)
        ])
        self.mamba_norms = nn.ModuleList([nn.LayerNorm(dim * 2) for _ in range(num_layers)])
        
        # 输出映射
        self.fc = nn.Linear(dim * 2, output_dim)
        self.norm_final = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, text_features):
        # NaN 检测与处理
        if torch.isnan(image_features).any() or torch.isnan(text_features).any():
            batch_size = image_features.size(0)
            return torch.zeros(batch_size, self.fc.out_features, device=image_features.device), \
                   torch.ones(batch_size, 2, device=image_features.device) * 0.5
        
        image_features = self.image_align(image_features)
        text_features = self.text_align(text_features)
        
        # 对齐后的 NaN 修正
        if torch.isnan(image_features).any() or torch.isnan(text_features).any():
            image_features = torch.where(torch.isnan(image_features), torch.zeros_like(image_features), image_features)
            text_features = torch.where(torch.isnan(text_features), torch.zeros_like(text_features), text_features)
        
        # 门控权重计算
        concat_features = torch.cat([image_features, text_features], dim=-1)
        gate_weights = self.gate(concat_features)
        
        if torch.isnan(gate_weights).any():
            gate_weights = torch.ones_like(gate_weights) * 0.5
        
        image_weight, text_weight = gate_weights[:, 0:1], gate_weights[:, 1:2]
        
        # 加权融合特征
        weighted_image = image_weight * image_features
        weighted_text = text_weight * text_features
        weighted_features = torch.cat([weighted_image, weighted_text], dim=-1).unsqueeze(1)
        
        # 多层 Mamba 迭代处理
        mamba_output = weighted_features
        for mamba, norm in zip(self.mamba_layers, self.mamba_norms):
            residual = mamba_output
            mamba_output = mamba(mamba_output)
            if torch.isnan(mamba_output).any():
                mamba_output = residual
            mamba_output = norm(mamba_output + residual)
        
        mamba_output = mamba_output.squeeze(1)
        
        # 输出投影
        fused_features = self.fc(mamba_output)
        fused_features = self.dropout(fused_features)
        fused_features = self.norm_final(fused_features)
        
        if torch.isnan(fused_features).any():
            fused_features = torch.zeros_like(fused_features)
        
        return fused_features, gate_weights


class SCAG_Gate(nn.Module):
    # S-CAG: 语义置信度感知门控。根据解耦质量（冲突分数）动态调节图像特征权重，实现"弃图保文"或"图文并重"
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.temperature = temperature

        # 置信度修正网络
        self.confidence_mlp = nn.Sequential(
            nn.Linear(1, dim // 4), nn.LayerNorm(dim // 4), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(dim // 4, dim // 2), nn.LayerNorm(dim // 2), nn.ReLU(),
            nn.Linear(dim // 2, dim)
        )

        # 互查询分支，用于特征交互
        self.mutual_proj = nn.Linear(dim * 2, dim * 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.confidence_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 最后一层初始化为 0，使初始状态主要受冲突分数影响
        if isinstance(self.confidence_mlp[-1], nn.Linear):
            nn.init.constant_(self.confidence_mlp[-1].weight, 0)
            nn.init.constant_(self.confidence_mlp[-1].bias, 0)

        nn.init.xavier_normal_(self.mutual_proj.weight, gain=1.0)
        if self.mutual_proj.bias is not None:
            nn.init.constant_(self.mutual_proj.bias, 0)

    def forward(self, id_feat, attr_feat, conflict_score):
        # NaN 鲁棒性检查
        if torch.isnan(id_feat).any() or torch.isnan(attr_feat).any():
            B, D = id_feat.shape
            return torch.zeros_like(id_feat), torch.zeros_like(attr_feat), torch.ones(B, D, device=id_feat.device) * 0.5
        
        if conflict_score.dim() == 1:
            conflict_score = conflict_score.unsqueeze(1)
        
        # 1. 计算置信度权重：冲突高则置信度低
        base_confidence = 1.0 - conflict_score
        delta_confidence = self.confidence_mlp(conflict_score)
        
        if torch.isnan(delta_confidence).any():
            delta_confidence = torch.zeros_like(delta_confidence)
        
        # 应用温度缩放并使用 Sigmoid 激活
        logits = torch.clamp((base_confidence + delta_confidence) / self.temperature, min=-10, max=10)
        confidence_weight = torch.sigmoid(logits)
        
        # 2. 互查询门控：计算 ID 和属性分支的相对权重
        joint_feat = torch.cat([id_feat, attr_feat], dim=-1)
        raw_gates = self.mutual_proj(joint_feat)
        
        if torch.isnan(raw_gates).any():
            raw_gates = torch.zeros_like(raw_gates)
        
        raw_gates = raw_gates.view(-1, 2, id_feat.shape[-1])
        gates = F.softmax(raw_gates, dim=1)

        g_id = gates[:, 0, :]
        g_attr = gates[:, 1, :]

        # 3. 应用置信度调节并配合残差连接，防止信息丢失
        alpha = 0.7
        id_out = alpha * (id_feat * g_id * confidence_weight) + (1 - alpha) * id_feat
        attr_out = alpha * (attr_feat * g_attr * confidence_weight) + (1 - alpha) * attr_feat

        return id_out, attr_out, confidence_weight


class RCSM_Fusion(nn.Module):
    # RCSM: 残差交叉扫描 Mamba 融合。通过正向和反向序列扫描，捕捉图像与文本特征之间的深度关联
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
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        
        self.out_proj = nn.Linear(dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, img_id, img_attr, txt_id, txt_attr):
        B, D = img_id.shape
        
        # 构建交替序列：[图像身份, 图像属性, 文本身份, 文本属性]
        s_inter = torch.stack([img_id, img_attr, txt_id, txt_attr], dim=1)
        
        x_norm = self.ln(s_inter)
        
        # 交叉扫描：正向与反向 SSM
        s_fwd = self.mamba_fwd(x_norm)
        s_rev = x_norm.flip(dims=[1])
        s_bwd = self.mamba_bwd(s_rev).flip(dims=[1])
        
        # 合并与残差投影
        s_mamba = s_inter + self.dropout(s_fwd + s_bwd)
        s_shortcut = self.injection_proj(s_inter)
        s_fused = F.layer_norm(s_mamba + s_shortcut, (D,))
        
        # 瓶颈层提取与聚合
        s_out = self.bottleneck(s_fused)
        fused_feat = s_out.mean(dim=1)
        
        out = self.out_norm(self.out_proj(fused_feat))
        return out


class ScagRcsmFusion(nn.Module):
    # S-CAG + RCSM 融合模块集成类：结合语义质量门控和交叉扫描融合，是系统的核心融合方案
    def __init__(self, dim=768, output_dim=256, **kwargs):
        super().__init__()
        self.scag = SCAG_Gate(dim)
        self.rcsm = RCSM_Fusion(dim, output_dim=output_dim, **kwargs)
        self.residual_proj = nn.Linear(dim, output_dim)

    def forward(self, img_id, img_attr, txt_id, txt_attr, conflict_score):
        # 1. 语义置信度门控
        img_id_gated, img_attr_gated, confidence_weight = self.scag(img_id, img_attr, conflict_score)

        # 2. 交叉扫描融合
        fused_embeds = self.rcsm(img_id_gated, img_attr_gated, txt_id, txt_attr)
        
        # 3. 跨模态残差连接：直接注入身份信息
        residual_weight = 0.3
        img_id_proj = self.residual_proj(img_id)
        fused_embeds = (1 - residual_weight) * fused_embeds + residual_weight * img_id_proj

        return fused_embeds, confidence_weight


def get_fusion_module(config):
    # 融合模块工厂函数：根据配置动态创建指定的融合组件
    fusion_type = config.get("type")

    if fusion_type in ["scag_rcsm", "samg_rcsm"]:
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'd_state', 'd_conv', 'output_dim', 'dropout']}
        return ScagRcsmFusion(**valid_params)

    elif fusion_type == "enhanced_mamba":
        valid_params = {k: v for k, v in config.items() if k in ['dim', 'd_state', 'd_conv', 'num_layers', 'output_dim', 'dropout']}
        return EnhancedMambaFusion(**valid_params)

    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Supported: ['scag_rcsm', 'samg_rcsm', 'enhanced_mamba']")
