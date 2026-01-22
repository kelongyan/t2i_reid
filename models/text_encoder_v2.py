import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入 Mamba，如果环境不支持（如Windows）则提供占位符或报错提示
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

class LayerNorm1d(nn.Module):
    """
    1D LayerNorm 适配器，支持 (B, C, L) 或 (B, L, C) 输入
    默认为 channels_last=True，即输入 (B, L, C)
    """
    def __init__(self, normalized_shape, eps=1e-6, channels_last=True):
        super().__init__()
        self.channels_last = channels_last
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        # x: [B, D, L] if channels_last=False
        # x: [B, L, D] if channels_last=True
        if not self.channels_last:
            x = x.transpose(1, 2) # [B, L, D]
            x = self.ln(x)
            x = x.transpose(1, 2) # [B, D, L]
        else:
            x = self.ln(x)
        return x

class ResidualBottleneck1d(nn.Module):
    """
    Stage 1: 局部残差瓶颈模块 (Local Residual Bottleneck)
    结构: 1x1 Conv (Squeeze) -> 3x3 Conv (Process) -> 1x1 Conv (Expand)
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = dim // reduction
        
        self.squeeze = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        self.process = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True)
        )
        
        self.expand = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(dim)
        )
        
        # 这里的残差连接在 forward 中处理

    def forward(self, x):
        # Input: [B, D, L]
        identity = x
        out = self.squeeze(x)
        out = self.process(out)
        out = self.expand(out)
        return identity + out

class GatedMambaBlock(nn.Module):
    """
    Stage 2: 门控长程 Mamba 模块 (Gated Residual Mamba)
    结构: Mamba -> Gating (Sigmoid) -> Residual
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba module not found. Please install mamba-ssm.")
            
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # 自适应门控生成器
        self.gate_fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input: [B, L, D]
        residual = x
        x_norm = self.norm(x)
        
        # Mamba 处理
        mamba_out = self.mamba(x_norm)
        
        # 门控机制: G = Sigmoid(Linear(x))
        # 使用原始输入 x (或者 x_norm) 来计算门控
        gate = torch.sigmoid(self.gate_fc(x_norm))
        
        # 调制
        out = mamba_out * gate
        out = self.dropout(out)
        
        return residual + out

class PyramidTextEncoder(nn.Module):
    """
    串行渐进式金字塔文本编码器 (Serial Progressive Pyramid Text Encoder)
    
    Architecture:
    Input (CLIP) -> [Stage 1: CNN Bottleneck] -> Attr Head (MaxPool)
                                     |
                                     v
                   [Stage 2: Gated Mamba] -> Input Injection -> ID Head (AvgPool)
    """
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim
        
        # === Stage 1: Local Attribute Extraction (CNN) ===
        self.stage1_bottleneck = ResidualBottleneck1d(dim, reduction=4)
        
        # Attr Head: LayerNorm -> MaxPool
        # 用于提取最显著的局部特征 (Keyphrase Activation)
        self.attr_norm = nn.LayerNorm(dim)
        
        # === Stage 2: Global Context Modeling (Mamba) ===
        self.stage2_mamba = GatedMambaBlock(dim=dim)
        
        # ID Head: LayerNorm -> AvgPool -> BNNeck
        # 用于聚合全局身份特征
        self.id_norm = nn.LayerNorm(dim)
        self.id_bn = nn.BatchNorm1d(dim)
        nn.init.constant_(self.id_bn.weight, 1.0)
        nn.init.constant_(self.id_bn.bias, 0.0)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input text embeddings from CLIP. Shape [B, L, D].
                              Expect L=77, D=768.
        Returns:
            dict: {
                'feat_attr': [B, D], # 属性特征
                'feat_id': [B, D],   # 身份特征
                'feat_id_bn': [B, D] # 经过BNNeck的身份特征(用于分类)
            }
        """
        B, L, D = x.shape
        
        # 保存原始输入用于 Input Injection
        x_raw = x
        
        # =================================================
        # Stage 1: Local Processing (CNN)
        # =================================================
        # Permute for Conv1d: [B, L, D] -> [B, D, L]
        x_cnn_in = x.transpose(1, 2)
        
        # Bottleneck processing
        x_cnn_out = self.stage1_bottleneck(x_cnn_in) # [B, D, L]
        
        # Permute back: [B, D, L] -> [B, L, D]
        x_stage1 = x_cnn_out.transpose(1, 2)
        
        # --- Attr Head Output ---
        # Norm -> MaxPool
        # MaxPool along sequence dim (dim=1)
        # x_stage1: [B, L, D]
        feat_attr = self.attr_norm(x_stage1)
        feat_attr = feat_attr.transpose(1, 2) # [B, D, L]
        feat_attr = F.adaptive_max_pool1d(feat_attr, 1).squeeze(2) # [B, D]
        
        # =================================================
        # Stage 2: Global Processing (Mamba)
        # =================================================
        # Mamba takes [B, L, D]
        x_mamba_out = self.stage2_mamba(x_stage1)
        
        # =================================================
        # Global Input Injection
        # =================================================
        # Feature Fusion: Stage 2 Output + Original Input
        # 防止深层遗忘，保留原始语义空间
        x_final = x_mamba_out + x_raw
        
        # --- ID Head Output ---
        # Norm -> AvgPool -> BNNeck
        x_final_norm = self.id_norm(x_final)
        x_final_norm = x_final_norm.transpose(1, 2) # [B, D, L]
        
        # AvgPool along sequence dim
        feat_id = F.adaptive_avg_pool1d(x_final_norm, 1).squeeze(2) # [B, D]
        
        # BNNeck (Classification Friendly)
        feat_id_bn = self.id_bn(feat_id)
        
        return {
            'feat_attr': feat_attr,
            'feat_id': feat_id,
            'feat_id_bn': feat_id_bn
        }
