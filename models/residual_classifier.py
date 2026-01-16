"""
残差分类器模块
实现深层残差结构用于大规模身份分类
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    残差块：支持残差连接的基本构建单元
    """
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.block(x)
        out = out + identity  # 残差连接
        out = self.relu(out)
        return out


class ResidualClassifier(nn.Module):
    """
    残差分类器：用于大规模身份分类
    
    架构：
    - 输入投影：768 → hidden_dim
    - N个残差块：在hidden_dim维度
    - 输出投影：hidden_dim → output_dim
    
    特点：
    - 深层网络不易梯度消失
    - BN稳定训练
    - Dropout防止过拟合
    """
    def __init__(self, in_dim=768, hidden_dim=1536, output_dim=1024, 
                 num_blocks=2, dropout=0.2):
        """
        Args:
            in_dim: 输入特征维度（默认768，来自G-S3）
            hidden_dim: 中间隐藏层维度（默认1536）
            output_dim: 输出特征维度（默认1024）
            num_blocks: 残差块数量（默认2）
            dropout: Dropout比率（默认0.2）
        """
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影：扩展特征维度
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),  # 输入层较小的dropout
        )
        
        # 残差块堆叠
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) 
            for _ in range(num_blocks)
        ])
        
        # 输出投影：降维到目标维度
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout * 1.5),  # 输出层较大的dropout
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化权重：使用Xavier初始化
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, in_dim]
            
        Returns:
            out: 输出特征 [batch_size, output_dim]
        """
        # 输入投影
        x = self.input_proj(x)
        
        # 残差块堆叠
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 输出投影
        out = self.output_proj(x)
        
        return out


class DeepResidualClassifier(nn.Module):
    """
    深层残差分类器：更深的版本（可选）
    
    相比ResidualClassifier：
    - 更多残差块（3-4个）
    - 更高的中间维度（2048）
    - 适合更大规模的数据集
    """
    def __init__(self, in_dim=768, hidden_dim=2048, output_dim=1024, 
                 num_blocks=3, dropout=0.25):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影：渐进扩展
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
        )
        
        # 残差块堆叠
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) 
            for _ in range(num_blocks)
        ])
        
        # 输出投影：渐进降维
        self.output_proj = nn.Sequential(
            nn.Dropout(dropout * 1.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_proj(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        out = self.output_proj(x)
        return out
