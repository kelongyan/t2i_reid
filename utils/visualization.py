# utils/visualization.py
"""
FSHD-Net可视化工具
包含频域热力图、注意力图、t-SNE降维、特征分布等
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class FSHDVisualizer:
    """
    FSHD-Net可视化器
    """
    def __init__(self, save_dir='visualizations', logger=None):
        """
        Args:
            save_dir: 可视化结果保存目录
            logger: 日志记录器
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['font.size'] = 10
        
        if logger:
            logger.debug_logger.info(f"✅ Visualizer initialized, save to: {self.save_dir}")
    
    def plot_frequency_masks(self, freq_info, epoch, batch_idx):
        """
        可视化频域掩码
        Args:
            freq_info: 频域信息字典
            epoch: 当前epoch
            batch_idx: 当前batch索引
        """
        if 'low_mask' not in freq_info or 'high_mask' not in freq_info:
            return
        
        low_mask = freq_info['low_mask'].cpu().numpy().squeeze()  # [H, W]
        high_mask = freq_info['high_mask'].cpu().numpy().squeeze()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 低频掩码
        im1 = axes[0].imshow(low_mask, cmap='hot', interpolation='bilinear')
        axes[0].set_title(f'Low-Frequency Mask (α={freq_info.get("alpha_low", 0):.3f})')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # 高频掩码
        im2 = axes[1].imshow(high_mask, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f'High-Frequency Mask (α={freq_info.get("alpha_high", 0):.3f})')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # 重叠区域
        overlap = low_mask * high_mask
        im3 = axes[2].imshow(overlap, cmap='RdYlGn_r', interpolation='bilinear')
        axes[2].set_title('Overlap (should be minimal)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        save_path = self.save_dir / f'freq_masks_epoch{epoch}_batch{batch_idx}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.debug_logger.info(f"📊 Saved frequency masks to {save_path}")
    
    def plot_attention_maps(self, id_attn_map, attr_attn_map, epoch, batch_idx, num_samples=4):
        """
        可视化注意力图
        Args:
            id_attn_map: ID分支注意力 [B, N, N]
            attr_attn_map: Attr分支注意力 [B, N, N]
            epoch: 当前epoch
            batch_idx: 当前batch
            num_samples: 可视化样本数
        """
        if id_attn_map is None or attr_attn_map is None:
            return
        
        id_attn = id_attn_map.cpu().numpy()[:num_samples]
        attr_attn = attr_attn_map.cpu().numpy()[:num_samples]
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 4))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # ID attention
            im1 = axes[i, 0].imshow(id_attn[i], cmap='viridis', aspect='auto')
            axes[i, 0].set_title(f'Sample {i+1}: ID Attention')
            axes[i, 0].set_xlabel('Key Position')
            axes[i, 0].set_ylabel('Query Position')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Attr attention
            im2 = axes[i, 1].imshow(attr_attn[i], cmap='plasma', aspect='auto')
            axes[i, 1].set_title(f'Sample {i+1}: Attr Attention')
            axes[i, 1].set_xlabel('Key Position')
            axes[i, 1].set_ylabel('Query Position')
            plt.colorbar(im2, ax=axes[i, 1])
        
        plt.tight_layout()
        save_path = self.save_dir / f'attention_maps_epoch{epoch}_batch{batch_idx}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.debug_logger.info(f"📊 Saved attention maps to {save_path}")
    
    def plot_feature_tsne(self, id_features, attr_features, labels, epoch, 
                         perplexity=30, n_iter=1000):
        """
        t-SNE降维可视化特征分布
        Args:
            id_features: ID特征 [N, D]
            attr_features: Attr特征 [N, D]
            labels: 标签 [N]
            epoch: 当前epoch
            perplexity: t-SNE参数
            n_iter: t-SNE迭代次数
        """
        id_feat = id_features.cpu().numpy()
        attr_feat = attr_features.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # 限制样本数量（t-SNE计算量大）
        max_samples = 2000
        if len(id_feat) > max_samples:
            indices = np.random.choice(len(id_feat), max_samples, replace=False)
            id_feat = id_feat[indices]
            attr_feat = attr_feat[indices]
            labels_np = labels_np[indices]
        
        # t-SNE降维
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        
        try:
            id_embedded = tsne.fit_transform(id_feat)
            attr_embedded = tsne.fit_transform(attr_feat)
        except Exception as e:
            if self.logger:
                self.logger.debug_logger.warning(f"t-SNE failed: {e}, using PCA instead")
            pca = PCA(n_components=2)
            id_embedded = pca.fit_transform(id_feat)
            attr_embedded = pca.fit_transform(attr_feat)
        
        # 绘制
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ID特征
        scatter1 = axes[0].scatter(id_embedded[:, 0], id_embedded[:, 1], 
                                   c=labels_np, cmap='tab20', s=10, alpha=0.6)
        axes[0].set_title(f'ID Features (Epoch {epoch})')
        axes[0].set_xlabel('t-SNE Dim 1')
        axes[0].set_ylabel('t-SNE Dim 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Attr特征
        scatter2 = axes[1].scatter(attr_embedded[:, 0], attr_embedded[:, 1],
                                   c=labels_np, cmap='tab20', s=10, alpha=0.6)
        axes[1].set_title(f'Attr Features (Epoch {epoch})')
        axes[1].set_xlabel('t-SNE Dim 1')
        axes[1].set_ylabel('t-SNE Dim 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        save_path = self.save_dir / f'tsne_epoch{epoch}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.debug_logger.info(f"📊 Saved t-SNE to {save_path}")
    
    def plot_feature_distribution(self, id_features, attr_features, epoch):
        """
        特征分布统计（范数、余弦相似度等）
        Args:
            id_features: ID特征 [N, D]
            attr_features: Attr特征 [N, D]
            epoch: 当前epoch
        """
        id_feat = id_features.cpu()
        attr_feat = attr_features.cpu()
        
        # 计算范数
        id_norms = torch.norm(id_feat, p=2, dim=1).numpy()
        attr_norms = torch.norm(attr_feat, p=2, dim=1).numpy()
        
        # 计算余弦相似度（ID vs Attr）
        id_normalized = F.normalize(id_feat, dim=1)
        attr_normalized = F.normalize(attr_feat, dim=1)
        cosine_sim = (id_normalized * attr_normalized).sum(dim=1).numpy()
        
        # 绘制
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 范数分布
        axes[0, 0].hist(id_norms, bins=50, alpha=0.7, label='ID', color='blue')
        axes[0, 0].hist(attr_norms, bins=50, alpha=0.7, label='Attr', color='red')
        axes[0, 0].set_title('Feature Norm Distribution')
        axes[0, 0].set_xlabel('L2 Norm')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 余弦相似度分布
        axes[0, 1].hist(cosine_sim, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('ID-Attr Cosine Similarity (should be near 0)')
        axes[0, 1].set_xlabel('Cosine Similarity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(0, color='red', linestyle='--', label='Target=0')
        axes[0, 1].legend()
        
        # 特征维度方差
        id_var = id_feat.var(dim=0).numpy()
        attr_var = attr_feat.var(dim=0).numpy()
        axes[1, 0].plot(id_var, label='ID', alpha=0.7)
        axes[1, 0].plot(attr_var, label='Attr', alpha=0.7)
        axes[1, 0].set_title('Per-dimension Variance')
        axes[1, 0].set_xlabel('Dimension')
        axes[1, 0].set_ylabel('Variance')
        axes[1, 0].legend()
        
        # 统计摘要
        stats_text = f"""
        Epoch {epoch} Statistics:
        
        ID Features:
          Mean Norm: {id_norms.mean():.4f}
          Std Norm:  {id_norms.std():.4f}
        
        Attr Features:
          Mean Norm: {attr_norms.mean():.4f}
          Std Norm:  {attr_norms.std():.4f}
        
        Orthogonality:
          Mean Cosine Sim: {cosine_sim.mean():.4f}
          Std Cosine Sim:  {cosine_sim.std():.4f}
          (Target: 0.0)
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / f'feature_dist_epoch{epoch}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.debug_logger.info(f"📊 Saved feature distribution to {save_path}")
    
    def plot_gate_statistics(self, gate_id, gate_attr, epoch, batch_idx):
        """
        门控统计可视化
        Args:
            gate_id: ID门控 [B, D]
            gate_attr: Attr门控 [B, D]
            epoch: 当前epoch
            batch_idx: 当前batch
        """
        gate_id_np = gate_id.cpu().numpy()
        gate_attr_np = gate_attr.cpu().numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 门控值分布
        axes[0, 0].hist(gate_id_np.flatten(), bins=50, alpha=0.7, label='ID Gate', color='blue')
        axes[0, 0].hist(gate_attr_np.flatten(), bins=50, alpha=0.7, label='Attr Gate', color='red')
        axes[0, 0].set_title('Gate Value Distribution')
        axes[0, 0].set_xlabel('Gate Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 门控均值（per sample）
        gate_id_mean = gate_id_np.mean(axis=1)
        gate_attr_mean = gate_attr_np.mean(axis=1)
        axes[0, 1].scatter(gate_id_mean, gate_attr_mean, alpha=0.5)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Equal')
        axes[0, 1].set_title('Per-sample Gate Mean')
        axes[0, 1].set_xlabel('ID Gate Mean')
        axes[0, 1].set_ylabel('Attr Gate Mean')
        axes[0, 1].legend()
        
        # 门控多样性（ID vs Attr）
        diversity = np.abs(gate_id_np - gate_attr_np).mean(axis=1)
        axes[1, 0].hist(diversity, bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Gate Diversity (|ID - Attr|)')
        axes[1, 0].set_xlabel('Diversity')
        axes[1, 0].set_ylabel('Frequency')
        
        # 统计摘要
        stats_text = f"""
        Epoch {epoch}, Batch {batch_idx}
        
        ID Gate:
          Mean: {gate_id_np.mean():.4f}
          Std:  {gate_id_np.std():.4f}
        
        Attr Gate:
          Mean: {gate_attr_np.mean():.4f}
          Std:  {gate_attr_np.std():.4f}
        
        Diversity:
          Mean: {diversity.mean():.4f}
          Std:  {diversity.std():.4f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                       verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        save_path = self.save_dir / f'gate_stats_epoch{epoch}_batch{batch_idx}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.debug_logger.info(f"📊 Saved gate statistics to {save_path}")
    
    def plot_frequency_energy_spectrum(self, freq_info, epoch, batch_idx):
        """
        频域能量谱可视化
        Args:
            freq_info: 频域信息
            epoch: 当前epoch
            batch_idx: 当前batch
        """
        if 'freq_magnitude' not in freq_info:
            return
        
        freq_mag = freq_info['freq_magnitude'].cpu().numpy()  # [B]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(freq_mag)), freq_mag, alpha=0.7, color='purple')
        ax.set_title(f'Frequency Energy Spectrum (Epoch {epoch})')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Frequency Magnitude')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / f'freq_energy_epoch{epoch}_batch{batch_idx}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        if self.logger:
            self.logger.debug_logger.info(f"📊 Saved frequency energy to {save_path}")
    
    def create_visualization_report(self, epoch, all_visualizations):
        """
        创建综合可视化报告（HTML格式）
        Args:
            epoch: 当前epoch
            all_visualizations: 所有可视化文件路径列表
        """
        html_content = f"""
        <html>
        <head>
            <title>FSHD-Net Visualization Report - Epoch {epoch}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .section {{ margin: 20px 0; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <h1>FSHD-Net Visualization Report - Epoch {epoch}</h1>
            <div class="section">
                <h2>Frequency Masks</h2>
                <img src="freq_masks_epoch{epoch}_batch0.png" alt="Frequency Masks">
            </div>
            <div class="section">
                <h2>Attention Maps</h2>
                <img src="attention_maps_epoch{epoch}_batch0.png" alt="Attention Maps">
            </div>
            <div class="section">
                <h2>Feature Distribution</h2>
                <img src="feature_dist_epoch{epoch}.png" alt="Feature Distribution">
            </div>
            <div class="section">
                <h2>t-SNE Visualization</h2>
                <img src="tsne_epoch{epoch}.png" alt="t-SNE">
            </div>
        </body>
        </html>
        """
        
        report_path = self.save_dir / f'report_epoch{epoch}.html'
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        if self.logger:
            self.logger.debug_logger.info(f"📊 Saved visualization report to {report_path}")
