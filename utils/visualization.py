# utils/visualization.py
"""
AH-Net Visualization Tools (Publication Quality)
Focus: Asymmetric Resolution, Semantic Decomposition, Overlay Heatmaps.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cv2
from pathlib import Path

class FSHDVisualizer:
    """
    Visualization for AH-Net.
    Generates high-quality heatmaps showing ID (structure) vs Attribute (texture) focus.
    """
    def __init__(self, save_dir='visualizations', logger=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        
        # Configure Matplotlib for academic style (Robust for Linux)
        plt.rcParams['font.family'] = 'serif'
        # Try Times New Roman first, then fallbacks common on Linux (DejaVu, Liberation), finally generic serif
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['figure.dpi'] = 300
        
        if logger:
            logger.debug_logger.info(f"✅ AH-Net Visualizer initialized at {self.save_dir}")

    def _denormalize_image(self, tensor):
        """
        Convert Image Tensor [3, H, W] (normalized) -> Numpy [H, W, 3] (0-255)
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    def _apply_heatmap(self, image, map_tensor, resolution_tag):
        """
        Overlay heatmap on image.
        Args:
            image: [H, W, 3] uint8
            map_tensor: [1, h, w] raw attention map (0-1)
            resolution_tag: 'low' or 'high'
        """
        H, W = image.shape[:2]
        
        # 1. Process Map
        attn = map_tensor.squeeze().cpu().numpy()
        
        # Normalize to 0-1 if not already
        if attn.max() > attn.min():
            attn = (attn - attn.min()) / (attn.max() - attn.min())
        
        # 2. Resize to Image Size
        # For 'low' resolution (ID map), use INTER_NEAREST to show blocky structure?
        # Or INTER_CUBIC for smooth looking?
        # Academic standard usually prefers smooth for heatmaps, but let's try to preserve
        # the "coarse" nature if possible. However, 12x4 is too coarse.
        # Let's use smooth resize for better aesthetics.
        attn_resized = cv2.resize(attn, (W, H), interpolation=cv2.INTER_CUBIC)
        
        # 3. Colorize
        heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 4. Blend
        alpha = 0.5
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay, attn_resized

    def plot_attention_maps(self, aux_info, epoch, batch_idx, images=None):
        """
        Plot AH-Net Attention Maps (Spatial/Semantic).
        Layout: [Original] | [ID Map (Structure)] | [Attr Map (Texture)]
        """
        # 🔥 修复: 先检查 aux_info 是否为 dict 类型
        if not isinstance(aux_info, dict):
            return
        
        if 'map_id' not in aux_info or 'map_attr' not in aux_info:
            return
            
        map_id = aux_info['map_id']     # [B, 1, H, W]
        map_attr = aux_info['map_attr'] # [B, 1, H, W]
        conflict_scores = aux_info.get('conflict_score', None)
        
        num_samples = min(4, map_id.size(0))
        
        # Prepare Figure
        fig = plt.figure(figsize=(10, 3 * num_samples))
        gs = GridSpec(num_samples, 3, figure=fig, wspace=0.05, hspace=0.1)
        
        for i in range(num_samples):
            # 1. Original Image
            if images is not None:
                img_np = self._denormalize_image(images[i])
            else:
                img_np = np.zeros((224, 224, 3), dtype=np.uint8) + 200 # Grey placeholder
            
            # 2. Generate Overlays
            overlay_id, _ = self._apply_heatmap(img_np, map_id[i], 'low')
            overlay_attr, _ = self._apply_heatmap(img_np, map_attr[i], 'high')
            
            # Conflict Score Text
            c_score = conflict_scores[i].item() if conflict_scores is not None else 0.0
            
            # 3. Plot
            # Col 0: Original
            ax0 = fig.add_subplot(gs[i, 0])
            ax0.imshow(img_np)
            if i == 0: ax0.set_title("Original", fontweight='bold')
            ax0.text(5, 20, f"C-Score: {c_score:.2f}", color='white', 
                     bbox=dict(facecolor='black', alpha=0.5))
            ax0.axis('off')
            
            # Col 1: ID Attention
            ax1 = fig.add_subplot(gs[i, 1])
            ax1.imshow(overlay_id)
            if i == 0: ax1.set_title(r"$\mathcal{M}_{ID}$ (Structure)", fontweight='bold')
            ax1.axis('off')
            
            # Col 2: Attr Attention
            ax2 = fig.add_subplot(gs[i, 2])
            ax2.imshow(overlay_attr)
            if i == 0: ax2.set_title(r"$\mathcal{M}_{Attr}$ (Texture)", fontweight='bold')
            ax2.axis('off')

        # Save
        save_path = self.save_dir / f'ahnet_maps_E{epoch}_B{batch_idx}.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        if self.logger:
            self.logger.debug_logger.info(f"📊 Saved AH-Net maps to {save_path}")