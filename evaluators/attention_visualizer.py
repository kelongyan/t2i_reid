# src/evaluation/attention_visualizer.py
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# 使用非交互后端
matplotlib.use('Agg')


def visualize_attention_maps(model, data_loader, output_dir, num_samples=10, device='cuda'):
    """
    可视化身份分支和服装分支的注意力热力图

    Args:
        model: Model 实例
        data_loader: 数据加载器
        output_dir: 输出目录
        num_samples: 可化样本数量
        device: 设备
    """
    model.eval()
    output_dir = Path(output_dir) / 'attention_maps'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 反归一化用于显示原图
    inv_normalize = transforms.Normalize(
        mean=[-0.5/0.5, -0.5/0.5, -0.5/0.5],
        std=[1/0.5, 1/0.5, 1/0.5]
    )
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            if sample_count >= num_samples:
                break
            
            imgs, cloth_captions, id_captions, pids, cam_ids, is_matched = data
            imgs = imgs.to(device)
            
            # 前向传播并获取注意力图
            outputs = model(image=imgs, cloth_instruction=cloth_captions, 
                          id_instruction=id_captions, return_attention=True)
            
            if len(outputs) < 12:
                logging.error("Model output does not contain attention maps")
                return
            
            id_attn_map = outputs[10]  # [batch_size, seq_len]
            cloth_attn_map = outputs[11]  # [batch_size, seq_len]
            
            if id_attn_map is None or cloth_attn_map is None:
                logging.warning("Attention maps are None, skipping visualization")
                continue
            
            # 处理每张图像
            batch_size = imgs.size(0)
            for i in range(min(batch_size, num_samples - sample_count)):
                # 获取原始图像
                img = imgs[i].cpu()
                img = inv_normalize(img)
                img = torch.clamp(img, 0, 1)
                img_np = img.permute(1, 2, 0).numpy()
                
                # 获取注意力图
                id_attn = id_attn_map[i].cpu().numpy()  # [seq_len]
                cloth_attn = cloth_attn_map[i].cpu().numpy()  # [seq_len]
                
                # 计算网格大小 (假设 ViT patch size=16, image size=224)
                h, w = 14, 14  # 224/16 = 14
                seq_len = id_attn.shape[0]
                
                # 确保序列长度匹配
                if seq_len != h * w:
                    logging.warning(f"Attention map size {seq_len} does not match expected {h*w}")
                    # 调整尺寸
                    h = w = int(np.sqrt(seq_len))
                
                # reshape 注意力图到 2D
                id_attn_2d = id_attn.reshape(h, w)
                cloth_attn_2d = cloth_attn.reshape(h, w)
                
                # 归一化到 [0, 1]
                id_attn_2d = (id_attn_2d - id_attn_2d.min()) / (id_attn_2d.max() - id_attn_2d.min() + 1e-8)
                cloth_attn_2d = (cloth_attn_2d - cloth_attn_2d.min()) / (cloth_attn_2d.max() - cloth_attn_2d.min() + 1e-8)
                
                # 上采样到原图大小
                id_attn_resized = np.array(Image.fromarray((id_attn_2d * 255).astype(np.uint8)).resize(
                    (img_np.shape[1], img_np.shape[0]), Image.BILINEAR)) / 255.0
                cloth_attn_resized = np.array(Image.fromarray((cloth_attn_2d * 255).astype(np.uint8)).resize(
                    (img_np.shape[1], img_np.shape[0]), Image.BILINEAR)) / 255.0
                
                # 绘制可视化
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # 原图
                axes[0].imshow(img_np)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # 身份分支注意力
                axes[1].imshow(img_np)
                im1 = axes[1].imshow(id_attn_resized, cmap='jet', alpha=0.5)
                axes[1].set_title('Identity Branch Attention')
                axes[1].axis('off')
                plt.colorbar(im1, ax=axes[1], fraction=0.046)
                
                # 服装分支注意力
                axes[2].imshow(img_np)
                im2 = axes[2].imshow(cloth_attn_resized, cmap='jet', alpha=0.5)
                axes[2].set_title('Clothing Branch Attention')
                axes[2].axis('off')
                plt.colorbar(im2, ax=axes[2], fraction=0.046)
                
                plt.tight_layout()
                
                # 保存
                save_path = output_dir / f'attention_sample_{sample_count:03d}_pid_{pids[i].item()}.png'
                plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
                plt.close()
                
                logging.info(f"Saved attention map to: {save_path}")
                sample_count += 1
                
                if sample_count >= num_samples:
                    break
    
    logging.info(f"Attention visualization completed. Saved {sample_count} samples to {output_dir}")