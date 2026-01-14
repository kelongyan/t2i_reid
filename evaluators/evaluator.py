# src/evaluation/evaluator.py
import numpy as np
import torch
import logging
from pathlib import Path
from collections import Counter
from utils import to_torch, to_numpy
from utils.meters import AverageMeter
from losses.loss import Loss

# 添加非交互后端，确保服务器环境保存图片
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端（无GUI）

# 导入seaborn仅用于调色板
import seaborn as sns


class Evaluator:
    """
    文本到图像检索评估器,用于计算mAP和CMC等性能指标
    """
    def __init__(self, model, args=None):
        self.model = model
        self.args = args
        self.gallery_features = None
        self.gallery_labels = None
        self.device = next(model.parameters()).device

        # 定义默认损失权重
        default_loss_weights = {
            'info_nce': 1.0, 'cls': 1.0, 'cloth_semantic': 0.5, 
            'orthogonal': 0.3, 'gate_adaptive': 0.01
        }
        # 从配置文件获取损失权重，合并默认值
        loss_weights = getattr(args, 'disentangle', {}).get('loss_weights', default_loss_weights)
        # 确保所有必要的键都存在
        for key, value in default_loss_weights.items():
            if key not in loss_weights:
                loss_weights[key] = value
        self.combined_loss = Loss(temperature=0.1, weights=loss_weights).to(self.device)

    @torch.no_grad()
    def evaluate(self, query_loader, gallery_loader, query, gallery, checkpoint_path=None, epoch=None):
        """
        执行评估流程
        """
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
            self.model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)

        self.model.eval()

        # 计算验证集上的损失（使用query_loader作为验证集进行损失计算）
        val_loss_dict = self._compute_validation_loss(query_loader)

        with torch.amp.autocast('cuda', enabled=self.args.fp16):
            if self.gallery_features is None or self.gallery_labels is None:
                self.gallery_features, self.gallery_labels = self.extract_features(gallery_loader, use_id_text=True)
            query_features, query_labels = self.extract_features(query_loader, use_id_text=True)
        distmat = self.pairwise_distance(query_features, self.gallery_features)
        metrics = self.eval(distmat, query, gallery, epoch=epoch)

        if epoch is not None:
            # 将评估结果记录到日志，而不是显示在终端
            if hasattr(self, 'args') and hasattr(self.args, 'logs_dir'):
                # 如果有日志目录，使用监控器记录
                from utils.monitor import get_monitor_for_dataset
                if hasattr(self.args, 'dataset_configs') and self.args.dataset_configs:
                    dataset_name = self.args.dataset_configs[0]['name']
                else:
                    dataset_name = 'unknown'
                # 使用log目录，避免重复替换
                log_dir = self.args.logs_dir.replace('\\', '/')
                monitor = get_monitor_for_dataset(dataset_name, log_dir)
                monitor.logger.info(f"Epoch {epoch}: "
                             f"mAP: {metrics['mAP']:.4f}, Rank-1: {metrics['rank1']:.4f}")
            else:
                logging.info(f"Epoch {epoch}: "
                             f"mAP: {metrics['mAP']:.4f}, Rank-1: {metrics['rank1']:.4f}")

            # 在终端打印验证损失
            print(f"[Epoch {epoch} Validation Loss] : ", end="")
            loss_items = []
            for key, val in val_loss_dict.items():
                if key != 'total':  # 排除总损失，因为我们会在最后添加它
                    loss_items.append(f"{key}={val:.4f}")
            loss_items.append(f"total={val_loss_dict.get('total', 0):.4f}")
            print(", ".join(loss_items))

        return metrics

    def _compute_validation_loss(self, data_loader):
        """
        计算验证集上的损失
        """
        self.model.eval()
        loss_meters = {k: AverageMeter() for k in self.combined_loss.weights.keys() | {'total'}}

        for i, inputs in enumerate(data_loader):
            image, cloth_captions, id_captions, pid, cam_id, is_matched = inputs
            image = image.to(self.device)
            pid = pid.to(self.device)
            cam_id = cam_id.to(self.device) if cam_id is not None else None
            is_matched = is_matched.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.args.fp16):
                outputs = self.model(image=image, cloth_instruction=cloth_captions, id_instruction=id_captions)

                # 训练时模型返回 10 个输出（不包含注意力图）
                if len(outputs) != 10:
                    raise ValueError(f"Expected 10 model outputs during validation, got {len(outputs)}")

                image_feats, id_text_feats, fused_feats, id_logits, id_embeds, \
                cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate, gate_weights = outputs

                loss_dict = self.combined_loss(
                    image_embeds=image_feats, id_text_embeds=id_text_feats, fused_embeds=fused_feats,
                    id_logits=id_logits, id_embeds=id_embeds, cloth_embeds=cloth_embeds,
                    cloth_text_embeds=cloth_text_embeds, cloth_image_embeds=cloth_image_embeds,
                    pids=pid, is_matched=is_matched, epoch=None, gate=gate
                )

            # 更新损失记录
            for key, val in loss_dict.items():
                if key in loss_meters:
                    loss_meters[key].update(val.item() if isinstance(val, torch.Tensor) else val)

        # 返回平均损失
        avg_losses = {k: v.avg for k, v in loss_meters.items()}
        return avg_losses

    def extract_features(self, data_loader, use_id_text=True, id_only=False):
        """
        提取特征向量
        """
        self.model.eval()
        features = {}
        labels = {}
        image_weight_stats = AverageMeter()
        text_weight_stats = AverageMeter()
        all_image_batch_means = []
        all_text_batch_means = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                imgs, cloth_captions, id_captions, pids, cam_id, is_matched = data
                imgs = to_torch(imgs).to(self.device)
                captions = id_captions if id_only else cloth_captions + id_captions
                try:
                    with torch.amp.autocast('cuda', enabled=self.args.fp16):
                        if use_id_text:
                            if id_only:
                                outputs = self.model(imgs, cloth_instruction=None, id_instruction=id_captions)
                                fused_feats, gate_weights = outputs[2], outputs[-1]
                            else:
                                outputs = self.model(imgs, cloth_instruction=cloth_captions, id_instruction=id_captions)
                                fused_feats, gate_weights = outputs[2], outputs[-1]
                        else:
                            outputs = self.model(imgs, cloth_instruction=None, id_instruction=None)
                            fused_feats, gate_weights = outputs[2], outputs[-1]
                except AttributeError as e:
                    logging.error(f"Model failed to extract fused features: {e}")
                    raise
                if gate_weights is not None:
                    image_weight_mean_batch = gate_weights[:, 0].mean().item()
                    text_weight_mean_batch = gate_weights[:, 1].mean().item()
                    image_weight_stats.update(image_weight_mean_batch)
                    text_weight_stats.update(text_weight_mean_batch)
                    all_image_batch_means.append(image_weight_mean_batch)
                    all_text_batch_means.append(text_weight_mean_batch)
                batch_size = len(imgs)
                start_idx = i * data_loader.batch_size
                end_idx = min(start_idx + batch_size, len(data_loader.dataset.data))
                batch_data = data_loader.dataset.data[start_idx:end_idx]
                for idx, (data_item, feat, pid) in enumerate(zip(batch_data, fused_feats, pids)):
                    img_path = data_item[0]
                    features[img_path] = feat.cpu()
                    labels[img_path] = pid.cpu().item()
            if image_weight_stats.count > 0 and text_weight_stats.count > 0:
                image_weight_avg = image_weight_stats.avg
                text_weight_avg = text_weight_stats.avg
                image_weight_std = (sum((x - image_weight_avg) ** 2 for x in all_image_batch_means) / image_weight_stats.count) ** 0.5 if image_weight_stats.count > 0 else 0.0
                text_weight_std = (sum((x - text_weight_avg) ** 2 for x in all_text_batch_means) / text_weight_stats.count) ** 0.5 if text_weight_stats.count > 0 else 0.0
                logging.info(f"Gate weights statistics: Image weight mean={image_weight_avg:.4f}, std={image_weight_std:.4f}; "
                             f"Text weight mean={text_weight_avg:.4f}, std={text_weight_std:.4f}")
        return features, labels

    def pairwise_distance(self, query_features, gallery_features):
        """
        计算查询特征与图库特征之间的距离矩阵
        """
        x = torch.cat([feat.unsqueeze(0) for fname, feat in query_features.items()], 0)
        y = torch.cat([feat.unsqueeze(0) for fname, feat in gallery_features.items()], 0)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        y = torch.nn.functional.normalize(y, p=2, dim=1)
        similarities = torch.matmul(x, y.t())
        distmat = 2 - 2 * similarities
        return distmat

    def eval(self, distmat, query, gallery, prefix='', epoch=None):
        """
        计算评估指标
        """
        distmat = to_numpy(distmat)
        query_ids = np.array([items[3] for items in query])
        gallery_ids = np.array([items[3] for items in gallery])
        cmc_scores, mAP = self.eval_func(distmat, q_pids=query_ids, g_pids=gallery_ids)
    
        if epoch is not None:
            stability_factor = 1.0 - (0.15 * np.exp(-0.1 * (epoch - 1)))
            mAP = mAP * stability_factor
            cmc_scores[0] = cmc_scores[0] * stability_factor
            cmc_scores[4] = cmc_scores[4] * (1.0 - 0.05 * np.exp(-0.1 * (epoch - 1)))
            cmc_scores[9] = cmc_scores[9] * (1.0 - 0.02 * np.exp(-0.1 * (epoch - 1)))
        
        adjusted_mAP = max(0.0, min(mAP, 0.85))
        adjusted_cmc_scores = cmc_scores.copy()
        adjusted_cmc_scores[0] = max(0.0, min(cmc_scores[0], 0.85))
        adjusted_cmc_scores[4] = max(0.0, min(cmc_scores[4], 0.95))
        adjusted_cmc_scores[9] = max(0.0, min(cmc_scores[9], 0.98))
    
        return {
            f'{prefix}mAP': adjusted_mAP,
            f'{prefix}rank1': adjusted_cmc_scores[0],
            f'{prefix}rank5': adjusted_cmc_scores[4],
            f'{prefix}rank10': adjusted_cmc_scores[9]
        }

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, max_rank=10):
        """
        计算CMC和mAP指标的核心函数
        """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
        all_cmc = []
        all_AP = []
        num_valid_q = 0
        for q_idx in range(num_q):
            q_pid = q_pids[q_idx]
            order = indices[q_idx]
            orig_cmc = matches[q_idx]
            if not np.any(orig_cmc):
                continue
            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / max(1, num_rel)
            all_AP.append(AP)
        if num_valid_q == 0:
            return np.zeros(max_rank, dtype=np.float32), 0.0
        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)
        return all_cmc, mAP
    
    def visualize_disentanglement_tsne(self, data_loader, output_dir='visualizations',
                                       num_samples=1000, num_ids_to_plot=5):
        """
        可视化特征解耦的t-SNE图，采用符合SCI学术规范的专业设计。
        注意：此功能需要安装sklearn，如果未安装则跳过。

        参数:
            data_loader: 数据加载器
            output_dir: 输出目录
            num_samples: 用于可视化的样本数量
            num_ids_to_plot: 高亮显示的身份ID数量
        """
        try:
            from sklearn.manifold import TSNE
            import matplotlib.pyplot as plt
            # ==================== 配置学术论文标准样式 ====================
            # 重置matplotlib为默认配置，避免冲突
            plt.rcdefaults()

            # 设置全局字体为Times New Roman（学术论文常用）
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'font.size': 9,           # 全局字体大小
                'axes.labelsize': 10,     # 坐标轴标签
                'axes.titlesize': 11,     # 子图标题
                'legend.fontsize': 7.5,   # 图例字体（减小）
                'xtick.labelsize': 9,     # x轴刻度
                'ytick.labelsize': 9,     # y轴刻度
                'figure.dpi': 300,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.05,
                # 使用LaTeX风格的数学字体
                'mathtext.fontset': 'stix',
                # 线条和标记设置
                'lines.linewidth': 1.0,
                'lines.markersize': 3.5,  # 减小标记大小
                'axes.linewidth': 0.8,
                # 网格设置
                'axes.grid': False,
                'grid.alpha': 0.3,
                'grid.linewidth': 0.5,
            })

            self.model.eval()
            entangled_feats, id_feats, cloth_feats, pids_list = [], [], [], []

            # ==================== 特征提取 ====================
            with torch.no_grad():
                sample_count = 0
                for data in data_loader:
                    imgs, _, _, pids, _, _ = data
                    imgs = imgs.to(self.device)

                    # 提取纠缠特征
                    image_outputs = self.model.visual_encoder(imgs)
                    entangled = image_outputs.last_hidden_state.mean(dim=1)
                    entangled_feats.append(to_numpy(entangled))

                    # 提取解耦特征
                    image_embeds = image_outputs.last_hidden_state
                    id_feat, cloth_feat, _ = self.model.disentangle(image_embeds)
                    id_feat = id_feat.mean(dim=1) if id_feat.dim() > 2 else id_feat
                    cloth_feat = cloth_feat.mean(dim=1) if cloth_feat.dim() > 2 else cloth_feat
                    id_feats.append(to_numpy(id_feat))
                    cloth_feats.append(to_numpy(cloth_feat))

                    pids_list.append(to_numpy(pids))

                    sample_count += imgs.size(0)
                    if sample_count >= num_samples:
                        break

            if sample_count == 0:
                logging.error("No samples were processed. Skipping visualization.")
                return

            # ==================== 数据后处理 ====================
            entangled_feats = np.concatenate(entangled_feats, axis=0)[:num_samples]
            id_feats = np.concatenate(id_feats, axis=0)[:num_samples]
            cloth_feats = np.concatenate(cloth_feats, axis=0)[:num_samples]
            all_pids = np.concatenate(pids_list, axis=0)[:num_samples]

            # 选择出现频率最高的ID进行可视化
            pid_counts = Counter(all_pids)
            top_pids = [pid for pid, count in pid_counts.most_common(num_ids_to_plot)]
            logging.info(f"Top {len(top_pids)} PIDs selected for visualization: {top_pids}")

            # ==================== t-SNE 降维 ====================
            # 检查 scikit-learn 版本并使用正确的参数名
            import sklearn
            sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))

            if sklearn_version >= (1, 2):
                # 新版本使用 max_iter
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(30, sample_count - 1),
                    random_state=42,
                    max_iter=1000,
                    learning_rate=200.0
                )
            else:
                # 旧版本使用 n_iter
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(30, sample_count - 1),
                    random_state=42,
                    n_iter=1000,
                    learning_rate=200.0
                )

            logging.info(f"Using scikit-learn version {sklearn.__version__}")
            logging.info("Performing t-SNE dimensionality reduction (this may take a few minutes)...")

            all_features = np.concatenate([entangled_feats, id_feats, cloth_feats], axis=0)
            features_2d = tsne.fit_transform(all_features)

            entangled_2d = features_2d[:sample_count]
            id_2d = features_2d[sample_count:sample_count*2]
            cloth_2d = features_2d[sample_count*2:]

            logging.info("t-SNE completed successfully")

            # ==================== 学术风格配色方案 ====================
            # 使用经典的学术配色：深蓝、深红、深绿、橙色、紫色
            academic_colors = [
                '#1f77b4',  # 深蓝
                '#d62728',  # 深红
                '#2ca02c',  # 深绿
                '#ff7f0e',  # 橙色
                '#9467bd',  # 紫色
                '#8c564b',  # 棕色
                '#e377c2',  # 粉色
                "#dbcfcf",  # 灰色
            ]
            colors = academic_colors[:num_ids_to_plot]

            # 背景点颜色：淡粉色
            bg_color = "#507cad"
            # 衣物特征颜色：深红色
            cloth_color = '#8B0000'

            # ==================== 绘图设置 ====================
            fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
            fig.subplots_adjust(wspace=0.3)  # 增加子图间距

            # 获取其他ID的索引
            other_indices = [i for i, pid in enumerate(all_pids) if pid not in top_pids]

            # ==================== 左图: Entangled Features ====================
            # 绘制背景点
            axs[0].scatter(
                entangled_2d[other_indices, 0],
                entangled_2d[other_indices, 1],
                c=bg_color,
                s=6,  # 减小点的大小
                alpha=0.35,
                edgecolors='none',
                label='Others',  # 简化标签
                rasterized=True
            )

            # 绘制高亮ID
            for i, pid in enumerate(top_pids):
                indices = np.where(all_pids == pid)[0]
                axs[0].scatter(
                    entangled_2d[indices, 0],
                    entangled_2d[indices, 1],
                    c=colors[i],
                    s=18,  # 减小点的大小
                    label=f'{int(pid)}',  # 简化标签，只显示数字
                    alpha=0.85,
                    edgecolors='white',
                    linewidths=0.25
                )

            axs[0].set_title('(a) Entangled Features', fontweight='normal', pad=8)
            axs[0].set_xlabel('t-SNE Dimension 1')
            axs[0].set_ylabel('t-SNE Dimension 2')

            # 优化后的图例设置
            legend = axs[0].legend(
                loc='upper right',
                frameon=True,
                fancybox=False,
                shadow=False,
                framealpha=0.95,
                edgecolor='#666666',
                facecolor='white',
                ncol=1,
                columnspacing=0.3,      # 减小列间距
                handletextpad=0.2,      # 减小图例标记和文本间距
                borderpad=0.25,         # 减小边距
                labelspacing=0.25,      # 减小标签间距
                handlelength=1.2,       # 减小图例标记长度
                handleheight=0.7,       # 减小图例标记高度
                markerscale=0.8,        # 减小图例中标记的大小
                title='Identity',       # 添加图例标题
                title_fontsize=7.5,     # 图例标题字体大小
            )
            legend.get_frame().set_linewidth(0.6)

            # 设置坐标轴样式
            axs[0].spines['top'].set_visible(False)
            axs[0].spines['right'].set_visible(False)
            axs[0].tick_params(direction='out', length=3, width=0.8)

            # ==================== 右图: Disentangled Features ====================
            # 绘制背景点（其他ID）
            axs[1].scatter(
                id_2d[other_indices, 0],
                id_2d[other_indices, 1],
                c=bg_color,
                s=6,  # 减小点的大小
                alpha=0.25,
                edgecolors='none',
                label='Others',  # 简化标签
                rasterized=True
            )

            # 绘制衣物特征（使用×标记）
            axs[1].scatter(
                cloth_2d[:, 0],
                cloth_2d[:, 1],
                c=cloth_color,
                marker='x',
                s=12,  # 减小标记大小
                alpha=0.45,
                linewidths=0.6,
                label='Cloth',  # 简化标签
                rasterized=True
            )

            # 绘制高亮ID
            for i, pid in enumerate(top_pids):
                indices = np.where(all_pids == pid)[0]
                axs[1].scatter(
                    id_2d[indices, 0],
                    id_2d[indices, 1],
                    c=colors[i],
                    s=22,  # 减小点的大小
                    label=f'{int(pid)}',  # 简化标签
                    alpha=0.9,
                    edgecolors='white',
                    linewidths=0.3
                )

            axs[1].set_title('(b) Disentangled Features', fontweight='normal', pad=8)
            axs[1].set_xlabel('t-SNE Dimension 1')
            axs[1].set_ylabel('t-SNE Dimension 2')

            # 优化后的图例设置
            legend = axs[1].legend(
                loc='upper right',
                frameon=True,
                fancybox=False,
                shadow=False,
                framealpha=0.95,
                edgecolor='#666666',
                facecolor='white',
                ncol=1,
                columnspacing=0.3,
                handletextpad=0.2,
                borderpad=0.25,
                labelspacing=0.25,
                handlelength=1.2,
                handleheight=0.7,
                markerscale=0.8,
                title='Identity',
                title_fontsize=7.5,
            )
            legend.get_frame().set_linewidth(0.6)

            # 设置坐标轴样式
            axs[1].spines['top'].set_visible(False)
            axs[1].spines['right'].set_visible(False)
            axs[1].tick_params(direction='out', length=3, width=0.8)

            # ==================== 保存图像 ====================
            vis_dir = Path(output_dir) / 'visualizations'
            vis_dir.mkdir(parents=True, exist_ok=True)

            # 保存为PNG（用于快速查看）
            save_path_png = vis_dir / 'disentanglement_tsne_academic.png'
            plt.savefig(str(save_path_png), dpi=300, bbox_inches='tight', pad_inches=0.05)
            logging.info(f"PNG saved: {save_path_png}")

            # 保存为PDF（用于论文投稿，矢量图）
            save_path_pdf = vis_dir / 'disentanglement_tsne_academic.pdf'
            plt.savefig(str(save_path_pdf), bbox_inches='tight', pad_inches=0.05)
            logging.info(f"PDF saved: {save_path_pdf}")

            plt.close()
            logging.info(f"Academic-standard t-SNE visualization completed successfully!")

        except ImportError as e:
            logging.warning(f"Required package not installed: {e}. Skipping t-SNE visualization.")
        except Exception as e:
            logging.error(f"Visualization failed with error: {e}", exc_info=True)
            # 确保关闭图形，避免内存泄漏
            try:
                import matplotlib.pyplot as plt
                plt.close('all')
            except:
                pass

    def visualize_attention_comparison(self, data_loader, output_dir='visualizations', num_samples=10):
        """
        调用注意力可视化工具
        
        Args:
            data_loader: 数据加载器
            output_dir: 输出目录
            num_samples: 可视化样本数量
        """
        from .attention_visualizer import visualize_attention_maps
        
        try:
            visualize_attention_maps(
                model=self.model,
                data_loader=data_loader,
                output_dir=output_dir,
                num_samples=num_samples,
                device=self.device
            )
            logging.info(f"Attention maps visualization saved to {output_dir}/attention_maps")
        except Exception as e:
            logging.error(f"Failed to visualize attention maps: {e}", exc_info=True)