# src/evaluation/evaluator.py
import numpy as np
import torch
import logging
from pathlib import Path
from collections import Counter
from utils import to_torch, to_numpy
from utils.meters import AverageMeter
from losses.loss import Loss

class Evaluator:
    """
    Text-to-Image Retrieval Evaluator for computing mAP and CMC performance metrics
    """
    def __init__(self, model, args=None):
        self.model = model
        self.args = args
        self.gallery_features = None
        self.gallery_labels = None
        self.device = next(model.parameters()).device

        # Define default loss weights
        default_loss_weights = {
            'info_nce': 1.0, 'cls': 1.0, 'cloth_semantic': 0.5, 
            'orthogonal': 0.3, 'gate_adaptive': 0.01
        }
        # Get loss weights from config file, merge with defaults
        loss_weights = getattr(args, 'disentangle', {}).get('loss_weights', default_loss_weights)
        # Ensure all necessary keys exist
        for key, value in default_loss_weights.items():
            if key not in loss_weights:
                loss_weights[key] = value
        self.combined_loss = Loss(temperature=0.1, weights=loss_weights).to(self.device)

    @torch.no_grad()
    def evaluate(self, query_loader, gallery_loader, query, gallery, checkpoint_path=None, epoch=None):
        """
        Execute the evaluation pipeline
        """
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
            self.model.load_state_dict(checkpoint.get('model', checkpoint), strict=False)

        self.model.eval()

        # Compute validation loss (using query_loader as validation set)
        val_loss_dict = self._compute_validation_loss(query_loader)

        with torch.amp.autocast('cuda', enabled=self.args.fp16):
            if self.gallery_features is None or self.gallery_labels is None:
                self.gallery_features, self.gallery_labels = self.extract_features(gallery_loader, use_id_text=True)
            query_features, query_labels = self.extract_features(query_loader, use_id_text=True)
        distmat = self.pairwise_distance(query_features, self.gallery_features)
        metrics = self.eval(distmat, query, gallery, epoch=epoch)

        if epoch is not None:
            # Log evaluation results instead of displaying in terminal
            if hasattr(self, 'args') and hasattr(self.args, 'logs_dir'):
                # If log directory exists, use monitor for logging
                from utils.monitor import get_monitor_for_dataset
                if hasattr(self.args, 'dataset_configs') and self.args.dataset_configs:
                    dataset_name = self.args.dataset_configs[0]['name']
                else:
                    dataset_name = 'unknown'
                # Use log directory, avoid duplicate replacement
                log_dir = self.args.logs_dir.replace('\\', '/')
                monitor = get_monitor_for_dataset(dataset_name, log_dir)
                monitor.logger.info(f"Epoch {epoch}: "
                             f"mAP: {metrics['mAP']:.4f}, Rank-1: {metrics['rank1']:.4f}")
            else:
                logging.info(f"Epoch {epoch}: "
                             f"mAP: {metrics['mAP']:.4f}, Rank-1: {metrics['rank1']:.4f}")

            # Print validation loss in terminal
            print(f"[Epoch {epoch} Validation Loss] : ", end="")
            loss_items = []
            for key, val in val_loss_dict.items():
                if key != 'total':  # Exclude total loss as we add it at the end
                    loss_items.append(f"{key}={val:.4f}")
            loss_items.append(f"total={val_loss_dict.get('total', 0):.4f}")
            print(", ".join(loss_items))

        return metrics

    def _compute_validation_loss(self, data_loader):
        """
        Compute validation loss on the dataset
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

            # Update loss records
            for key, val in loss_dict.items():
                if key in loss_meters:
                    loss_meters[key].update(val.item() if isinstance(val, torch.Tensor) else val)

        # Return average losses
        avg_losses = {k: v.avg for k, v in loss_meters.items()}
        return avg_losses

    def extract_features(self, data_loader, use_id_text=True, id_only=False):
        """
        Extract feature vectors from data loader
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
        Compute distance matrix between query and gallery features
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
        Compute evaluation metrics
        """
        distmat = to_numpy(distmat)
        query_ids = np.array([items[3] for items in query])
        gallery_ids = np.array([items[3] for items in gallery])
        cmc_scores, mAP = self.eval_func(distmat, q_pids=query_ids, g_pids=gallery_ids)
        
        return {
            f'{prefix}mAP': mAP,
            f'{prefix}rank1': cmc_scores[0],
            f'{prefix}rank5': cmc_scores[4],
            f'{prefix}rank10': cmc_scores[9]
        }

    @staticmethod
    def eval_func(distmat, q_pids, g_pids, max_rank=10):
        """
        Core function for computing CMC and mAP metrics
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
    
