# evaluators/evaluator.py
"""
T2I-ReID Evaluator
用于评估模型在测试集上的表现
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score


class Evaluator:
    """
    T2I-ReID 评估器
    """
    def __init__(self, model, args=None):
        """
        初始化评估器
        
        Args:
            model: 待评估的模型
            args: 配置参数
        """
        self.model = model
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate(self, query_loader, gallery_loader, query_data, gallery_data, 
                 checkpoint_path=None, epoch=None):
        """
        执行评估 - Text-to-Image ReID
        
        🔥 修复：正确的Text-to-Image ReID评估
        - Query: 使用文本特征 (encode_text)
        - Gallery: 使用图像特征 (encode_image)
        
        Args:
            query_loader: Query数据加载器（文本）
            gallery_loader: Gallery数据加载器（图像）
            query_data: Query数据集
            gallery_data: Gallery数据集
            checkpoint_path: 检查点路径（可选）
            epoch: 当前epoch（可选）
        
        Returns:
            dict: 包含mAP, rank1, rank5, rank10的字典
        """
        self.model.eval()
        
        with torch.no_grad():
            # 🔥 修复：Query使用文本特征
            query_features = []
            query_pids = []
            query_camids = []
            
            for batch in tqdm(query_loader, desc="Extracting query text features"):
                images, _, captions, pids, cam_ids, _ = batch
                
                # 🔥 使用文本编码器而不是图像编码器
                text_embeds = self.model.encode_text(captions)
                
                query_features.append(text_embeds.cpu())
                query_pids.append(pids)
                query_camids.append(cam_ids)
            
            query_features = torch.cat(query_features, dim=0)
            query_pids = torch.cat(query_pids, dim=0).numpy()
            query_camids = torch.cat(query_camids, dim=0).numpy()
            
            # Gallery使用图像特征（这部分是正确的）
            gallery_features = []
            gallery_pids = []
            gallery_camids = []
            
            for batch in tqdm(gallery_loader, desc="Extracting gallery image features"):
                images, _, captions, pids, cam_ids, _ = batch
                images = images.to(self.device)
                
                # 使用图像编码器
                image_embeds = self.model.encode_image(images)
                
                gallery_features.append(image_embeds.cpu())
                gallery_pids.append(pids)
                gallery_camids.append(cam_ids)
            
            gallery_features = torch.cat(gallery_features, dim=0)
            gallery_pids = torch.cat(gallery_pids, dim=0).numpy()
            gallery_camids = torch.cat(gallery_camids, dim=0).numpy()
        
        # 计算相似度矩阵（文本 x 图像）
        query_features = query_features / query_features.norm(dim=1, keepdim=True)
        gallery_features = gallery_features / gallery_features.norm(dim=1, keepdim=True)
        
        similarity_matrix = torch.mm(query_features, gallery_features.t()).numpy()
        
        # 计算指标
        cmc, mAP = self.compute_metrics(
            similarity_matrix,
            query_pids,
            gallery_pids,
            query_camids,
            gallery_camids
        )
        
        metrics = {
            'mAP': mAP,
            'rank1': cmc[0],
            'rank5': cmc[4] if len(cmc) > 4 else cmc[-1],
            'rank10': cmc[9] if len(cmc) > 9 else cmc[-1]
        }
        
        return metrics
    
    def compute_metrics(self, similarity_matrix, query_pids, gallery_pids, 
                       query_camids, gallery_camids):
        """
        计算CMC和mAP指标
        
        Args:
            similarity_matrix: 相似度矩阵 [num_query, num_gallery]
            query_pids: Query的person IDs
            gallery_pids: Gallery的person IDs
            query_camids: Query的camera IDs
            gallery_camids: Gallery的camera IDs
        
        Returns:
            tuple: (cmc, mAP)
        """
        num_query = similarity_matrix.shape[0]
        
        # 存储所有query的AP值
        all_AP = []
        all_cmc = []
        
        for i in range(num_query):
            # 获取当前query
            query_pid = query_pids[i]
            query_camid = query_camids[i]
            
            # 获取相似度分数（降序排列的索引）
            scores = similarity_matrix[i]
            indices = np.argsort(-scores)
            
            # 获取匹配情况
            matches = (gallery_pids[indices] == query_pid)
            
            # === 🔥 修复：优化同摄像头过滤逻辑 ===
            # 如果所有camera_id都相同（如全为0），则不进行camera过滤
            unique_cameras = np.unique(np.concatenate([query_camids, gallery_camids]))
            if len(unique_cameras) > 1:
                # 多个camera，正常过滤同camera的正样本
                same_camera = (gallery_camids[indices] == query_camid)
                valid = ~(matches & same_camera)  # 移除同camera的正样本
                matches = matches[valid]
            # 否则不过滤（所有样本都是同一个camera）
            
            if not np.any(matches):
                continue
            
            # 计算CMC
            cmc = matches.cumsum()
            cmc[cmc > 1] = 1
            all_cmc.append(cmc)
            
            # 计算AP
            num_rel = matches.sum()
            tmp_cmc = matches.cumsum()
            tmp_cmc = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
            tmp_cmc = tmp_cmc * matches
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)
        
        # === 🔥 修复：处理空CMC列表 ===
        if len(all_cmc) == 0 or len(all_AP) == 0:
            # 如果没有有效的query-gallery匹配，返回0指标
            print(f"⚠️  Warning: No valid query-gallery matches found!")
            print(f"   Query samples: {num_query}")
            print(f"   Valid matches: 0")
            # 返回全0的CMC和mAP
            return np.zeros(100), 0.0
        
        # 平均CMC
        max_len = max([len(cmc) for cmc in all_cmc])
        for i in range(len(all_cmc)):
            if len(all_cmc[i]) < max_len:
                # 填充最后一个值
                all_cmc[i] = np.concatenate([
                    all_cmc[i],
                    np.ones(max_len - len(all_cmc[i])) * all_cmc[i][-1]
                ])
        
        all_cmc = np.array(all_cmc).astype(float)
        all_cmc = all_cmc.sum(axis=0) / len(all_cmc)
        
        # 计算mAP
        mAP = np.mean(all_AP)
        
        return all_cmc, mAP
