# src/models/model.py
import logging
from pathlib import Path
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, ViTModel
from utils.serialization import copy_state_dict
from .fusion import get_fusion_module
from .gs3_module import GS3Module

# 设置transformers库的日志级别为ERROR，减少不必要的日志输出
logging.getLogger("transformers").setLevel(logging.ERROR)


class DisentangleModule(nn.Module):
    """
    特征分离模块的基类（保留用于向后兼容）
    实际使用中建议直接使用 GS3Module
    """
    def __init__(self, dim):
        """
        简化的特征分离模块（消融实验版本），移除复杂的注意力机制。

        Args:
            dim (int): 输入特征的维度（每个 token 的维度）。
        """
        super().__init__()
        # 简化为基本的线性变换
        self.id_linear = nn.Linear(dim, dim)
        self.cloth_linear = nn.Linear(dim, dim)
        
        # 保留门控机制以维持接口一致性
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, x, return_attention=False):
        """
        简化的前向传播，移除注意力机制，使用基本的线性变换和池化。

        Args:
            x (torch.Tensor): 输入特征，形状为 [batch_size, seq_len, dim]。
            return_attention (bool): 是否返回注意力权重（用于可视化）。

        Returns:
            tuple: 如果 return_attention=False，返回 (id_feat, cloth_feat, gate)
                  如果 return_attention=True，返回 (id_feat, cloth_feat, gate, id_attn_map, cloth_attn_map)
        """
        batch_size, seq_len, dim = x.size()

        # 简化的身份特征处理：线性变换 + 全局平均池化
        id_feat = self.id_linear(x)  # [batch_size, seq_len, dim]
        
        # 计算身份分支的注意力权重（基于特征幅值）
        id_attn_map = None
        if return_attention:
            # 使用 L2 范数作为注意力权重的近似
            id_attn_map = torch.norm(id_feat, p=2, dim=-1)  # [batch_size, seq_len]
            id_attn_map = torch.softmax(id_attn_map, dim=-1)  # 归一化
        
        id_feat = id_feat.mean(dim=1)  # [batch_size, dim]

        # 简化的服装特征处理：线性变换 + 全局平均池化  
        cloth_feat = self.cloth_linear(x)  # [batch_size, seq_len, dim]
        
        # 计算服装分支的注意力权重（基于特征幅值）
        cloth_attn_map = None
        if return_attention:
            cloth_attn_map = torch.norm(cloth_feat, p=2, dim=-1)  # [batch_size, seq_len]
            cloth_attn_map = torch.softmax(cloth_attn_map, dim=-1)  # 归一化
        
        cloth_feat = cloth_feat.mean(dim=1)  # [batch_size, dim]

        # 保持门控机制以维持原有接口
        gate = self.gate(torch.cat([id_feat, cloth_feat], dim=-1))  # [batch_size, dim]
        id_feat = gate * id_feat
        cloth_feat = (1 - gate) * cloth_feat
        
        if return_attention:
            return id_feat, cloth_feat, gate, id_attn_map, cloth_attn_map
        else:
            return id_feat, cloth_feat, gate


class Model(nn.Module):
    def __init__(self, net_config):
        """
        文本-图像行人重识别模型（消融实验版本），移除了复杂的解纠缠模块。

        Args:
            net_config (dict): 模型配置字典，包含BERT路径、ViT路径、融合模块配置等。
        """
        super().__init__()
        self.net_config = net_config
        bert_base_path = Path(net_config.get('bert_base_path', 'pretrained/bert-base-uncased'))
        vit_base_path = Path(net_config.get('vit_pretrained', 'pretrained/vit-base-patch16-224'))
        fusion_config = net_config.get('fusion', {})
        num_classes = net_config.get('num_classes', 8000)

        # 验证预训练模型路径
        if not bert_base_path.exists() or not vit_base_path.exists():
            raise FileNotFoundError(f"Model path not found: {bert_base_path} or {vit_base_path}")

        # 初始化文本编码器和分词器
        self.tokenizer = BertTokenizer.from_pretrained(str(bert_base_path), do_lower_case=True, use_fast=True)
        self.text_encoder = BertModel.from_pretrained(str(bert_base_path), weights_only=False)
        self.text_width = self.text_encoder.config.hidden_size  # BERT隐藏层维度（768）

        # 初始化图像编码器
        self.visual_encoder = ViTModel.from_pretrained(str(vit_base_path), weights_only=False)

        # 初始化 G-S3 特征分离模块（魔改版本）
        # 可通过配置选择使用简化版本(DisentangleModule)或G-S3版本
        disentangle_type = net_config.get('disentangle_type', 'gs3')
        if disentangle_type == 'gs3':
            # 使用增强的 G-S3 模块
            gs3_config = net_config.get('gs3', {})
            self.disentangle = GS3Module(
                dim=self.text_width,
                num_heads=gs3_config.get('num_heads', 8),
                d_state=gs3_config.get('d_state', 16),
                d_conv=gs3_config.get('d_conv', 4),
                dropout=gs3_config.get('dropout', 0.1)
            )
            logging.info("Using G-S3 (Geometry-Guided Selective State Space) disentangle module")
        else:
            # 使用简化版本（消融实验）
            self.disentangle = DisentangleModule(dim=self.text_width)
            logging.info("Using simplified disentangle module")

        # 初始化身份分类器
        self.id_classifier = nn.Linear(self.text_width, num_classes)

        # 初始化共享MLP
        self.shared_mlp = nn.Linear(self.text_width, 512)

        # 初始化图像特征MLP
        self.image_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )

        # 初始化文本特征MLP
        self.text_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )

        # 修改为 3 层文本自注意力模块
        self.text_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=self.text_width, num_heads=4, dropout=0.1) for _ in range(3)
        ])
        self.text_attn_norm_layers = nn.ModuleList([
            nn.LayerNorm(self.text_width) for _ in range(3)
        ])

        # 初始化融合模块
        self.fusion = get_fusion_module(fusion_config) if fusion_config else None
        self.feat_dim = fusion_config.get("output_dim", 256) if fusion_config else 256

        # 初始化可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)

        # 日志记录初始化信息
        logging.info(
            f"Initialized model with scale: {self.scale.item():.4f}, fusion: {fusion_config.get('type', 'None')}")

        # 文本分词结果缓存
        self.text_cache = {}

    def encode_image(self, image):
        """
        编码图像，提取图像特征并进行标准化，使用 ViT 整个序列。

        Args:
            image (torch.Tensor): 输入图像，形状为 [batch_size, channels, height, width] 或更高维。

        Returns:
            torch.Tensor: 标准化后的图像嵌入，形状为 [batch_size, 256]。
        """
        if image is None:
            return None
        device = next(self.parameters()).device
        if image.dim() == 5:
            image = image.squeeze(-1)
        image = image.to(device)
        image_outputs = self.visual_encoder(image)
        image_embeds = image_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        id_embeds, _, _ = self.disentangle(image_embeds)  # [batch_size, hidden_size]
        image_embeds = self.shared_mlp(id_embeds)
        image_embeds = self.image_mlp(image_embeds)
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
        return image_embeds

    def encode_text(self, instruction):
        """
        编码文本，提取文本特征并进行标准化，使用所有 token 进行全局建模。

        Args:
            instruction (str or list): 输入文本，单个字符串或字符串列表。

        Returns:
            torch.Tensor: 标准化后的文本嵌入，形状为 [batch_size, 256] 或 [256]（单文本）。
        """
        if instruction is None:
            return None
        device = next(self.parameters()).device
        if isinstance(instruction, list):
            texts = instruction
        else:
            texts = [instruction]

        # 检查缓存以复用分词结果
        cache_key = tuple(texts)
        if cache_key in self.text_cache:
            tokenized = self.text_cache[cache_key]
        else:
            tokenized = self.tokenizer(
                texts,
                padding='max_length',
                max_length=64,  # 适合CUHK-PEDES数据集的文本长度
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True
            )
            self.text_cache[cache_key] = tokenized

        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        # BERT编码，禁用梯度以提升效率
        with torch.no_grad():
            text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]

        # 3 层自注意力处理
        text_embeds = text_embeds.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        for attn, norm in zip(self.text_attn_layers, self.text_attn_norm_layers):
            attn_output, _ = attn(
                query=text_embeds,
                key=text_embeds,
                value=text_embeds,
                key_padding_mask=~attention_mask.bool()  # 忽略填充 token
            )
            text_embeds = attn_output + text_embeds  # 残差连接
            text_embeds = norm(text_embeds)
        text_embeds = text_embeds.transpose(0, 1)  # [batch_size, seq_len, hidden_size]

        # 均值池化，结合 attention_mask 忽略填充 token
        attention_mask = attention_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
        text_embeds = torch.sum(text_embeds * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        # 形状: [batch_size, hidden_size]

        # 降维和标准化
        text_embeds = self.shared_mlp(text_embeds)
        text_embeds = self.text_mlp(text_embeds)
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1)

        if not isinstance(instruction, list):
            text_embeds = text_embeds.squeeze(0)
        return text_embeds

    def forward(self, image=None, cloth_instruction=None, id_instruction=None, return_attention=False):
        """
        前向传播，处理图像和文本输入，输出多模态特征和分类结果。

        Args:
            image (torch.Tensor, optional): 输入图像，形状为 [batch_size, channels, height, width]。
            cloth_instruction (str or list, optional): 服装描述文本。
            id_instruction (str or list, optional): 身份描述文本。
            return_attention (bool): 是否返回注意力图（用于可视化）。

        Returns:
            tuple: (image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                    cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate, gate_weights)
                   或包含注意力图的扩展元组
        """
        device = next(self.parameters()).device
        id_logits, id_embeds, cloth_embeds, gate = None, None, None, None
        id_attn_map, cloth_attn_map = None, None
        
        if image is not None:
            if image.dim() == 5:
                image = image.squeeze(-1)
            image = image.to(device)
            image_outputs = self.visual_encoder(image)
            image_embeds = image_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            # 调用解耦模块，支持返回注意力图
            if return_attention:
                id_embeds, cloth_embeds, gate, id_attn_map, cloth_attn_map = self.disentangle(image_embeds, return_attention=True)
            else:
                id_embeds, cloth_embeds, gate = self.disentangle(image_embeds, return_attention=False)
            
            # 存储中间特征用于调试（仅在 debug 模式下）
            if hasattr(self, '_debug_mode') and self._debug_mode:
                self._debug_features = {
                    'image_embeds_raw': image_embeds,
                    'id_embeds': id_embeds,
                    'cloth_embeds': cloth_embeds,
                    'gate': gate
                }
                # 如果使用 G-S3 模块，也记录其内部状态
                if hasattr(self.disentangle, '_debug_info'):
                    self._debug_features.update(self.disentangle._debug_info)
            
            id_logits = self.id_classifier(id_embeds)
            image_embeds = self.shared_mlp(id_embeds)
            image_embeds = self.image_mlp(image_embeds)
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)
            cloth_image_embeds = self.shared_mlp(cloth_embeds)
            cloth_image_embeds = self.image_mlp(cloth_image_embeds)
            cloth_image_embeds = torch.nn.functional.normalize(cloth_image_embeds, dim=-1)
        else:
            image_embeds = None
            cloth_image_embeds = None
        
        cloth_text_embeds = self.encode_text(cloth_instruction)
        id_text_embeds = self.encode_text(id_instruction)
        fused_embeds, gate_weights = None, None
        if self.fusion and image_embeds is not None and id_text_embeds is not None:
            fused_embeds, gate_weights = self.fusion(image_embeds, id_text_embeds)
            fused_embeds = self.scale * torch.nn.functional.normalize(fused_embeds, dim=-1)
        
        base_outputs = (image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                       cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate, gate_weights)
        
        if return_attention:
            return base_outputs + (id_attn_map, cloth_attn_map)
        else:
            return base_outputs

    def load_param(self, trained_path):
        """
        加载预训练模型参数。

        Args:
            trained_path (str): 预训练模型文件路径。

        Returns:
            T2IReIDModel: 加载参数后的模型。
        """
        trained_path = Path(trained_path)
        checkpoint = torch.load(trained_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        self = copy_state_dict(state_dict, self)
        logging.info(f"Loaded checkpoint from {trained_path}, scale: {self.scale.item():.4f}")
        return self