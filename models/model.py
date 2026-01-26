# models/model.py
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel, ViTModel
from safetensors.torch import load_file
from utils.serialization import copy_state_dict
from .fusion import get_fusion_module, ScagRcsmFusion
from .ahnet_module import AHNetModule  # 使用新的 AH-Net 模块
from .semantic_guidance import SemanticGuidedDecoupling
from .vim import VisionMamba

# 尝试导入 Mamba
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    Mamba = None
    HAS_MAMBA = False

# 设置transformers库日志级别
import logging as _logging
import warnings
_logging.getLogger("transformers").setLevel(_logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=(), mid_cls=True, logger=None):
    """
    Rescale the grid of position embeddings when loading from state_dict. Adapted from DEIT/Vim.
    """
    ntok_new = posemb_new.shape[1]
    
    if num_tokens:
        if mid_cls:
            old_cls_idx = posemb.shape[1] // 2
            posemb_tok = posemb[:, old_cls_idx:old_cls_idx+num_tokens]
            posemb_grid = torch.cat([posemb[:, :old_cls_idx], posemb[:, old_cls_idx+num_tokens:]], dim=1)
        else:
            posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[:, num_tokens:]
            
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb
    
    gs_old = int(math.sqrt(len(posemb_grid[0])))
    
    if ntok_new != len(posemb_grid[0]):
        posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
        posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).flatten(1, 2)
        
        if mid_cls:
            new_cls_idx = posemb_grid.shape[1] // 2
            posemb = torch.cat([
                posemb_grid[:, :new_cls_idx], 
                posemb_tok, 
                posemb_grid[:, new_cls_idx:]
            ], dim=1)
        else:
            posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
            
    return posemb


class ResidualBottleneck1d(nn.Module):
    """
    Stage 1: 局部残差瓶颈模块
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

    def forward(self, x):
        identity = x
        out = self.squeeze(x)
        out = self.process(out)
        out = self.expand(out)
        return identity + out

class GatedMambaBlock(nn.Module):
    """
    Stage 2: 门控长程 Mamba 模块
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        if Mamba is None:
            raise ImportError("Mamba module not found. Please install mamba-ssm.")
            
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate_fc = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        mamba_out = self.mamba(x_norm)
        gate = torch.sigmoid(self.gate_fc(x_norm))
        out = mamba_out * gate
        out = self.dropout(out)
        return residual + out

class PyramidTextEncoder(nn.Module):
    """
    串行渐进式金字塔文本编码器
    """
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim
        self.stage1_bottleneck = ResidualBottleneck1d(dim, reduction=4)
        self.attr_norm = nn.LayerNorm(dim)
        self.stage2_mamba = GatedMambaBlock(dim=dim)
        self.id_norm = nn.LayerNorm(dim)
        self.id_bn = nn.BatchNorm1d(dim)
        nn.init.constant_(self.id_bn.weight, 1.0)
        nn.init.constant_(self.id_bn.bias, 0.0)
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
        B, L, D = x.shape
        x_raw = x
        x_cnn_in = x.transpose(1, 2)
        x_cnn_out = self.stage1_bottleneck(x_cnn_in)
        x_stage1 = x_cnn_out.transpose(1, 2)
        
        feat_attr = self.attr_norm(x_stage1)
        feat_attr = feat_attr.transpose(1, 2)
        feat_attr = F.adaptive_max_pool1d(feat_attr, 1).squeeze(2)
        
        x_mamba_out = self.stage2_mamba(x_stage1)
        x_final = x_mamba_out + x_raw
        
        x_final_norm = self.id_norm(x_final)
        x_final_norm = x_final_norm.transpose(1, 2)
        feat_id = F.adaptive_avg_pool1d(x_final_norm, 1).squeeze(2)
        feat_id_bn = self.id_bn(feat_id)
        
        return {
            'feat_attr': feat_attr,
            'feat_id': feat_id,
            'feat_id_bn': feat_id_bn
        }


class Model(nn.Module):
    def __init__(self, net_config, logger=None):
        super().__init__()
        self.net_config = net_config
        self.logger = logger
        
        clip_base_path = Path(net_config.get('clip_pretrained', 'pretrained/openai/clip-vit-base-patch16'))
        vit_base_path = Path(net_config.get('vit_pretrained', 'pretrained/vit-base-patch16-224'))
        fusion_config = net_config.get('fusion', {})
        
        # 1. Text Encoder (CLIP)
        if not clip_base_path.exists():
             fallback = list(Path("pretrained").glob("**/clip-vit-base-patch16"))
             if fallback: clip_base_path = fallback[0]
        
        self.tokenizer = CLIPTokenizer.from_pretrained(str(clip_base_path))
        self.text_encoder = CLIPTextModel.from_pretrained(str(clip_base_path))
        
        safetensors_path = clip_base_path / "model.safetensors"
        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path))
            new_state_dict = {k[len("text_model."):]: v for k, v in state_dict.items() if k.startswith("text_model.")}
            self.text_encoder.load_state_dict(new_state_dict, strict=False)
        
        self.clip_dim = self.text_encoder.config.hidden_size
        self.text_width = 768
        
        if self.clip_dim != self.text_width:
            self.text_proj = nn.Sequential(
                nn.Linear(self.clip_dim, self.text_width),
                nn.LayerNorm(self.text_width),
                nn.GELU()
            )
        else:
            self.text_proj = nn.Identity()

        # 2. Visual Encoder (Vim / ViT)
        self.vision_backbone_type = net_config.get('vision_backbone', 'vit')
        self.img_size = net_config.get('img_size', (224, 224))
        if isinstance(self.img_size, int): self.img_size = (self.img_size, self.img_size)
        
        if self.vision_backbone_type == 'vim':
            vim_pretrained_path = net_config.get('vim_pretrained', 'pretrained/Vision Mamba/vim_s_midclstok.pth')
            self.visual_encoder = VisionMamba(img_size=self.img_size, patch_size=16, embed_dim=384, depth=24)
            if Path(vim_pretrained_path).exists():
                checkpoint = torch.load(vim_pretrained_path, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('model', checkpoint)
                if 'pos_embed' in state_dict:
                    grid_h = self.img_size[0] // 16
                    grid_w = self.img_size[1] // 16
                    state_dict['pos_embed'] = resize_pos_embed(
                        state_dict['pos_embed'], self.visual_encoder.pos_embed, 
                        num_tokens=1, gs_new=(grid_h, grid_w), mid_cls=True
                    )
                self.visual_encoder.load_state_dict(state_dict, strict=False)
            self.visual_proj = nn.Linear(384, self.text_width)
        else:
            self.visual_encoder = ViTModel.from_pretrained(str(vit_base_path), weights_only=False)
            self.visual_proj = nn.Identity()

        # 3. Disentangle Module (AH-Net)
        # 强制使用 AH-Net
        gs3_config = net_config.get('gs3', {})
        self.disentangle = AHNetModule(
            dim=self.text_width,
            img_size=self.img_size,
            patch_size=16,
            d_state=gs3_config.get('d_state', 16),
            d_conv=gs3_config.get('d_conv', 4),
            logger=self.logger
        )
        if self.logger:
            self.logger.debug_logger.info("🔥 Using AH-Net Module (Spatial-Structure Dual Stream)")

        # 4. Semantic Guidance
        self.semantic_guidance = SemanticGuidedDecoupling(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            dim=self.text_width,
            logger=self.logger
        )

        # 5. Retrieval Projection Heads
        self.shared_mlp = nn.Linear(self.text_width, 512)
        self.image_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )
        self.text_mlp = nn.Sequential(
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256)
        )

        # 6. Text Encoder V2 (Pyramid Text Encoder)
        if HAS_MAMBA:
            self.text_encoder_v2 = PyramidTextEncoder(dim=self.text_width)
        else:
            raise ImportError("Mamba module is required. Please install mamba-ssm.")

        # 7. Fusion
        self.fusion = get_fusion_module(fusion_config) if fusion_config else None
        self.scale = nn.Parameter(torch.ones(1), requires_grad=True)
        self.text_cache = {}

    def _seq_to_grid(self, seq_feat):
        """
        Helper: Convert Sequence [B, L, D] to Grid [B, D, H, W]
        Handling CLS removal.
        """
        B, L, D = seq_feat.shape
        
        # Calculate Grid Size
        patch_size = 16
        H, W = self.img_size
        h_grid, w_grid = H // patch_size, W // patch_size
        
        expected_len = h_grid * w_grid
        
        # Determine CLS position and Remove
        if L == expected_len + 1:
            # Assume 1 CLS token
            if self.vision_backbone_type == 'vim':
                # Mid-CLS
                cls_idx = L // 2
                seq_patches = torch.cat([seq_feat[:, :cls_idx], seq_feat[:, cls_idx+1:]], dim=1)
            else:
                # ViT: First-CLS
                seq_patches = seq_feat[:, 1:]
        elif L == expected_len:
            seq_patches = seq_feat
        else:
            raise ValueError(f"Sequence length {L} does not match grid {h_grid}x{w_grid} ({expected_len})")
            
        # Reshape to Grid: [B, H*W, D] -> [B, D, H, W]
        grid_feat = seq_patches.transpose(1, 2).reshape(B, D, h_grid, w_grid)
        return grid_feat

    def encode_image(self, image):
        if image is None: return None
        device = next(self.parameters()).device
        if image.dim() == 5: image = image.squeeze(-1)
        image = image.to(device)
        
        # Visual Encoder
        if self.vision_backbone_type == 'vim':
            with torch.amp.autocast('cuda', enabled=False):
                image_embeds_raw = self.visual_encoder(image.float())
            image_embeds_raw = image_embeds_raw.to(image.dtype)
        else:
            image_embeds_raw = self.visual_encoder(image).last_hidden_state
        
        image_embeds_raw = self.visual_proj(image_embeds_raw) # [B, L, D]
        
        # Seq -> Grid for AH-Net
        image_grid = self._seq_to_grid(image_embeds_raw) # [B, D, H, W]
        
        # AH-Net Disentangle
        id_embeds, _, _ = self.disentangle(image_grid)
        
        # Projection
        image_embeds = self.shared_mlp(id_embeds)
        image_embeds = self.image_mlp(image_embeds)
        image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1, eps=1e-8)
        return image_embeds

    def encode_text(self, instruction):
        """Encode text instruction to embedding."""
        if instruction is None: return None
        device = next(self.parameters()).device
        if isinstance(instruction, (list, tuple)): texts = list(instruction)
        else: texts = [instruction]

        cache_key = tuple(texts)
        if cache_key in self.text_cache: tokenized = self.text_cache[cache_key]
        else:
            tokenized = self.tokenizer(texts, padding='max_length', max_length=77, truncation=True, return_tensors="pt", return_attention_mask=True)
            self.text_cache[cache_key] = tokenized

        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        text_embeds = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        text_embeds = self.text_proj(text_embeds)

        if self.text_encoder_v2:
            out_v2 = self.text_encoder_v2(text_embeds)
            text_embeds = out_v2['feat_id']
        else:
            text_embeds = text_embeds.mean(dim=1) 

        text_embeds = self.shared_mlp(text_embeds)
        text_embeds = self.text_mlp(text_embeds)
        text_embeds = torch.nn.functional.normalize(text_embeds, dim=-1, eps=1e-8)
        if not isinstance(instruction, list): text_embeds = text_embeds.squeeze(0)
        return text_embeds

    def forward(self, image=None, cloth_instruction=None, id_instruction=None, return_attention=False):
        device = next(self.parameters()).device
        id_logits, id_embeds, cloth_embeds = None, None, None
        aux_info = None
        
        # === Image Branch ===
        if image is not None:
            if image.dim() == 5: image = image.squeeze(-1)
            image = image.to(device)
            
            # Encoder
            if self.vision_backbone_type == 'vim':
                with torch.amp.autocast('cuda', enabled=False):
                    image_embeds_raw = self.visual_encoder(image.float())
            else:
                image_embeds_raw = self.visual_encoder(image).last_hidden_state
            
            # Proj & Reshape
            image_embeds_raw = self.visual_proj(image_embeds_raw)
            image_grid = self._seq_to_grid(image_embeds_raw) # [B, D, H, W]
            
            # AH-Net Disentangle
            id_embeds, cloth_embeds, aux_info = self.disentangle(image_grid)
            
            # Projection
            image_embeds = self.shared_mlp(id_embeds)
            image_embeds = self.image_mlp(image_embeds)
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1, eps=1e-8)
            
            cloth_image_embeds = self.shared_mlp(cloth_embeds)
            cloth_image_embeds = self.image_mlp(cloth_image_embeds)
            cloth_image_embeds = torch.nn.functional.normalize(cloth_image_embeds, dim=-1, eps=1e-8)
        else:
            image_embeds, cloth_image_embeds = None, None

        # === Text Branch ===
        main_instruction = id_instruction if id_instruction else cloth_instruction
        feat_attr_raw, feat_id_raw = None, None
        
        if self.text_encoder_v2 and main_instruction:
            if isinstance(main_instruction, (list, tuple)): texts = list(main_instruction)
            else: texts = [main_instruction]
            
            cache_key = tuple(texts)
            if cache_key in self.text_cache: tokenized = self.text_cache[cache_key]
            else:
                tokenized = self.tokenizer(texts, padding='max_length', max_length=77, truncation=True, return_tensors="pt", return_attention_mask=True)
                self.text_cache[cache_key] = tokenized
            
            input_ids = tokenized['input_ids'].to(device)
            attention_mask = tokenized['attention_mask'].to(device)
            
            text_seq = self.text_encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            text_seq = self.text_proj(text_seq)
            
            out_v2 = self.text_encoder_v2(text_seq)
            feat_attr_raw, feat_id_raw = out_v2['feat_attr'], out_v2['feat_id']
            
            cloth_text_embeds = self.shared_mlp(feat_attr_raw)
            cloth_text_embeds = self.text_mlp(cloth_text_embeds)
            cloth_text_embeds = torch.nn.functional.normalize(cloth_text_embeds, dim=-1, eps=1e-8)
            
            id_text_embeds = self.shared_mlp(feat_id_raw)
            id_text_embeds = self.text_mlp(id_text_embeds)
            id_text_embeds = torch.nn.functional.normalize(id_text_embeds, dim=-1, eps=1e-8)
        else:
            cloth_text_embeds = self.encode_text(cloth_instruction)
            id_text_embeds = self.encode_text(id_instruction)

        # === Fusion ===
        fused_embeds, gate_weights = None, None
        if self.fusion and image_embeds is not None and id_text_embeds is not None:
            if isinstance(self.fusion, ScagRcsmFusion) and feat_id_raw is not None:
                # 🔥 方案书核心实现：使用 conflict_score 替代 energy_ratio
                # conflict_score 来自 aux_info，由 AH-Net Module 计算得出
                conflict_score = aux_info.get('conflict_score',
                                               torch.zeros(id_embeds.size(0), device=device))

                fused_embeds, gate_weights = self.fusion(
                    img_id=id_embeds, img_attr=cloth_embeds,
                    txt_id=feat_id_raw, txt_attr=feat_attr_raw,
                    conflict_score=conflict_score  # ← 传递冲突分数而非能量比
                )
            else:
                fused_embeds, gate_weights = self.fusion(image_embeds, id_text_embeds)
            fused_embeds = self.scale * torch.nn.functional.normalize(fused_embeds, dim=-1, eps=1e-8)
        elif image_embeds is not None:
            fused_embeds = image_embeds

        # Returns
        # aux_info (9th element) 包含 conflict_score, attention maps, reconstruction features
        # gate_weights (10th element) 包含 confidence weights from S-CAG
        base_outputs = (image_embeds, id_text_embeds, fused_embeds, id_logits, id_embeds,
                       cloth_embeds, cloth_text_embeds, cloth_image_embeds, aux_info, gate_weights,
                       None, None)

        if return_attention and aux_info:
            return base_outputs + (aux_info.get('map_id'), aux_info.get('map_attr'))
        else:
            return base_outputs

    def load_param(self, trained_path):
        trained_path = Path(trained_path)
        checkpoint = torch.load(trained_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
        copy_state_dict(state_dict, self)
        if self.logger:
            self.logger.debug_logger.info(f"Loaded checkpoint from {trained_path}")
        return self