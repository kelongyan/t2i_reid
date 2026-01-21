import argparse
import ast
import gc
import logging
import random
import sys
from pathlib import Path
import torch
from torch.backends import cudnn
from torch.cuda.amp import GradScaler

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.serialization import save_checkpoint
from models.model import Model
from datasets.data_builder import DataBuilder
from trainers.trainer import Trainer
from utils.lr_scheduler import WarmupMultiStepLR
from utils.monitor import get_monitor_for_dataset

def configuration():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train T2I-ReID model (CLIP Upgrade)")
    parser.add_argument('--root', type=str, default=str(ROOT_DIR / 'datasets'),
                       help='Root directory of the dataset')
    parser.add_argument('--dataset-configs', nargs='+', type=str, help='List of dataset configurations in JSON format')
    parser.add_argument('--loss-weights', type=str, help='Loss weights in JSON format')
    parser.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('-j', '--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--warmup-step', type=int, default=500, help='Warmup steps')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 60], help='Milestones for LR scheduler')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    
    # [Modify] Pretrained Model Paths
    # BERT path kept for compatibility if needed, but CLIP is default now
    parser.add_argument('--bert-base-path', type=str, default=str(ROOT_DIR / 'pretrained' / 'bert-base-uncased'),
                       help='Path to BERT model (Legacy)')
    
    # [New] CLIP Path
    parser.add_argument('--clip-pretrained', type=str, 
                       default=str(ROOT_DIR / 'pretrained' / 'clip-vit-base-patch16'),
                       help='Path to CLIP text encoder model')
    
    parser.add_argument('--vit-pretrained', type=str, default=str(ROOT_DIR / 'pretrained' / 'vit-base-patch16-224'),
                       help='Path to ViT model')
    parser.add_argument('--vision-backbone', type=str, default='vim', choices=['vit', 'vim'],
                       help='Vision backbone type: vit or vim')
    parser.add_argument('--vim-pretrained', type=str, default=str(ROOT_DIR / 'pretrained' / 'Vision Mamba' / 'vim_s_midclstok.pth'),
                       help='Path to Vision Mamba model')
    parser.add_argument('--logs-dir', type=str, default=str(ROOT_DIR / 'log'), help='Directory for logs')
    parser.add_argument('--num-classes', type=int, default=8000, help='Number of identity classes')

    # Fusion module parameters
    parser.add_argument('--fusion-type', type=str, default='enhanced_mamba', help='Type of fusion module')
    parser.add_argument('--fusion-dim', type=int, default=256, help='Fusion module dimension')
    parser.add_argument('--fusion-d-state', type=int, default=16, help='Fusion module d_state')
    parser.add_argument('--fusion-d-conv', type=int, default=4, help='Fusion module d_conv')
    parser.add_argument('--fusion-num-layers', type=int, default=2, help='Fusion module number of layers')
    parser.add_argument('--fusion-output-dim', type=int, default=256, help='Fusion module output dimension')
    parser.add_argument('--fusion-dropout', type=float, default=0.1, help='Fusion module dropout')

    # Disentangle module parameters
    parser.add_argument('--id-projection-dim', type=int, default=768, help='ID projection dimension')
    parser.add_argument('--cloth-projection-dim', type=int, default=768, help='Cloth projection dimension')
    
    # G-S3/FSHD module parameters
    # [Modify] Fixed to 'fshd' as default, removed 'gs3' option
    parser.add_argument('--disentangle-type', type=str, default='fshd', 
                       choices=['fshd', 'simple'],
                       help='Type of disentangle module: fshd (FSHD-Net) or simple (DisentangleModule)')
    
    parser.add_argument('--gs3-num-heads', type=int, default=8, 
                       help='Number of attention heads in G-S3 OPA')
    parser.add_argument('--gs3-d-state', type=int, default=16, 
                       help='State dimension for G-S3 Mamba filter')
    parser.add_argument('--gs3-d-conv', type=int, default=4, 
                       help='Convolution kernel size for G-S3 Mamba filter')
    parser.add_argument('--gs3-dropout', type=float, default=0.1, 
                       help='Dropout rate for G-S3 module')
    
    # [New] FSHD specific parameters
    # [Modify] Removed freq-type choice (fixed to dct)
    parser.add_argument('--gs3-use-multi-scale-cnn', type=str, default='true',
                       help='Whether to use multi-scale CNN in FSHD (true/false)')
    parser.add_argument('--gs3-img-size', nargs=2, type=int, default=[14, 14],
                       help='Image patch grid size (h, w) for FSHD frequency splitting')

    # Loss weights
    parser.add_argument('--loss-info-nce', type=float, default=1.0, help='InfoNCE loss weight')
    parser.add_argument('--loss-cls', type=float, default=0.5, help='Classification loss weight')
    parser.add_argument('--loss-cloth-semantic', type=float, default=2.0, help='Cloth semantic loss weight')
    parser.add_argument('--loss-gate-adaptive', type=float, default=0.05, help='Gate adaptive loss weight')
    
    # [New] Relax & Constrain Losses
    parser.add_argument('--loss-id-triplet', type=float, default=1.0, help='ID Triplet loss weight')
    parser.add_argument('--loss-anti-collapse', type=float, default=1.0, help='Anti-collapse loss weight')
    parser.add_argument('--loss-reconstruction', type=float, default=0.1, help='Reconstruction loss weight')
    parser.add_argument('--loss-orthogonal', type=float, default=0.1, help='Orthogonal loss weight')
    parser.add_argument('--loss-semantic-alignment', type=float, default=0.1, help='Semantic alignment loss weight')
    parser.add_argument('--loss-freq-consistency', type=float, default=0.5, help='Frequency consistency loss weight')
    parser.add_argument('--loss-freq-separation', type=float, default=0.2, help='Frequency separation loss weight')

    # [New] Visualization parameters
    parser.add_argument('--visualization-enabled', action='store_true', help='Enable FSHD visualization')
    parser.add_argument('--visualization-save-dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--visualization-frequency', type=int, default=5, help='Frequency (epochs) to save visualizations')
    parser.add_argument('--visualization-batch-interval', type=int, default=200, help='Batch interval to save visualizations')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Scheduler type')

    parser.add_argument('--finetune-from', type=str, help='Path to checkpoint to finetune from')

    args = parser.parse_args()

    # 初始化 disentangle 字典
    args.disentangle = {}

    # 处理布尔值字符串
    args.gs3_use_multi_scale_cnn = args.gs3_use_multi_scale_cnn.lower() == 'true'

    # 处理损失权重
    if args.loss_weights:
        args.disentangle['loss_weights'] = ast.literal_eval(args.loss_weights)
    else:
        args.disentangle['loss_weights'] = {
            'info_nce': args.loss_info_nce,
            'cls': args.loss_cls,
            'cloth_semantic': args.loss_cloth_semantic,
            'gate_adaptive': args.loss_gate_adaptive,
            'id_triplet': args.loss_id_triplet,
            'anti_collapse': args.loss_anti_collapse,
            'reconstruction': args.loss_reconstruction,
            'orthogonal': args.loss_orthogonal,
            'semantic_alignment': args.loss_semantic_alignment,
            'freq_consistency': args.loss_freq_consistency,
            'freq_separation': args.loss_freq_separation
        }
    
    # 初始化可视化配置
    args.visualization = {
        'enabled': args.visualization_enabled,
        'save_dir': args.visualization_save_dir,
        'frequency': args.visualization_frequency,
        'batch_interval': args.visualization_batch_interval
    }

    # 处理数据集配置
    if args.dataset_configs:
        dataset_configs = []
        for cfg in args.dataset_configs:
            parsed = ast.literal_eval(cfg)
            dataset_configs.extend(parsed if isinstance(parsed, list) else [parsed])
        args.dataset_configs = dataset_configs
    else:
        # Default Datasets
        args.dataset_configs = [
            {
                'name': 'CUHK-PEDES',
                'root': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES'),
                'json_file': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_all.json'),
                'cloth_json': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_cloth.json'),
                'id_json': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_id.json')
            }
        ]

    args.bert_base_path = str(Path(args.bert_base_path))
    args.clip_pretrained = str(Path(args.clip_pretrained))
    args.vit_pretrained = str(Path(args.vit_pretrained))
    args.logs_dir = str(Path(args.logs_dir))
    args.root = str(Path(args.root))

    if not Path(args.vit_pretrained).exists():
        raise FileNotFoundError(f"ViT base path not found at: {args.vit_pretrained}")

    args.img_size = (args.height, args.width)
    args.task_name = 't2i'
    return args, {}

class Runner:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16) if self.device.type == 'cuda' else None
        
        self.args.original_logs_dir = args.logs_dir

        # === 统一数据集名称映射 ===
        if hasattr(args, 'dataset_configs') and args.dataset_configs:
            dataset_full_name = args.dataset_configs[0]['name'].lower() if args.dataset_configs else 'unknown'
            # 映射到短名称
            if 'cuhk' in dataset_full_name:
                dataset_name = 'cuhk_pedes'
            elif 'rstp' in dataset_full_name:
                dataset_name = 'rstp'
            elif 'icfg' in dataset_full_name:
                dataset_name = 'icfg'
            else:
                dataset_name = dataset_full_name
        else:
            dataset_name = 'unknown'
        
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        log_base_dir = str(project_root / 'log')

        self.monitor = get_monitor_for_dataset(dataset_name, log_base_dir)

    def verify_freeze_status(self, model):
        """
        验证冻结状态 - 适配 CLIP + Vim
        """
        vit_frozen = sum(p.numel() for n, p in model.named_parameters() 
                        if 'visual_encoder' in n and not p.requires_grad)
        vit_total = sum(p.numel() for n, p in model.named_parameters() 
                       if 'visual_encoder' in n)
        vit_trainable = vit_total - vit_frozen
        
        # 统计文本部分 (CLIP)
        # text_encoder 是 CLIP, text_proj 是 Adapter
        text_frozen = sum(p.numel() for n, p in model.named_parameters() 
                         if ('text_encoder' in n) and not p.requires_grad)
        text_total = sum(p.numel() for n, p in model.named_parameters() 
                        if ('text_encoder' in n))
        text_trainable = text_total - text_frozen
        
        # Adapter 应该始终可训练
        adapter_trainable = sum(p.numel() for n, p in model.named_parameters() if 'text_proj' in n and p.requires_grad)
        adapter_total = sum(p.numel() for n, p in model.named_parameters() if 'text_proj' in n)
        
        # 其他任务参数
        task_trainable = sum(p.numel() for n, p in model.named_parameters() 
                            if 'visual_encoder' not in n 
                            and 'text_encoder' not in n 
                            and 'text_proj' not in n
                            and p.requires_grad)
        
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logging.info("=" * 70)
        logging.info("📊 Freeze Status Verification (CLIP Upgrade)")
        logging.info("=" * 70)
        logging.info(f"Visual: {vit_trainable:,}/{vit_total:,} trainable ({100*vit_trainable/(vit_total+1):.1f}%)")
        logging.info(f"Text(CLIP): {text_trainable:,}/{text_total:,} trainable ({100*text_trainable/(text_total+1):.1f}%)")
        logging.info(f"Adapter: {adapter_trainable:,}/{adapter_total:,} trainable")
        logging.info(f"Task:    {task_trainable:,} trainable")
        logging.info(f"Total:   {total_trainable:,}/{total_params:,} trainable")
        logging.info("=" * 70)
        
        return {}
    
    def freeze_text_layers(self, model, unfreeze_from_layer=None):
        """
        [New] Freeze logic for CLIP Text Encoder
        Structure: text_model.encoder.layers.X (0-11)
        
        Args:
            unfreeze_from_layer:
                None -> Freeze All
                8 -> Unfreeze layers 8-11 (Last 4)
                0 -> Unfreeze All (Including Embeddings)
        """
        # 1. Freeze all CLIP parameters first
        for name, param in model.named_parameters():
            if 'text_encoder' in name:
                param.requires_grad = False
            
            # [CRITICAL] Adapter must ALWAYS be trainable
            if 'text_proj' in name:
                param.requires_grad = True
                
        # 2. Unfreeze specific layers
        if unfreeze_from_layer is not None:
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if 'text_encoder' in name:
                    # Case 1: Layers (text_model.encoder.layers.X)
                    if 'layers.' in name:
                        try:
                            # Extract layer number
                            # name like: text_encoder.text_model.encoder.layers.11.self_attn...
                            parts = name.split('layers.')[1].split('.')
                            layer_num = int(parts[0])
                            
                            if layer_num >= unfreeze_from_layer:
                                param.requires_grad = True
                                unfrozen_count += 1
                        except:
                            pass
                    
                    # Case 2: Final LayerNorm (Always unfreeze if any layer is unfrozen)
                    if 'final_layer_norm' in name:
                        param.requires_grad = True
                        unfrozen_count += 1
                        
                    # Case 3: Embeddings (Only if unfreeze_from_layer == 0)
                    if unfreeze_from_layer == 0 and ('embeddings' in name):
                        param.requires_grad = True
                        unfrozen_count += 1

            logging.info(f"CLIP: Unfrozen layers starting from {unfreeze_from_layer} (Total params groups: {unfrozen_count})")
        else:
            logging.info(f"CLIP: All layers frozen (Adapter remains trainable)")

    def freeze_vit_layers(self, model, unfreeze_from_layer=None):
        """
        Freeze logic for ViT/Vim. (Kept similar to previous version)
        """
        is_vim = getattr(model, 'vision_backbone_type', 'vit') == 'vim'
        total_layers = 24 if is_vim else 12
        
        target_start_layer = None
        if unfreeze_from_layer is not None:
            if unfreeze_from_layer == 0:
                target_start_layer = 0
            elif unfreeze_from_layer == 8: # Last 4
                target_start_layer = total_layers - 4 
            elif unfreeze_from_layer == 4: # Last 8
                target_start_layer = total_layers - 8
            else:
                target_start_layer = unfreeze_from_layer if not is_vim else unfreeze_from_layer * 2

        # 1. Freeze all visual encoder
        for name, param in model.named_parameters():
            if 'visual_encoder' in name:
                param.requires_grad = False
                if 'visual_proj' in name: # Projection always trainable
                    param.requires_grad = True
        
        # 2. Unfreeze specific layers
        if target_start_layer is not None:
            for name, param in model.named_parameters():
                if 'visual_encoder' in name:
                    if target_start_layer == 0 and ('embeddings' in name or 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name):
                        param.requires_grad = True
                        continue

                    layer_num = -1
                    if is_vim and 'layers.' in name:
                        try:
                            parts = name.split('layers.')[1].split('.')
                            layer_num = int(parts[0])
                        except: pass
                    elif not is_vim and 'encoder.layer.' in name:
                        try:
                            parts = name.split('encoder.layer.')[1].split('.')
                            layer_num = int(parts[0])
                        except: pass
                    
                    if layer_num != -1 and layer_num >= target_start_layer:
                        param.requires_grad = True
                    
                    # Unfreeze final norms
                    if target_start_layer == 0 or target_start_layer < total_layers:
                        if is_vim and 'norm_f' in name:
                            param.requires_grad = True
                        elif not is_vim and ('layernorm' in name or 'pooler' in name):
                            param.requires_grad = True

            logging.info(f"{'Vim' if is_vim else 'ViT'}: Unfrozen params from layer {target_start_layer}/{total_layers}")
        else:
            logging.info(f"{'Vim' if is_vim else 'ViT'}: All layers frozen")

    def get_param_groups_with_diff_lr(self, model, base_lr, stage):
        """
        Advanced LR Grouping for CLIP + Vim
        
        Key Policy:
        1. CLIP Text Encoder: 0.1x LR (Very Sensitive)
        2. Text Adapter (New): 1.0x LR
        3. Visual Backbone: Graded LR (Low/Mid/High)
        4. Task Modules: 1.0x LR
        """
        is_vim = getattr(model, 'vision_backbone_type', 'vit') == 'vim'
        
        # --- Grouping Containers ---
        clip_params = []
        text_adapter_params = []
        
        vit_low_params = []      
        vit_mid_params = []      
        vit_high_params = []     
        vit_embed_params = []
        
        task_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 1. CLIP Params
            if 'text_encoder' in name:
                clip_params.append(param)
            
            # 2. Text Adapter (New)
            elif 'text_proj' in name:
                text_adapter_params.append(param)
                
            # 3. Visual Encoder
            elif 'visual_encoder' in name:
                if 'embeddings' in name or 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
                    vit_embed_params.append(param)
                else:
                    layer_num = -1
                    if is_vim and 'layers.' in name:
                        try: layer_num = int(name.split('layers.')[1].split('.')[0])
                        except: pass
                    elif not is_vim and 'encoder.layer.' in name:
                        try: layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                        except: pass
                    
                    if layer_num != -1:
                        if is_vim: # 24 layers
                            if layer_num < 8: vit_low_params.append(param)
                            elif layer_num < 16: vit_mid_params.append(param)
                            else: vit_high_params.append(param)
                        else: # 12 layers
                            if layer_num < 4: vit_low_params.append(param)
                            elif layer_num < 8: vit_mid_params.append(param)
                            else: vit_high_params.append(param)
                    elif 'layernorm' in name or 'pooler' in name or 'norm_f' in name:
                        vit_high_params.append(param)
                    else:
                        task_params.append(param)
            
            # 4. Others (Fusion, Classifier, Heads)
            else:
                task_params.append(param)
        
        # --- Stage Logic ---
        
        # Common settings
        clip_lr_ratio = 0.1  # CLIP needs lower LR
        
        groups = []
        
        # Always add task and adapter params with full LR
        groups.append({'params': task_params, 'lr': base_lr, 'name': 'task_modules'})
        groups.append({'params': text_adapter_params, 'lr': base_lr, 'name': 'text_adapter'})
        
        if stage == 1: 
            # Stage 1: ViT High (0.3x)
            groups.append({'params': vit_high_params, 'lr': base_lr * 0.3, 'name': 'vit_high'})
            # CLIP is frozen in Stage 1 usually, but if not, give it very low LR
            if clip_params:
                groups.append({'params': clip_params, 'lr': base_lr * clip_lr_ratio * 0.5, 'name': 'clip_encoder'})

        elif stage == 2: 
            # Stage 2: ViT High (0.5x) + CLIP High (if unfrozen)
            groups.append({'params': vit_high_params, 'lr': base_lr * 0.5, 'name': 'vit_high'})
            if clip_params:
                groups.append({'params': clip_params, 'lr': base_lr * clip_lr_ratio, 'name': 'clip_encoder'})

        elif stage == 3: 
            # Stage 3: ViT Mid+High (0.6x)
            groups.append({'params': vit_mid_params + vit_high_params, 'lr': base_lr * 0.6, 'name': 'vit_mid_high'})
            if clip_params:
                groups.append({'params': clip_params, 'lr': base_lr * clip_lr_ratio, 'name': 'clip_encoder'})

        elif stage == 4: 
            # Stage 4: All
            groups.append({'params': vit_embed_params, 'lr': base_lr * 0.05, 'name': 'vit_embed'})
            groups.append({'params': vit_low_params, 'lr': base_lr * 0.2, 'name': 'vit_low'})
            groups.append({'params': vit_mid_params, 'lr': base_lr * 0.4, 'name': 'vit_mid'})
            groups.append({'params': vit_high_params, 'lr': base_lr * 0.6, 'name': 'vit_high'})
            
            if clip_params:
                groups.append({'params': clip_params, 'lr': base_lr * clip_lr_ratio, 'name': 'clip_encoder'})
                
        else:
            raise ValueError(f"Invalid stage: {stage}")
            
        return groups

    def build_optimizer(self, model, stage=1):
        param_groups = self.get_param_groups_with_diff_lr(model, self.args.lr, stage)
        
        optimizer_type = self.args.optimizer.lower()
        if optimizer_type == 'adamw':
            return torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)
        elif optimizer_type == 'adam':
            return torch.optim.Adam(param_groups, eps=1e-8, betas=(0.9, 0.999))
        else:
            return torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))

    def build_scheduler(self, optimizer):
        if self.args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.epochs, eta_min=1e-6
            )
        return WarmupMultiStepLR(
            optimizer, self.args.milestones, gamma=0.1,
            warmup_factor=0.1, warmup_iters=self.args.warmup_step
        )

    def load_param(self, model, trained_path):
        param_dict = torch.load(trained_path, map_location=self.device, weights_only=False)
        param_dict = param_dict.get('state_dict', param_dict.get('model', param_dict))
        model_dict = model.state_dict()
        for i in param_dict:
            if i in model_dict and model_dict[i].shape == param_dict[i].shape:
                model_dict[i] = param_dict[i]
        model.load_state_dict(model_dict, strict=False)
        logging.info(f"Loaded pretrained weights from {trained_path}")

    def run(self):
        args = self.args
        config = self.config
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # [修复日志泄露] 清除根日志记录器的所有处理器，防止 DEBUG 信息输出到终端
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # 获取数据集日志目录（TrainingMonitor已经创建）
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        dataset_log_dir = self.monitor.dataset_log_dir

        # 1. 详细日志记录器 (仅限文件)
        detailed_logger = logging.getLogger('detailed')
        detailed_logger.setLevel(logging.DEBUG)
        detailed_logger.propagate = False
        for handler in detailed_logger.handlers[:]: 
            detailed_logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(dataset_log_dir / 'log.txt', mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        detailed_logger.addHandler(file_handler)

        # 2. 调试日志记录器 (Root Logger, 仅限文件)
        debug_log_file = dataset_log_dir / 'debug.txt'
        root_file_handler = logging.FileHandler(debug_log_file, mode='a', encoding='utf-8')
        root_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.root.addHandler(root_file_handler)
        logging.root.setLevel(logging.DEBUG)

        # 3. 控制台记录器 (专门用于终端输出，INFO级别)
        console_logger = logging.getLogger('console')
        console_logger.setLevel(logging.INFO)
        console_logger.propagate = False # 防止向上级传播
        for handler in console_logger.handlers[:]: console_logger.removeHandler(handler)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        console_logger.addHandler(console_handler)

        # 构建数据集
        console_logger.info("Building dataset...")
        data_builder = DataBuilder(args, is_distributed=False)
        args.num_classes = data_builder.get_num_classes()
        detailed_logger.info(f"Set num_classes = {args.num_classes}")

        console_logger.info("Loading training data...")
        train_loader, _ = data_builder.build_data(is_train=True)
        console_logger.info("Loading query and gallery data...")
        query_loader, gallery_loader = data_builder.build_data(is_train=False)

        # Model Config
        model_config = {
            'clip_pretrained': args.clip_pretrained, # [New]
            'vit_pretrained': args.vit_pretrained,
            'vision_backbone': args.vision_backbone,
            'vim_pretrained': args.vim_pretrained,
            'img_size': (args.height, args.width),
            'num_classes': args.num_classes,
            'disentangle_type': args.disentangle_type,
            'gs3': {
                'num_heads': args.gs3_num_heads,
                'd_state': args.gs3_d_state,
                'd_conv': args.gs3_d_conv,
                'dropout': args.gs3_dropout,
                'use_multi_scale_cnn': args.gs3_use_multi_scale_cnn,
                'img_size': tuple(args.gs3_img_size)
            },
            'fusion': {
                'type': args.fusion_type,
                'dim': args.fusion_dim,
                'd_state': args.fusion_d_state,
                'd_conv': args.fusion_d_conv,
                'num_layers': args.fusion_num_layers,
                'output_dim': args.fusion_output_dim,
                'dropout': args.fusion_dropout
            }
        }

        console_logger.info("Initializing model (CLIP + Vim)...")
        model = Model(net_config=model_config).to(self.device)
        
        if args.finetune_from:
            self.load_param(model, args.finetune_from)

        # === Stage 1 Initialization ===
        console_logger.info("=" * 60)
        console_logger.info("🚀 Training Start: CLIP + Vim Architecture")
        console_logger.info("   Stage 1: Freeze CLIP / Unfreeze Vim Last 4")
        console_logger.info("=" * 60)
        
        # Initial Freeze State
        self.freeze_text_layers(model, unfreeze_from_layer=None) # Freeze CLIP
        self.freeze_vit_layers(model, unfreeze_from_layer=8)     # Unfreeze Vim Last 4

        console_logger.info("Building optimizer...")
        optimizer = self.build_optimizer(model, stage=1)
        lr_scheduler = self.build_scheduler(optimizer)
        
        self.verify_freeze_status(model)

        console_logger.info("Starting training loop...")
        trainer = Trainer(model, args, self.monitor, runner=self)
        # Pass dataset_log_dir as checkpoint_dir so trainer can create model/ subdir inside it
        trainer.train(
            train_loader, optimizer, lr_scheduler, query_loader, gallery_loader, checkpoint_dir=str(dataset_log_dir)
        )

if __name__ == '__main__':
    args, config = configuration()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    runner = Runner(args, config)
    runner.run()
