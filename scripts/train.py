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
    parser = argparse.ArgumentParser(description="Train T2I-ReID model")
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
    parser.add_argument('--bert-base-path', type=str, default=str(ROOT_DIR / 'pretrained' / 'bert-base-uncased'),
                       help='Path to BERT model')
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
    
    # G-S3 module parameters
    parser.add_argument('--disentangle-type', type=str, default='gs3', 
                       choices=['gs3', 'simple'],
                       help='Type of disentangle module: gs3 (G-S3 Module) or simple (DisentangleModule)')
    parser.add_argument('--gs3-num-heads', type=int, default=8, 
                       help='Number of attention heads in G-S3 OPA')
    parser.add_argument('--gs3-d-state', type=int, default=16, 
                       help='State dimension for G-S3 Mamba filter')
    parser.add_argument('--gs3-d-conv', type=int, default=4, 
                       help='Convolution kernel size for G-S3 Mamba filter')
    parser.add_argument('--gs3-dropout', type=float, default=0.1, 
                       help='Dropout rate for G-S3 module')

    # Loss weights（重新调整权重以平衡各损失项，避免竞争）
    parser.add_argument('--loss-info-nce', type=float, default=1.0, help='InfoNCE loss weight')
    parser.add_argument('--loss-cls', type=float, default=0.05, help='Classification loss weight')
    parser.add_argument('--loss-cloth-semantic', type=float, default=0.2, help='Cloth semantic loss weight (从0.5降到0.2)')
    parser.add_argument('--loss-orthogonal', type=float, default=0.3, help='Orthogonal loss weight (从0.1提高到0.3)')
    parser.add_argument('--loss-gate-adaptive', type=float, default=0.15, help='Gate adaptive loss weight (从0.05提高到0.15)')

    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Scheduler type')

    # [修改点 1] 添加 finetune-from 参数
    parser.add_argument('--finetune-from', type=str, help='Path to checkpoint to finetune from')

    args = parser.parse_args()

    # 初始化 disentangle 字典
    args.disentangle = {}

    # 处理损失权重
    if args.loss_weights:
        args.disentangle['loss_weights'] = ast.literal_eval(args.loss_weights)
    else:
        # 设置默认损失权重
        args.disentangle['loss_weights'] = {
            'info_nce': args.loss_info_nce,
            'cls': args.loss_cls,
            'cloth_semantic': args.loss_cloth_semantic,
            'orthogonal': args.loss_orthogonal,
            'gate_adaptive': args.loss_gate_adaptive
        }

    # 处理数据集配置
    if args.dataset_configs:
        dataset_configs = []
        for cfg in args.dataset_configs:
            parsed = ast.literal_eval(cfg)
            dataset_configs.extend(parsed if isinstance(parsed, list) else [parsed])
        args.dataset_configs = dataset_configs
    else:
        args.dataset_configs = [
            {
                'name': 'CUHK-PEDES',
                'root': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES'),
                'json_file': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_all.json'),
                'cloth_json': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_cloth.json'),
                'id_json': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_id.json')
            },
            {
                'name': 'ICFG-PEDES',
                'root': str(ROOT_DIR / 'datasets' / 'ICFG-PEDES'),
                'json_file': str(ROOT_DIR / 'datasets' / 'ICFG-PEDES' / 'annotations' / 'ICFG-PEDES.json'),
                'cloth_json': str(ROOT_DIR / 'datasets' / 'ICFG-PEDES' / 'annotations' / 'caption_cloth.json'),
                'id_json': str(ROOT_DIR / 'datasets' / 'ICFG-PEDES' / 'annotations' / 'caption_id.json')
            },
            {
                'name': 'RSTPReid',
                'root': str(ROOT_DIR / 'datasets' / 'RSTPReid'),
                'json_file': str(ROOT_DIR / 'datasets' / 'RSTPReid' / 'annotations' / 'data_captions.json'),
                'cloth_json': str(ROOT_DIR / 'datasets' / 'RSTPReid' / 'annotations' / 'caption_cloth.json'),
                'id_json': str(ROOT_DIR / 'datasets' / 'RSTPReid' / 'annotations' / 'caption_id.json')
            }
        ]

    # 确保路径使用 Path 对象
    args.bert_base_path = str(Path(args.bert_base_path))
    args.vit_pretrained = str(Path(args.vit_pretrained))
    args.logs_dir = str(Path(args.logs_dir))
    args.root = str(Path(args.root))

    # 验证路径有效性
    if not Path(args.bert_base_path).exists():
        raise FileNotFoundError(f"BERT base path not found at: {args.bert_base_path}")
    if not Path(args.vit_pretrained).exists():
        raise FileNotFoundError(f"ViT base path not found at: {args.vit_pretrained}")

    args.img_size = (args.height, args.width)
    args.task_name = 't2i'
    return args, {}

class Runner:
    def __init__(self, args, config):
        # 初始化运行器，设置参数和设备
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16) if self.device.type == 'cuda' else None
        if args.fp16 and self.device.type != 'cuda':
            logging.warning("FP16 is enabled but no CUDA device is available. Disabling mixed precision.")

        # 保存原始的日志目录，以防在训练过程中被修改
        self.args.original_logs_dir = args.logs_dir

        # 初始化监控器
        # 从数据集配置中获取数据集名称
        if hasattr(args, 'dataset_configs') and args.dataset_configs:
            dataset_name = args.dataset_configs[0]['name'] if args.dataset_configs else 'unknown'
        else:
            dataset_name = 'unknown'
        # 使用基于项目根目录的 log 目录，而不是依赖args.logs_dir
        # 获取项目根目录
        script_dir = Path(__file__).parent  # scripts/
        project_root = script_dir.parent   # 项目根目录
        log_base_dir = str(project_root / 'log')

        # 传递原始的dataset_name给get_monitor_for_dataset，它会内部规范化
        self.monitor = get_monitor_for_dataset(dataset_name, log_base_dir)

    def verify_freeze_status(self, model):
        """
        验证冻结状态，确保freeze函数生效
        
        Returns:
            dict: 包含各模块的冻结统计信息
        """
        vit_frozen = sum(p.numel() for n, p in model.named_parameters() 
                        if 'visual_encoder' in n and not p.requires_grad)
        vit_total = sum(p.numel() for n, p in model.named_parameters() 
                       if 'visual_encoder' in n)
        vit_trainable = vit_total - vit_frozen
        
        bert_frozen = sum(p.numel() for n, p in model.named_parameters() 
                         if 'text_encoder' in n and not p.requires_grad)
        bert_total = sum(p.numel() for n, p in model.named_parameters() 
                        if 'text_encoder' in n)
        bert_trainable = bert_total - bert_frozen
        
        task_trainable = sum(p.numel() for n, p in model.named_parameters() 
                            if 'visual_encoder' not in n and 'text_encoder' not in n and p.requires_grad)
        
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        stats = {
            'vit_frozen': vit_frozen,
            'vit_trainable': vit_trainable,
            'vit_total': vit_total,
            'bert_frozen': bert_frozen,
            'bert_trainable': bert_trainable,
            'bert_total': bert_total,
            'task_trainable': task_trainable,
            'total_trainable': total_trainable,
            'total_params': total_params
        }
        
        logging.info("=" * 70)
        logging.info("📊 Freeze Status Verification")
        logging.info("=" * 70)
        logging.info(f"ViT:  {vit_trainable:,}/{vit_total:,} trainable "
                    f"({100*vit_trainable/vit_total:.1f}%), "
                    f"{vit_frozen:,} frozen ({100*vit_frozen/vit_total:.1f}%)")
        logging.info(f"BERT: {bert_trainable:,}/{bert_total:,} trainable "
                    f"({100*bert_trainable/bert_total:.1f}%), "
                    f"{bert_frozen:,} frozen ({100*bert_frozen/bert_total:.1f}%)")
        logging.info(f"Task: {task_trainable:,} trainable")
        logging.info(f"Total: {total_trainable:,}/{total_params:,} trainable "
                    f"({100*total_trainable/total_params:.1f}%)")
        logging.info("=" * 70)
        
        return stats
    
    def freeze_bert_layers(self, model, unfreeze_from_layer=None):
        """
        冻结/解冻BERT的指定层（基于BERT-Base 12层结构）
        
        Args:
            model: 完整的T2I-ReID模型
            unfreeze_from_layer: 
                - None: 冻结所有BERT层
                - 8: 解冻layer 8-11（后4层）
                - 4: 解冻layer 4-11（后8层）
                - 0: 解冻所有12层
        """
        # 步骤1: 先冻结所有BERT参数
        for name, param in model.named_parameters():
            if 'text_encoder' in name:
                param.requires_grad = False
        
        # 步骤2: 如果指定了解冻层，则解冻对应的层
        if unfreeze_from_layer is not None:
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if 'text_encoder' in name:
                    # 当完全解冻时(unfreeze_from_layer=0)，解冻embeddings
                    if unfreeze_from_layer == 0 and 'embeddings' in name:
                        param.requires_grad = True
                        unfrozen_count += 1
                    
                    # 解冻指定层及其之后的所有层
                    # BERT命名: encoder.layer.X (X=0-11)
                    if 'encoder.layer.' in name:
                        try:
                            # 提取层号
                            parts = name.split('encoder.layer.')[1].split('.')
                            layer_num = int(parts[0])
                            
                            # 验证层号范围 (0-11)
                            if 0 <= layer_num <= 11 and layer_num >= unfreeze_from_layer:
                                param.requires_grad = True
                                unfrozen_count += 1
                        except (IndexError, ValueError) as e:
                            logging.warning(f"Could not parse BERT layer number from: {name}, error: {e}")
                            continue
                    
                    # 解冻pooler（如果完全解冻）
                    if unfreeze_from_layer == 0 and 'pooler' in name:
                        param.requires_grad = True
                        unfrozen_count += 1
            
            logging.info(f"BERT: Unfrozen {unfrozen_count} parameter groups from layer {unfreeze_from_layer}")
        else:
            logging.info(f"BERT: All layers frozen")
        
        # 步骤3: 统计并记录可训练参数
        bert_trainable = sum(p.numel() for n, p in model.named_parameters() 
                            if p.requires_grad and 'text_encoder' in n)
        bert_total = sum(p.numel() for n, p in model.named_parameters() if 'text_encoder' in n)
        
        logging.info(f"BERT: {bert_trainable:,}/{bert_total:,} trainable ({100*bert_trainable/bert_total:.1f}%)")
    
    def freeze_vit_layers(self, model, unfreeze_from_layer=None):
        """
        冻结/解冻ViT或Vim的指定层
        
        ViT-Base: 12层 (encoder.layer.X)
        Vim-S: 24层 (layers.X)
        
        Args:
            unfreeze_from_layer:
                对于ViT (12层):
                - None: 全冻结
                - 8: 解冻 8-11 (后4层)
                - 4: 解冻 4-11 (后8层)
                - 0: 全解冻
                
                对于Vim (24层) - 自动映射:
                - None: 全冻结
                - 8 -> 映射为 16 (解冻 16-23, 后8层) -> 等等，保持比例
                  为了简化逻辑，我们按照"后N层"的概念:
                  Stage 1 (ViT后4层) -> Vim后4层 (20-23)
                  Stage 2 (ViT后8层) -> Vim后8层 (16-23)
                  Stage 3 (ViT后8层) -> Vim后12层 (12-23) ? 
                  Let's align by ratio or fixed blocks.
                  
                  定义映射:
                  unfreeze_from_layer=8 (ViT 8/12, last 4) -> Vim 20/24 (last 4)
                  unfreeze_from_layer=4 (ViT 4/12, last 8) -> Vim 16/24 (last 8)
                  unfreeze_from_layer=0 (ViT 0/12, all)    -> Vim 0/24 (all)
        """
        is_vim = getattr(model, 'vision_backbone_type', 'vit') == 'vim'
        total_layers = 24 if is_vim else 12
        
        # 确定解冻的起始层索引
        target_start_layer = None
        if unfreeze_from_layer is not None:
            if unfreeze_from_layer == 0:
                target_start_layer = 0
            elif unfreeze_from_layer == 8: # ViT后4层
                target_start_layer = total_layers - 4 # Vim: 20, ViT: 8
            elif unfreeze_from_layer == 4: # ViT后8层
                target_start_layer = total_layers - 8 # Vim: 16, ViT: 4
            else:
                # 默认映射
                target_start_layer = unfreeze_from_layer if not is_vim else unfreeze_from_layer * 2

        # 步骤1: 先冻结所有视觉编码器参数
        for name, param in model.named_parameters():
            if 'visual_encoder' in name:
                param.requires_grad = False
                # 投影层始终训练
                if 'visual_proj' in name:
                    param.requires_grad = True
        
        # 步骤2: 解冻指定层
        if target_start_layer is not None:
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if 'visual_encoder' in name:
                    # 解冻embeddings (当完全解冻时)
                    if target_start_layer == 0 and ('embeddings' in name or 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name):
                        param.requires_grad = True
                        unfrozen_count += 1
                        continue

                    # 识别层号
                    layer_num = -1
                    if is_vim:
                        # Vim命名: visual_encoder.layers.X
                        if 'layers.' in name:
                            try:
                                parts = name.split('layers.')[1].split('.')
                                layer_num = int(parts[0])
                            except (IndexError, ValueError):
                                pass
                    else:
                        # ViT命名: visual_encoder.encoder.layer.X
                        if 'encoder.layer.' in name:
                            try:
                                parts = name.split('encoder.layer.')[1].split('.')
                                layer_num = int(parts[0])
                            except (IndexError, ValueError):
                                pass
                    
                    # 判断是否解冻
                    if layer_num != -1 and layer_num >= target_start_layer:
                        param.requires_grad = True
                        unfrozen_count += 1
                    
                    # 解冻最后的Norm层 (如果完全解冻或部分解冻)
                    # Vim: norm_f, ViT: layernorm/pooler
                    if target_start_layer == 0 or target_start_layer < total_layers:
                        if is_vim and 'norm_f' in name:
                            param.requires_grad = True
                        elif not is_vim and ('layernorm' in name or 'pooler' in name):
                            param.requires_grad = True

            logging.info(f"{'Vim' if is_vim else 'ViT'}: Unfrozen params from layer {target_start_layer}/{total_layers}")
        else:
            logging.info(f"{'Vim' if is_vim else 'ViT'}: All layers frozen")
        
        # 步骤3: 统计
        vit_trainable = sum(p.numel() for n, p in model.named_parameters() 
                           if p.requires_grad and 'visual_encoder' in n)
        vit_total = sum(p.numel() for n, p in model.named_parameters() if 'visual_encoder' in n)
        
        logging.info(f"Visual Encoder: {vit_trainable:,}/{vit_total:,} trainable ({100*vit_trainable/vit_total:.1f}%)")
        
        # 总体统计
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logging.info(f"Overall: {trainable:,}/{total:,} trainable ({100*trainable/total:.1f}%)")
    
    def get_param_groups_with_diff_lr(self, model, base_lr, stage):
        """
        为不同的模块设置不同的学习率（方案B：渐进解冻）
        支持 ViT (12层) 和 Vim (24层)
        
        Vim 分组映射:
        - Low: layers 0-7
        - Mid: layers 8-15
        - High: layers 16-23
        """
        is_vim = getattr(model, 'vision_backbone_type', 'vit') == 'vim'
        
        # 初始化参数组
        bert_embed_params = []
        bert_low_params = []     # BERT layer 0-3
        bert_mid_params = []     # BERT layer 4-7
        bert_high_params = []    # BERT layer 8-11
        bert_other_params = []
        
        vit_embed_params = []
        vit_low_params = []      # ViT 0-3 / Vim 0-7
        vit_mid_params = []      # ViT 4-7 / Vim 8-15
        vit_high_params = []     # ViT 8-11 / Vim 16-23
        vit_other_params = []
        
        task_params = []         # G-S3, Fusion, Classifier, Projection等
        
        for name, param in model.named_parameters():
            # 【关键修复】跳过冻结的参数
            if not param.requires_grad:
                continue
            
            # 处理BERT参数
            if 'text_encoder' in name:
                if 'embeddings' in name:
                    bert_embed_params.append(param)
                elif 'encoder.layer.' in name:
                    try:
                        parts = name.split('encoder.layer.')[1].split('.')
                        layer_num = int(parts[0])
                        
                        if 0 <= layer_num <= 3:
                            bert_low_params.append(param)
                        elif 4 <= layer_num <= 7:
                            bert_mid_params.append(param)
                        elif 8 <= layer_num <= 11:
                            bert_high_params.append(param)
                    except (IndexError, ValueError):
                        task_params.append(param)
                elif 'pooler' in name:
                    bert_other_params.append(param)
                else:
                    task_params.append(param)
            
            # 处理视觉编码器参数 (ViT or Vim)
            elif 'visual_encoder' in name:
                # Embeddings
                if 'embeddings' in name or 'patch_embed' in name or 'cls_token' in name or 'pos_embed' in name:
                    vit_embed_params.append(param)
                    continue
                    
                # Layers
                layer_num = -1
                if is_vim and 'layers.' in name: # Vim: layers.X
                    try:
                        parts = name.split('layers.')[1].split('.')
                        layer_num = int(parts[0])
                    except (IndexError, ValueError):
                        pass
                elif not is_vim and 'encoder.layer.' in name: # ViT: encoder.layer.X
                    try:
                        parts = name.split('encoder.layer.')[1].split('.')
                        layer_num = int(parts[0])
                    except (IndexError, ValueError):
                        pass
                
                if layer_num != -1:
                    if is_vim:
                        # Vim 24 layers grouping
                        if 0 <= layer_num <= 7:
                            vit_low_params.append(param)
                        elif 8 <= layer_num <= 15:
                            vit_mid_params.append(param)
                        elif 16 <= layer_num <= 23:
                            vit_high_params.append(param)
                        else:
                            vit_other_params.append(param)
                    else:
                        # ViT 12 layers grouping
                        if 0 <= layer_num <= 3:
                            vit_low_params.append(param)
                        elif 4 <= layer_num <= 7:
                            vit_mid_params.append(param)
                        elif 8 <= layer_num <= 11:
                            vit_high_params.append(param)
                        else:
                            vit_other_params.append(param)
                # Other parts (Norms etc.)
                elif 'layernorm' in name or 'pooler' in name or 'norm_f' in name:
                    vit_other_params.append(param)
                else:
                    # 无法分类的视觉参数放入 task_params 或其他
                    task_params.append(param)
            
            # 其他参数（任务特定模块 + 投影层）
            else:
                task_params.append(param)
        
        # 根据训练阶段设置学习率
        if stage == 1:  # Stage 1 (Epoch 1-10): ViT后4层 + 任务模块
            param_groups = [
                {'params': vit_high_params, 'lr': base_lr * 0.3, 'weight_decay': 0.0001, 'name': 'vit_high'},
                {'params': task_params, 'lr': base_lr * 1.0, 'weight_decay': 0.0001, 'name': 'task_modules'}
            ]
            logging.info(f"Stage 1 LR: vit_high={base_lr*0.3:.2e}, task={base_lr*1.0:.2e}")
            
        elif stage == 2:  # Stage 2 (Epoch 11-30): BERT和ViT后4层
            backbone_high_params = bert_high_params + vit_high_params
            param_groups = [
                {'params': backbone_high_params, 'lr': base_lr * 0.5, 'weight_decay': 0.0001, 'name': 'backbone_high'},
                {'params': task_params, 'lr': base_lr * 1.0, 'weight_decay': 0.0001, 'name': 'task_modules'}
            ]
            logging.info(f"Stage 2 LR: backbone_high={base_lr*0.5:.2e}, task={base_lr*1.0:.2e}")
            
        elif stage == 3:  # Stage 3 (Epoch 31-60): BERT和ViT后8层
            backbone_mid_high_params = bert_mid_params + bert_high_params + vit_mid_params + vit_high_params
            param_groups = [
                {'params': backbone_mid_high_params, 'lr': base_lr * 0.6, 'name': 'backbone_mid_high'},
                {'params': task_params, 'lr': base_lr * 1.0, 'name': 'task_modules'}
            ]
            logging.info(f"Stage 3 LR: backbone_mid_high={base_lr*0.6:.2e}, task={base_lr*1.0:.2e}")
            
        elif stage == 4:  # Stage 4 (Epoch 61-80): 全部解冻，分层学习率
            all_embed_params = bert_embed_params + bert_other_params + vit_embed_params + vit_other_params
            all_low_params = bert_low_params + vit_low_params
            all_mid_params = bert_mid_params + vit_mid_params
            all_high_params = bert_high_params + vit_high_params
            
            param_groups = [
                {'params': all_embed_params, 'lr': base_lr * 0.05, 'name': 'backbone_embed'},
                {'params': all_low_params, 'lr': base_lr * 0.2, 'name': 'backbone_low'},
                {'params': all_mid_params, 'lr': base_lr * 0.4, 'name': 'backbone_mid'},
                {'params': all_high_params, 'lr': base_lr * 0.6, 'name': 'backbone_high'},
                {'params': task_params, 'lr': base_lr * 0.8, 'name': 'task_modules'}
            ]
            logging.info(f"Stage 4 LR: embed={base_lr*0.05:.2e}, low={base_lr*0.2:.2e}, "
                        f"mid={base_lr*0.4:.2e}, high={base_lr*0.6:.2e}, task={base_lr*0.8:.2e}")
            
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1-4.")
        
        # 统计每组参数数量
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            logging.info(f"  {group['name']}: {num_params:,} parameters")
        
        return param_groups

    def build_optimizer(self, model, stage=1):
        # 创建优化器，根据训练阶段使用不同的参数组
        param_groups = self.get_param_groups_with_diff_lr(model, self.args.lr, stage)
        
        # 根据配置选择优化器
        optimizer_type = self.args.optimizer.lower()
        if optimizer_type == 'adamw':
            return torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
        elif optimizer_type == 'adam':
            return torch.optim.Adam(param_groups, eps=1e-8, betas=(0.9, 0.999))
        else:
            logging.warning(f"Unknown optimizer {self.args.optimizer}, defaulting to AdamW")
            return torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))

    def build_scheduler(self, optimizer):
        # 创建学习率调度器
        if self.args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.epochs, eta_min=1e-6
            )
        return WarmupMultiStepLR(
            optimizer, self.args.milestones, gamma=0.1,
            warmup_factor=0.1, warmup_iters=self.args.warmup_step
        )

    def load_param(self, model, trained_path):
        # 加载预训练模型参数
        param_dict = torch.load(trained_path, map_location=self.device, weights_only=False)
        param_dict = param_dict.get('state_dict', param_dict.get('model', param_dict))
        model_dict = model.state_dict()
        for i in param_dict:
            # 这里的形状检查非常关键：顺序训练不同数据集时，ID分类器(id_classifier)的维度不同
            # 形状检查可以确保不加载形状不匹配的分类头，从而让新阶段从随机初始化的分类器开始
            if i in model_dict and model_dict[i].shape == param_dict[i].shape:
                model_dict[i] = param_dict[i]
        model.load_state_dict(model_dict, strict=False)
        logging.info(f"Loaded pretrained weights from {trained_path}")

    def run(self):
        # 执行训练和评估流程
        args = self.args
        config = self.config  # 现在config为空字典，但我们直接使用args中的参数
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 创建两个logger：一个用于详细日志，一个用于重要信息显示
        detailed_logger = logging.getLogger('detailed')
        detailed_logger.setLevel(logging.DEBUG)

        # 清除已有处理器
        for handler in detailed_logger.handlers[:]:
            detailed_logger.removeHandler(handler)

        # 为detailed_logger添加处理器，写入数据集特定的日志文件
        # 获取数据集名称以确定日志文件位置
        if hasattr(args, 'dataset_configs') and args.dataset_configs:
            dataset_full_name = args.dataset_configs[0]['name'].lower()
            if 'cuhk' in dataset_full_name:
                dataset_dir_name = 'cuhk'
            elif 'rstp' in dataset_full_name:
                dataset_dir_name = 'rstp'
            elif 'icfg' in dataset_full_name:
                dataset_dir_name = 'icfg'
            else:
                dataset_dir_name = dataset_full_name
        else:
            dataset_dir_name = 'unknown'

        # 确保使用 log 目录而不是 logs 目录
        # 获取项目根目录
        script_dir = Path(__file__).parent  # scripts/
        project_root = script_dir.parent   # 项目根目录
        log_base_dir = project_root / 'log'
        dataset_log_dir = log_base_dir / dataset_dir_name
        dataset_log_dir.mkdir(parents=True, exist_ok=True)

        # 创建数据集特定的主要日志文件
        main_log_file = dataset_log_dir / 'log.txt'

        # 为detailed_logger添加文件处理器
        file_handler = logging.FileHandler(main_log_file, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        detailed_logger.addHandler(file_handler)

        # 控制台logger - 只显示重要信息
        console_logger = logging.getLogger('console')
        console_logger.setLevel(logging.INFO)

        # 清除已有处理器
        for handler in console_logger.handlers[:]:
            console_logger.removeHandler(handler)

        # 控制台处理器 - 只显示重要信息
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('%(message)s')  # 简化格式，只显示消息
        console_handler.setFormatter(console_formatter)
        console_logger.addHandler(console_handler)

        # 设置基础日志配置，将调试信息写入数据集特定的日志文件
        # 获取项目根目录
        script_dir = Path(__file__).parent  # scripts/
        project_root = script_dir.parent   # 项目根目录
        log_base_dir = project_root / 'log'
        dataset_log_dir = log_base_dir / dataset_dir_name
        dataset_log_dir.mkdir(parents=True, exist_ok=True)

        # 创建数据集特定的调试日志文件
        debug_log_file = dataset_log_dir / 'debug.txt'

        # 设置基础日志配置，将调试信息写入数据集特定的调试日志文件
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(debug_log_file, mode='a', encoding='utf-8'),  # 写入调试日志文件
                logging.StreamHandler(sys.stdout)  # 同时输出到控制台（可以移除此项以仅写入文件）
            ]
        )

        # 但为了满足要求，只将调试信息写入文件，不输出到控制台
        # 重新配置，只写入文件
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        file_handler = logging.FileHandler(debug_log_file, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
        logging.root.setLevel(logging.DEBUG)

        # 构建数据集
        console_logger.info("Building dataset...")
        data_builder = DataBuilder(args, is_distributed=False)
        args.num_classes = data_builder.get_num_classes()
        detailed_logger.info(f"Set num_classes = {args.num_classes}")

        console_logger.info("Loading training data...")
        train_loader, _ = data_builder.build_data(is_train=True)
        console_logger.info("Loading query and gallery data...")
        query_loader, gallery_loader = data_builder.build_data(is_train=False)
        console_logger.info(f"Train data size: {len(train_loader.dataset.data)}")
        console_logger.info(f"Query data size: {len(query_loader.dataset.data)}")
        console_logger.info(f"Gallery data size: {len(gallery_loader.dataset.data)}")
        detailed_logger.info(f"Train data size: {len(train_loader.dataset.data)}")
        detailed_logger.info(f"Query data size: {len(query_loader.dataset.data)}")


        # 构建模型配置字典
        model_config = {
            'bert_base_path': args.bert_base_path,
            'vit_pretrained': args.vit_pretrained,
            'vision_backbone': args.vision_backbone,
            'vim_pretrained': args.vim_pretrained,
            'num_classes': args.num_classes,
            'disentangle_type': args.disentangle_type,
            'gs3': {
                'num_heads': args.gs3_num_heads,
                'd_state': args.gs3_d_state,
                'd_conv': args.gs3_d_conv,
                'dropout': args.gs3_dropout
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

        # 初始化模型
        console_logger.info("Initializing model...")
        model = Model(net_config=model_config).to(self.device)
        detailed_logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

        # 记录训练开始信息
        self.monitor.log_training_start(model, args)

        # [修改点 2] 如果指定了 finetune-from，则在构建优化器之前加载权重
        if args.finetune_from:
            detailed_logger.info(f"Finetuning: Loading checkpoint from {args.finetune_from}")
            console_logger.info(f"Loading checkpoint from {args.finetune_from}")
            self.load_param(model, args.finetune_from)

        # 【方案B：渐进解冻策略 - Stage 1】Epoch 1-10: 解冻ViT后4层 (关键修复!)
        console_logger.info("=" * 70)
        console_logger.info("🔓 Progressive Unfreezing Strategy - Solution B")
        console_logger.info("=" * 70)
        console_logger.info("Stage 1 (Epoch 1-10): Unfreeze ViT last 4 layers (layer 8-11)")
        console_logger.info("                      Keep ViT first 8 layers frozen (layer 0-7)")
        console_logger.info("                      Keep all BERT layers frozen")
        console_logger.info("")
        console_logger.info("🎯 Key Fix: Let CLS loss backpropagate through ViT!")
        console_logger.info("   - id_embeds will update via ViT gradients")
        console_logger.info("   - Classification head can learn properly")
        console_logger.info("=" * 70)
        
        # 冻结策略：只解冻ViT后4层，其余全部冻结
        self.freeze_bert_layers(model, unfreeze_from_layer=None)  # 全部冻结
        self.freeze_vit_layers(model, unfreeze_from_layer=8)      # 解冻layer 8-11

        # 构建优化器和调度器
        console_logger.info("Building optimizer and scheduler...")
        optimizer = self.build_optimizer(model, stage=1)
        lr_scheduler = self.build_scheduler(optimizer)
        
        # 【关键验证】确保freeze生效
        console_logger.info("\n🔍 Verifying freeze status after optimizer build...")
        self.verify_freeze_status(model)

        # 训练模型
        console_logger.info("Starting training...")
        trainer = Trainer(model, args, self.monitor, runner=self)  # 传递runner引用以便调用freeze方法
        trainer.train(
            train_loader, optimizer, lr_scheduler, query_loader, gallery_loader, checkpoint_dir=args.logs_dir
        )

        # 评估模型 - 直接使用训练好的模型，不再保存和加载最终检查点
        console_logger.info("Evaluating model...")
        from evaluators.evaluator import Evaluator
        # 直接使用当前训练好的模型进行评估，无需保存和重新加载
        evaluator = Evaluator(model, args=args)
        metrics = evaluator.evaluate(
            query_loader, gallery_loader, query_loader.dataset.data,
            gallery_loader.dataset.data, checkpoint_path=None
        )
        console_logger.info(f"Evaluation Results: {metrics}")
        detailed_logger.info(f"Evaluation Results: {metrics}")
        # 记录训练结束信息
        self.monitor.log_training_end(metrics)

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