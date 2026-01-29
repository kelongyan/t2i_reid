# scripts/train.py
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

# 设置根目录并添加到路径，确保可以导入内部模块
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.serialization import save_checkpoint
from models.model import Model
from datasets.data_builder import DataBuilder
from trainers.trainer import Trainer
from utils.lr_scheduler import WarmupMultiStepLR
from utils.monitor import get_monitor_for_dataset

def configuration():
    # 配置命令行参数，定义训练所需的超参数、路径和模型组件选项
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
    parser.add_argument('--warmup-step', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 60], help='Milestones for LR scheduler')
    parser.add_argument('--epochs', type=int, default=80, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision training')
    
    # 预训练模型路径配置
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

    # 融合模块配置参数
    parser.add_argument('--fusion-type', type=str, default='enhanced_mamba', help='Type of fusion module')
    parser.add_argument('--fusion-dim', type=int, default=256, help='Fusion module dimension')
    parser.add_argument('--fusion-d-state', type=int, default=16, help='Fusion module d_state')
    parser.add_argument('--fusion-d-conv', type=int, default=4, help='Fusion module d_conv')
    parser.add_argument('--fusion-num-layers', type=int, default=2, help='Fusion module number of layers')
    parser.add_argument('--fusion-output-dim', type=int, default=256, help='Fusion module output dimension')
    parser.add_argument('--fusion-dropout', type=float, default=0.1, help='Fusion module dropout')

    # 解耦模块（AH-Net/G-S3）相关参数
    parser.add_argument('--id-projection-dim', type=int, default=768, help='ID projection dimension')
    parser.add_argument('--cloth-projection-dim', type=int, default=768, help='Cloth projection dimension')
    parser.add_argument('--gs3-num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--gs3-d-state', type=int, default=16, help='State dimension for G-S3')
    parser.add_argument('--gs3-d-conv', type=int, default=4, help='Conv kernel size for G-S3')
    parser.add_argument('--gs3-dropout', type=float, default=0.1, help='Dropout rate for G-S3')
    parser.add_argument('--gs3-img-size', nargs=2, type=int, default=[14, 14], help='Image patch grid size')

    # 各项损失函数权重初始值
    parser.add_argument('--loss-info-nce', type=float, default=1.0, help='InfoNCE loss weight')
    parser.add_argument('--loss-id-triplet', type=float, default=2.0, help='ID Triplet loss weight')
    parser.add_argument('--loss-cloth-semantic', type=float, default=0.1, help='Cloth semantic loss weight')
    parser.add_argument('--loss-spatial-orthogonal', type=float, default=0.0, help='Spatial Orthogonal loss weight')
    parser.add_argument('--loss-semantic-alignment', type=float, default=0.0, help='Semantic Alignment loss weight')
    parser.add_argument('--loss-ortho-reg', type=float, default=0.0, help='Query Orthogonality weight')

    # 对抗性解耦损失权重
    parser.add_argument('--loss-adversarial-attr', type=float, default=0.0, help='Adversarial Attribute weight')
    parser.add_argument('--loss-adversarial-domain', type=float, default=0.0, help='Adversarial Domain weight')
    parser.add_argument('--loss-discriminator-attr', type=float, default=0.0, help='Discriminator Attribute weight')
    parser.add_argument('--loss-discriminator-domain', type=float, default=0.0, help='Discriminator Domain weight')

    # 可视化相关配置
    parser.add_argument('--visualization-enabled', action='store_true', help='Enable visualization')
    parser.add_argument('--visualization-save-dir', type=str, default='visualizations', help='Dir to save visualizations')
    parser.add_argument('--visualization-frequency', type=int, default=5, help='Frequency to save visualizations')
    parser.add_argument('--visualization-batch-interval', type=int, default=200, help='Batch interval for visualizations')

    # 优化器与调度器配置
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine', help='Scheduler type')
    parser.add_argument('--finetune-from', type=str, help='Checkpoint path to finetune from')

    args = parser.parse_args()

    # 聚合损失权重到 disentangle 字典
    args.disentangle = {}
    if args.loss_weights:
        args.disentangle['loss_weights'] = ast.literal_eval(args.loss_weights)
    else:
        args.disentangle['loss_weights'] = {
            'info_nce': args.loss_info_nce,
            'cloth_semantic': args.loss_cloth_semantic,
            'id_triplet': args.loss_id_triplet,
            'spatial_orthogonal': args.loss_spatial_orthogonal,
            'semantic_alignment': args.loss_semantic_alignment,
            'ortho_reg': args.loss_ortho_reg,
            'adversarial_attr': args.loss_adversarial_attr,
            'adversarial_domain': args.loss_adversarial_domain,
            'discriminator_attr': args.loss_discriminator_attr,
            'discriminator_domain': args.loss_discriminator_domain
        }
    
    # 聚合可视化配置
    args.visualization = {
        'enabled': args.visualization_enabled,
        'save_dir': args.visualization_save_dir,
        'frequency': args.visualization_frequency,
        'batch_interval': args.visualization_batch_interval
    }

    # 处理多数据集配置
    if args.dataset_configs:
        dataset_configs = []
        for cfg in args.dataset_configs:
            parsed = ast.literal_eval(cfg)
            dataset_configs.extend(parsed if isinstance(parsed, list) else [parsed])
        args.dataset_configs = dataset_configs
    else:
        args.dataset_configs = [{
            'name': 'CUHK-PEDES',
            'root': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES'),
            'json_file': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_all.json')
        }]

    # 规范化所有路径
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
    # 训练运行器：管理模型生命周期、学习率调度、参数冻结以及日志监控
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16) if self.device.type == 'cuda' else None
        self.args.original_logs_dir = args.logs_dir

        # 数据集名称映射，用于获取对应的监控器
        if hasattr(args, 'dataset_configs') and args.dataset_configs:
            dataset_full_name = args.dataset_configs[0]['name'].lower()
            if 'cuhk' in dataset_full_name: dataset_name = 'cuhk_pedes'
            elif 'rstp' in dataset_full_name: dataset_name = 'rstp'
            elif 'icfg' in dataset_full_name: dataset_name = 'icfg'
            else: dataset_name = dataset_full_name
        else:
            dataset_name = 'unknown'
        
        project_root = Path(__file__).parent.parent
        log_base_dir = str(project_root / 'log')
        self.monitor = get_monitor_for_dataset(dataset_name, log_base_dir)

    def verify_freeze_status(self, model):
        # 验证并打印模型各部分的冻结/可训练参数状态
        vit_frozen = sum(p.numel() for n, p in model.named_parameters() if 'visual_encoder' in n and not p.requires_grad)
        vit_total = sum(p.numel() for n, p in model.named_parameters() if 'visual_encoder' in n)
        vit_trainable = vit_total - vit_frozen
        
        text_frozen = sum(p.numel() for n, p in model.named_parameters() if 'text_encoder' in n and not p.requires_grad)
        text_total = sum(p.numel() for n, p in model.named_parameters() if 'text_encoder' in n)
        text_trainable = text_total - text_frozen
        
        adapter_trainable = sum(p.numel() for n, p in model.named_parameters() if 'text_proj' in n and p.requires_grad)
        adapter_total = sum(p.numel() for n, p in model.named_parameters() if 'text_proj' in n)
        
        task_trainable = sum(p.numel() for n, p in model.named_parameters() 
                            if 'visual_encoder' not in n and 'text_encoder' not in n and 'text_proj' not in n and p.requires_grad)
        
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logging.info("=" * 70)
        logging.info("📊 模型冻结状态验证")
        logging.info("=" * 70)
        logging.info(f"视觉骨干: {vit_trainable:,}/{vit_total:,} 可训练 ({100*vit_trainable/(vit_total+1):.1f}%)")
        logging.info(f"文本骨干 (CLIP): {text_trainable:,}/{text_total:,} 可训练 ({100*text_trainable/(text_total+1):.1f}%)")
        logging.info(f"文本适配器: {adapter_trainable:,}/{adapter_total:,} 可训练")
        logging.info(f"任务特定模块: {task_trainable:,} 可训练")
        logging.info(f"总计可训练: {total_trainable:,}/{total_params:,}")
        logging.info("=" * 70)
        return {}
    
    def freeze_text_layers(self, model, unfreeze_from_layer=None):
        # 冻结 CLIP 文本编码器。unfreeze_from_layer 定义从哪一层开始解冻。
        for name, param in model.named_parameters():
            if 'text_encoder' in name:
                param.requires_grad = False
            if 'text_proj' in name: # 适配器层始终可训练
                param.requires_grad = True
                
        if unfreeze_from_layer is not None:
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if 'text_encoder' in name:
                    if 'layers.' in name:
                        try:
                            layer_num = int(name.split('layers.')[1].split('.')[0])
                            if layer_num >= unfreeze_from_layer:
                                param.requires_grad = True
                                unfrozen_count += 1
                        except: pass
                    if 'final_layer_norm' in name:
                        param.requires_grad = True
                        unfrozen_count += 1
                    if unfreeze_from_layer == 0 and ('embeddings' in name):
                        param.requires_grad = True
                        unfrozen_count += 1
            logging.info(f"CLIP: 已解冻从第 {unfreeze_from_layer} 层起的参数 (共 {unfrozen_count} 组)")
        else:
            logging.info(f"CLIP: 所有层已冻结 (适配器保持可训练)")

    def freeze_vit_layers(self, model, unfreeze_from_layer=None):
        # 冻结视觉编码器（ViT/Vim）。处理逻辑与文本分支类似。
        is_vim = getattr(model, 'vision_backbone_type', 'vit') == 'vim'
        total_layers = 24 if is_vim else 12
        
        target_start_layer = None
        if unfreeze_from_layer is not None:
            if unfreeze_from_layer == 0: target_start_layer = 0
            elif unfreeze_from_layer == 8: target_start_layer = total_layers - 4 
            elif unfreeze_from_layer == 4: target_start_layer = total_layers - 8
            else: target_start_layer = unfreeze_from_layer if not is_vim else unfreeze_from_layer * 2

        for name, param in model.named_parameters():
            if 'visual_encoder' in name:
                param.requires_grad = False
                if 'visual_proj' in name: param.requires_grad = True
        
        if target_start_layer is not None:
            for name, param in model.named_parameters():
                if 'visual_encoder' in name:
                    if target_start_layer == 0 and any(k in name for k in ['embeddings', 'patch_embed', 'cls_token', 'pos_embed']):
                        param.requires_grad = True
                        continue
                    layer_num = -1
                    if is_vim and 'layers.' in name:
                        try: layer_num = int(name.split('layers.')[1].split('.')[0])
                        except: pass
                    elif not is_vim and 'encoder.layer.' in name:
                        try: layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                        except: pass
                    if layer_num != -1 and layer_num >= target_start_layer:
                        param.requires_grad = True
                    if target_start_layer == 0 or target_start_layer < total_layers:
                        if is_vim and 'norm_f' in name: param.requires_grad = True
                        elif not is_vim and ('layernorm' in name or 'pooler' in name): param.requires_grad = True
            logging.info(f"{'Vim' if is_vim else 'ViT'}: 已解冻从第 {target_start_layer}/{total_layers} 层起的参数")
        else:
            logging.info(f"{'Vim' if is_vim else 'ViT'}: 所有层已冻结")

    def get_param_groups_with_diff_lr(self, model, base_lr, stage):
        # 实现分层学习率策略：敏感的骨干网络使用极低学习率，任务相关模块使用全速学习率
        is_vim = getattr(model, 'vision_backbone_type', 'vit') == 'vim'
        clip_params, text_adapter_params, vit_low_params, vit_mid_params, vit_high_params, vit_embed_params, task_params = [], [], [], [], [], [], []
        
        for name, param in model.named_parameters():
            if not param.requires_grad: continue
            if 'text_encoder' in name: clip_params.append(param)
            elif 'text_proj' in name: text_adapter_params.append(param)
            elif 'visual_encoder' in name:
                if any(k in name for k in ['embeddings', 'patch_embed', 'cls_token', 'pos_embed']): vit_embed_params.append(param)
                else:
                    layer_num = -1
                    if is_vim and 'layers.' in name:
                        try: layer_num = int(name.split('layers.')[1].split('.')[0])
                        except: pass
                    elif not is_vim and 'encoder.layer.' in name:
                        try: layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
                        except: pass
                    if layer_num != -1:
                        if is_vim:
                            if layer_num < 8: vit_low_params.append(param)
                            elif layer_num < 16: vit_mid_params.append(param)
                            else: vit_high_params.append(param)
                        else:
                            if layer_num < 4: vit_low_params.append(param)
                            elif layer_num < 8: vit_mid_params.append(param)
                            else: vit_high_params.append(param)
                    elif any(k in name for k in ['layernorm', 'pooler', 'norm_f']): vit_high_params.append(param)
                    else: task_params.append(param)
            else: task_params.append(param)
        
        clip_lr_ratio = 0.05
        groups = [
            {'params': task_params, 'lr': base_lr, 'name': 'task_modules'},
            {'params': text_adapter_params, 'lr': base_lr, 'name': 'text_adapter'}
        ]
        if vit_embed_params: groups.append({'params': vit_embed_params, 'lr': base_lr * 0.01, 'name': 'vit_embed'})
        if vit_low_params: groups.append({'params': vit_low_params, 'lr': base_lr * 0.05, 'name': 'vit_low'})
        if vit_mid_params: groups.append({'params': vit_mid_params, 'lr': base_lr * 0.1, 'name': 'vit_mid'})
        if vit_high_params: groups.append({'params': vit_high_params, 'lr': base_lr * 0.2, 'name': 'vit_high'})
        if clip_params: groups.append({'params': clip_params, 'lr': base_lr * clip_lr_ratio, 'name': 'clip_encoder'})
        return groups

    def build_optimizer(self, model, stage=1):
        # 根据配置构建优化器，支持 AdamW 和 Adam
        param_groups = self.get_param_groups_with_diff_lr(model, self.args.lr, stage)
        optimizer_type = self.args.optimizer.lower()
        if optimizer_type == 'adamw':
            return torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)
        elif optimizer_type == 'adam':
            return torch.optim.Adam(param_groups, eps=1e-8, betas=(0.9, 0.999))
        return torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))

    def build_scheduler(self, optimizer):
        # 构建学习率调度器，支持余弦退火和带预热的多步下降
        if self.args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=1e-6)
        return WarmupMultiStepLR(optimizer, self.args.milestones, gamma=0.1, warmup_factor=0.1, warmup_iters=self.args.warmup_step)

    def load_param(self, model, trained_path):
        # 加载预训练参数，并过滤维度不匹配的层
        param_dict = torch.load(trained_path, map_location=self.device, weights_only=False)
        param_dict = param_dict.get('state_dict', param_dict.get('model', param_dict))
        model_dict = model.state_dict()
        for i in param_dict:
            if i in model_dict and model_dict[i].shape == param_dict[i].shape:
                model_dict[i] = param_dict[i]
        model.load_state_dict(model_dict, strict=False)
        logging.info(f"已从 {trained_path} 加载预训练权重")

    def run(self):
        # 训练主循环入口：初始化日志系统、构建数据流、实例化模型、应用冻结策略并启动训练
        args = self.args
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        dataset_log_dir = self.monitor.dataset_log_dir

        # 1. 详细日志（文件）
        detailed_logger = logging.getLogger('detailed')
        detailed_logger.setLevel(logging.DEBUG)
        detailed_logger.propagate = False
        file_handler = logging.FileHandler(dataset_log_dir / 'log.txt', mode='a', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        detailed_logger.addHandler(file_handler)

        # 2. 调试日志（文件）
        root_file_handler = logging.FileHandler(dataset_log_dir / 'debug.txt', mode='a', encoding='utf-8')
        root_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.root.addHandler(root_file_handler)
        logging.root.setLevel(logging.DEBUG)

        # 3. 控制台日志（终端）
        console_logger = logging.getLogger('console')
        console_logger.setLevel(logging.INFO)
        console_logger.propagate = False
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        console_logger.addHandler(console_handler)

        console_logger.info("正在构建数据集...")
        data_builder = DataBuilder(args, is_distributed=False)
        args.num_classes = data_builder.get_num_classes()
        
        console_logger.info("正在加载训练与测试数据...")
        train_loader, _ = data_builder.build_data(is_train=True)
        query_loader, gallery_loader = data_builder.build_data(is_train=False)

        # 初始化模型架构（CLIP + Vim）
        model_config = {
            'clip_pretrained': args.clip_pretrained,
            'vit_pretrained': args.vit_pretrained,
            'vision_backbone': args.vision_backbone,
            'vim_pretrained': args.vim_pretrained,
            'img_size': (args.height, args.width),
            'num_classes': args.num_classes,
            'gs3': {
                'num_heads': args.gs3_num_heads,
                'd_state': args.gs3_d_state,
                'd_conv': args.gs3_d_conv,
                'dropout': args.gs3_dropout,
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

        console_logger.info("正在初始化模型结构...")
        model = Model(net_config=model_config).to(self.device)
        if args.finetune_from: self.load_param(model, args.finetune_from)

        # 应用预热冻结策略，防止训练初期梯度不稳定
        console_logger.info("=" * 60)
        console_logger.info("🚀 训练启动: CLIP + Vim 混合架构")
        console_logger.info("   ❄️  策略: 初始阶段冻结骨干网络 (Epoch 0-5)")
        console_logger.info("=" * 60)
        
        self.freeze_text_layers(model, unfreeze_from_layer=None)
        self.freeze_vit_layers(model, unfreeze_from_layer=None)

        optimizer = self.build_optimizer(model, stage=1)
        lr_scheduler = self.build_scheduler(optimizer)
        self.verify_freeze_status(model)

        # 启动 Trainer 进行正式训练
        trainer = Trainer(model, args, self.monitor, runner=self)
        trainer.train(train_loader, optimizer, lr_scheduler, query_loader, gallery_loader, checkpoint_dir=str(dataset_log_dir))


if __name__ == '__main__':
    args, config = configuration()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    runner = Runner(args, config)
    runner.run()
