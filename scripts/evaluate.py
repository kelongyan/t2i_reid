# scripts/evaluate.py
import argparse
import ast
import sys
from pathlib import Path
import torch
import logging
import random

# 设置根目录并添加到 Python 路径，确保可以正确导入 models, datasets 等模块
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.model import Model
from datasets.data_builder import DataBuilder
from evaluators.evaluator import Evaluator


class StreamToLogger:
    # 将 sys.stdout 重定向到日志记录器的辅助类，方便在日志中查看原本打印到终端的内容
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


def parse_args():
    # 解析命令行参数：包括路径配置、超参数、模型结构选择等
    parser = argparse.ArgumentParser(description="Evaluate T2I-ReID model")
    parser.add_argument('--root', type=str, default=str(ROOT_DIR / 'data'),
                        help='Root directory of the dataset')
    parser.add_argument('--dataset-configs', nargs='+', type=str, required=True,
                        help='Dataset configurations in JSON format')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--fp16', action='store_true', help='Use mixed precision evaluation')
    parser.add_argument('--logs-dir', type=str, default=str(ROOT_DIR / 'logs'), help='Directory for logs')
    parser.add_argument('--vision-backbone', type=str, default='vim', choices=['vit', 'vim'],
                       help='Vision backbone type: vit or vim')
    parser.add_argument('--vim-pretrained', type=str, default=str(ROOT_DIR / 'pretrained' / 'Vision Mamba' / 'vim_s_midclstok.pth'),
                       help='Path to Vision Mamba model')

    # AH-Net/G-S3 解耦模块相关参数
    parser.add_argument('--gs3-d-state', type=int, default=16, help='State dimension for G-S3')
    parser.add_argument('--gs3-d-conv', type=int, default=4, help='Conv kernel size for G-S3')
    parser.add_argument('--gs3-dropout', type=float, default=0.1, help='Dropout rate for G-S3')
    parser.add_argument('--gs3-img-size', nargs=2, type=int, default=[14, 14], help='Image patch grid size')

    # 融合模块相关参数（RCSM + S-CAG）
    parser.add_argument('--fusion-type', type=str, default='samg_rcsm', help='Type of fusion module')
    parser.add_argument('--fusion-dim', type=int, default=256, help='Fusion module dimension')
    parser.add_argument('--fusion-d-state', type=int, default=16, help='Fusion module d_state')
    parser.add_argument('--fusion-d-conv', type=int, default=4, help='Fusion module d_conv')
    parser.add_argument('--fusion-num-layers', type=int, default=3, help='Fusion module number of layers')
    parser.add_argument('--fusion-output-dim', type=int, default=256, help='Fusion module output dimension')
    parser.add_argument('--fusion-dropout', type=float, default=0.15, help='Fusion module dropout')

    # CLIP 文本编码器路径
    parser.add_argument('--clip-pretrained', type=str, default=str(ROOT_DIR / 'pretrained' / 'clip-vit-base-patch16'),
                       help='Path to CLIP text encoder model')

    args = parser.parse_args()

    # 路径格式规范化
    args.logs_dir = str(Path(args.logs_dir))
    args.root = str(Path(args.root))
    args.checkpoint = str(Path(args.checkpoint))
    args.clip_pretrained = str(Path(args.clip_pretrained))

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {args.checkpoint}")
    
    return args


def main():
    # 执行评估主流程：初始化日志、构建数据加载器、加载模型、执行评估并输出指标
    args = parse_args()
    
    # 设置基础日志配置，仅输出到控制台，不重复创建 log.txt
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger()
    sys.stdout = StreamToLogger(logger, logging.INFO)
    
    # 打印评估配置信息
    args_dict = {k: v for k, v in vars(args).items() if not isinstance(v, (dict, list))}
    logger.info(f"==========\nEvaluation Args: {args_dict}\nDataset Configs: {args.dataset_configs}\n==========")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Starting evaluation")
    
    # 构建数据加载器，并动态获取类别数
    data_builder = DataBuilder(args, is_distributed=False)
    args.num_classes = data_builder.get_num_classes()
    
    query_loader, gallery_loader = data_builder.build_data(is_train=False)
    logger.info(f"Query data size: {len(query_loader.dataset.data)}")
    logger.info(f"Gallery data size: {len(gallery_loader.dataset.data)}")

    # 初始化模型结构
    net_config = {
        'clip_pretrained': args.clip_pretrained,
        'vit_pretrained': str(ROOT_DIR / 'pretrained' / 'vit-base-patch16-224'),
        'vision_backbone': args.vision_backbone,
        'vim_pretrained': args.vim_pretrained,
        'img_size': (getattr(args, 'height', 224), getattr(args, 'width', 224)),
        'num_classes': args.num_classes,
        'gs3': {
            'd_state': args.gs3_d_state,
            'd_conv': args.gs3_d_conv,
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
    
    model = Model(net_config=net_config)

    # 加载已训练好的权重（允许部分匹配，以兼容旧版检查点）
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # 执行检索评估，计算 mAP 和 Rank-k 等指标
    evaluator = Evaluator(model, args=args)
    with torch.amp.autocast('cuda', enabled=args.fp16):
        metrics = evaluator.evaluate(
            query_loader,
            gallery_loader,
            query_loader.dataset.data,
            gallery_loader.dataset.data,
            checkpoint_path=None
        )
    
    # 输出最终评估指标
    logger.info("Evaluation results:")
    logger.info(f"mAP:    {metrics['mAP']:.4f} ({metrics['mAP']*100:.2f}%)")
    logger.info(f"Rank-1: {metrics['rank1']:.4f} ({metrics['rank1']*100:.2f}%)")
    logger.info(f"Rank-5: {metrics['rank5']:.4f} ({metrics['rank5']*100:.2f}%)")
    logger.info(f"Rank-10: {metrics['rank10']:.4f} ({metrics['rank10']*100:.2f}%)")
    logger.info("Evaluation completed")


if __name__ == '__main__':
    main()
