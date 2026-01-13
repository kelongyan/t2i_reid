# scripts/visualize_tsne.py
"""
专门用于生成 t-SNE 可视化的脚本
"""
import sys
from pathlib import Path
import torch
import yaml
import logging
import argparse

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 添加当前目录到Python路径，确保能找到所有模块
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from models.model import Model
from datasets.data_builder import DataBuilder
from evaluators.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Generate t-SNE visualization for T2I-ReID model")
    parser.add_argument('--config', default='configs/config_cuhk_pedes.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='CUHK-PEDES',
                        choices=['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid'],
                        help='Dataset name')
    parser.add_argument('--output-dir', type=str, default='logs/visualization',
                        help='Output directory for visualization')
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Number of samples to visualize')
    parser.add_argument('--num-ids', type=int, default=5,
                        help='Number of identity IDs to highlight')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 加载配置文件
    logger.info(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 筛选出指定的数据集
    all_datasets = config['dataset_configs']
    dataset_config = None
    for dc in all_datasets:
        if dc['name'] == args.dataset:
            dataset_config = dc
            break
    
    if not dataset_config:
        logger.error(f"Dataset {args.dataset} not found in config file")
        return
    
    logger.info(f"Using dataset: {args.dataset}")
    logger.info(f"Dataset config: {dataset_config}")
    
    # 创建参数命名空间
    eval_args = argparse.Namespace(
        root='data',
        batch_size=args.batch_size,
        workers=args.workers,
        height=config.get('height', 224),
        width=config.get('width', 224),
        fp16=config.get('fp16', True),
        num_classes=config['model']['num_classes'],
        dataset_configs=[dataset_config]
    )
    
    # 创建数据加载器
    logger.info("Building data loaders...")
    data_builder = DataBuilder(eval_args, is_distributed=False)
    
    try:
        query_loader, gallery_loader = data_builder.build_data(is_train=False)
        logger.info(f"✓ Successfully created data loaders")
        logger.info(f"  Query: {len(query_loader.dataset)} samples")
        logger.info(f"  Gallery: {len(gallery_loader.dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to build data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载模型
    logger.info("Loading model...")
    net_config = config['model'].copy()
    net_config['num_classes'] = eval_args.num_classes
    model = Model(net_config=net_config)
    
    # 加载检查点
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cuda', weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    
    # 调整分类器维度
    if 'id_classifier.weight' in state_dict:
        if state_dict['id_classifier.weight'].shape[0] != net_config['num_classes']:
            logger.info(f"Adapting classifier dimensions: "
                       f"{state_dict['id_classifier.weight'].shape[0]} -> {net_config['num_classes']}")
            state_dict['id_classifier.weight'] = state_dict['id_classifier.weight'][:net_config['num_classes'], :]
            state_dict['id_classifier.bias'] = state_dict['id_classifier.bias'][:net_config['num_classes']]
    
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()
    logger.info("✓ Model loaded successfully")
    
    # 创建评估器
    evaluator = Evaluator_t2i(model, args=eval_args)
    
    # 生成可视化
    logger.info("="*70)
    logger.info(f"Generating t-SNE visualization for {args.dataset}...")
    logger.info(f"  Samples: {args.num_samples}")
    logger.info(f"  Highlighted IDs: {args.num_ids}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("="*70)
    
    evaluator.visualize_disentanglement_tsne(
        data_loader=gallery_loader,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        num_ids_to_plot=args.num_ids
    )
    
    logger.info("="*70)
    logger.info("✓ Visualization completed successfully!")
    logger.info(f"Output files saved to: {args.output_dir}/visualizations/")
    logger.info("  - disentanglement_tsne_academic.png (raster image)")
    logger.info("  - disentanglement_tsne_academic.pdf (vector image for publication)")
    logger.info("="*70)

if __name__ == '__main__':
    main()