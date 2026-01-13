# test_visualization_simple.py
import sys
from pathlib import Path
import torch
import yaml
import logging
import argparse

ROOT_DIR = Path(__file__).parent
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # 加载配置
    with open('configs/config_cuhk_pedes.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建参数对象
    args = argparse.Namespace(
        root='data',
        batch_size=128,
        workers=4,
        height=224,
        width=224,
        fp16=True,
        num_classes=8000,
        # 直接从配置文件使用 dataset_configs
        dataset_configs=config['dataset_configs']
    )
    
    logging.info(f"Root directory: {args.root}")
    logging.info(f"Dataset configs: {args.dataset_configs}")
    
    # 验证文件存在
    for dc in args.dataset_configs:
        if dc['name'] == 'CUHK-PEDES':
            json_path = Path(args.root) / dc['json_file']
            logging.info(f"Checking: {json_path}")
            logging.info(f"  Exists: {json_path.exists()}")
            if json_path.exists():
                logging.info(f"  Is file: {json_path.is_file()}")
                break
    
    # 创建数据加载器
    logging.info("Building data loaders...")
    data_builder = DataBuilder(args, is_distributed=False)
    
    try:
        query_loader, gallery_loader = data_builder.build_data(is_train=False)
        logging.info(f"✓ Gallery loader created with {len(gallery_loader.dataset)} samples")
    except Exception as e:
        logging.error(f"Failed to build data loaders: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载模型
    logging.info("Loading model...")
    net_config = config['model'].copy()
    net_config['num_classes'] = args.num_classes
    model = Model(net_config=net_config)
    
    checkpoint_path = 'checkpoints/cuhk_pedes/checkpoint_epoch_final.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
    
    # 调整分类器维度
    if 'id_classifier.weight' in state_dict:
        if state_dict['id_classifier.weight'].shape[0] != net_config['num_classes']:
            logging.info(f"Adapting classifier: {state_dict['id_classifier.weight'].shape[0]} -> {net_config['num_classes']}")
            state_dict['id_classifier.weight'] = state_dict['id_classifier.weight'][:net_config['num_classes'], :]
            state_dict['id_classifier.bias'] = state_dict['id_classifier.bias'][:net_config['num_classes']]
    
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    model.eval()
    
    # 创建评估器
    evaluator = Evaluator_t2i(model, args=args)
    
    # 生成可视化
    logging.info("="*60)
    logging.info("Generating t-SNE visualization...")
    logging.info("="*60)
    
    evaluator.visualize_disentanglement_tsne(
        data_loader=gallery_loader,
        output_dir='logs/visualization_test',
        num_samples=1000,
        num_ids_to_plot=5
    )
    
    logging.info("="*60)
    logging.info("✓ Visualization completed!")
    logging.info("Output files:")
    logging.info("  PNG: logs/visualization_test/visualizations/disentanglement_tsne_academic.png")
    logging.info("  PDF: logs/visualization_test/visualizations/disentanglement_tsne_academic.pdf")
    logging.info("="*60)

if __name__ == '__main__':
    main()