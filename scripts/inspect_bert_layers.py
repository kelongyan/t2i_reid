"""
BERT层结构检查脚本
检查BERT模型的层命名和参数分布，用于设计逐层解冻策略
"""
import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from transformers import BertModel


def inspect_bert_layers():
    """检查BERT模型的层结构"""
    print("=" * 80)
    print("BERT模型层结构检查")
    print("=" * 80)
    
    # 加载BERT模型
    bert_path = ROOT_DIR / 'pretrained' / 'bert-base-uncased'
    
    if not bert_path.exists():
        print(f"\n❌ 错误：BERT模型路径不存在: {bert_path}")
        print("请确保预训练模型已下载到正确位置")
        return
    
    print(f"\n正在加载BERT模型: {bert_path}")
    model = BertModel.from_pretrained(str(bert_path))
    print("✓ 模型加载成功\n")
    
    # 统计所有参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"BERT总参数量: {total_params:,}\n")
    
    # 按层分组统计
    print("=" * 80)
    print("按模块分组的参数统计")
    print("=" * 80)
    
    # 1. Embeddings层
    print("\n【1. Embeddings层】")
    embeddings_params = {}
    embeddings_total = 0
    for name, param in model.named_parameters():
        if name.startswith('embeddings'):
            embeddings_params[name] = param.numel()
            embeddings_total += param.numel()
    
    print(f"  总参数量: {embeddings_total:,} ({embeddings_total/total_params*100:.2f}%)")
    print(f"  参数详情:")
    for name, count in sorted(embeddings_params.items()):
        print(f"    - {name}: {count:,}")
    
    # 2. Encoder层（12层）
    print("\n【2. Encoder层（12层）】")
    layer_params = {}
    for i in range(12):
        layer_prefix = f'encoder.layer.{i}'
        layer_total = 0
        layer_details = {}
        
        for name, param in model.named_parameters():
            if name.startswith(layer_prefix):
                layer_details[name] = param.numel()
                layer_total += param.numel()
        
        layer_params[i] = {
            'total': layer_total,
            'details': layer_details
        }
    
    # 打印每层的统计信息
    for i in range(12):
        layer_info = layer_params[i]
        print(f"\n  Layer {i}:")
        print(f"    总参数量: {layer_info['total']:,} ({layer_info['total']/total_params*100:.2f}%)")
        print(f"    参数详情: ({len(layer_info['details'])} 个参数组)")
        
        # 按子模块分组显示
        attention_params = {k: v for k, v in layer_info['details'].items() if 'attention' in k}
        intermediate_params = {k: v for k, v in layer_info['details'].items() if 'intermediate' in k}
        output_params = {k: v for k, v in layer_info['details'].items() if 'output' in k and 'attention' not in k}
        
        if attention_params:
            attn_total = sum(attention_params.values())
            print(f"      * Attention: {attn_total:,}")
            for name in sorted(attention_params.keys())[:3]:  # 只显示前3个
                short_name = name.replace(f'encoder.layer.{i}.', '')
                print(f"          - {short_name}: {attention_params[name]:,}")
            if len(attention_params) > 3:
                print(f"          ... (共 {len(attention_params)} 个参数)")
        
        if intermediate_params:
            inter_total = sum(intermediate_params.values())
            print(f"      * Intermediate: {inter_total:,}")
            for name in sorted(intermediate_params.keys()):
                short_name = name.replace(f'encoder.layer.{i}.', '')
                print(f"          - {short_name}: {intermediate_params[name]:,}")
        
        if output_params:
            out_total = sum(output_params.values())
            print(f"      * Output: {out_total:,}")
            for name in sorted(output_params.keys()):
                short_name = name.replace(f'encoder.layer.{i}.', '')
                print(f"          - {short_name}: {output_params[name]:,}")
    
    # 3. Pooler层
    print("\n【3. Pooler层】")
    pooler_params = {}
    pooler_total = 0
    for name, param in model.named_parameters():
        if name.startswith('pooler'):
            pooler_params[name] = param.numel()
            pooler_total += param.numel()
    
    print(f"  总参数量: {pooler_total:,} ({pooler_total/total_params*100:.2f}%)")
    print(f"  参数详情:")
    for name, count in sorted(pooler_params.items()):
        print(f"    - {name}: {count:,}")
    
    # 4. 完整的参数列表（用于精确匹配）
    print("\n" + "=" * 80)
    print("完整参数列表（用于代码实现）")
    print("=" * 80)
    
    print("\n所有参数名称（按层分组）:\n")
    
    # Embeddings
    print("# Embeddings层")
    for name, param in model.named_parameters():
        if name.startswith('embeddings'):
            print(f"  {name}: shape={list(param.shape)}")
    
    # Encoder层（分组显示）
    print("\n# Encoder层（12层）")
    for i in range(12):
        print(f"\n## Layer {i}")
        layer_prefix = f'encoder.layer.{i}'
        layer_names = [name for name, _ in model.named_parameters() if name.startswith(layer_prefix)]
        for name in sorted(layer_names):
            param = dict(model.named_parameters())[name]
            print(f"  {name}: shape={list(param.shape)}")
    
    # Pooler
    print("\n# Pooler层")
    for name, param in model.named_parameters():
        if name.startswith('pooler'):
            print(f"  {name}: shape={list(param.shape)}")
    
    # 5. 生成冻结/解冻代码模板
    print("\n" + "=" * 80)
    print("BERT逐层解冻策略建议")
    print("=" * 80)
    
    print("\n基于BERT-Base的12层结构，建议的逐层解冻策略：\n")
    
    strategies = [
        {
            'stage': 'Stage 1 (Epoch 1-5)',
            'description': '冻结所有BERT层',
            'freeze': 'all',
            'unfreeze': []
        },
        {
            'stage': 'Stage 2 (Epoch 6-20)',
            'description': '解冻BERT后4层 (layer 8-11)',
            'freeze': [0, 1, 2, 3, 4, 5, 6, 7],
            'unfreeze': [8, 9, 10, 11]
        },
        {
            'stage': 'Stage 3 (Epoch 21-40)',
            'description': '解冻BERT后8层 (layer 4-11)',
            'freeze': [0, 1, 2, 3],
            'unfreeze': [4, 5, 6, 7, 8, 9, 10, 11]
        },
        {
            'stage': 'Stage 4 (Epoch 41-60)',
            'description': '解冻所有BERT层',
            'freeze': [],
            'unfreeze': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        },
        {
            'stage': 'Stage 5 (Epoch 61-80)',
            'description': '全局微调（降低学习率）',
            'freeze': [],
            'unfreeze': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        }
    ]
    
    for strategy in strategies:
        print(f"{strategy['stage']}: {strategy['description']}")
        if strategy['freeze'] == 'all':
            print(f"  冻结: 所有层")
        elif strategy['freeze']:
            print(f"  冻结: Layer {strategy['freeze']}")
        else:
            print(f"  冻结: 无")
        
        if strategy['unfreeze']:
            print(f"  解冻: Layer {strategy['unfreeze']}")
        print()
    
    # 6. 代码实现建议
    print("=" * 80)
    print("代码实现参考")
    print("=" * 80)
    
    print("""
def freeze_bert_layers(model, unfreeze_from_layer=None):
    \"\"\"
    冻结/解冻BERT的指定层
    
    Args:
        model: 完整的T2I-ReID模型
        unfreeze_from_layer: 
            - None: 冻结所有BERT层
            - 8: 解冻layer 8-11（后4层）
            - 4: 解冻layer 4-11（后8层）
            - 0: 解冻所有12层
    \"\"\"
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
                if 'encoder.layer.' in name:
                    try:
                        # 提取层号
                        parts = name.split('encoder.layer.')[1].split('.')
                        layer_num = int(parts[0])
                        
                        # 验证层号范围 (0-11)
                        if 0 <= layer_num <= 11 and layer_num >= unfreeze_from_layer:
                            param.requires_grad = True
                            unfrozen_count += 1
                    except (IndexError, ValueError):
                        continue
                
                # 解冻pooler（如果完全解冻）
                if unfreeze_from_layer == 0 and 'pooler' in name:
                    param.requires_grad = True
                    unfrozen_count += 1
        
        print(f"BERT: Unfrozen {unfrozen_count} parameter groups from layer {unfreeze_from_layer}")
    else:
        print(f"BERT: All layers frozen")
    
    # 步骤3: 统计并记录可训练参数
    bert_trainable = sum(p.numel() for n, p in model.named_parameters() 
                        if p.requires_grad and 'text_encoder' in n)
    bert_total = sum(p.numel() for n, p in model.named_parameters() if 'text_encoder' in n)
    
    print(f"BERT: {bert_trainable:,}/{bert_total:,} trainable ({100*bert_trainable/bert_total:.1f}%)")
""")
    
    print("\n" + "=" * 80)
    print("检查完成！")
    print("=" * 80)


if __name__ == '__main__':
    try:
        inspect_bert_layers()
    except Exception as e:
        print(f"\n❌ 检查过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
