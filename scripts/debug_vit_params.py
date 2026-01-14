"""
调试脚本：打印ViT模型的参数命名格式
用于确定正确的层命名规则
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import ViTModel

def main():
    # 加载ViT模型
    vit_base_path = ROOT_DIR / 'pretrained' / 'vit-base-patch16-224'
    
    if not vit_base_path.exists():
        print(f"Error: ViT model not found at {vit_base_path}")
        return
    
    print(f"Loading ViT from: {vit_base_path}")
    model = ViTModel.from_pretrained(str(vit_base_path))
    
    print("\n" + "="*80)
    print("ViT Model Parameter Names")
    print("="*80)
    
    # 收集不同类型的参数
    embeddings_params = []
    layer_params = {}
    other_params = []
    
    for name, param in model.named_parameters():
        if 'embeddings' in name:
            embeddings_params.append(name)
        elif 'encoder.layer.' in name:
            # BERT风格: encoder.layer.X
            layer_num = int(name.split('encoder.layer.')[1].split('.')[0])
            if layer_num not in layer_params:
                layer_params[layer_num] = []
            layer_params[layer_num].append(name)
        elif 'encoder.layers.' in name:
            # 标准ViT风格: encoder.layers.X
            layer_num = int(name.split('encoder.layers.')[1].split('.')[0])
            if layer_num not in layer_params:
                layer_params[layer_num] = []
            layer_params[layer_num].append(name)
        else:
            other_params.append(name)
    
    # 打印embeddings参数
    if embeddings_params:
        print(f"\n[Embeddings Parameters] ({len(embeddings_params)} params)")
        for name in embeddings_params[:5]:  # 只显示前5个
            print(f"  - {name}")
        if len(embeddings_params) > 5:
            print(f"  ... and {len(embeddings_params) - 5} more")
    
    # 打印layer参数
    if layer_params:
        print(f"\n[Encoder Layers] ({len(layer_params)} layers)")
        for layer_num in sorted(layer_params.keys()):
            params = layer_params[layer_num]
            print(f"\n  Layer {layer_num}: ({len(params)} params)")
            # 显示该层的前3个参数示例
            for name in params[:3]:
                print(f"    - {name}")
            if len(params) > 3:
                print(f"    ... and {len(params) - 3} more")
    
    # 打印其他参数
    if other_params:
        print(f"\n[Other Parameters] ({len(other_params)} params)")
        for name in other_params[:5]:
            print(f"  - {name}")
        if len(other_params) > 5:
            print(f"  ... and {len(other_params) - 5} more")
    
    # 总结
    print("\n" + "="*80)
    print("Summary:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Embeddings: {len(embeddings_params)} parameter groups")
    print(f"  Encoder Layers: {len(layer_params)} layers")
    print(f"  Other: {len(other_params)} parameter groups")
    print("="*80)
    
    # 检测命名格式
    print("\n[Naming Format Detection]")
    has_bert_style = any('encoder.layer.' in name for name, _ in model.named_parameters())
    has_vit_style = any('encoder.layers.' in name for name, _ in model.named_parameters())
    
    if has_bert_style:
        print("  ✓ Detected BERT-style naming: 'encoder.layer.X'")
    if has_vit_style:
        print("  ✓ Detected ViT-style naming: 'encoder.layers.X'")
    if not has_bert_style and not has_vit_style:
        print("  ⚠ Warning: No standard encoder layer naming detected!")
        print("  Please check the 'Other Parameters' section above")

if __name__ == '__main__':
    main()
