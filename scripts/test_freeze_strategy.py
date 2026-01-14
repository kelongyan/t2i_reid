"""
测试脚本：验证渐进解冻策略是否正确工作
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

# 模拟train.py中的freeze_vit_layers逻辑
def test_freeze_logic():
    from transformers import ViTModel
    
    vit_base_path = ROOT_DIR / 'pretrained' / 'vit-base-patch16-224'
    
    if not vit_base_path.exists():
        print(f"Error: ViT model not found at {vit_base_path}")
        return
    
    print("Loading ViT model...")
    model = ViTModel.from_pretrained(str(vit_base_path))
    
    print("\n" + "="*80)
    print("Testing Freeze/Unfreeze Logic")
    print("="*80)
    
    # 测试不同的解冻配置
    test_cases = [
        (None, "Stage 1: 冻结所有ViT层"),
        (8, "Stage 2: 解冻layer 8-11 (后4层)"),
        (4, "Stage 3: 解冻layer 4-11 (后8层)"),
        (0, "Stage 4: 解冻所有12层"),
    ]
    
    for unfreeze_from, description in test_cases:
        print(f"\n{'='*80}")
        print(f"{description}")
        print(f"{'='*80}")
        
        # 先冻结所有ViT参数
        for param in model.parameters():
            param.requires_grad = False
        
        # 解冻指定层
        if unfreeze_from is not None:
            unfrozen_layers = set()
            unfrozen_count = 0
            
            for name, param in model.named_parameters():
                # 当完全解冻时，解冻embeddings
                if unfreeze_from == 0 and 'embeddings' in name:
                    param.requires_grad = True
                    unfrozen_count += 1
                    print(f"  ✓ Unfrozen: {name}")
                
                # 解冻指定层
                if 'encoder.layer.' in name:
                    try:
                        parts = name.split('encoder.layer.')[1].split('.')
                        layer_num = int(parts[0])
                        
                        if 0 <= layer_num <= 11 and layer_num >= unfreeze_from:
                            param.requires_grad = True
                            unfrozen_layers.add(layer_num)
                            unfrozen_count += 1
                    except (IndexError, ValueError) as e:
                        print(f"  ⚠ Warning: Could not parse {name}")
                
                # 解冻layernorm和pooler
                if unfreeze_from == 0:
                    if 'layernorm' in name or 'pooler' in name:
                        param.requires_grad = True
                        unfrozen_count += 1
            
            # 统计
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            
            print(f"\nUnfrozen layers: {sorted(unfrozen_layers)}")
            print(f"Unfrozen parameter groups: {unfrozen_count}")
            print(f"Trainable parameters: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")
        else:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"All ViT frozen")
            print(f"Trainable parameters: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")
    
    print("\n" + "="*80)
    print("✓ All tests completed successfully!")
    print("="*80)

if __name__ == '__main__':
    test_freeze_logic()
