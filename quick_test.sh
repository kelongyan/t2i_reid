#!/bin/bash

# ============================================================================
# Quick Test Script - 快速测试脚本（10 epochs）
# ============================================================================
# 用途：快速验证代码修复是否生效
# 特点：
#   1. 仅训练10个epoch快速验证
#   2. 小batch size减少内存占用
#   3. 使用RSTPReid数据集（中等规模）
# ============================================================================

# 清理缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

python scripts/train.py \
    --root datasets \
    --dataset-configs "[{'name': 'RSTPReid', 'root': 'RSTPReid/imgs', 'json_file': 'RSTPReid/annotations/data_captions.json', 'cloth_json': 'RSTPReid/annotations/caption_cloth.json', 'id_json': 'RSTPReid/annotations/caption_id.json'}]" \
    --batch-size 64 \
    --lr 0.00012 \
    --weight-decay 0.0015 \
    --epochs 10 \
    --milestones 40 60 \
    --warmup-step 200 \
    --workers 4 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 3701 \
    --disentangle-type gs3 \
    --gs3-num-heads 8 \
    --gs3-d-state 20 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.12 \
    --fusion-type "enhanced_mamba" \
    --fusion-dim 256 \
    --fusion-d-state 20 \
    --fusion-d-conv 4 \
    --fusion-num-layers 2 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.12 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --loss-info-nce 1.2 \
    --loss-cls 0.04 \
    --loss-cloth-semantic 0.55 \
    --loss-orthogonal 0.12 \
    --loss-gate-adaptive 0.08 \
    --optimizer "AdamW" \
    --scheduler "cosine"

echo ""
echo "============================================================================"
echo "Quick test completed! Check the logs for:"
echo "  1. gate_adaptive loss should be > 0 from epoch 1"
echo "  2. cls loss should decrease faster"
echo "  3. Total loss should be more balanced"
echo "============================================================================"
