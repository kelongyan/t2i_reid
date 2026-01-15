#!/bin/bash

# 清理__pycache__缓存文件
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true

# ============================================================================
# CUHK-PEDES Training Script with G-S3 Module (Optimized)
# ============================================================================
# 数据集特征：
#   - 训练样本: 34,054 images, 11,003 identities
#   - 最大、最复杂的数据集
#   - 需要更强的正则化和更长的训练时间
#
# 优化策略：
#   1. 较大batch size (96) 利用数据规模
#   2. 更强的dropout (0.15) 防止过拟合
#   3. 更高的weight decay (0.002) 
#   4. 使用AdamW优化器，更好的权重衰减
#   5. Warmup 1000步，充分预热
#
# 渐进解冻策略 (Progressive Unfreezing Strategy):
#   Stage 1 (Epoch 1-5):   冻结所有BERT+ViT，只训练任务模块
#   Stage 2 (Epoch 6-20):  解冻BERT+ViT后4层 (layer 8-11)
#   Stage 3 (Epoch 21-40): 解冻BERT+ViT后8层 (layer 4-11)
#   Stage 4 (Epoch 41-60): 解冻所有BERT+ViT层
#   Stage 5 (Epoch 61-80): 降低学习率精细微调
# ============================================================================

python scripts/train.py \
    --root datasets \
    --dataset-configs "[{'name': 'CUHK-PEDES', 'root': 'CUHK-PEDES/imgs', 'json_file': 'CUHK-PEDES/annotations/caption_all.json', 'cloth_json': 'CUHK-PEDES/annotations/caption_cloth.json', 'id_json': 'CUHK-PEDES/annotations/caption_id.json'}]" \
    --batch-size 96 \
    --lr 0.00015 \
    --weight-decay 0.002 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 1000 \
    --workers 8 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 11003 \
    --disentangle-type gs3 \
    --gs3-num-heads 8 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.15 \
    --fusion-type "enhanced_mamba" \
    --fusion-dim 256 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 3 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.15 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --loss-info-nce 1.0 \
    --loss-cls 0.05 \
    --loss-cloth-semantic 0.6 \
    --loss-orthogonal 0.15 \
    --loss-gate-adaptive 0.08 \
    --optimizer "AdamW" \
    --scheduler "cosine"