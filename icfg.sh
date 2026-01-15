#!/bin/bash

# 清理__pycache__缓存文件
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true

# ============================================================================
# ICFG-PEDES Training Script with G-S3 Module (Optimized)
# ============================================================================
# 数据集特征：
#   - 训练样本: ~54,000 images, 4,102 identities
#   - 跨数据集，domain gap较大
#   - 需要强泛化能力
#
# 优化策略：
#   1. 大batch size (112) 利用大规模数据
#   2. 较高的dropout (0.18) 应对domain gap
#   3. 更强的正交约束 (0.2) 增强解耦
#   4. 增加Mamba层数 (3层) 增强建模能力
#   5. 较低的cls权重 (0.03) 因为跨域分类难度高
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
    --dataset-configs "[{'name': 'ICFG-PEDES', 'root': 'ICFG-PEDES', 'json_file': 'ICFG-PEDES/annotations/ICFG-PEDES.json', 'cloth_json': 'ICFG-PEDES/annotations/caption_cloth.json', 'id_json': 'ICFG-PEDES/annotations/caption_id.json'}]" \
    --batch-size 112 \
    --lr 0.00018 \
    --weight-decay 0.0025 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 800 \
    --workers 8 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 4102 \
    --disentangle-type gs3 \
    --gs3-num-heads 12 \
    --gs3-d-state 24 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.18 \
    --fusion-type "enhanced_mamba" \
    --fusion-dim 256 \
    --fusion-d-state 24 \
    --fusion-d-conv 4 \
    --fusion-num-layers 3 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.18 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --loss-info-nce 1.5 \
    --loss-cls 0.03 \
    --loss-cloth-semantic 0.7 \
    --loss-orthogonal 0.2 \
    --loss-gate-adaptive 0.1 \
    --optimizer "AdamW" \
    --scheduler "cosine"
