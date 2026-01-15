#!/bin/bash

# 清理__pycache__缓存文件
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true

# ICFG-PEDES Training Script with G-S3 Module
# 使用 G-S3 (Geometry-Guided Selective State Space) 解耦模块
# 对比实验：将 --disentangle-type 改为 simple 可使用简化版本
#
# 渐进解冻策略 (Progressive Unfreezing Strategy):
#   Stage 1 (Epoch 1-5):   冻结所有ViT层，只训练任务模块
#   Stage 2 (Epoch 6-20):  解冻ViT后4层 (layer 8-11)
#   Stage 3 (Epoch 21-40): 解冻ViT后8层 (layer 4-11)
#   Stage 4 (Epoch 41-60): 解冻所有ViT层
#   Stage 5 (Epoch 61-80): 降低学习率进行微调

python scripts/train.py \
    --root datasets \
    --dataset-configs "[{'name': 'ICFG-PEDES', 'root': 'ICFG-PEDES', 'json_file': 'ICFG-PEDES/annotations/ICFG-PEDES.json', 'cloth_json': 'ICFG-PEDES/annotations/caption_cloth.json', 'id_json': 'ICFG-PEDES/annotations/caption_id.json'}]" \
    --batch-size 128 \
    --lr 0.0001 \
    --weight-decay 0.001 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 500 \
    --workers 4 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 8000 \
    --disentangle-type gs3 \
    --gs3-num-heads 8 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.1 \
    --fusion-type "enhanced_mamba" \
    --fusion-dim 256 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 2 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.1 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --loss-info-nce 1.0 \
    --loss-cls 0.1 \
    --loss-cloth-semantic 0.5 \
    --loss-orthogonal 0.1 \
    --loss-gate-adaptive 0.01 \
    --optimizer "Adam" \
    --scheduler "cosine"
