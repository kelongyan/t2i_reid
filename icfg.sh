#!/bin/bash

# ============================================================================
# ICFG-PEDES Training Script - 深度重构版本
# ============================================================================
# 实施P0+P1+P2方案，修复损失函数问题
# ============================================================================

# 清理缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

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
    --vision-backbone vim \
    --vim-pretrained "pretrained/Vision Mamba/vim_s_midclstok.pth" \
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
    --loss-info-nce 1.0 \
    --loss-cls 0.5 \
    --loss-cloth-semantic 0.1 \
    --loss-orthogonal 0.8 \
    --loss-gate-adaptive 0.05 \
    --optimizer "AdamW" \
    --scheduler "cosine"
