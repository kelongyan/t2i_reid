#!/bin/bash

# 清理__pycache__缓存文件
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true

# ICFG-PEDES Training Script

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
    --loss-cls 1.0 \
    --loss-cloth 0.5 \
    --loss-cloth-adv 0.1 \
    --loss-cloth-match 1.0 \
    --loss-decouple 0.1 \
    --loss-gate-regularization 0.01 \
    --loss-projection-l2 0.0001 \
    --loss-uniformity 0.01 \
    --optimizer "Adam" \
    --scheduler "cosine"
