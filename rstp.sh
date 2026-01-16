#!/bin/bash

# ============================================================================
# RSTPReid Training Script - 方案B：渐进解冻策略 (Vim 版)
# ============================================================================
# 核心修复：
#   ✅ Stage 1 (Epoch 1-10):  解冻Vim后4层 (layers 20-23)
#   ✅ Stage 2 (Epoch 11-30): 解冻Vim后8层 (16-23) + BERT后4层
#   ✅ Stage 3 (Epoch 31-60): 解冻Vim后12层 + BERT后8层
#   ✅ Stage 4 (Epoch 61-80): 全部解冻，分层学习率
#
# 预期效果：
#   - Epoch 10: CLS 8.4 → 4.5-5.5 (下降40%+)
#   - Epoch 30: CLS < 2.0, mAP 0.75-0.78
#   - Epoch 60: mAP 0.78-0.81 (峰值)
#   - Epoch 80: mAP 0.78-0.81 (稳定)
#
# 关键改进：
#   🎯 让CLS损失从一开始就能反向传播到Vim
#   🎯 id_embeds不再固定，分类头能正常学习
#   🎯 渐进解冻保证训练稳定性
# ============================================================================

# 清理缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

python scripts/train.py \
    --root datasets \
    --dataset-configs "[{'name': 'RSTPReid', 'root': 'RSTPReid/imgs', 'json_file': 'RSTPReid/annotations/data_captions.json', 'cloth_json': 'RSTPReid/annotations/caption_cloth.json', 'id_json': 'RSTPReid/annotations/caption_id.json'}]" \
    --batch-size 80 \
    --lr 0.00012 \
    --weight-decay 0.0015 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 500 \
    --workers 6 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 3701 \
    --vision-backbone vim \
    --vim-pretrained "pretrained/Vision Mamba/vim_s_midclstok.pth" \
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
    --loss-info-nce 1.0 \
    --loss-cls 0.1 \
    --loss-cloth-semantic 0.15 \
    --loss-orthogonal 0.3 \
    --loss-gate-adaptive 0.02 \
    --optimizer "AdamW" \
    --scheduler "cosine"
