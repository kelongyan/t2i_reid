#!/bin/bash

# 清理__pycache__缓存文件
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
find . -type f -name "*.pyd" -delete 2>/dev/null || true

# ============================================================================
# RSTPReid Training Script with G-S3 Module (Optimized)
# ============================================================================
# 数据集特征：
#   - 训练样本: 18,505 images, 3,701 identities
#   - 中等规模数据集
#   - 较好的数据质量，收敛快
#
# 优化策略：
#   1. 较小batch size (80) 平衡速度和精度
#   2. 适中的dropout (0.12) 
#   3. 更快的warmup (600步)
#   4. 稍高的学习率 (1.2e-4) 加速收敛
#   5. 增强G-S3参数以更好解耦
#
# 根据日志分析：
#   - Epoch 1: mAP 0.4742 → Epoch 8: mAP 0.6932
#   - cls损失下降: 8.3 → 6.3（需要进一步优化）
#   - gate_adaptive在epoch 6才激活（需要提前）
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
    --dataset-configs "[{'name': 'RSTPReid', 'root': 'RSTPReid/imgs', 'json_file': 'RSTPReid/annotations/data_captions.json', 'cloth_json': 'RSTPReid/annotations/caption_cloth.json', 'id_json': 'RSTPReid/annotations/caption_id.json'}]" \
    --batch-size 80 \
    --lr 0.00012 \
    --weight-decay 0.0015 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 600 \
    --workers 6 \
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
