#!/bin/bash

# ============================================================================
# RSTPReid Training Script - 方案B：渐进解冻策略 (Vim 版)
# ============================================================================
# 核心修复：
#   ✅ Stage 1 (Epoch 1-10): 解冻Vim后4层 (layers 20-23)
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
#
# 新增优化策略：
#   ⚡ 早停机制（patience=10, min_delta=0.001）
#   ⚡ 学习率预热（warmup_steps=1000）
#   ⚡ CLIP文本编码器bias重新初始化
#   ⚡ 分层学习率优化（Stage 2+）
#   ⚡ 分层梯度裁剪
#   ⚡ BatchNorm预热（momentum=0.01）
#   ⚡ 改进G-S3门控机制（熵正则+差异正则）
#   ⚡ 优化损失权重动态调整
# ============================================================================

# 解析参数
ENABLE_OPTIMIZATIONS=true
RESUME_PATH=""

for arg in "$@"; do
    case $arg in
        --enable-optimizations)
            ENABLE_OPTIMIZATIONS=true
            shift
            ;;
        --resume)
            RESUME_PATH="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# 清理缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# 构建基础命令
BASE_CMD="python scripts/train.py \
    --root datasets \
    --dataset-configs '[{\"name\": \"RSTPReid\", \"root\": \"RSTPReid/imgs\", \"json_file\": \"RSTPReid/annotations/data_captions.json\", \"cloth_json\": \"RSTPReid/annotations/caption_cloth.json\", \"id_json\": \"RSTPReid/annotations/caption_id.json\"}]' \
    --batch-size 80 \
    --lr 0.00012 \
    --weight-decay 0.00015 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 1000 \
    --workers 6 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 3701 \
    --clip-pretrained \"pretrained/clip-vit-base-patch16\" \
    --vision-backbone vim \
    --vim-pretrained \"pretrained/Vision Mamba/vim_s_midclstok.pth\" \
    --disentangle-type gs3 \
    --gs3-num-heads 8 \
    --gs3-d-state 20 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.12 \
    --fusion-type \"enhanced_mamba\" \
    --fusion-dim 256 \
    --fusion-d-state 20 \
    --fusion-d-conv 4 \
    --fusion-num-layers 2 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.12 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --optimizer \"AdamW\" \
    --scheduler \"cosine\""

# 添加损失权重
BASE_CMD="$BASE_CMD \
    --loss-info-nce 1.0 \
    --loss-cls 0.5 \
    --loss-cloth-semantic 2.0 \
    --loss-gate-adaptive 0.05 \
    --loss-id-triplet 1.0 \
    --loss-anti-collapse 1.5 \
    --loss-reconstruction 0.1"

# 如果有resume路径，添加--resume参数
if [ -n "$RESUME_PATH" ]; then
    BASE_CMD="$BASE_CMD --resume \"$RESUME_PATH\""
    echo "📂 从检查点恢复训练：$RESUME_PATH"
    echo ""
fi

echo "🚀 开始训练 RSTPReid 数据集..."
echo ""

# 执行训练
eval $BASE_CMD

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ RSTPReid 训练完成！"
else
    echo ""
    echo "❌ RSTPReid 训练失败，退出码：$exit_code"
fi

exit $exit_code
