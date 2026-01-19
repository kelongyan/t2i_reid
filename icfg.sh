#!/bin/bash

# ============================================================================
# ICFG-PEDES Training Script - 渐进解冻策略
# ============================================================================
# 实施P0+P1+P2方案，修复损失函数问题
# ============================================================================
# 核心修复：
#   ✅ Stage 1 (Epoch 1-10): 解冻Vim后4层 (layers 20-23)
#   ✅ Stage 2 (Epoch 11-30): 解冻Vim后8层 (16-23) + CLIP后1层
#   ✅ Stage 3 (Epoch 31-60): 解冻Vim后12层 + CLIP后6层
#   ✅ Stage 4 (Epoch 61-80): 全部解冻，分层学习率
#
# ICFG-PEDES特点（相比其他数据集）：
#   - 中等类别数 (4,102 vs RSTP 3,701, CUHK 11,003)
#   - 最大数据集 (~54k训练样本)
#   - 更大batch_size (112)
#   - 更多G-S3 heads (12)
#
# 预期性能：
#   - Epoch 10: CLS 8.5 → 4.5-5.5
#   - Epoch 30: CLS < 2.0, mAP 0.72-0.75
#   - Epoch 60: mAP 0.75-0.78 (峰值)
#   - Epoch 80: mAP 0.75-0.78 (稳定)
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
    --dataset-configs '[{\"name\": \"ICFG-PEDES\", \"root\": \"ICFG-PEDES\", \"json_file\": \"ICFG-PEDES/annotations/ICFG-PEDES.json\", \"cloth_json\": \"ICFG-PEDES/annotations/caption_cloth.json\", \"id_json\": \"ICFG-PEDES/annotations/caption_id.json\"}]' \
    --batch-size 112 \
    --lr 0.00018 \
    --weight-decay 0.00025 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 1000 \
    --workers 8 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 4102 \
    --clip-pretrained \"pretrained/clip-vit-base-patch16\" \
    --vision-backbone vim \
    --vim-pretrained \"pretrained/Vision Mamba/vim_s_midclstok.pth\" \
    --disentangle-type gs3 \
    --gs3-num-heads 12 \
    --gs3-d-state 24 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.18 \
    --fusion-type \"enhanced_mamba\" \
    --fusion-dim 256 \
    --fusion-d-state 24 \
    --fusion-d-conv 4 \
    --fusion-num-layers 3 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.18 \
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

echo "🚀 开始训练 ICFG-PEDES 数据集..."
echo ""

# 执行训练
eval $BASE_CMD

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ ICFG-PEDES 训练完成！"
else
    echo ""
    echo "❌ ICFG-PEDES 训练失败，退出码：$exit_code"
fi

exit $exit_code
