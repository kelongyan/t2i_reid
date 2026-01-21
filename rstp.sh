#!/bin/bash

# ============================================================================
# RSTPReid Training Script - FSHD-Net Version
# ============================================================================
# 支持FSHD模块配置
# 预期效果：
#   - FSHD-Full: mAP 78-81%
#   - FSHD-Lite: mAP 77-80%
#   - Baseline G-S3: mAP 76-79%
# ============================================================================

# 默认参数配置（FSHD-Full完整版）
DISENTANGLE_TYPE="fshd"
FREQ_TYPE="dct"
USE_MULTI_SCALE_CNN=true
ENABLE_VISUALIZATION=true
RESUME_PATH=""

echo "🔥 默认配置: FSHD-Full (disentangle=fshd, freq=dct, multi_scale_cnn=true, visualization=true)"
echo "   可通过参数覆盖，例如: bash rstp.sh --disentangle-type=gs3 --no-viz"
echo ""

for arg in "$@"; do
    case $arg in
        --disentangle-type=*) 
            DISENTANGLE_TYPE="${arg#*=}"
            shift
            ;; 
        --freq-type=*) 
            FREQ_TYPE="${arg#*=}"
            shift
            ;; 
        --use-cnn) 
            USE_MULTI_SCALE_CNN=true
            shift
            ;; 
        --no-cnn) 
            USE_MULTI_SCALE_CNN=false
            shift
            ;; 
        --no-viz) 
            ENABLE_VISUALIZATION=false
            shift
            ;; 
        --resume=*) 
            RESUME_PATH="${arg#*=}"
            shift
            ;; 
        *) 
            shift
            ;; 
    esac
done

# 清理缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 构建基础命令
BASE_CMD="python scripts/train.py \
    --root datasets \
    --dataset-configs '[{\"name\": \"RSTPReid\", \"root\": \"RSTPReid/imgs\", \"json_file\": \"RSTPReid/annotations/data_captions.json\", \"cloth_json\": \"RSTPReid/annotations/caption_cloth.json\", \"id_json\": \"RSTPReid/annotations/caption_id.json\"}]' \
    --batch-size 64 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 800 \
    --workers 8 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 3701 \
    --clip-pretrained \"pretrained/clip-vit-base-patch16\" \
    --vision-backbone vim \
    --vim-pretrained \"pretrained/Vision Mamba/vim_s_midclstok.pth\""

# 添加解耦模块配置
BASE_CMD="$BASE_CMD \
    --disentangle-type $DISENTANGLE_TYPE"

if [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    BASE_CMD="$BASE_CMD \
    --gs3-freq-type $FREQ_TYPE \
    --gs3-use-multi-scale-cnn $USE_MULTI_SCALE_CNN \
    --gs3-img-size 14 14"
    echo "🔥 使用FSHD模块: freq_type=$FREQ_TYPE, multi_scale_cnn=$USE_MULTI_SCALE_CNN"
elif [ "$DISENTANGLE_TYPE" = "gs3" ]; then
    echo "📊 使用Baseline G-S3模块"
else
    echo "🔧 使用简化解耦模块"
fi

BASE_CMD="$BASE_CMD \
    --gs3-num-heads 8 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.15 \
    --fusion-type \"enhanced_mamba\" \
    --fusion-dim 256 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 3 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.15 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --optimizer \"AdamW\" \
    --scheduler \"cosine\""

# 损失权重
BASE_CMD="$BASE_CMD \
    --loss-info-nce 1.0 \
    --loss-cls 0.05 \
    --loss-cloth-semantic 1.0 \
    --loss-orthogonal 0.1 \
    --loss-gate-adaptive 0.02 \
    --loss-id-triplet 0.5 \
    --loss-anti-collapse 1.0 \
    --loss-reconstruction 0.5"

if [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    BASE_CMD="$BASE_CMD \
    --loss-freq-consistency 0.5 \
    --loss-freq-separation 0.2"
fi

if [ "$ENABLE_VISUALIZATION" = true ]; then
    BASE_CMD="$BASE_CMD \
    --visualization-enabled \
    --visualization-save-dir \"visualizations/rstp_${DISENTANGLE_TYPE}_${FREQ_TYPE}\" \
    --visualization-frequency 5 \
    --visualization-batch-interval 200"
fi

if [ -n "$RESUME_PATH" ]; then
    BASE_CMD="$BASE_CMD --resume \"$RESUME_PATH\""
fi

echo ""
echo "🚀 开始训练 RSTPReid 数据集 (${DISENTANGLE_TYPE}模式)"
echo ""

eval $BASE_CMD

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "✅ RSTPReid 训练完成！"
else
    echo "❌ RSTPReid 训练失败，退出码：$exit_code"
fi

exit $exit_code