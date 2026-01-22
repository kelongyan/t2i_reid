#!/bin/bash

# ============================================================================
# RSTPReid Training Script - FSHD-Net Version
# ============================================================================
# 支持FSHD模块配置
# 预期效果：
#   - FSHD-Full: mAP 78-81%
#   - FSHD-Lite: mAP 77-80%
# ============================================================================

# 默认参数配置（FSHD-Full完整版）
DISENTANGLE_TYPE="fshd"
USE_MULTI_SCALE_CNN=true
ENABLE_VISUALIZATION=true
RESUME_PATH=""

echo "🔥 默认配置: FSHD-Full (disentangle=fshd, multi_scale_cnn=true, visualization=true)"
echo "   可通过参数覆盖，例如: bash rstp.sh --disentangle-type=simple --no-viz"
echo ""

for arg in "$@"; do
    case $arg in
        --disentangle-type=*) 
            DISENTANGLE_TYPE="${arg#*=}"
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

# JSON Config String (Single quoted for safety)
DATASET_CONFIG="[{'name': 'RSTPReid', 'root': 'RSTPReid/imgs', 'json_file': 'RSTPReid/annotations/data_captions.json'}]"

# 构建基础命令
CMD="python scripts/train.py \
    --root datasets \
    --dataset-configs \"${DATASET_CONFIG}\" \
    --batch-size 64 \
    --lr 0.00003 \
    --weight-decay 0.0001 \
    --epochs 50 \
    --milestones 25 40 \
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
CMD="$CMD \
    --disentangle-type $DISENTANGLE_TYPE"

if [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    CMD="$CMD \
    --gs3-use-multi-scale-cnn $USE_MULTI_SCALE_CNN \
    --gs3-img-size 14 14"
    echo "🔥 使用FSHD模块: multi_scale_cnn=$USE_MULTI_SCALE_CNN (Frequency: DCT fixed)"
else
    echo "🔧 使用简化解耦模块"
fi

CMD="$CMD \
    --gs3-num-heads 8 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.15 \
    --fusion-type \"samg_rcsm\" \
    --fusion-dim 256 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 3 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.15 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --optimizer \"AdamW\" \
    --scheduler \"cosine\" \
    --loss-info-nce 1.0 \
    --loss-cls 0.15 \
    --loss-cloth-semantic 0.2 \
    --loss-orthogonal 0.3 \
    --loss-id-triplet 0.8 \
    --loss-anti-collapse 1.5 \
    --loss-reconstruction 0.2 \
    --loss-gate-adaptive 0.0 \
    --loss-semantic-alignment 0.0 \
    --loss-freq-consistency 0.0 \
    --loss-freq-separation 0.0"

echo "🔥 架构升级: SAMG + R-CSM (Pyramid Text Encoder)"
echo "   - anti_collapse: EMA追踪 (修复loss=0 BUG), 权重1.5"
echo "   - cls权重: 0.15 (降低60%, 避免过拟合)"
echo "   - cloth_semantic: 0.2 (降低60% + 延迟激活)"
echo "   - orthogonal: 0.3 (提升100%, 强化解耦)"
echo "   - gating: OFC-Gate (Physics-Aware + Ortho-Suppression)"
echo "   - gate_clamp: [0.05, 0.95] (放宽范围)"
echo "   - prompts: 7+23个细粒度描述"

if [ "$ENABLE_VISUALIZATION" = true ]; then
    CMD="$CMD \
    --visualization-enabled \
    --visualization-save-dir \"visualizations/rstp_${DISENTANGLE_TYPE}\" \
    --visualization-frequency 5 \
    --visualization-batch-interval 200"
fi

if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume \"$RESUME_PATH\""
fi

echo ""
echo "🚀 开始训练 RSTPReid 数据集 (${DISENTANGLE_TYPE}模式)"
echo "Executing command..."
echo ""

eval $CMD

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "✅ RSTPReid 训练完成！"
else
    echo "❌ RSTPReid 训练失败，退出码：$exit_code"
fi

exit $exit_code