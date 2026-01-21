#!/bin/bash

# ============================================================================
# CUHK-PEDES Training Script - FSHD-Net Version
# ============================================================================
# 支持FSHD模块配置
# 核心特性：
#   ✅ 支持FSHD/Simple两种解耦模式
#   ✅ 频域分解固化为DCT
#   ✅ 异构双流配置（Multi-scale CNN开关）
#   ✅ 可视化配置
#   ✅ 渐进解冻策略
#
# 预期性能：
#   - FSHD-Full: mAP 68-70%
#   - FSHD-Lite: mAP 67-69%
# ============================================================================

# 默认参数配置（FSHD-Full完整版）
DISENTANGLE_TYPE="fshd"  # fshd | simple
USE_MULTI_SCALE_CNN=true # true | false
ENABLE_VISUALIZATION=true
RESUME_PATH=""

echo "🔥 默认配置: FSHD-Full (disentangle=fshd, multi_scale_cnn=true, visualization=true)"
echo "   可通过参数覆盖，例如: bash cuhk.sh --disentangle-type=simple --no-viz"
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
DATASET_CONFIG="[{'name': 'CUHK-PEDES', 'root': 'CUHK-PEDES/imgs', 'json_file': 'CUHK-PEDES/annotations/caption_all.json', 'cloth_json': 'CUHK-PEDES/annotations/caption_cloth.json', 'id_json': 'CUHK-PEDES/annotations/caption_id.json'}]"

# 构建基础命令
CMD="python scripts/train.py \
    --root datasets \
    --dataset-configs \"${DATASET_CONFIG}\" \
    --batch-size 96 \
    --lr 0.00003 \
    --weight-decay 0.0002 \
    --epochs 80 \
    --milestones 40 60 \
    --warmup-step 1000 \
    --workers 8 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 11003 \
    --clip-pretrained \"pretrained/clip-vit-base-patch16\" \
    --vision-backbone vim \
    --vim-pretrained \"pretrained/Vision Mamba/vim_s_midclstok.pth\""

# 添加解耦模块配置
CMD="$CMD \
    --disentangle-type $DISENTANGLE_TYPE"

# FSHD特定配置
if [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    CMD="$CMD \
    --gs3-use-multi-scale-cnn $USE_MULTI_SCALE_CNN \
    --gs3-img-size 14 14"
    echo "🔥 使用FSHD模块: multi_scale_cnn=$USE_MULTI_SCALE_CNN (Frequency: DCT fixed)"
else
    echo "🔧 使用简化解耦模块"
fi

# G-S3/FSHD通用配置
CMD="$CMD \
    --gs3-num-heads 8 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.15"

# Fusion配置
CMD="$CMD \
    --fusion-type \"enhanced_mamba\" \
    --fusion-dim 256 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 3 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.15"

# 投影维度
CMD="$CMD \
    --id-projection-dim 768 \
    --cloth-projection-dim 768"

# 优化器
CMD="$CMD \
    --optimizer \"AdamW\" \
    --scheduler \"cosine\""

# 损失权重（优化版 - 平衡权重，提升辅助损失）
CMD="$CMD \
    --loss-info-nce 1.2 \
    --loss-cls 0.05 \
    --loss-cloth-semantic 1.0 \
    --loss-orthogonal 0.12 \
    --loss-gate-adaptive 0.05 \
    --loss-id-triplet 0.8 \
    --loss-anti-collapse 2.0 \
    --loss-reconstruction 1.5 \
    --loss-semantic-alignment 0.0 \
    --loss-freq-consistency 0.0 \
    --loss-freq-separation 0.0"

echo "🚀 优化模式："
echo "   - 学习率: 3e-5"
echo "   - anti_collapse: 修复自适应margin，权重2.0"
echo "   - gate_adaptive: 添加类间分离，权重0.05"
echo "   - reconstruction: 增强版（多样性+能量守恒），权重1.5"
echo "   - cls/orthogonal/id_triplet: 权重提升"
echo "   - 频域/语义损失: 阶段1禁用，Epoch 21+激活"

# 可视化配置
if [ "$ENABLE_VISUALIZATION" = true ]; then
    CMD="$CMD \
    --visualization-enabled \
    --visualization-save-dir \"visualizations/cuhk_${DISENTANGLE_TYPE}\" \
    --visualization-frequency 5 \
    --visualization-batch-interval 200"
    echo "📊 可视化已启用，保存到: visualizations/cuhk_${DISENTANGLE_TYPE}"
fi

# Resume
if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume \"$RESUME_PATH\""
    echo "📂 从检查点恢复训练：$RESUME_PATH"
fi

echo ""
echo "🚀 开始训练 CUHK-PEDES 数据集 (${DISENTANGLE_TYPE}模式)"
echo "Executing command..."
echo ""

# 执行训练
eval $CMD

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ CUHK-PEDES 训练完成！"
else
    echo ""
    echo "❌ CUHK-PEDES 训练失败，退出码：$exit_code"
fi

exit $exit_code
