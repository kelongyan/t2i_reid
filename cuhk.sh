#!/bin/bash

# ============================================================================
# CUHK-PEDES Training Script - AH-Net Version
# ============================================================================
# AH-Net: Asymmetric Heterogeneous Network
# 核心特性：
#   • 不对称双流架构 (Mamba结构流 + CNN纹理流)
#   • 空间结构解耦
#   • 原型引导的语义交互
#   • 空间互斥与重构损失
# ============================================================================

# 默认参数配置
DISENTANGLE_TYPE="ahnet"  # ahnet | simple
ENABLE_VISUALIZATION=true
RESUME_PATH=""

echo "=========================================="
echo "  CUHK-PEDES Training Script"
echo "  Architecture: AH-Net (Asymmetric Heterogeneous Network)"
echo "=========================================="
echo ""
echo "默认配置: disentangle=ahnet, visualization=true"
echo "参数覆盖示例: bash cuhk.sh --disentangle-type=simple --no-viz"
echo ""

for arg in "$@"; do
    case $arg in
        --disentangle-type=*)
            DISENTANGLE_TYPE="${arg#*=}"
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

# 清理Python缓存
echo "清理缓存文件..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# JSON Config String
DATASET_CONFIG="[{'name': 'CUHK-PEDES', 'root': 'CUHK-PEDES/imgs', 'json_file': 'CUHK-PEDES/annotations/caption_all.json'}]"

# 构建基础命令
CMD="python scripts/train.py \
    --root datasets \
    --dataset-configs \"${DATASET_CONFIG}\" \
    --batch-size 96 \
    --lr 0.00003 \
    --weight-decay 0.0002 \
    --epochs 60 \
    --milestones 30 50 \
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
CMD="$CMD --disentangle-type $DISENTANGLE_TYPE"

if [ "$DISENTANGLE_TYPE" = "ahnet" ] || [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    CMD="$CMD --gs3-img-size 14 14"
    echo "✓ 解耦模块: AH-Net (Mamba Structure + CNN Texture)"
else
    echo "✓ 解耦模块: Simple"
fi

# AH-Net 配置参数
CMD="$CMD \
    --gs3-num-heads 8 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.15"

# Fusion 配置 (SAMG-RCSM)
CMD="$CMD \
    --fusion-type \"samg_rcsm\" \
    --fusion-dim 768 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 3 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.15"

# 投影维度
CMD="$CMD \
    --id-projection-dim 768 \
    --cloth-projection-dim 768"

# 优化器配置
CMD="$CMD \
    --optimizer \"AdamW\" \
    --scheduler \"cosine\""

# 损失权重配置 (AH-Net + 方案书 Phase 3)
CMD="$CMD \
    --loss-info-nce 1.0 \
    --loss-id-triplet 1.0 \
    --loss-cloth-semantic 0.5 \
    --loss-reconstruction 0.5 \
    --loss-spatial-orthogonal 0.1 \
    --loss-semantic-alignment 0.1"

# 可视化配置
if [ "$ENABLE_VISUALIZATION" = true ]; then
    CMD="$CMD \
    --visualization-enabled \
    --visualization-save-dir \"visualizations/cuhk_${DISENTANGLE_TYPE}\" \
    --visualization-frequency 5 \
    --visualization-batch-interval 200"
    echo "✓ 可视化: enabled (visualizations/cuhk_${DISENTANGLE_TYPE})"
else
    echo "✓ 可视化: disabled"
fi

# Resume training
if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume \"$RESUME_PATH\""
    echo "✓ 从检查点恢复: $RESUME_PATH"
fi

echo ""
echo "=========================================="
echo "  配置摘要"
echo "=========================================="
echo "数据集: CUHK-PEDES (11,003 IDs)"
echo "架构: AH-Net (Asymmetric Heterogeneous) + S-CAG Fusion"
echo "创新点: Conflict Score驱动动态融合"
echo "损失权重: info_nce=1.0, id_triplet=1.0, cloth_semantic=0.5"
echo "        reconstruction=0.5, spatial_orthogonal=0.1"
echo "        semantic_alignment=0.1 (Phase 3)"
echo "=========================================="
echo ""
echo "🚀 开始训练..."
echo ""

# 执行训练
eval $CMD

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "=========================================="
    echo "✅ CUHK-PEDES 训练完成！"
    echo "=========================================="
else
    echo "=========================================="
    echo "❌ 训练失败 (退出码: $exit_code)"
    echo "=========================================="
fi

exit $exit_code
