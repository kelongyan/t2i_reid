#!/bin/bash

# ============================================================================
# RSTPReid Training Script - Updated Version
# ============================================================================
# Current Architecture: CLIP + Vision Mamba (Vim) with Disentanglement Module
# Core Features:
#   • CLIP-based Text-Image ReID
#   • Vision Mamba backbone
#   • Disentanglement module (AH-Net)
#   • Multi-modal fusion
# ============================================================================

# Default parameter configuration
DISENTANGLE_TYPE="ahnet"  # ahnet | simple
ENABLE_VISUALIZATION=true
RESUME_PATH=""
FINETUNE_FROM=""

echo "=========================================="
echo "  RSTPReid Training Script"
echo "  Architecture: CLIP + Vision Mamba with Disentanglement"
echo "=========================================="
echo ""
echo "Default config: disentangle=ahnet, visualization=true"
echo "Parameter override examples: bash rstp.sh --disentangle-type=simple --no-viz"
echo ""

for arg in "$@"; do
    case $arg in
        --disentangle-type=*)
            DISENTANGLE_TYPE="${arg#*=}"
            ;;
        --no-viz)
            ENABLE_VISUALIZATION=false
            ;;
        --resume=*)
            RESUME_PATH="${arg#*=}"
            ;;
        --finetune-from=*)
            FINETUNE_FROM="${arg#*=}"
            ;;
        *)
            echo "Unknown option: $arg"
            ;;
    esac
done

# Clean Python cache
echo "Cleaning cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# JSON Config String
DATASET_CONFIG="[{'name': 'RSTPReid', 'root': 'RSTPReid', 'json_file': 'RSTPReid/annotations/caption_all.json'}]"

# Build base command
CMD="python scripts/train.py \
    --root datasets \
    --dataset-configs \"${DATASET_CONFIG}\" \
    --batch-size 64 \
    --lr 0.00003 \
    --weight-decay 0.0001 \
    --epochs 120 \
    --milestones 50 80 \
    --warmup-step 800 \
    --workers 8 \
    --height 224 \
    --width 224 \
     --print-freq 200 \
     --fp16 \
     --num-classes 4101 \
     --clip-pretrained \"pretrained/clip-vit-base-patch16\" \
    --vision-backbone vim \
    --vim-pretrained \"pretrained/Vision Mamba/vim_s_midclstok.pth\""

# Add disentangle module configuration
CMD="$CMD --disentangle-type $DISENTANGLE_TYPE"

if [ "$DISENTANGLE_TYPE" = "ahnet" ]; then
    CMD="$CMD --gs3-img-size 14 14"
    echo "✓ Disentangle module: AH-Net (Mamba Structure + CNN Texture)"
else
    echo "✓ Disentangle module: Simple"
fi

# FSHD/AH-Net configuration parameters
CMD="$CMD \
    --gs3-num-heads 8 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.1"

# Fusion configuration
CMD="$CMD \
    --fusion-type \"enhanced_mamba\" \
    --fusion-dim 256 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 2 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.1"

# Projection dimensions
CMD="$CMD \
    --id-projection-dim 768 \
    --cloth-projection-dim 768"

# Optimizer configuration
CMD="$CMD \
    --optimizer \"AdamW\" \
    --scheduler \"cosine\""

# Loss weights configuration (🔥 修复后的权重)
# 修复说明：id_triplet 从 10.0 降到 2.0，cloth_semantic 从 0.01 提升到 0.1
# Phase 1 初始值，后续会被 CurriculumScheduler 动态调整
CMD="$CMD \
    --loss-info-nce 1.0 \
    --loss-id-triplet 2.0 \
    --loss-cloth-semantic 0.1 \
    --loss-spatial-orthogonal 0.0 \
    --loss-semantic-alignment 0.0 \
    --loss-ortho-reg 0.0 \
    --loss-adversarial-attr 0.0 \
    --loss-adversarial-domain 0.0 \
    --loss-discriminator-attr 0.0 \
    --loss-discriminator-domain 0.0"

# Visualization configuration
if [ "$ENABLE_VISUALIZATION" = true ]; then
    CMD="$CMD \
    --visualization-enabled \
    --visualization-save-dir \"visualizations/rstp_${DISENTANGLE_TYPE}\" \
    --visualization-frequency 5 \
    --visualization-batch-interval 200"
    echo "✓ Visualization: enabled (visualizations/rstp_${DISENTANGLE_TYPE})"
else
    echo "✓ Visualization: disabled"
fi

# Resume training
if [ -n "$RESUME_PATH" ]; then
    CMD="$CMD --resume \"$RESUME_PATH\""
    echo "✓ Resuming from checkpoint: $RESUME_PATH"
fi

# Finetune from checkpoint
if [ -n "$FINETUNE_FROM" ]; then
    CMD="$CMD --finetune-from \"$FINETUNE_FROM\""
    echo "✓ Finetuning from checkpoint: $FINETUNE_FROM"
fi

echo ""
echo "=========================================="
echo "  Configuration Summary"
echo "=========================================="
echo "Dataset: RSTPReid (4,101 IDs)"
echo "Architecture: CLIP + Vision Mamba with Disentanglement Module"
echo "Features: Multi-modal fusion, AH-Net disentanglement"
echo "Epochs: 120 (smaller dataset, longer training)"
echo "🔥 Loss weights (Phase 1 Initial - Fixed):"
echo "              info_nce=1.0, id_triplet=2.0, cloth_semantic=0.1"
echo "              spatial_orthogonal=0.0, semantic_alignment=0.0, ortho_reg=0.0"
echo "              (Dynamic adjustment by CurriculumScheduler)"
echo "=========================================="
echo ""
echo "🚀 Starting training..."
echo ""

# Execute training
eval $CMD

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "=========================================="
    echo "✅ RSTPReid Training completed!"
    echo "=========================================="
else
    echo "=========================================="
    echo "❌ Training failed (Exit code: $exit_code)"
    echo "=========================================="
fi

exit $exit_code
