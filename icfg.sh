#!/bin/bash
# ============================================================================
# ICFG-PEDES Training Protocol | Optimized Configuration
# ============================================================================
# Model:    AH-Net + S-CAG/RCSM Fusion + Adversarial Disentanglement
# Backbone: CLIP-ViT-B/16 + Vim-S (Vision Mamba)
# Dataset:  ICFG-PEDES (4,102 identities)
# Strategy: 3-Phase Curriculum Learning (20-50-100 epochs)
# ============================================================================

export OMP_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID

LOG_DIR="log/icfg"
mkdir -p "$LOG_DIR/model" "$LOG_DIR/vis"

echo "================================================================================"
echo "🚀 ICFG-PEDES Training | Optimized Protocol"
echo "================================================================================"
echo "  📊 Dataset:       ICFG-PEDES (4,102 IDs)"
echo "  🧠 Backbone:      CLIP-ViT-B/16 + Vim-S"
echo "  🔧 Architecture:  AH-Net + S-CAG/RCSM + Adversarial"
echo "  📚 Curriculum:    3-Phase (20-50-100 epochs)"
echo "================================================================================"

# Clean cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

DATASET_CONFIG="[{'name': 'ICFG-PEDES', 'root': 'ICFG-PEDES', 'json_file': 'ICFG-PEDES/annotations/caption_all.json'}]"

python scripts/train.py \
    --root datasets \
    --dataset-configs "${DATASET_CONFIG}" \
    --logs-dir "$LOG_DIR" \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.00005 \
    --weight-decay 0.0002 \
    --warmup-step 2000 \
    --milestones 20 50 \
    --scheduler "cosine" \
    --optimizer "AdamW" \
    --workers 8 \
    --height 224 \
    --width 224 \
    --fp16 \
    --num-classes 4102 \
    --clip-pretrained "pretrained/clip-vit-base-patch16" \
    --vision-backbone vim \
    --vim-pretrained "pretrained/Vision Mamba/vim_s_midclstok.pth" \
    --gs3-img-size 14 14 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --fusion-type "enhanced_mamba" \
    --fusion-dim 256 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 2 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.1 \
    --loss-info-nce 1.0 \
    --loss-id-triplet 5.0 \
    --loss-cloth-semantic 1.0 \
    --loss-spatial-orthogonal 0.0 \
    --loss-semantic-alignment 0.0 \
    --loss-ortho-reg 0.0 \
    --loss-adversarial-attr 0.0 \
    --loss-adversarial-domain 0.0 \
    --loss-discriminator-attr 0.0 \
    --loss-discriminator-domain 0.0 \
    --visualization-enabled \
    --visualization-save-dir "${LOG_DIR}/vis" \
    --visualization-frequency 10 \
    --visualization-batch-interval 500

EXIT_CODE=$?

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ICFG-PEDES Training Completed Successfully"
    echo "   Results:  ${LOG_DIR}/res.txt"
    echo "   Best Model: ${LOG_DIR}/model/best_icfg.pth"
    echo "   Visualizations: ${LOG_DIR}/vis/"
else
    echo "❌ Training Failed (Exit Code: $EXIT_CODE)"
    echo "   Check: ${LOG_DIR}/log.txt"
fi
echo "================================================================================"

exit $EXIT_CODE
