#!/bin/bash
# ============================================================================
# CUHK-PEDES Training Protocol | Stage: Deployment
# ============================================================================
# Project:  Text-Image ReID with State-Space Models (Mamba)
# Backbone: CLIP-ViT-B/16 + Vim-S (Vision Mamba)
# Module:   AH-Net Disentanglement & ScagRcsmFusion
# Config:   Standard Benchmark Protocol (Phase 1-3)
# ============================================================================

# Environment Setup
export OMP_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Uncomment to specify GPU
# export CUDA_VISIBLE_DEVICES=0,1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="log/cuhk_pedes/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "🚀 Initiating Training Protocol: CUHK-PEDES"
echo "================================================================================"
echo "  📅 Timestamp:      $TIMESTAMP"
echo "  📂 Log Directory:  $LOG_DIR"
echo "  🔧 Configuration:  AH-Net + Vim-S + CLIP"
echo "================================================================================"

# Clean Python cache to prevent stale bytecode execution
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Dataset Configuration
# Standard partition for CUHK-PEDES (11,003 identities)
DATASET_CONFIG="[{'name': 'CUHK-PEDES', 'root': 'CUHK-PEDES', 'json_file': 'CUHK-PEDES/annotations/caption_all.json'}]"

# Execution Command
# Using 'tee' to capture stdout/stderr to log file for audit
python scripts/train.py \
    --root datasets \
    --dataset-configs "${DATASET_CONFIG}" \
    --logs-dir "${LOG_DIR}" \
    --batch-size 64 \
    --lr 0.0001 \
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
    --clip-pretrained "pretrained/clip-vit-base-patch16" \
    --vision-backbone vim \
    --vim-pretrained "pretrained/Vision Mamba/vim_s_midclstok.pth" \
    --disentangle-type ahnet \
    --gs3-img-size 14 14 \
    --gs3-num-heads 8 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.1 \
    --fusion-type "enhanced_mamba" \
    --fusion-dim 256 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 2 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.1 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --optimizer "AdamW" \
    --scheduler "cosine" \
    --loss-info-nce 1.0 \
    --loss-id-triplet 2.0 \
    --loss-cloth-semantic 0.1 \
    --loss-spatial-orthogonal 0.0 \
    --loss-semantic-alignment 0.0 \
    --loss-ortho-reg 0.0 \
    --loss-adversarial-attr 0.0 \
    --loss-adversarial-domain 0.0 \
    --loss-discriminator-attr 0.0 \
    --loss-discriminator-domain 0.0 \
    --visualization-enabled \
    --visualization-save-dir "${LOG_DIR}/vis" \
    --visualization-frequency 5 \
    --visualization-batch-interval 200 \
    2>&1 | tee "${LOG_DIR}/training_console.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training Protocol Completed Successfully."
    echo "   Log saved to: ${LOG_DIR}/training_console.log"
else
    echo "❌ Training Protocol Failed (Exit Code: $EXIT_CODE)."
    echo "   Check log for details: ${LOG_DIR}/training_console.log"
fi
echo "================================================================================"

exit $EXIT_CODE