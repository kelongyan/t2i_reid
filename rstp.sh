#!/bin/bash
# ============================================================================
# RSTPReid Training Protocol | Stage: Deployment
# ============================================================================
# Project:  Text-Image ReID with State-Space Models (Mamba)
# Dataset:  RSTPReid (Real Scenario Text-based Person ReID)
# Backbone: CLIP-ViT-B/16 + Vim-S
# ============================================================================

export OMP_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0,1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="log/rstp_reid/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "🚀 Initiating Training Protocol: RSTPReid"
echo "================================================================================"
echo "  📅 Timestamp:      $TIMESTAMP"
echo "  📂 Log Directory:  $LOG_DIR"
echo "================================================================================"

# Cache cleanup
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Dataset Config
DATASET_CONFIG="[{'name': 'RSTPReid', 'root': 'RSTPReid', 'json_file': 'RSTPReid/annotations/caption_all.json'}]"

# Execute
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
    --num-classes 4101 \
    --clip-pretrained "pretrained/clip-vit-base-patch16" \
    --vision-backbone vim \
    --vim-pretrained "pretrained/Vision Mamba/vim_s_midclstok.pth" \
    --disentangle-type ahnet \
    --gs3-img-size 14 14 \
    --optimizer "AdamW" \
    --scheduler "cosine" \
    --loss-info-nce 1.0 \
    --loss-id-triplet 2.0 \
    --loss-cloth-semantic 0.1 \
    2>&1 | tee "${LOG_DIR}/training_console.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training Protocol Completed Successfully."
else
    echo "❌ Training Protocol Failed (Exit Code: $EXIT_CODE)."
fi
exit $EXIT_CODE