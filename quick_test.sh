#!/bin/bash

# ============================================================================
# Quick Test Script - 核心验证脚本 (15 Epochs)
# ============================================================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 解析参数
ENABLE_OPTIMIZATIONS=true
for arg in "$@"; do
    case $arg in
        --enable-optimizations) ENABLE_OPTIMIZATIONS=true; shift ;;
        *) shift ;;
    esac
done

# 1. 环境准备
echo -e "${BLUE}[1/4] 清理环境...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✓ 缓存已清除${NC}"

# 2. 配置汇总
echo -e "${BLUE}[2/4] 测试配置 (Quick Mode):${NC}"
echo -e "  • 模式: RSTPReid | 15 Epochs | Batch 64"
echo -e "  • 核心: Vision Mamba (Vim-S) + CLIP + G-S3"
if [ "$ENABLE_OPTIMIZATIONS" = true ]; then
    echo -e "  • 策略: ${GREEN}优化模式已开启${NC} (分层学习率/梯度裁剪/BN预热/早停)"
else
    echo -e "  • 策略: ${YELLOW}基础模式${NC}"
fi

# 3. 执行训练
echo -e "${BLUE}[3/4] 启动训练程序...${NC}"

BASE_CMD="python scripts/train.py \
    --root datasets \
    --dataset-configs '[{\"name\": \"RSTPReid\", \"root\": \"RSTPReid/imgs\", \"json_file\": \"RSTPReid/annotations/data_captions.json\", \"cloth_json\": \"RSTPReid/annotations/caption_cloth.json\", \"id_json\": \"RSTPReid/annotations/caption_id.json\"}]' \
    --batch-size 64 \
    --lr 0.00012 \
    --weight-decay 0.00015 \
    --epochs 15 \
    --milestones 40 60 \
    --warmup-step 1000 \
    --workers 4 \
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
    --scheduler \"cosine\" \
    --loss-info-nce 1.0 \
    --loss-cls 0.5 \
    --loss-cloth-semantic 2.0 \
    --loss-gate-adaptive 0.05 \
    --loss-id-triplet 1.0 \
    --loss-anti-collapse 1.5 \
    --loss-reconstruction 0.1"

eval $BASE_CMD
TRAIN_EXIT_CODE=$?

# 4. 结果分析
echo -e "${BLUE}[4/4] 快速分析结果...${NC}"
LOG_FILE="log/rstp/log.txt"

if [ -f "$LOG_FILE" ] && [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ 训练正常结束${NC}"
    
    echo -e "\n${BLUE}📊 关键指标 (Epoch 15):${NC}"
    grep -E "Epoch \[?15\]?:" "$LOG_FILE" | grep -E "mAP|R1" | tail -n 1
    
    echo -e "\n${BLUE}🔄 状态检查:${NC}"
    # 检查Stage切换
    grep "Progressive Unfreezing: Stage" "$LOG_FILE" | tail -n 1 || echo "  Stage: No switching in 15 epochs"
    
    # 检查梯度消失
    WARN_COUNT=$(grep -i "vanishing gradient" "$LOG_FILE" | wc -l)
    if [ $WARN_COUNT -eq 0 ]; then
        echo -e "  梯度状态: ${GREEN}正常${NC}"
    else
        echo -e "  梯度状态: ${YELLOW}异常 ($WARN_COUNT warnings)${NC}"
    fi

    # 简要建议
    FINAL_MAP=$(grep -E "Epoch \[?15\]?:" "$LOG_FILE" | grep "mAP" | sed 's/.*mAP=\([0-9.]*\).*/\1/' | tail -n 1)
    if [[ $(echo "$FINAL_MAP > 0.60" | bc -l 2>/dev/null) -eq 1 ]]; then
        echo -e "\n${GREEN}🚀 验证通过！建议启动完整训练脚本: bash rstp.sh --enable-optimizations${NC}"
    else
        echo -e "\n${YELLOW}⚠️  mAP较低，请检查特征对齐情况或Loss曲线${NC}"
    fi
else
    echo -e "${RED}❌ 训练失败或日志缺失${NC}"
    exit 1
fi

echo -e "\n${BLUE}🎉 Quick Test 完成${NC}"
exit 0
