#!/bin/bash

# ============================================================================
# Quick Test Script - FSHD快速验证脚本 (优化版)
# ============================================================================
# 目标: 在小数据集上快速测试优化后的FSHD模块
# 数据集: RSTPReid (3,701类，较小)
# Epochs: 20 (推荐) 或 10 (快速验证)
# 
# 优化内容:
#   - anti_collapse: 自适应margin + 方差正则
#   - gate_adaptive: 添加类间分离损失
#   - reconstruction: 多样性 + 能量守恒
#   - 权重再平衡: 提升辅助损失
# 
# 预期: mAP 70%+ (20 epochs)
# ============================================================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}🧪 FSHD-Net 优化版快速验证测试${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 默认参数
DISENTANGLE_TYPE="fshd"  # fshd | simple
USE_MULTI_SCALE_CNN=true # true | false
ENABLE_VISUALIZATION=true 
EPOCHS=20  # 修改默认值为20

# 解析参数
for arg in "$@"; do
    case $arg in
        --disentangle-type=*) 
            DISENTANGLE_TYPE="${arg#*=}"
            shift
            ;; 
        --no-cnn) 
            USE_MULTI_SCALE_CNN=false
            shift
            ;; 
        --epochs=*) 
            EPOCHS="${arg#*=}"
            shift
            ;; 
        --no-viz) 
            ENABLE_VISUALIZATION=false
            shift
            ;; 
        *) 
            shift
            ;; 
    esac
done

# 1. 环境准备
echo -e "${BLUE}[1/5] 清理环境...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
echo -e "${GREEN}✓ 缓存已清除${NC}"
echo ""

# 2. 配置汇总
echo -e "${BLUE}[2/5] 测试配置:${NC}"
echo -e "  • 数据集: ${GREEN}RSTPReid${NC}"
echo -e "  • Epochs: ${GREEN}${EPOCHS}${NC}"
echo -e "  • Batch Size: ${GREEN}64${NC}"
echo -e "  • 学习率: ${GREEN}3e-5${NC} (优化版)"
echo -e "  • 解耦模块: ${GREEN}${DISENTANGLE_TYPE}${NC}"
if [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    echo -e "  • Multi-scale CNN: ${GREEN}${USE_MULTI_SCALE_CNN}${NC}"
    echo -e "  • Frequency: ${GREEN}DCT${NC} (固化)"
fi
echo -e "  • 可视化: ${ENABLE_VISUALIZATION}"
echo ""
echo -e "${YELLOW}✨ 优化亮点:${NC}"
echo -e "  ✅ anti_collapse: 修复自适应margin (权重2.0)"
echo -e "  ✅ Architecture: Pyramid Text + OFC-Gate + SAMG-RCSM"
echo -e "  ✅ Gating: Physics-Aware (DCT) + Ortho-Suppression"
echo -e "  ✅ Fusion: Residual Cross-Scan Mamba (Dim=768)"
echo -e "  ✅ Loss: Anti-Collapse(1.5) + Ortho(0.3) + Reconstruction(0.2)"
echo ""

# 3. 检查预训练模型
echo -e "${BLUE}[3/5] 检查预训练模型...${NC}"
CLIP_PATH="pretrained/clip-vit-base-patch16"
VIM_PATH="pretrained/Vision Mamba/vim_s_midclstok.pth"

if [ ! -d "$CLIP_PATH" ]; then
    echo -e "${YELLOW}⚠️  CLIP模型不存在: $CLIP_PATH，请确保路径正确${NC}"
fi

if [ ! -f "$VIM_PATH" ]; then
    echo -e "${YELLOW}⚠️  Vision Mamba模型不存在: $VIM_PATH${NC}"
fi

# 4. 构建训练命令
echo -e "${BLUE}[4/5] 启动训练程序...${NC}"
echo ""

# JSON Config String
DATASET_CONFIG="[{'name': 'RSTPReid', 'root': 'RSTPReid/imgs', 'json_file': 'RSTPReid/annotations/data_captions.json'}]"

# 构建命令字符串
CMD="python scripts/train.py \
    --root datasets \
    --dataset-configs \"${DATASET_CONFIG}\" \
    --batch-size 64 \
    --lr 0.00003 \
    --weight-decay 0.0001 \
    --epochs ${EPOCHS} \
    --milestones 15 \
    --warmup-step 500 \
    --workers 4 \
    --height 224 \
    --width 224 \
    --print-freq 20 \
    --fp16 \
    --num-classes 3701 \
    --clip-pretrained \"${CLIP_PATH}\" \
    --vision-backbone vim \
    --vim-pretrained \"${VIM_PATH}\""

# 添加解耦模块配置
CMD="$CMD \
    --disentangle-type ${DISENTANGLE_TYPE}"

# FSHD特定配置
if [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    CMD="$CMD \
    --gs3-use-multi-scale-cnn ${USE_MULTI_SCALE_CNN} \
    --gs3-img-size 14 14"
fi

# 通用G-S3配置
CMD="$CMD \
    --gs3-num-heads 8 \
    --gs3-d-state 16 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.15 \
    --fusion-type \"samg_rcsm\" \
    --fusion-dim 768 \
    --fusion-d-state 16 \
    --fusion-d-conv 4 \
    --fusion-num-layers 3 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.15 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --optimizer \"AdamW\" \
    --scheduler \"cosine\" \
    --logs-dir \"log/quick_test\""

# 损失权重（优化版）
CMD="$CMD \
    --loss-info-nce 1.0 \
    --loss-cls 0.15 \
    --loss-cloth-semantic 0.2 \
    --loss-orthogonal 0.3 \
    --loss-gate-adaptive 0.0 \
    --loss-id-triplet 0.8 \
    --loss-anti-collapse 1.5 \
    --loss-reconstruction 0.2 \
    --loss-semantic-alignment 0.0 \
    --loss-freq-consistency 0.0 \
    --loss-freq-separation 0.0"

# 可视化配置
if [ "$ENABLE_VISUALIZATION" = true ]; then
    CMD="$CMD \
    --visualization-enabled \
    --visualization-save-dir \"visualizations/quick_test_${DISENTANGLE_TYPE}\" \
    --visualization-frequency 4 \
    --visualization-batch-interval 100"
fi

# 执行训练
echo "Executing: $CMD"
eval $CMD
TRAIN_EXIT_CODE=$?

echo ""
echo -e "${BLUE}[5/5] 测试汇总...${NC}"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 快速测试完成！${NC}"
    LOG_FILE="log/quick_test/RSTPReid/log.txt"
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}📊 最终性能:${NC}"
        grep "mAP=" "$LOG_FILE" | tail -n 1 | awk -F'mAP=' '{print "  • mAP: "$2}' | cut -d',' -f1
        echo ""
        echo -e "${BLUE}📈 损失值检查:${NC}"
        echo -e "  查看 $LOG_FILE 中的最后几行，关注："
        echo -e "  • anti_collapse: 应>0（修复前为0.0000）"
        echo -e "  • gate_adaptive: 应>0.02（修复前~0.005）"
        echo -e "  • reconstruction: 应>0.05（修复前~0.008）"
    fi
    echo ""
    echo -e "${YELLOW}💡 提示:${NC}"
    echo -e "  - 如果性能良好，可运行完整训练: bash rstp.sh"
    echo -e "  - 对比修改前后的损失曲线和mAP变化"
else
    echo -e "${RED}❌ 测试失败，退出码: ${TRAIN_EXIT_CODE}${NC}"
fi
