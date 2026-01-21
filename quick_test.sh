#!/bin/bash

# ============================================================================
# Quick Test Script - FSHD快速验证脚本 (RSTPReid 3 Epochs)
# ============================================================================
# 目标: 在小数据集上快速测试FSHD模块是否正常工作
# 数据集: RSTPReid (3,701类，较小)
# Epochs: 3 (快速验证)
# 预期: 能够正常前向/反向传播，损失正常下降
# ============================================================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 默认参数（FSHD-Full完整版）
DISENTANGLE_TYPE="fshd"  # fshd | gs3 | simple
FREQ_TYPE="dct"          # dct | wavelet
USE_MULTI_SCALE_CNN=true # true | false
ENABLE_VISUALIZATION=false # 快速测试关闭可视化
EPOCHS=3

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}🧪 FSHD-Net 快速验证测试${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# 解析参数
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
        --no-cnn)
            USE_MULTI_SCALE_CNN=false
            shift
            ;;
        --epochs=*)
            EPOCHS="${arg#*=}"
            shift
            ;;
        --enable-viz)
            ENABLE_VISUALIZATION=true
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
echo -e "  • 数据集: ${GREEN}RSTPReid${NC} (3,701类，最小数据集)"
echo -e "  • Epochs: ${GREEN}${EPOCHS}${NC} (快速验证)"
echo -e "  • Batch Size: ${GREEN}32${NC} (降低显存占用)"
echo -e "  • 解耦模块: ${GREEN}${DISENTANGLE_TYPE}${NC}"

if [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    echo -e "  • 频域类型: ${GREEN}${FREQ_TYPE}${NC}"
    echo -e "  • Multi-scale CNN: ${GREEN}${USE_MULTI_SCALE_CNN}${NC}"
fi

echo -e "  • 可视化: ${ENABLE_VISUALIZATION}"
echo -e "  • 目标: 验证模型能够正常前向/反向传播"
echo ""

# 3. 检查预训练模型
echo -e "${BLUE}[3/5] 检查预训练模型...${NC}"
CLIP_PATH="pretrained/clip-vit-base-patch16"
VIM_PATH="pretrained/Vision Mamba/vim_s_midclstok.pth"

if [ ! -d "$CLIP_PATH" ]; then
    echo -e "${RED}❌ CLIP模型不存在: $CLIP_PATH${NC}"
    echo -e "请下载并放置到pretrained目录"
    exit 1
fi

if [ ! -f "$VIM_PATH" ]; then
    echo -e "${RED}❌ Vision Mamba模型不存在: $VIM_PATH${NC}"
    echo -e "请下载并放置到pretrained目录"
    exit 1
fi

echo -e "${GREEN}✓ 预训练模型检查通过${NC}"
echo ""

# 4. 构建训练命令
echo -e "${BLUE}[4/5] 启动训练程序...${NC}"
echo ""

BASE_CMD="python scripts/train.py \
    --root datasets \
    --dataset-configs '[{\"name\": \"RSTPReid\", \"root\": \"RSTPReid/imgs\", \"json_file\": \"RSTPReid/annotations/data_captions.json\", \"cloth_json\": \"RSTPReid/annotations/caption_cloth.json\", \"id_json\": \"RSTPReid/annotations/caption_id.json\"}]' \
    --batch-size 32 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --epochs ${EPOCHS} \
    --milestones 40 60 \
    --warmup-step 200 \
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
BASE_CMD="$BASE_CMD \
    --disentangle-type ${DISENTANGLE_TYPE}"

# FSHD特定配置
if [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    BASE_CMD="$BASE_CMD \
    --gs3-freq-type ${FREQ_TYPE} \
    --gs3-use-multi-scale-cnn ${USE_MULTI_SCALE_CNN} \
    --gs3-img-size 14 14"
fi

# 通用G-S3配置
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

# 损失权重（FSHD优化版）
BASE_CMD="$BASE_CMD \
    --loss-info-nce 1.0 \
    --loss-cls 0.05 \
    --loss-cloth-semantic 1.0 \
    --loss-orthogonal 0.1 \
    --loss-gate-adaptive 0.02 \
    --loss-id-triplet 0.5 \
    --loss-anti-collapse 1.0 \
    --loss-reconstruction 0.5"

# FSHD频域损失
if [ "$DISENTANGLE_TYPE" = "fshd" ]; then
    BASE_CMD="$BASE_CMD \
    --loss-freq-consistency 0.5 \
    --loss-freq-separation 0.2"
fi

# 可视化配置
if [ "$ENABLE_VISUALIZATION" = true ]; then
    BASE_CMD="$BASE_CMD \
    --visualization-enabled \
    --visualization-save-dir \"visualizations/quick_test_${DISENTANGLE_TYPE}_${FREQ_TYPE}\" \
    --visualization-frequency 1 \
    --visualization-batch-interval 100"
fi

# 执行训练
eval $BASE_CMD
TRAIN_EXIT_CODE=$?

echo ""
echo -e "${BLUE}[5/5] 分析测试结果...${NC}"
echo ""

# 5. 结果分析
LOG_FILE="log/RSTPReid/log.txt"
DEBUG_FILE="log/RSTPReid/debug.txt"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 训练正常完成！${NC}"
    echo ""
    
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}📊 关键指标 (Epoch ${EPOCHS}):${NC}"
        
        # 提取最后一个epoch的指标
        LAST_LOSS=$(grep -E "E${EPOCHS} \[" "$LOG_FILE" | grep "Total:" | tail -n 1 | sed 's/.*Total:\([0-9.]*\).*/\1/')
        
        if [ -n "$LAST_LOSS" ]; then
            echo -e "  • 最终Loss: ${GREEN}${LAST_LOSS}${NC}"
        fi
        
        # 检查损失是否下降
        FIRST_LOSS=$(grep -E "E1 \[" "$LOG_FILE" | grep "Total:" | head -n 1 | sed 's/.*Total:\([0-9.]*\).*/\1/')
        
        if [ -n "$FIRST_LOSS" ] && [ -n "$LAST_LOSS" ]; then
            echo -e "  • 初始Loss: ${FIRST_LOSS}"
            echo -e "  • 损失下降: $(echo "$FIRST_LOSS - $LAST_LOSS" | bc -l 2>/dev/null || echo "N/A")"
        fi
    fi
    
    echo ""
    echo -e "${BLUE}🔍 健康度检查:${NC}"
    
    # 检查NaN/INF
    if [ -f "$DEBUG_FILE" ]; then
        NAN_COUNT=$(grep -i "NAN DETECTED" "$DEBUG_FILE" 2>/dev/null | wc -l)
        INF_COUNT=$(grep -i "INF DETECTED" "$DEBUG_FILE" 2>/dev/null | wc -l)
        EXPLODE_COUNT=$(grep -i "EXPLODING Gradients" "$DEBUG_FILE" 2>/dev/null | wc -l)
        VANISH_COUNT=$(grep -i "VANISHING Gradients" "$DEBUG_FILE" 2>/dev/null | wc -l)
        
        if [ $NAN_COUNT -eq 0 ] && [ $INF_COUNT -eq 0 ]; then
            echo -e "  • 数值稳定性: ${GREEN}正常${NC}"
        else
            echo -e "  • 数值稳定性: ${RED}异常 (NaN:$NAN_COUNT, INF:$INF_COUNT)${NC}"
        fi
        
        if [ $EXPLODE_COUNT -eq 0 ]; then
            echo -e "  • 梯度爆炸: ${GREEN}无${NC}"
        else
            echo -e "  • 梯度爆炸: ${YELLOW}检测到 $EXPLODE_COUNT 次${NC}"
        fi
        
        if [ $VANISH_COUNT -eq 0 ]; then
            echo -e "  • 梯度消失: ${GREEN}无${NC}"
        else
            echo -e "  • 梯度消失: ${YELLOW}检测到 $VANISH_COUNT 次${NC}"
        fi
    fi
    
    echo ""
    echo -e "${BLUE}📁 生成的文件:${NC}"
    echo -e "  • 训练日志: ${LOG_FILE}"
    echo -e "  • 调试日志: ${DEBUG_FILE}"
    
    if [ -d "log/RSTPReid/model" ]; then
        MODEL_COUNT=$(ls log/RSTPReid/model/*.pth 2>/dev/null | wc -l)
        echo -e "  • 模型文件: ${MODEL_COUNT} 个"
    fi
    
    if [ "$ENABLE_VISUALIZATION" = true ] && [ -d "visualizations/quick_test_${DISENTANGLE_TYPE}_${FREQ_TYPE}" ]; then
        VIZ_COUNT=$(ls visualizations/quick_test_${DISENTANGLE_TYPE}_${FREQ_TYPE}/*.png 2>/dev/null | wc -l)
        echo -e "  • 可视化图: ${VIZ_COUNT} 个"
    fi
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}✅ 快速测试通过！模型可以正常工作${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${BLUE}📋 下一步建议:${NC}"
    echo -e "  1. 完整训练 RSTPReid:"
    echo -e "     ${YELLOW}bash rstp.sh${NC}"
    echo ""
    echo -e "  2. 训练 CUHK-PEDES:"
    echo -e "     ${YELLOW}bash cuhk.sh${NC}"
    echo ""
    echo -e "  3. 对比Baseline:"
    echo -e "     ${YELLOW}bash rstp.sh --disentangle-type=gs3${NC}"
    echo ""
    
    exit 0
else
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${RED}❌ 训练失败，退出码: ${TRAIN_EXIT_CODE}${NC}"
    echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}🔍 排查建议:${NC}"
    echo -e "  1. 检查数据集路径: datasets/RSTPReid/"
    echo -e "  2. 检查预训练模型: $CLIP_PATH 和 $VIM_PATH"
    echo -e "  3. 查看错误日志: cat ${LOG_FILE}"
    echo -e "  4. 查看调试日志: cat ${DEBUG_FILE}"
    echo -e "  5. 检查显存: nvidia-smi"
    echo ""
    
    if [ -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}最后10行日志:${NC}"
        tail -n 10 "$LOG_FILE"
    fi
    
    exit 1
fi
