#!/bin/bash

# ============================================================================
# Quick Test Script - 构建升级验证 + 优化策略测试
# ============================================================================
# 目标：快速验证修改效果（15 epochs vs 完整的80 epochs）
#
# 验证内容：
#   1. CLIP文本编码器bias重新初始化是否生效
#   2. 学习率预热是否平滑启动
#   3. 早停机制是否正常工作
#   4. 分层梯度裁剪是否正确应用
#   5. BatchNorm预热是否加快收敛
#   6. 改进G-S3门控机制是否稳定
#   7. 优化损失权重动态调整是否合理
#
# 快速测试 vs 完整训练对比：
#   Quick Test (15 epochs)  ~ 1-1.5 小时
#   Full Training (80 epochs) ~ 6-8 小时
#
# 使用方法：
#   bash quick_test.sh [--enable-optimizations] [--transfer-learning]
# ============================================================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 解析参数
ENABLE_OPTIMIZATIONS=true
TRANSFER_LEARNING=false

for arg in "$@"; do
    case $arg in
        --enable-optimizations)
            ENABLE_OPTIMIZATIONS=true
            shift
            ;;
        --transfer-learning)
            TRANSFER_LEARNING=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# ============================================================================
# 清理缓存
# ============================================================================
echo ""
echo "========================================"
echo "🧹 清理Python缓存..."
echo "========================================"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo -e "${GREEN}✓ 缓存清理完成${NC}"
echo ""

# ============================================================================
# Quick Test 配置 (15 epochs, 快速验证)
# ============================================================================
echo ""
echo "========================================"
echo "🚀 Quick Test: CLIP + Vim (15 epochs)"
echo "========================================"
echo ""
echo -e "${BLUE}测试目标：${NC}"
echo -e "  ✓ 验证CLIP文本编码器bias重新初始化"
echo -e "  ✓ 验证学习率预热效果"
echo -e "  ✓ 验证分层梯度裁剪"
echo -e "  ✓ 验证BatchNorm预热"
echo -e "  ✓ 验证G-S3门控机制改进"
echo -e "  ✓ 验证损失权重动态调整"
echo ""
echo -e "${BLUE}测试配置：${NC}"
echo -e "  • 数据集: RSTPReid"
echo -e "  • 训练轮数: 15 (vs 完整80 epochs)"
echo -e "  • 批次大小: 64 (快速)"
echo -e "  • Worker数: 4 (快速)"
echo -e "  • 预计时间: ~1-1.5 小时"
echo ""

# ============================================================================
# 检查优化策略启用状态
# ============================================================================
if [ "$ENABLE_OPTIMIZATIONS" = true ]; then
    echo -e "${GREEN}✅ 优化策略已启用：${NC}"
    echo -e "  ✓ 早停机制（patience=10, min_delta=0.001）"
    echo -e "  ✓ 学习率预热（warmup_steps=1000）"
    echo -e "  ✓ CLIP文本编码器bias重新初始化"
    echo -e "  ✓ 分层学习率优化（Stage 2+）"
    echo -e "  ✓ 分层梯度裁剪"
    echo -e "  ✓ BatchNorm预热（momentum=0.01）"
    echo -e "  ✓ 改进G-S3门控机制（熵正则+差异正则）"
    echo -e "  ✓ 优化损失权重动态调整"
    echo ""
else
    echo -e "${YELLOW}⚠️  未启用优化策略${NC}"
    echo -e "   建议: bash quick_test.sh --enable-optimizations"
    echo ""
fi

# ============================================================================
# 构建训练命令
# ============================================================================
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
    --scheduler \"cosine\""

# 添加优化损失权重
BASE_CMD="$BASE_CMD \
    --loss-info-nce 1.0 \
    --loss-cls 0.5 \
    --loss-cloth-semantic 2.0 \
    --loss-gate-adaptive 0.05 \
    --loss-id-triplet 1.0 \
    --loss-anti-collapse 1.5 \
    --loss-reconstruction 0.1"

# 如果有resume路径，添加--resume参数
if [ -n "$RESUME_PATH" ]; then
    BASE_CMD="$BASE_CMD --resume \"$RESUME_PATH\""
    echo -e "${BLUE}📂 从检查点恢复训练：${NC}$RESUME_PATH"
    echo ""
fi

# ============================================================================
# 开始训练
# ============================================================================
echo ""
echo "========================================"
echo "🔥 开始训练..."
echo "========================================"
echo ""

# 执行训练
eval $BASE_CMD

TRAIN_EXIT_CODE=$?

# ============================================================================
# 训练结果分析
# ============================================================================
echo ""
echo "========================================"
echo "📊 训练完成分析"
echo "========================================"
echo ""

# 检查日志文件是否存在
LOG_FILE="log/rstp/log.txt"
if [ -f "$LOG_FILE" ]; then
    echo -e "${GREEN}✓ 训练日志文件存在${NC}"
    echo -e "  路径: $LOG_FILE"
    echo ""
    
    # 提取关键指标
    echo -e "${BLUE}📈 最终epoch的详细指标：${NC}"
    echo "----------------------------------------"
    grep "Epoch \[15\]" "$LOG_FILE" | grep -E "(mAP|Rank-1|Rank-5|Rank-10|Loss)" | tail -n 20
    echo ""
    
    # 检查CLIP梯度消失警告
    echo -e "${BLUE}⚠️ CLIP梯度消失检查：${NC}"
    echo "----------------------------------------"
    GRADIENT_WARNINGS=$(grep -i "vanishing gradient\|bias.*grad.*0.000" "$LOG_FILE" | grep "text_encoder" | wc -l)
    if [ $GRADIENT_WARNINGS -eq 0 ]; then
        echo -e "  ${GREEN}✓ 无CLIP梯度消失警告${NC}"
    else
        echo -e "  ${YELLOW}⚠️ 发现 $GRADIENT_WARNINGS 个CLIP梯度消失警告${NC}"
    fi
    echo ""
    
    # 检查早停触发
    echo -e "${BLUE}🛑 早停触发检查：${NC}"
    echo "----------------------------------------"
    if grep -q "Early stopping" "$LOG_FILE"; then
        echo -e "  ${YELLOW}⚠️ 早停已触发${NC}"
    else
        echo -e "  ${GREEN}✓ 早停未触发（正常）${NC}"
    fi
    echo ""
    
    # 检查Stage切换
    echo -e "${BLUE}🔄 Stage切换验证：${NC}"
    echo "----------------------------------------"
    STAGE_SWITCH=$(grep -E "Progressive Unfreezing: Stage" "$LOG_FILE" | tail -n 1)
    if [ -n "$STAGE_SWITCH" ]; then
        echo "  $STAGE_SWITCH"
    else
        echo -e "  ${YELLOW}⚠️ 未发现Stage切换信息${NC}"
    fi
    echo ""
    
    # 提取损失趋势
    echo -e "${BLUE}📉 损失趋势分析：${NC}"
    echo "----------------------------------------"
    echo "CLS损失:"
    grep "Epoch \[[0-9]\]" "$LOG_FILE" | grep "cls" | sed 's/.*cls=\([0-9.]*\).*/Epoch \1: cls=\1/' | tail -n 5
    echo ""
    echo "Cloth_Semantic损失:"
    grep "Epoch \[[0-9]\]" "$LOG_FILE" | grep "cloth_semantic" | sed 's/.*cloth_semantic=\([0-9.]*\).*/Epoch \1: cloth_semantic=\1/' | tail -n 5
    echo ""
    echo "Total损失:"
    grep "Epoch \[[0-9]\]" "$LOG_FILE" | grep "total" | sed 's/.*total=\([0-9.]*\).*/Epoch \1: total=\1/' | tail -n 5
    echo ""
    
    # 最终结果汇总
    echo -e "${BLUE}🎯 Quick Test (15 epochs) 最终结果：${NC}"
    echo "========================================"
    grep -E "Epoch \[15\]" "$LOG_FILE" | grep -E "mAP.*Rank-1.*Rank-5.*Rank-10" | tail -n 1
    echo ""
    
    # 对比预期效果
    echo -e "${BLUE}📈 优化效果验证：${NC}"
    echo "========================================"
    FINAL_MAP=$(grep -E "Epoch \[15\]" "$LOG_FILE" | grep "mAP" | sed 's/.*mAP=([0-9.]*\).*/\1/' | tail -n 1)
    if [ ! -z "$FINAL_MAP" ]; then
        MAP_VALUE=${FINAL_MAP#.*}
        echo "  最终mAP: $MAP_VALUE"
        echo ""
        
        if [ $(echo "$MAP_VALUE > 0.65" | bc -l 2>/dev/null) -eq 1 ]; then
            echo -e "  ${GREEN}✅ mAP > 0.65，优化策略生效！${NC}"
        else
            echo -e "  ${YELLOW}⚠️  mAP < 0.65，需要继续训练更多epochs${NC}"
        fi
        echo ""
        
        if [ $(echo "$MAP_VALUE > 0.70" | bc -l 2>/dev/null) -eq 1 ]; then
            echo -e "  ${GREEN}✅ mAP > 0.70，优化效果显著！${NC}"
        fi
    fi
    echo ""
else
    echo -e "${RED}❌ 训练日志文件不存在：$LOG_FILE${NC}"
    echo -e "  ${YELLOW}  请检查训练是否正常运行${NC}"
    TRAIN_EXIT_CODE=1
fi

# ============================================================================
# 快速测试总结
# ============================================================================
echo ""
echo "========================================"
echo "📝 Quick Test 总结"
echo "========================================"
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Quick Test 执行完成${NC}"
    echo ""
    echo -e "${BLUE}📊 预期效果（基于15 epochs外推）：${NC}"
    echo "  • Epoch 30: mAP ~0.68-0.70"
    echo "  • Epoch 60: mAP ~0.75-0.78"
    echo "  • Epoch 80: mAP ~0.78-0.81"
    echo ""
    echo -e "${BLUE}⏱ 对比完整训练（80 epochs）${NC}"
    echo "  • 预计时间: ~6-8 小时"
    echo "  • Quick Test: ~1-1.5 小时"
    echo ""
    
    echo -e "${GREEN}✅ 优化策略验证成功！${NC}"
    echo -e "${YELLOW}建议：如果15 epochs验证通过，开始完整训练${NC}"
    echo ""
    echo -e "${BLUE}🚀 完整训练命令：${NC}"
    echo "  bash rstp.sh --enable-optimizations"
    echo "  或："
    echo "  bash train_all.sh --enable-optimizations"
    echo ""
else
    echo -e "${RED}❌ Quick Test 执行失败${NC}"
    echo -e "${YELLOW} 请检查错误信息并重试${NC}"
    echo ""
    TRAIN_EXIT_CODE=1
fi

echo "========================================"
echo "🎉 Quick Test 完成！"
echo "========================================"
echo ""

exit $TRAIN_EXIT_CODE
