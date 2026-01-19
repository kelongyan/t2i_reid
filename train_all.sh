#!/bin/bash

# ============================================================================
# 联合训练脚本 - 顺序训练三个数据集
# ============================================================================
# 执行顺序：RSTPReid → CUHK-PEDES → ICFG-PEDES
# 
# 训练策略：
#   1. 每个数据集独立训练80 epochs
#   2. 下一个数据集使用上一个数据集的最佳模型作为初始化（可选）
#   3. 每个数据集的日志和模型独立保存
#
# 预计总耗时：
#   - RSTPReid:   ~6-8小时  (3,701类, ~34k样本)
#   - CUHK-PEDES:  ~8-10小时 (11,003类, ~34k样本)
#   - ICFG-PEDES: ~7-9小时  (4,102类, ~54k样本)
#   总计：        ~21-27小时
#
# 使用方法：
#   bash train_all.sh [--continue-on-error] [--transfer-learning] [--enable-optimizations]
#
# 参数说明：
#   --continue-on-error: 如果某个数据集训练失败，继续训练下一个
#   --transfer-learning: 使用迁移学习（前一个数据集的权重初始化）
#   --enable-optimizations: 启用优化策略（早停、学习率预热、BatchNorm预热等）
# ============================================================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 解析参数
CONTINUE_ON_ERROR=false
TRANSFER_LEARNING=false
ENABLE_OPTIMIZATIONS=false

for arg in "$@"; do
    case $arg in
        --continue-on-error)
            CONTINUE_ON_ERROR=true
            shift
            ;;
        --transfer-learning)
            TRANSFER_LEARNING=true
            shift
            ;;
        --enable-optimizations)
            ENABLE_OPTIMIZATIONS=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# 清理缓存函数
clean_cache() {
    echo -e "${BLUE}🧹 Cleaning Python cache...${NC}"
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    echo -e "${GREEN}✓ Cache cleaned${NC}"
}

# 打印分隔线
print_separator() {
    echo ""
    echo "============================================================================"
    echo "$1"
    echo "============================================================================"
    echo ""
}

# 训练函数
train_dataset() {
    local dataset_name=$1
    local script_name=$2
    local pretrained_path=$3
    
    print_separator "🚀 Training $dataset_name"
    
    echo -e "${BLUE}Dataset:${NC} $dataset_name"
    echo -e "${BLUE}Script:${NC} $script_name"
    echo -e "${BLUE}Start Time:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    
    if [ "$ENABLE_OPTIMIZATIONS" = true ]; then
        echo -e "${GREEN}已启用优化策略${NC}"
        echo -e "  ✓ 早停机制（patience=10, min_delta=0.001）"
        echo -e "  ✓ 学习率预热（warmup_steps=1000）"
        echo -e "  ✓ CLIP文本编码器bias重新初始化"
        echo -e "  ✓ 分层学习率优化（Stage 2+）"
        echo -e "  ✓ 分层梯度裁剪"
        echo -e "  ✓ BatchNorm预热（momentum=0.01）"
        echo -e "  ✓ 改进G-S3门控机制（熵正则+差异正则）"
        echo -e "  ✓ 优化损失权重动态调整"
    fi
    
    if [ "$TRANSFER_LEARNING" = true ] && [ -n "$pretrained_path" ]; then
        echo -e "${YELLOW}Using transfer learning from: $pretrained_path${NC}"
    fi
    
    echo ""
    
    # 清理缓存
    clean_cache
    
    # 记录开始时间
    local start_time=$(date +%s)
    
    # 构建训练命令
    TRAIN_CMD="bash \"$script_name\""
    
    if [ "$ENABLE_OPTIMIZATIONS" = true ]; then
        TRAIN_CMD="$TRAIN_CMD --enable-optimizations"
    fi
    
    if [ "$TRANSFER_LEARNING" = true ] && [ -n "$pretrained_path" ]; then
        TRAIN_CMD="$TRAIN_CMD --resume \"$pretrained_path\""
    fi
    
    # 执行训练
    eval $TRAIN_CMD
    
    local exit_code=$?
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$((($duration % 3600) / 60))
    
    echo ""
    echo -e "${BLUE}End Time:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BLUE}Duration:${NC} ${hours}h ${minutes}m"
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✓ $dataset_name training completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ $dataset_name training failed with exit code $exit_code${NC}"
        return $exit_code
    fi
}

# 主训练流程
main() {
    print_separator "🎯 联合训练开始"
    
    echo -e "${BLUE}训练配置:${NC}"
    echo -e "  Continue on error: $CONTINUE_ON_ERROR"
    echo -e "  Transfer learning: $TRANSFER_LEARNING"
    echo -e "  Enable optimizations: $ENABLE_OPTIMIZATIONS"
    echo -e "${BLUE}训练顺序:${NC}"
    echo -e "  1. RSTPReid"
    echo -e "  2. CUHK-PEDES"
    echo -e "  3. ICFG-PEDES"
    echo ""
    
    # 记录总开始时间
    total_start_time=$(date +%s)
    
    # 统计变量
    success_count=0
    failed_count=0
    failed_datasets=""
    
    # ========================================================================
    # 第1阶段：训练 RSTPReid
    # ========================================================================
    train_dataset "RSTPReid" "rstp.sh" ""
    rstp_exit=$?
    
    if [ $rstp_exit -eq 0 ]; then
        success_count=$((success_count + 1))
        rstp_best_model="log/rstp/model/best_rstp.pth"
    else
        failed_count=$((failed_count + 1))
        failed_datasets="$failed_datasets RSTPReid"
        rstp_best_model=""
        
        if [ "$CONTINUE_ON_ERROR" = false ]; then
            echo -e "${RED}❌ Stopping due to RSTPReid training failure${NC}"
            exit $rstp_exit
        fi
    fi
    
    # ========================================================================
    # 第2阶段：训练 CUHK-PEDES
    # ========================================================================
    if [ "$CONTINUE_ON_ERROR" = true ] || [ $rstp_exit -eq 0 ]; then
        train_dataset "CUHK-PEDES" "cuhk.sh" "$rstp_best_model"
        cuhk_exit=$?
        
        if [ $cuhk_exit -eq 0 ]; then
            success_count=$((success_count + 1))
            cuhk_best_model="log/cuhk/model/best_cuhk.pth"
        else
            failed_count=$((failed_count + 1))
            failed_datasets="$failed_datasets CUHK-PEDES"
            cuhk_best_model=""
            
            if [ "$CONTINUE_ON_ERROR" = false ]; then
                echo -e "${RED}❌ Stopping due to CUHK-PEDES training failure${NC}"
                exit $cuhk_exit
            fi
        fi
    fi
    
    # ========================================================================
    # 第3阶段：训练 ICFG-PEDES
    # ========================================================================
    if [ "$CONTINUE_ON_ERROR" = true ] || ([ $rstp_exit -eq 0 ] && [ $cuhk_exit -eq 0 ]); then
        train_dataset "ICFG-PEDES" "icfg.sh" "$cuhk_best_model"
        icfg_exit=$?
        
        if [ $icfg_exit -eq 0 ]; then
            success_count=$((success_count + 1))
        else
            failed_count=$((failed_count + 1))
            failed_datasets="$failed_datasets ICFG-PEDES"
        fi
    fi
    
    # ========================================================================
    # 总结报告
    # ========================================================================
    total_end_time=$(date +%s)
    total_duration=$((total_end_time - total_start_time))
    total_hours=$((total_duration / 3600))
    total_minutes=$((($total_duration % 3600) / 60))
    
    print_separator "📊 训练总结"
    
    echo -e "${BLUE}完成时间:${NC} $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BLUE}总耗时:${NC} ${total_hours}h ${total_minutes}m"
    echo ""
    echo -e "${BLUE}训练统计:${NC}"
    echo -e "  成功: ${GREEN}$success_count${NC}/3"
    echo -e "  失败: ${RED}$failed_count${NC}/3"
    
    if [ $failed_count -gt 0 ]; then
        echo -e "  失败数据集:${RED}$failed_datasets${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}各数据集结果:${NC}"
    
    # RSTPReid结果
    if [ $rstp_exit -eq 0 ]; then
        echo -e "  1. RSTPReid:   ${GREEN}✓ Success${NC}"
        if [ -f "log/rstp/model/best_rstp.pth" ]; then
            echo -e "     最佳模型: log/rstp/model/best_rstp.pth"
        fi
    else
        echo -e "  1. RSTPReid:   ${RED}✗ Failed${NC}"
    fi
    
    # CUHK-PEDES结果
    if [ -n "$cuhk_exit" ]; then
        if [ $cuhk_exit -eq 0 ]; then
            echo -e "  2. CUHK-PEDES: ${GREEN}✓ Success${NC}"
            if [ -f "log/cuhk/model/best_cuhk.pth" ]; then
                echo -e "     最佳模型: log/cuhk/model/best_cuhk.pth"
            fi
        else
            echo -e "  2. CUHK-PEDES: ${RED}✗ Failed${NC}"
        fi
    else
        echo -e "  2. CUHK-PEDES: ${YELLOW}⊘ Skipped${NC}"
    fi
    
    # ICFG-PEDES结果
    if [ -n "$icfg_exit" ]; then
        if [ $icfg_exit -eq 0 ]; then
            echo -e "  3. ICFG-PEDES: ${GREEN}✓ Success${NC}"
            if [ -f "log/icfg/model/best_icfg.pth" ]; then
                echo -e "     最佳模型: log/icfg/model/best_icfg.pth"
            fi
        else
            echo -e "  3. ICFG-PEDES: ${RED}✗ Failed${NC}"
        fi
    else
        echo -e "  3. ICFG-PEDES: ${YELLOW}⊘ Skipped${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}日志文件:${NC}"
    echo -e "  RSTPReid:   log/rstp/log.txt"
    echo -e "  CUHK-PEDES: log/cuhk/log.txt"
    echo -e "  ICFG-PEDES: log/icfg/log.txt"
    
    echo ""
    
    if [ $failed_count -eq 0 ]; then
        echo -e "${GREEN}🎉 所有数据集训练成功完成！${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  训练完成，但有 $failed_count 个数据集失败${NC}"
        exit 1
    fi
}

# 运行主函数
main
