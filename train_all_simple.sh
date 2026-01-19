#!/bin/bash

# ============================================================================
# 快速联合训练脚本 - 简化版
# ============================================================================
# 用法：bash train_all_simple.sh
# ============================================================================

echo "=================================================="
echo "🎯 联合训练开始"
echo "=================================================="
echo "训练顺序: RSTPReid → CUHK-PEDES → ICFG-PEDES"
echo "预计总耗时: ~21-27小时"
echo ""

# 记录开始时间
START_TIME=$(date +%s)

# 训练 RSTPReid
echo ""
echo "=========================================="
echo "1/3 训练 RSTPReid..."
echo "=========================================="
bash rstp.sh
if [ $? -ne 0 ]; then
    echo "❌ RSTPReid 训练失败"
    exit 1
fi
echo "✓ RSTPReid 完成"

# 训练 CUHK-PEDES
echo ""
echo "=========================================="
echo "2/3 训练 CUHK-PEDES..."
echo "=========================================="
bash cuhk.sh
if [ $? -ne 0 ]; then
    echo "❌ CUHK-PEDES 训练失败"
    exit 1
fi
echo "✓ CUHK-PEDES 完成"

# 训练 ICFG-PEDES
echo ""
echo "=========================================="
echo "3/3 训练 ICFG-PEDES..."
echo "=========================================="
bash icfg.sh
if [ $? -ne 0 ]; then
    echo "❌ ICFG-PEDES 训练失败"
    exit 1
fi
echo "✓ ICFG-PEDES 完成"

# 计算总耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "=================================================="
echo "🎉 所有训练完成！"
echo "=================================================="
echo "总耗时: ${HOURS}h ${MINUTES}m"
echo ""
echo "模型保存位置:"
echo "  - log/rstp/model/best_rstp.pth"
echo "  - log/cuhk/model/best_cuhk.pth"
echo "  - log/icfg/model/best_icfg.pth"
echo ""
echo "日志文件:"
echo "  - log/rstp/log.txt"
echo "  - log/cuhk/log.txt"
echo "  - log/icfg/log.txt"
echo "=================================================="
