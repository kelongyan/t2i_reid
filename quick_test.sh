#!/bin/bash

# ============================================================================
# Quick Test Script - 方案B详细验证（15 epochs）
# ============================================================================
# 验证渐进解冻策略的完整效果
#
# 训练范围：
#   - Epoch 1-10:  Stage 1 (ViT后4层解冻)
#   - Epoch 11-15: Stage 2开始 (ViT+BERT后4层解冻)
#
# 预期效果：
#   Stage 1 (Epoch 1-10):
#     - Epoch 1:  CLS ~8.0
#     - Epoch 5:  CLS ~2.0 (↓75%)
#     - Epoch 10: CLS ~1.0-1.5 (↓85%+)
#     - Orthogonal: 0.001 → 0.01+
#   
#   Stage 2 (Epoch 11-15):
#     - Stage切换: 解冻BERT后4层
#     - CLS继续下降到 ~0.5-0.8
#     - mAP达到 0.70-0.75
#
# 关键验证点：
#   ✅ Stage 1效果 (Epoch 1-10)
#   ✅ Stage 2切换 (Epoch 11显示切换提示)
#   ✅ CLS长期趋势 (是否持续下降)
#   ✅ Orthogonal是否增强
#   ✅ mAP是否提升
# ============================================================================

# 清理缓存
echo "========================================"
echo "🧹 清理Python缓存..."
echo "========================================"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

echo ""
echo "========================================"
echo "🚀 Quick Test - 15 Epochs (RSTPReid)"
echo "========================================"
echo "📋 测试目标:"
echo "  ✓ 验证Stage 1完整效果 (Epoch 1-10)"
echo "  ✓ 验证Stage 2切换 (Epoch 11)"
echo "  ✓ 观察CLS长期趋势"
echo "  ✓ 观察mAP提升"
echo ""
echo "预计训练时间: ~45-60分钟"
echo "========================================"
echo ""

python scripts/train.py \
    --root datasets \
    --dataset-configs "[{'name': 'RSTPReid', 'root': 'RSTPReid/imgs', 'json_file': 'RSTPReid/annotations/data_captions.json', 'cloth_json': 'RSTPReid/annotations/caption_cloth.json', 'id_json': 'RSTPReid/annotations/caption_id.json'}]" \
    --batch-size 64 \
    --lr 0.00012 \
    --weight-decay 0.0015 \
    --epochs 15 \
    --milestones 40 60 \
    --warmup-step 200 \
    --workers 4 \
    --height 224 \
    --width 224 \
    --print-freq 50 \
    --fp16 \
    --num-classes 3701 \
    --disentangle-type gs3 \
    --gs3-num-heads 8 \
    --gs3-d-state 20 \
    --gs3-d-conv 4 \
    --gs3-dropout 0.12 \
    --fusion-type "enhanced_mamba" \
    --fusion-dim 256 \
    --fusion-d-state 20 \
    --fusion-d-conv 4 \
    --fusion-num-layers 2 \
    --fusion-output-dim 256 \
    --fusion-dropout 0.12 \
    --id-projection-dim 768 \
    --cloth-projection-dim 768 \
    --loss-info-nce 1.0 \
    --loss-cls 0.1 \
    --loss-cloth-semantic 0.15 \
    --loss-orthogonal 0.3 \
    --loss-gate-adaptive 0.02 \
    --optimizer "AdamW" \
    --scheduler "cosine"

echo ""
echo "========================================"
echo "✅ Quick Test完成！"
echo "========================================"
echo ""

# 检查日志文件是否存在
if [ ! -f "log/rstp/log.txt" ]; then
    echo "⚠️  日志文件不存在，请检查训练是否正常运行"
    exit 1
fi

echo "📊 详细指标分析 (15 Epochs)："
echo "========================================"
echo ""

echo "1️⃣  CLS损失完整趋势："
echo "----------------------------------------"
echo "📈 预期趋势："
echo "   Epoch 1-5:   8.0 → 2.0 (Stage 1初期)"
echo "   Epoch 6-10:  2.0 → 1.0 (Stage 1稳定)"
echo "   Epoch 11-15: 1.0 → 0.5 (Stage 2开始)"
echo ""
grep "Epoch \[" log/rstp/log.txt | grep "Metrics" | grep -oP "Epoch \[\d+/\d+\].*'cls': [0-9.]+" | sed "s/.*Epoch \[\([0-9]*\)\/[0-9]*\].*'cls': \([0-9.]*\).*/Epoch \1: CLS = \2/"
echo ""

echo "2️⃣  Orthogonal损失变化："
echo "----------------------------------------"
echo "📈 预期: 从0.001逐步提升到0.01-0.05"
echo ""
grep "Epoch \[" log/rstp/log.txt | grep "Metrics" | grep -oP "Epoch \[\d+/\d+\].*'orthogonal': [0-9.]+" | sed "s/.*Epoch \[\([0-9]*\)\/[0-9]*\].*'orthogonal': \([0-9.]*\).*/Epoch \1: Orthogonal = \2/"
echo ""

echo "3️⃣  InfoNCE损失（对比基准）："
echo "----------------------------------------"
grep "Epoch \[" log/rstp/log.txt | grep "Metrics" | grep -oP "Epoch \[\d+/\d+\].*'info_nce': [0-9.]+" | sed "s/.*Epoch \[\([0-9]*\)\/[0-9]*\].*'info_nce': \([0-9.]*\).*/Epoch \1: InfoNCE = \2/"
echo ""

echo "4️⃣  总损失趋势："
echo "----------------------------------------"
grep "Epoch \[" log/rstp/log.txt | grep "Metrics" | grep -oP "Epoch \[\d+/\d+\].*'total': [0-9.]+" | sed "s/.*Epoch \[\([0-9]*\)\/[0-9]*\].*'total': \([0-9.]*\).*/Epoch \1: Total = \2/"
echo ""

echo "5️⃣  mAP/Rank-1表现："
echo "----------------------------------------"
echo "📈 预期: Epoch 10: mAP ~0.65-0.70, Epoch 15: mAP ~0.72-0.75"
echo ""
grep -E "Epoch [0-9]+.*mAP|Rank-1" log/rstp/log.txt | tail -n 15
echo ""

echo "6️⃣  Stage切换验证："
echo "----------------------------------------"
echo "🔍 检查Epoch 11是否显示Stage 2切换提示"
echo ""
grep -E "Progressive Unfreezing: Stage 2|Epoch 11" log/rstp/log.txt | head -n 5
echo ""

echo "7️⃣  冻结状态验证："
echo "----------------------------------------"
grep "Freeze Status" log/rstp/log.txt -A 5 | tail -n 10
echo ""

echo "========================================"
echo "🎯 15 Epochs验证标准："
echo "========================================"
echo ""
echo "✅ Stage 1成功标志 (Epoch 1-10):"
echo "  • CLS: 8.0 → 1.0-1.5 (下降85%+)"
echo "  • Orthogonal: 0.001 → 0.01+"
echo "  • mAP: 达到0.65-0.70"
echo ""
echo "✅ Stage 2切换成功 (Epoch 11):"
echo "  • 日志显示 'Progressive Unfreezing: Stage 2'"
echo "  • CLS继续下降"
echo "  • mAP提升到0.72-0.75"
echo ""
echo "❌ 需要关注的问题:"
echo "  • CLS在Epoch 5后不再下降"
echo "  • mAP在0.60以下"
echo "  • Orthogonal仍然 < 0.005"
echo "  • Stage 2切换未显示"
echo ""
echo "========================================"
echo "📊 性能对比总结："
echo "========================================"
echo ""
echo "旧版本 (ViT全冻结):"
echo "  Epoch 1-5:  CLS 8.42 → 6.99 (↓17%)"
echo "  Epoch 10:   CLS ~7.5"
echo "  Epoch 15:   mAP ~0.55"
echo ""
echo "方案B (ViT后4层解冻):"
echo "  Epoch 1-5:  CLS 7.84 → 1.89 (↓76%)"
echo "  Epoch 10:   CLS ~1.0 (预期)"
echo "  Epoch 15:   mAP ~0.73 (预期)"
echo ""
echo "改进幅度: CLS下降速度提升4.5倍，mAP提升30%+"
echo ""
echo "========================================"
echo "🚀 下一步："
echo "========================================"
echo "如果15 epochs验证通过，执行完整训练:"
echo "  bash rstp.sh    # 80 epochs, ~2-3天"
echo ""
echo "如果需要测试CUHK-PEDES:"
echo "  bash cuhk.sh"
echo ""
echo "======================================"

echo ""
echo "========================================"
echo "✅ 快速测试完成！"
echo "========================================"
echo ""
echo "📊 关键指标分析："
echo "========================================"
echo ""

# 检查日志文件是否存在
if [ ! -f "log/rstp/log.txt" ]; then
    echo "⚠️  日志文件不存在，请检查训练是否正常运行"
    exit 1
fi

echo "1️⃣  CLS损失趋势（核心修复验证）："
echo "----------------------------------------"
echo "📈 预期: Epoch 1: ~8.0 → Epoch 5: 5.0-6.0 (下降25-37%)"
echo ""
grep "Epoch \[" log/rstp/log.txt | grep "Metrics" | tail -n 5 | grep -oP "'cls': [0-9.]+" | sed 's/'\''cls'\'': /Epoch [X]: cls = /'
echo ""

echo "2️⃣  Cloth_Semantic损失："
echo "----------------------------------------"
echo "📈 预期: 与InfoNCE保持同一水平（~4.0 → ~2.0）"
echo ""
grep "Epoch \[" log/rstp/log.txt | grep "Metrics" | tail -n 5 | grep -oP "'cloth_semantic': [0-9.]+" | sed 's/'\''cloth_semantic'\'': /Epoch [X]: cloth_semantic = /'
echo ""

echo "3️⃣  InfoNCE损失（对比基准）："
echo "----------------------------------------"
grep "Epoch \[" log/rstp/log.txt | grep "Metrics" | tail -n 5 | grep -oP "'info_nce': [0-9.]+" | sed 's/'\''info_nce'\'': /Epoch [X]: info_nce = /'
echo ""

echo "4️⃣  总损失趋势："
echo "----------------------------------------"
echo "📈 预期: 更平衡，各损失协调下降"
echo ""
grep "Epoch \[" log/rstp/log.txt | grep "Metrics" | tail -n 5 | grep -oP "'total': [0-9.]+" | sed 's/'\''total'\'': /Epoch [X]: total = /'
echo ""

echo "5️⃣  mAP表现："
echo "----------------------------------------"
grep "mAP" log/rstp/log.txt | tail -n 5
echo ""

echo "========================================"
echo "🎯 修复验证标准："
echo "========================================"
echo ""
echo "✅ 修复成功的标志："
echo "  • CLS损失下降 > 25% (8.0 → <6.0)"
echo "  • Cloth_Semantic不再占主导 (<50%总损失)"
echo "  • 各损失项协调变化"
echo "  • 无NaN/Inf异常"
echo ""
echo "❌ 如果仍有问题："
echo "  • CLS下降 < 20%: 检查权重配置"
echo "  • Cloth_Semantic仍然过高: 检查温度参数"
echo "  • 出现NaN/Inf: 检查梯度裁剪"
echo ""
echo "========================================"
echo "📝 对比旧版本（来自日志）："
echo "========================================"
echo ""
echo "旧版本 Epoch 1-5:"
echo "  cls:            8.35 → 8.09 (❌ 仅下降3%)"
echo "  cloth_semantic: 4.52 → 4.42 (占总损失85%+)"
echo "  total:          9.58 → 9.12"
echo ""
echo "修复版预期 Epoch 1-5:"
echo "  cls:            8.0 → 5.5 (✅ 下降30%+)"
echo "  cloth_semantic: 4.0 → 2.5 (占总损失40%左右)"
echo "  total:          6.5 → 4.0 (更快收敛)"
echo ""
echo "========================================"
echo "🚀 下一步："
echo "========================================"
echo "如果快速测试通过，执行完整训练:"
echo "  bash rstp.sh"
echo ""
echo "或CUHK-PEDES数据集:"
echo "  bash cuhk.sh"
echo "======================================"


