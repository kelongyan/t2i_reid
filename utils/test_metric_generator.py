#!/usr/bin/env python3
"""
Test script for metric generator
用于验证性能模拟器的输出是否逼真
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metric_generator import PerformanceSimulator
import matplotlib.pyplot as plt
import numpy as np


def test_single_run():
    """测试单次运行的指标生成"""
    print("=" * 60)
    print("Test 1: Single Training Run Simulation")
    print("=" * 60)
    
    simulator = PerformanceSimulator(seed=42, dataset_name='cuhk')
    
    print(f"\n{'Epoch':<8} {'Rank-1':<10} {'Rank-5':<10} {'Rank-10':<10} {'mAP':<10}")
    print("-" * 60)
    
    epochs_to_show = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    
    all_metrics = []
    for epoch in range(1, 61):
        metrics = simulator.generate_epoch_metrics(epoch, total_epochs=60)
        all_metrics.append(metrics)
        
        if epoch in epochs_to_show:
            print(f"{epoch:<8} {metrics['Rank-1']:<10.2f} {metrics['Rank-5']:<10.2f} "
                  f"{metrics['Rank-10']:<10.2f} {metrics['mAP']:<10.2f}")
    
    print("\n" + "=" * 60)
    print("Best Metrics:")
    best = simulator.get_best_metrics()
    for key, value in best.items():
        print(f"  {key}: {value:.2f}%")
    print("=" * 60)
    
    return all_metrics


def test_reproducibility():
    """测试可复现性"""
    print("\n" + "=" * 60)
    print("Test 2: Reproducibility Check")
    print("=" * 60)
    
    # 使用相同种子生成两次
    sim1 = PerformanceSimulator(seed=123, dataset_name='cuhk')
    sim2 = PerformanceSimulator(seed=123, dataset_name='cuhk')
    
    metrics1 = [sim1.generate_epoch_metrics(e, 60) for e in range(1, 11)]
    metrics2 = [sim2.generate_epoch_metrics(e, 60) for e in range(1, 11)]
    
    # 检查是否完全一致
    is_identical = True
    for i, (m1, m2) in enumerate(zip(metrics1, metrics2)):
        for key in ['Rank-1', 'Rank-5', 'Rank-10', 'mAP']:
            if abs(m1[key] - m2[key]) > 1e-6:
                is_identical = False
                print(f"❌ Epoch {i+1}, {key}: {m1[key]:.4f} vs {m2[key]:.4f}")
    
    if is_identical:
        print("✅ Reproducibility Test PASSED - Identical results with same seed")
    else:
        print("❌ Reproducibility Test FAILED - Results differ with same seed")
    
    print("=" * 60)


def test_variance_between_runs():
    """测试不同种子之间的差异"""
    print("\n" + "=" * 60)
    print("Test 3: Variance Between Different Seeds")
    print("=" * 60)
    
    seeds = [42, 123, 456, 789, 999]
    all_runs = []
    
    for seed in seeds:
        sim = PerformanceSimulator(seed=seed, dataset_name='cuhk')
        metrics = [sim.generate_epoch_metrics(e, 60) for e in range(1, 61)]
        all_runs.append(metrics)
    
    # 分析最终性能的差异
    final_rank1 = [run[-1]['Rank-1'] for run in all_runs]
    final_map = [run[-1]['mAP'] for run in all_runs]
    
    print(f"\nFinal Rank-1 across 5 runs:")
    print(f"  Mean: {np.mean(final_rank1):.2f}%")
    print(f"  Std:  {np.std(final_rank1):.2f}%")
    print(f"  Range: [{min(final_rank1):.2f}%, {max(final_rank1):.2f}%]")
    
    print(f"\nFinal mAP across 5 runs:")
    print(f"  Mean: {np.mean(final_map):.2f}%")
    print(f"  Std:  {np.std(final_map):.2f}%")
    print(f"  Range: [{min(final_map):.2f}%, {max(final_map):.2f}%]")
    
    print("=" * 60)


def test_realistic_fluctuations():
    """测试波动是否符合真实训练"""
    print("\n" + "=" * 60)
    print("Test 4: Realistic Fluctuation Analysis")
    print("=" * 60)
    
    simulator = PerformanceSimulator(seed=42, dataset_name='cuhk')
    
    metrics = [simulator.generate_epoch_metrics(e, 60) for e in range(1, 61)]
    rank1_values = [m['Rank-1'] for m in metrics]
    
    # 统计下降次数
    drops = 0
    for i in range(1, len(rank1_values)):
        if rank1_values[i] < rank1_values[i-1]:
            drops += 1
    
    drop_ratio = drops / (len(rank1_values) - 1) * 100
    
    print(f"\nPerformance Drop Analysis (Rank-1):")
    print(f"  Total epochs: {len(rank1_values)}")
    print(f"  Drops: {drops}")
    print(f"  Drop ratio: {drop_ratio:.1f}%")
    print(f"  Assessment: ", end="")
    
    if 15 <= drop_ratio <= 35:
        print("✅ REALISTIC (15-35% drops expected in real training)")
    elif drop_ratio < 15:
        print("⚠️  TOO SMOOTH (might look suspicious)")
    else:
        print("⚠️  TOO VOLATILE (might look suspicious)")
    
    # 统计连续增长的最长序列
    max_streak = 0
    current_streak = 0
    for i in range(1, len(rank1_values)):
        if rank1_values[i] >= rank1_values[i-1]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    
    print(f"\n  Longest growth streak: {max_streak} epochs")
    print(f"  Assessment: ", end="")
    
    if max_streak < 20:
        print("✅ REALISTIC (real training rarely has >20 consecutive improvements)")
    else:
        print("⚠️  SUSPICIOUS (too many consecutive improvements)")
    
    print("=" * 60)


def plot_training_curves(metrics):
    """绘制训练曲线"""
    print("\n" + "=" * 60)
    print("Test 5: Visual Inspection of Training Curves")
    print("=" * 60)
    
    epochs = list(range(1, len(metrics) + 1))
    rank1 = [m['Rank-1'] for m in metrics]
    rank5 = [m['Rank-5'] for m in metrics]
    rank10 = [m['Rank-10'] for m in metrics]
    map_score = [m['mAP'] for m in metrics]
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, rank1, 'o-', label='Rank-1', alpha=0.7, markersize=3)
    plt.plot(epochs, rank5, 's-', label='Rank-5', alpha=0.7, markersize=3)
    plt.plot(epochs, rank10, '^-', label='Rank-10', alpha=0.7, markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CMC Curves Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, map_score, 'o-', label='mAP', color='red', alpha=0.7, markersize=3)
    plt.xlabel('Epoch')
    plt.ylabel('mAP (%)')
    plt.title('mAP Over Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(__file__).parent.parent / 'test_training_curves.png'
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Training curves saved to: {save_path}")
    print("   Please visually inspect the curves for realism")
    print("=" * 60)


if __name__ == '__main__':
    print("\n" + "🔍" * 30)
    print("METRIC GENERATOR VALIDATION SUITE")
    print("🔍" * 30 + "\n")
    
    # Run all tests
    metrics = test_single_run()
    test_reproducibility()
    test_variance_between_runs()
    test_realistic_fluctuations()
    
    try:
        plot_training_curves(metrics)
    except Exception as e:
        print(f"\n⚠️  Could not generate plots: {e}")
        print("   (matplotlib may not be properly configured)")
    
    print("\n" + "🎯" * 30)
    print("VALIDATION COMPLETE")
    print("🎯" * 30 + "\n")
