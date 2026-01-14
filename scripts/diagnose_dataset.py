#!/usr/bin/env python
"""
数据集划分诊断脚本
用于检查Query和Gallery的实际情况，找出高mAP的根本原因
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
import argparse

def load_dataset_info(dataset_root):
    """加载数据集信息"""
    data_file = Path(dataset_root) / "RSTPReid" / "annotations" / "data_captions.json"
    
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return None
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # 按PID分组
    items_by_pid = defaultdict(list)
    for item in data:
        pid = item['id']
        img_path = item['file_path']
        items_by_pid[pid].append(img_path)
    
    return items_by_pid


def analyze_split(query_data, gallery_data):
    """分析Query和Gallery的实际划分情况"""
    
    print("=" * 80)
    print("数据集划分诊断报告")
    print("=" * 80)
    print()
    
    # 1. 基本统计
    print("【1. 基本统计】")
    query_images = {item[0] for item in query_data}
    gallery_images = {item[0] for item in gallery_data}
    query_pids = {item[3] for item in query_data}
    gallery_pids = {item[3] for item in gallery_data}
    
    print(f"  Query 图像数: {len(query_images)}")
    print(f"  Gallery 图像数: {len(gallery_images)}")
    print(f"  Query 人物ID数: {len(query_pids)}")
    print(f"  Gallery 人物ID数: {len(gallery_pids)}")
    print()
    
    # 2. 图像重叠检查（关键！）
    print("【2. 图像重叠检查】⚠️ 关键诊断")
    image_overlap = query_images & gallery_images
    if image_overlap:
        print(f"  ❌ 发现 {len(image_overlap)} 个重复图像！")
        print(f"  重复率: {len(image_overlap)/len(query_images)*100:.2f}%")
        print(f"  样例重复图像（前10个）:")
        for img in list(image_overlap)[:10]:
            print(f"    - {img}")
        print()
        print("  ⚠️ 这就是高mAP的原因：数据泄漏！")
    else:
        print(f"  ✅ 没有重复图像")
    print()
    
    # 3. ID重叠检查
    print("【3. ID重叠检查】")
    id_overlap = query_pids & gallery_pids
    print(f"  重叠ID数: {len(id_overlap)}")
    print(f"  Query专属ID: {len(query_pids - gallery_pids)}")
    print(f"  Gallery专属ID: {len(gallery_pids - query_pids)}")
    print(f"  ID重叠率: {len(id_overlap)/len(query_pids)*100:.2f}%")
    print()
    
    # 4. 每个ID的图像数量分布
    print("【4. 每个ID的图像数量分布】")
    query_pid_counts = defaultdict(int)
    gallery_pid_counts = defaultdict(int)
    
    for item in query_data:
        query_pid_counts[item[3]] += 1
    
    for item in gallery_data:
        gallery_pid_counts[item[3]] += 1
    
    # 检查重叠ID的图像分布
    overlap_analysis = []
    for pid in list(id_overlap)[:10]:  # 只检查前10个
        q_count = query_pid_counts[pid]
        g_count = gallery_pid_counts[pid]
        total = q_count + g_count
        overlap_analysis.append((pid, q_count, g_count, total))
    
    print("  重叠ID的图像分布（前10个ID）:")
    print("  PID | Query图像数 | Gallery图像数 | 总数")
    print("  " + "-" * 50)
    for pid, q_count, g_count, total in overlap_analysis:
        print(f"  {pid:4d} | {q_count:11d} | {g_count:13d} | {total:4d}")
    print()
    
    # 5. 实际检查是否有图像重复（逐ID检查）
    print("【5. 逐ID检查图像重复】")
    query_by_pid = defaultdict(list)
    gallery_by_pid = defaultdict(list)
    
    for item in query_data:
        query_by_pid[item[3]].append(item[0])
    
    for item in gallery_data:
        gallery_by_pid[item[3]].append(item[0])
    
    total_duplicates = 0
    duplicate_pids = []
    
    for pid in id_overlap:
        q_imgs = set(query_by_pid[pid])
        g_imgs = set(gallery_by_pid[pid])
        dup = q_imgs & g_imgs
        if dup:
            total_duplicates += len(dup)
            duplicate_pids.append((pid, len(dup)))
    
    if total_duplicates > 0:
        print(f"  ❌ 发现 {total_duplicates} 个重复图像")
        print(f"  涉及 {len(duplicate_pids)} 个ID")
        print(f"  重复最多的ID（前5个）:")
        for pid, dup_count in sorted(duplicate_pids, key=lambda x: x[1], reverse=True)[:5]:
            print(f"    PID {pid}: {dup_count} 个重复图像")
            print(f"      Query: {query_by_pid[pid][:3]}")
            print(f"      Gallery: {gallery_by_pid[pid][:3]}")
    else:
        print(f"  ✅ 没有发现图像重复")
    print()
    
    # 6. 诊断结论
    print("=" * 80)
    print("【诊断结论】")
    print("=" * 80)
    
    if image_overlap or total_duplicates > 0:
        print("⚠️⚠️⚠️ 数据泄漏检测阳性")
        print()
        print("问题：Query和Gallery包含相同的图像")
        print(f"  - 重复图像数: {len(image_overlap) if image_overlap else total_duplicates}")
        print(f"  - 这导致模型可以直接记住图像，而非学习语义匹配")
        print(f"  - 这就是第一个epoch就有 {0.7974:.2%} mAP 的原因")
        print()
        print("建议：")
        print("  1. 检查 data_builder.py 的划分逻辑")
        print("  2. 确保每个ID只shuffle一次")
        print("  3. Query和Gallery使用同一次shuffle的不同部分")
    else:
        print("✅ 数据集划分正常")
        print()
        print("如果仍然出现异常高的mAP，可能原因：")
        print("  1. 训练集和测试集有ID重叠（检查 train/test split）")
        print("  2. 模型过拟合")
        print("  3. 评估代码有bug")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='诊断数据集划分问题')
    parser.add_argument('--root', type=str, default='datasets', help='数据集根目录')
    parser.add_argument('--query-file', type=str, help='Query数据文件（可选）')
    parser.add_argument('--gallery-file', type=str, help='Gallery数据文件（可选）')
    args = parser.parse_args()
    
    # 方法1: 如果提供了具体的query/gallery文件
    if args.query_file and args.gallery_file:
        print("从文件加载Query和Gallery数据...")
        # 这里需要根据实际数据格式解析
        pass
    
    # 方法2: 从训练脚本中提取（推荐）
    print("请在训练脚本中添加以下代码来导出Query和Gallery数据：")
    print()
    print("```python")
    print("# 在 scripts/train.py 的数据加载后添加")
    print("import pickle")
    print("with open('debug_query_data.pkl', 'wb') as f:")
    print("    pickle.dump(args.query_data, f)")
    print("with open('debug_gallery_data.pkl', 'wb') as f:")
    print("    pickle.dump(args.gallery_data, f)")
    print("```")
    print()
    print("然后运行：")
    print("python diagnose_dataset.py --query-file debug_query_data.pkl --gallery-file debug_gallery_data.pkl")
    print()
    
    # 如果文件存在，加载并分析
    if Path('debug_query_data.pkl').exists() and Path('debug_gallery_data.pkl').exists():
        import pickle
        print("发现调试文件，开始分析...")
        with open('debug_query_data.pkl', 'rb') as f:
            query_data = pickle.load(f)
        with open('debug_gallery_data.pkl', 'rb') as f:
            gallery_data = pickle.load(f)
        
        analyze_split(query_data, gallery_data)
    else:
        print("未找到调试文件，请先运行训练脚本生成数据")


if __name__ == "__main__":
    main()
