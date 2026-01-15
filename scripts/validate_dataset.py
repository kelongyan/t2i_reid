"""
数据集验证脚本
检查训练集和测试集的ID划分是否正确，是否存在数据泄漏
"""
import sys
from pathlib import Path
import argparse

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from datasets.data_builder import DataBuilder


def validate_dataset_split(args):
    """验证数据集划分是否正确"""
    print("=" * 80)
    print("数据集划分验证")
    print("=" * 80)
    
    # 1. 构建数据加载器
    print("\n[步骤 1] 构建数据加载器...")
    data_builder = DataBuilder(args, is_distributed=False)
    
    # 2. 加载训练集
    print("\n[步骤 2] 加载训练集数据...")
    train_loader, _ = data_builder.build_data(is_train=True)
    train_data = train_loader.dataset.data
    
    # 3. 加载测试集（query + gallery）
    print("\n[步骤 3] 加载测试集数据（query和gallery）...")
    query_loader, gallery_loader = data_builder.build_data(is_train=False)
    query_data = query_loader.dataset.data
    gallery_data = gallery_loader.dataset.data
    
    # 4. 提取ID集合
    print("\n[步骤 4] 提取ID集合...")
    
    # 训练集ID（注意：训练集包含正负样本对，需要去重）
    train_pids = set()
    train_samples_count = 0
    train_positive_samples = 0
    train_negative_samples = 0
    
    for item in train_data:
        # item格式: (image_path, cloth_caption, id_caption, pid, cam_id, is_matched)
        pid = item[3]
        is_matched = item[5]
        train_pids.add(pid)
        train_samples_count += 1
        if is_matched == 1:
            train_positive_samples += 1
        else:
            train_negative_samples += 1
    
    # 测试集ID
    query_pids = set(item[3] for item in query_data)
    gallery_pids = set(item[3] for item in gallery_data)
    
    # 5. 统计信息
    print("\n" + "=" * 80)
    print("数据集统计信息")
    print("=" * 80)
    print(f"\n训练集:")
    print(f"  - 总样本数: {train_samples_count:,}")
    print(f"  - 正样本数: {train_positive_samples:,}")
    print(f"  - 负样本数: {train_negative_samples:,}")
    print(f"  - 唯一ID数: {len(train_pids):,}")
    print(f"  - 正负比例: 1:{train_negative_samples/train_positive_samples:.2f}")
    
    print(f"\nQuery集:")
    print(f"  - 总样本数: {len(query_data):,}")
    print(f"  - 唯一ID数: {len(query_pids):,}")
    
    print(f"\nGallery集:")
    print(f"  - 总样本数: {len(gallery_data):,}")
    print(f"  - 唯一ID数: {len(gallery_pids):,}")
    
    # 6. 检查数据泄漏
    print("\n" + "=" * 80)
    print("数据泄漏检查")
    print("=" * 80)
    
    # 训练集与Query集的ID重叠
    train_query_overlap = train_pids & query_pids
    train_query_overlap_ratio = len(train_query_overlap) / len(query_pids) * 100 if query_pids else 0
    
    print(f"\n训练集 vs Query集:")
    print(f"  - Query集中的ID数: {len(query_pids):,}")
    print(f"  - 重叠的ID数: {len(train_query_overlap):,}")
    print(f"  - 重叠比例: {train_query_overlap_ratio:.2f}%")
    
    if train_query_overlap_ratio > 0:
        print(f"  ⚠️  警告：训练集和Query集存在ID重叠！可能存在数据泄漏！")
        # 打印前10个重叠的ID
        overlap_list = sorted(list(train_query_overlap))[:10]
        print(f"  - 重叠ID示例（前10个）: {overlap_list}")
    else:
        print(f"  ✓ 训练集和Query集没有ID重叠")
    
    # 训练集与Gallery集的ID重叠
    train_gallery_overlap = train_pids & gallery_pids
    train_gallery_overlap_ratio = len(train_gallery_overlap) / len(gallery_pids) * 100 if gallery_pids else 0
    
    print(f"\n训练集 vs Gallery集:")
    print(f"  - Gallery集中的ID数: {len(gallery_pids):,}")
    print(f"  - 重叠的ID数: {len(train_gallery_overlap):,}")
    print(f"  - 重叠比例: {train_gallery_overlap_ratio:.2f}%")
    
    if train_gallery_overlap_ratio > 0:
        print(f"  ⚠️  警告：训练集和Gallery集存在ID重叠！可能存在数据泄漏！")
        # 打印前10个重叠的ID
        overlap_list = sorted(list(train_gallery_overlap))[:10]
        print(f"  - 重叠ID示例（前10个）: {overlap_list}")
    else:
        print(f"  ✓ 训练集和Gallery集没有ID重叠")
    
    # Query与Gallery的重叠（这是正常的）
    query_gallery_overlap = query_pids & gallery_pids
    query_gallery_overlap_ratio = len(query_gallery_overlap) / len(query_pids) * 100 if query_pids else 0
    
    print(f"\nQuery集 vs Gallery集:")
    print(f"  - Query集中的ID数: {len(query_pids):,}")
    print(f"  - Gallery集中的ID数: {len(gallery_pids):,}")
    print(f"  - 重叠的ID数: {len(query_gallery_overlap):,}")
    print(f"  - 重叠比例: {query_gallery_overlap_ratio:.2f}%")
    print(f"  ℹ️  注意：Query和Gallery的ID重叠是正常的（用于检索评估）")
    
    # 7. 检查样本质量
    print("\n" + "=" * 80)
    print("样本质量检查")
    print("=" * 80)
    
    # 检查训练集前10个样本
    print(f"\n训练集前10个样本:")
    for i, item in enumerate(train_data[:10]):
        image_path, cloth_caption, id_caption, pid, cam_id, is_matched = item
        match_label = "正样本" if is_matched == 1 else "负样本"
        print(f"  [{i+1}] PID={pid}, {match_label}")
        print(f"      图像: {Path(image_path).name}")
        print(f"      服装描述: {cloth_caption[:60]}...")
        print(f"      身份描述: {id_caption[:60]}...")
    
    # 检查Query集前5个样本
    print(f"\nQuery集前5个样本:")
    for i, item in enumerate(query_data[:5]):
        image_path, cloth_caption, id_caption, pid, cam_id, is_matched = item
        print(f"  [{i+1}] PID={pid}")
        print(f"      图像: {Path(image_path).name}")
        print(f"      服装描述: {cloth_caption[:60]}...")
        print(f"      身份描述: {id_caption[:60]}...")
    
    # 8. 检查每个ID的样本数分布
    print("\n" + "=" * 80)
    print("ID样本数分布统计")
    print("=" * 80)
    
    # 统计训练集中每个ID的样本数（只统计正样本）
    train_pid_counts = {}
    for item in train_data:
        pid = item[3]
        is_matched = item[5]
        if is_matched == 1:  # 只统计正样本
            train_pid_counts[pid] = train_pid_counts.get(pid, 0) + 1
    
    # 统计Query集中每个ID的样本数
    query_pid_counts = {}
    for item in query_data:
        pid = item[3]
        query_pid_counts[pid] = query_pid_counts.get(pid, 0) + 1
    
    # 统计Gallery集中每个ID的样本数
    gallery_pid_counts = {}
    for item in gallery_data:
        pid = item[3]
        gallery_pid_counts[pid] = gallery_pid_counts.get(pid, 0) + 1
    
    print(f"\n训练集ID样本数分布（正样本）:")
    train_counts_list = sorted(train_pid_counts.values())
    print(f"  - 最少样本数: {min(train_counts_list)}")
    print(f"  - 最多样本数: {max(train_counts_list)}")
    print(f"  - 平均样本数: {sum(train_counts_list)/len(train_counts_list):.2f}")
    print(f"  - 中位数样本数: {train_counts_list[len(train_counts_list)//2]}")
    
    print(f"\nQuery集ID样本数分布:")
    query_counts_list = sorted(query_pid_counts.values())
    print(f"  - 最少样本数: {min(query_counts_list)}")
    print(f"  - 最多样本数: {max(query_counts_list)}")
    print(f"  - 平均样本数: {sum(query_counts_list)/len(query_counts_list):.2f}")
    print(f"  - 中位数样本数: {query_counts_list[len(query_counts_list)//2]}")
    
    print(f"\nGallery集ID样本数分布:")
    gallery_counts_list = sorted(gallery_pid_counts.values())
    print(f"  - 最少样本数: {min(gallery_counts_list)}")
    print(f"  - 最多样本数: {max(gallery_counts_list)}")
    print(f"  - 平均样本数: {sum(gallery_counts_list)/len(gallery_counts_list):.2f}")
    print(f"  - 中位数样本数: {gallery_counts_list[len(gallery_counts_list)//2]}")
    
    # 9. 最终结论
    print("\n" + "=" * 80)
    print("验证结论")
    print("=" * 80)
    
    has_data_leak = train_query_overlap_ratio > 0 or train_gallery_overlap_ratio > 0
    
    if has_data_leak:
        print("\n❌ 数据集划分存在问题！")
        print("   训练集和测试集存在ID重叠，可能导致过拟合和虚高的性能指标。")
        print("   建议重新检查数据集的split字段或文件路径划分逻辑。")
    else:
        print("\n✓ 数据集划分正确！")
        print("  训练集和测试集没有ID重叠，数据划分符合要求。")
    
    print("\n" + "=" * 80)
    
    return not has_data_leak


def configuration():
    """配置参数"""
    parser = argparse.ArgumentParser(description="Validate dataset split")
    parser.add_argument('--root', type=str, default=str(ROOT_DIR / 'datasets'),
                       help='Root directory of the dataset')
    parser.add_argument('--dataset-configs', nargs='+', type=str, 
                       help='List of dataset configurations in JSON format')
    parser.add_argument('-b', '--batch-size', type=int, default=64, 
                       help='Batch size (not used, just for compatibility)')
    parser.add_argument('-j', '--workers', type=int, default=4, 
                       help='Number of data loading workers')
    parser.add_argument('--height', type=int, default=224, help='Image height')
    parser.add_argument('--width', type=int, default=224, help='Image width')
    
    args = parser.parse_args()
    
    # 使用默认的CUHK-PEDES配置
    if not args.dataset_configs:
        args.dataset_configs = [
            {
                'name': 'CUHK-PEDES',
                'root': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES'),
                'json_file': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_all.json'),
                'cloth_json': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_cloth.json'),
                'id_json': str(ROOT_DIR / 'datasets' / 'CUHK-PEDES' / 'annotations' / 'caption_id.json')
            }
        ]
    
    args.img_size = (args.height, args.width)
    args.root = str(Path(args.root))
    
    return args


if __name__ == '__main__':
    args = configuration()
    
    try:
        is_valid = validate_dataset_split(args)
        sys.exit(0 if is_valid else 1)
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
