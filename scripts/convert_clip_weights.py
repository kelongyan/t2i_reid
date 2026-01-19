import torch
from safetensors.torch import save_file, load_file
import os
import logging
import argparse
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')

def convert(args):
    print("=" * 60)
    print("📦 CLIP 权重格式转换器 (.bin -> .safetensors)")
    print("=" * 60)

    # 路径处理
    base_path = Path(args.model_path)
    bin_path = base_path / "pytorch_model.bin"
    safe_path = base_path / "model.safetensors"

    if not bin_path.exists():
        print(f"❌ 错误: 在该路径下找不到 pytorch_model.bin: {base_path}")
        print("   请确认路径是否正确，或者模型是否已经下载。")
        return

    if safe_path.exists() and not args.force:
        print(f"⚠️  警告: model.safetensors 已经存在: {safe_path}")
        print("   使用 --force 参数可覆盖。")
        return

    print(f"📂 正在加载旧权重: {bin_path}")
    print("   注意: 这可能需要几秒钟，并且会消耗内存...")
    
    try:
        # 使用 CPU 加载以节省显存
        # 强制允许 pickle 加载，因为这是我们自己的转换脚本
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
        print(f"   ✅ 加载成功，包含 {len(state_dict)} 个张量")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        print("   (提示: 请确保您的 torch 版本支持 weights_only 参数，或尝试更新脚本)")
        return

    print(f"💾 正在保存为 SafeTensors: {safe_path}")
    try:
        save_file(state_dict, safe_path)
        print("   ✅ 保存成功")
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return

    # 验证步骤
    print("\n🔍 正在验证新文件...")
    try:
        loaded_dict = load_file(safe_path)
        # 简单比对 key 数量
        if len(loaded_dict) == len(state_dict):
            print("   ✅ 验证通过！文件可读且 key 数量一致。")
        else:
            print(f"   ⚠️ 警告: Key 数量不一致 ({len(loaded_dict)} vs {len(state_dict)})")
    except Exception as e:
        print(f"❌ 验证失败 (文件可能已损坏): {e}")
        return

    print("\n🎉 转换完成！您现在可以安全地使用 CLIP 了。")
    if args.delete_old:
        print(f"🗑️  正在删除旧文件: {bin_path}")
        try:
            os.remove(bin_path)
            print("   已删除。")
        except OSError as e:
            print(f"   ❌ 删除失败: {e}")
    else:
        print(f"ℹ️  旧文件已保留: {bin_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to SafeTensors for CLIP")
    parser.add_argument('--model-path', type=str, default="pretrained/clip-vit-base-patch16", 
                        help="包含 pytorch_model.bin 的文件夹路径")
    parser.add_argument('--force', action='store_true', help="如果目标文件存在，强制覆盖")
    parser.add_argument('--delete-old', action='store_true', help="转换成功后删除 .bin 文件")
    
    args = parser.parse_args()
    convert(args)
