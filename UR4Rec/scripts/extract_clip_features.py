"""
Extract CLIP image features for MovieLens items.

This script:
1. Loads item images from the Multimodal_Datasets directory
2. Uses CLIP to extract visual features
3. Saves features for fast loading during training
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from UR4Rec.models.clip_image_encoder import CLIPImageEncoder


def main():
    parser = argparse.ArgumentParser(description='Extract CLIP features for items')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='UR4Rec/data/Multimodal_Datasets',
        help='Path to multimodal dataset directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='UR4Rec/data/clip_features.pt',
        help='Path to save extracted features'
    )
    parser.add_argument(
        '--clip_model',
        type=str,
        default='ViT-B/32',
        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
        help='CLIP model variant'
    )
    parser.add_argument(
        '--output_dim',
        type=int,
        default=512,
        help='Output embedding dimension'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for feature extraction'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CLIP Feature Extraction")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output path: {args.output_path}")
    print(f"CLIP model: {args.clip_model}")
    print(f"Output dimension: {args.output_dim}")
    print(f"Device: {args.device}")
    print()

    # Create CLIP encoder
    print("Loading CLIP model...")
    clip_encoder = CLIPImageEncoder(
        model_name=args.clip_model,
        output_dim=args.output_dim,
        freeze_encoder=True,
        device=args.device
    )
    print()

    # Build item_id -> image_path mapping
    data_dir = Path(args.data_dir)

    # MovieLens images are in M_ML-100K/image directory
    images_dir = data_dir / 'M_ML-100K' / 'image'

    if not images_dir.exists():
        print(f"错误: 找不到图片目录: {images_dir}")
        return

    # Count total items from image files
    # Images are named: 1.png, 2.png, ..., 1682.png
    all_image_files = list(images_dir.glob('*.png'))
    if not all_image_files:
        print(f"错误: 图片目录中没有找到PNG文件: {images_dir}")
        return

    # Extract item IDs from filenames to determine range
    item_ids_from_files = []
    for img_path in all_image_files:
        try:
            item_id = int(img_path.stem)
            item_ids_from_files.append(item_id)
        except ValueError:
            print(f"警告: 忽略非数字文件名: {img_path.name}")

    if not item_ids_from_files:
        print("错误: 没有找到有效的物品图片")
        return

    num_items = max(item_ids_from_files)
    print(f"检测到的物品总数: {num_items}")
    print(f"找到的图片文件总数: {len(all_image_files)}")
    print()

    # Build item_id -> image_path mapping with verification
    item_image_paths = {}
    print("构建 item_id -> image_path 映射...")

    for item_id in range(1, num_items + 1):
        # Try different image formats
        for ext in ['.png', '.jpg', '.jpeg']:
            image_path = images_dir / f"{item_id}{ext}"
            if image_path.exists():
                item_image_paths[item_id] = str(image_path)
                break

    print(f"✓ 成功映射 {len(item_image_paths)}/{num_items} 个物品到图片文件")

    # Verification: Show sample mappings to ensure correctness
    print("\n验证映射 (前10个物品):")
    for item_id in range(1, min(11, num_items + 1)):
        if item_id in item_image_paths:
            print(f"  Item {item_id:4d} -> {Path(item_image_paths[item_id]).name}")
        else:
            print(f"  Item {item_id:4d} -> [无图片]")
    print()

    # Extract features
    print("提取 CLIP 特征...")
    item_features = clip_encoder.precompute_item_features(
        item_image_paths=item_image_paths,
        num_items=num_items,
        batch_size=args.batch_size
    )
    print()

    # Save features
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip_encoder.save_features(item_features, str(output_path))

    print()
    print("=" * 60)
    print("验证 Item ID 对应关系")
    print("=" * 60)

    # Critical verification: Ensure item_features[i] corresponds to item i
    # We'll spot-check by re-encoding a few items and comparing
    test_item_ids = [1, 10, 100, min(500, num_items), min(1000, num_items)]
    print("抽查验证 (重新编码几个物品并比较特征):")

    # Ensure model is in eval mode for deterministic verification
    clip_encoder.eval()

    all_match = True
    for item_id in test_item_ids:
        if item_id not in item_image_paths:
            continue

        # Re-encode this single item
        with torch.no_grad():  # Disable gradients for verification
            single_features = clip_encoder.encode_images_batch([item_image_paths[item_id]])
            single_projected = clip_encoder.forward(single_features)

        # Compare with stored features
        stored_features = item_features[item_id].unsqueeze(0)
        cosine_sim = F.cosine_similarity(single_projected, stored_features, dim=-1).item()

        match_status = "✓ 匹配" if cosine_sim > 0.999 else "✗ 不匹配"
        if cosine_sim <= 0.999:
            all_match = False

        print(f"  Item {item_id:4d} ({Path(item_image_paths[item_id]).name:15s}): "
              f"相似度 = {cosine_sim:.6f} {match_status}")

    print()
    if all_match:
        print("✓ 验证通过: 所有抽查的物品特征都正确对应!")
    else:
        print("✗ 警告: 发现对应关系不匹配，请检查!")
        print("注意: 如果相似度接近1.0 (>0.95)，这通常是由于数值精度造成的，不影响使用。")
        return

    print()
    print("=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"特征已保存至: {output_path}")
    print(f"特征形状: {item_features.shape}")
    print(f"特征维度: {args.output_dim}")
    print(f"有效物品数: {len(item_image_paths)}/{num_items}")
    print()
    print("✓ Item ID 对应关系已验证正确，不会张冠李戴!")
    print()
    print("使用方法:")
    print(f"在训练脚本中添加参数: --clip_features_path {output_path}")


if __name__ == '__main__':
    main()
