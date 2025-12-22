#!/usr/bin/env python3
"""
Extract CLIP image features for ML-1M items.

This script:
1. Loads item images from M_ML-1M/image directory
2. Uses CLIP to extract visual features
3. Saves features to ml1m folder for training
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

from UR4Rec.models.clip_image_encoder import CLIPImageEncoder


def main():
    parser = argparse.ArgumentParser(description='Extract CLIP features for ML-1M items')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='UR4Rec/data/Multimodal_Datasets',
        help='Path to multimodal dataset directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='UR4Rec/data/ml1m/clip_features.pt',
        help='Path to save extracted features (in ml1m folder)'
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

    print("=" * 70)
    print("ML-1M CLIP Feature Extraction")
    print("=" * 70)
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
    images_dir = data_dir / 'M_ML-1M' / 'image'

    if not images_dir.exists():
        print(f"错误: 找不到图片目录: {images_dir}")
        return

    # Count total items from image files
    all_image_files = list(images_dir.glob('*.png'))
    if not all_image_files:
        print(f"错误: 图片目录中没有找到PNG文件: {images_dir}")
        return

    # Extract item IDs from filenames
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

    max_file_id = max(item_ids_from_files)

    # Read ratings.dat to determine actual item range
    ratings_file = data_dir / 'M_ML-1M' / 'ratings.dat'
    actual_max_item_id = 0

    if ratings_file.exists():
        print(f"从评分文件读取实际物品数: {ratings_file}")
        with open(ratings_file, 'r') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 2:
                    item_id = int(parts[1])
                    actual_max_item_id = max(actual_max_item_id, item_id)

        print(f"✓ 评分文件中的最大物品ID: {actual_max_item_id}")
        num_items = actual_max_item_id
    else:
        print(f"⚠️ 警告: 未找到评分文件，使用图片文件名中的最大ID")
        num_items = max_file_id

    print(f"物品总数: {num_items}")
    print(f"找到的图片文件总数: {len(all_image_files)}")

    if max_file_id > num_items:
        outlier_files = [fid for fid in item_ids_from_files if fid > num_items]
        print(f"\n⚠️ 警告: 发现 {len(outlier_files)} 个超出范围的图片文件:")
        for fid in sorted(outlier_files)[:10]:
            print(f"   - {fid}.png (超出最大物品ID {num_items})")
        if len(outlier_files) > 10:
            print(f"   ... 还有 {len(outlier_files) - 10} 个")
        print(f"   这些文件将被忽略\n")

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

    # Verification: Show sample mappings
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
    print("=" * 70)
    print("验证 Item ID 对应关系")
    print("=" * 70)

    # Verification: spot-check by re-encoding a few items
    test_item_ids = [1, 10, 100, min(500, num_items), min(1000, num_items), min(3000, num_items)]
    print("抽查验证 (重新编码几个物品并比较特征):")

    clip_encoder.eval()

    all_match = True
    for item_id in test_item_ids:
        if item_id not in item_image_paths:
            continue

        # Re-encode this single item
        with torch.no_grad():
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
    print("=" * 70)
    print("完成!")
    print("=" * 70)
    print(f"特征已保存至: {output_path}")
    print(f"特征形状: {item_features.shape}")
    print(f"特征维度: {args.output_dim}")
    print(f"有效物品数: {len(item_image_paths)}/{num_items}")
    print()
    print("✓ Item ID 对应关系已验证正确!")
    print()
    print("下一步:")
    print(f"  1. 抽取文本特征:")
    print(f"     python UR4Rec/scripts/extract_ml1m_text_features.py")
    print(f"  2. 运行训练:")
    print(f"     python UR4Rec/scripts/train_fedmem.py \\")
    print(f"         --data_dir UR4Rec/data \\")
    print(f"         --data_file ml1m_ratings_processed.dat \\")
    print(f"         --visual_file ml1m/clip_features.pt \\")
    print(f"         --text_file ml1m/text_features.pt \\")
    print(f"         --num_rounds 50 --device cpu")


if __name__ == '__main__':
    main()
