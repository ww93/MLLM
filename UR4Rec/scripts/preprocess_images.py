"""
图片预处理

使用 CLIP 提取图片特征，保存为 embeddings
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict


def preprocess_images_with_clip(
    image_dir: str,
    output_path: str,
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    使用 CLIP 提取图片特征

    Args:
        image_dir: 图片目录
        output_path: 输出文件路径（.npy 或 .pt）
        model_name: CLIP 模型名称
        batch_size: 批次大小
        device: 设备
    """
    print("\n" + "=" * 60)
    print("使用 CLIP 提取图片特征")
    print("=" * 60)
    print(f"图片目录: {image_dir}")
    print(f"输出路径: {output_path}")
    print(f"模型: {model_name}")
    print(f"设备: {device}")

    # 加载 CLIP 模型
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("错误: 需要安装 transformers 和 torch")
        print("运行: pip install transformers torch pillow")
        return

    print("\n加载 CLIP 模型...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    print("模型加载完成！")

    # 获取所有图片文件
    image_path = Path(image_dir)
    image_files = sorted(list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")))

    print(f"\n找到 {len(image_files)} 张图片")

    # 存储特征
    image_features = {}
    image_embeddings = []
    image_ids = []

    # 批量处理
    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size), desc="提取特征"):
            batch_files = image_files[i:i+batch_size]

            # 加载图片
            batch_images = []
            batch_ids = []

            for img_file in batch_files:
                try:
                    img = Image.open(img_file).convert('RGB')
                    batch_images.append(img)
                    batch_ids.append(img_file.stem)  # 文件名（不含扩展名）
                except Exception as e:
                    print(f"\n警告: 无法加载图片 {img_file}: {e}")
                    continue

            if not batch_images:
                continue

            # 预处理
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 提取特征
            outputs = model.get_image_features(**inputs)

            # 归一化
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)

            # 保存
            for item_id, features in zip(batch_ids, outputs):
                image_features[item_id] = features.cpu().numpy()
                image_embeddings.append(features.cpu().numpy())
                image_ids.append(item_id)

    # 转换为数组
    image_embeddings = np.array(image_embeddings)

    print(f"\n提取的特征形状: {image_embeddings.shape}")

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == '.npy':
        # 保存为 numpy 数组 + ID 映射
        np.save(output_path, image_embeddings)

        # 保存 ID 映射
        id_map_path = output_path.parent / f"{output_path.stem}_ids.json"
        with open(id_map_path, 'w') as f:
            json.dump(image_ids, f, indent=2)

        print(f"\n特征已保存至: {output_path}")
        print(f"ID 映射已保存至: {id_map_path}")

    elif output_path.suffix == '.pt':
        # 保存为 PyTorch 字典
        torch.save({
            'embeddings': torch.from_numpy(image_embeddings),
            'ids': image_ids,
            'features_dict': image_features
        }, output_path)

        print(f"\n特征已保存至: {output_path}")

    else:
        # 保存为字典（pickle）
        save_dict = {
            'embeddings': image_embeddings,
            'ids': image_ids,
            'features_dict': image_features
        }
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(save_dict, f)

        print(f"\n特征已保存至: {output_path}")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


def create_image_dataloader_cache(
    image_dir: str,
    output_path: str,
    target_size: tuple = (224, 224)
):
    """
    创建图片数据加载缓存（调整大小后的图片）

    Args:
        image_dir: 图片目录
        output_path: 输出目录
        target_size: 目标尺寸
    """
    print("\n" + "=" * 60)
    print("创建图片缓存")
    print("=" * 60)

    image_path = Path(image_dir)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")))

    print(f"找到 {len(image_files)} 张图片")
    print(f"目标尺寸: {target_size}")

    for img_file in tqdm(image_files, desc="处理图片"):
        try:
            # 加载并调整大小
            img = Image.open(img_file).convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)

            # 保存
            output_file = output_path / img_file.name
            img.save(output_file, 'JPEG', quality=95)

        except Exception as e:
            print(f"\n处理失败 {img_file}: {e}")

    print(f"\n缓存已保存至: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='图片预处理')

    parser.add_argument('--image_dir', type=str, required=True,
                        help='图片目录')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出路径')

    parser.add_argument('--mode', type=str, default='clip',
                        choices=['clip', 'resize'],
                        help='处理模式: clip (提取特征) | resize (调整大小)')

    parser.add_argument('--model_name', type=str,
                        default='openai/clip-vit-base-patch32',
                        help='CLIP 模型名称')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')

    parser.add_argument('--target_size', type=int, nargs=2, default=[224, 224],
                        help='目标尺寸（宽 高）')

    args = parser.parse_args()

    if args.mode == 'clip':
        preprocess_images_with_clip(
            image_dir=args.image_dir,
            output_path=args.output_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device
        )
    elif args.mode == 'resize':
        create_image_dataloader_cache(
            image_dir=args.image_dir,
            output_path=args.output_path,
            target_size=tuple(args.target_size)
        )


if __name__ == "__main__":
    main()
