"""
ML-1M子集真实多模态特征提取脚本

使用真实的电影海报图片和文本描述提取特征：
1. CLIP视觉特征（从真实图片）: image/*.png
2. Sentence-BERT文本特征（从text.xls描述）

输出：
- data/ml-1m/clip_features.pt: [max_item_id+1, 512] CLIP特征
- data/ml-1m/text_features.pt: [max_item_id+1, 384] 文本特征
"""

import os
import sys
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set
from PIL import Image
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_subset_items(subset_file: str) -> Set[int]:
    """
    从子集数据中提取unique物品集合

    Args:
        subset_file: 子集数据文件路径

    Returns:
        item_ids: 物品ID集合
    """
    print(f"加载子集物品: {subset_file}")

    item_ids = set()
    with open(subset_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                item_id = int(parts[1])
                item_ids.add(item_id)

    print(f"✓ 子集包含 {len(item_ids)} 个unique物品")
    print(f"  ItemID范围: [{min(item_ids)}, {max(item_ids)}]")

    return item_ids


def load_text_descriptions(text_file: str, subset_items: Set[int]) -> Dict[int, str]:
    """
    从Excel文件加载文本描述

    Args:
        text_file: text.xls文件路径
        subset_items: 子集物品ID集合

    Returns:
        descriptions: {item_id: description}
    """
    print(f"\n加载文本描述: {text_file}")

    df = pd.read_excel(text_file)
    print(f"✓ 加载了 {len(df)} 条描述")

    descriptions = {}
    missing_count = 0

    for item_id in subset_items:
        # 查找描述
        row = df[df['MovieID'] == item_id]

        if len(row) > 0 and pd.notna(row.iloc[0]['description']):
            descriptions[item_id] = str(row.iloc[0]['description'])
        else:
            # 如果没有描述，使用默认
            descriptions[item_id] = f"Movie {item_id}"
            missing_count += 1

    print(f"✓ 为 {len(descriptions)} 个物品准备了描述")
    if missing_count > 0:
        print(f"  警告: {missing_count} 个物品缺少描述，使用默认文本")

    return descriptions


def check_available_images(image_dir: str, subset_items: Set[int]) -> Set[int]:
    """
    检查哪些物品有图片

    Args:
        image_dir: 图片目录
        subset_items: 子集物品ID集合

    Returns:
        available_items: 有图片的物品ID集合
    """
    print(f"\n检查可用图片: {image_dir}")

    available_items = set()

    for item_id in subset_items:
        image_path = os.path.join(image_dir, f"{item_id}.png")
        if os.path.exists(image_path):
            available_items.add(item_id)

    print(f"✓ 找到 {len(available_items)} 个物品的图片 ({len(available_items)/len(subset_items)*100:.1f}%)")
    missing_count = len(subset_items) - len(available_items)
    if missing_count > 0:
        print(f"  警告: {missing_count} 个物品缺少图片")

    return available_items


def extract_clip_visual_features(
    image_dir: str,
    subset_items: Set[int],
    available_images: Set[int],
    max_item_id: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    使用CLIP提取真实图片的视觉特征

    Args:
        image_dir: 图片目录
        subset_items: 子集物品ID集合
        available_images: 有图片的物品ID集合
        max_item_id: 最大物品ID
        device: 计算设备

    Returns:
        clip_features: [max_item_id+1, 512] CLIP特征张量
    """
    print("\n提取CLIP视觉特征（从真实图片）...")
    print("正在加载CLIP模型...")

    try:
        import clip

        # 加载CLIP模型
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("✓ CLIP模型加载成功 (ViT-B/32)")

        # 创建特征矩阵 [max_item_id+1, 512]
        clip_features = torch.zeros(max_item_id + 1, 512, dtype=torch.float32)

        # 批量处理图片
        batch_size = 32
        item_list = sorted(list(available_images))

        with torch.no_grad():
            for i in tqdm(range(0, len(item_list), batch_size), desc="提取CLIP特征"):
                batch_items = item_list[i:i+batch_size]
                batch_images = []

                # 加载并预处理图片
                for item_id in batch_items:
                    image_path = os.path.join(image_dir, f"{item_id}.png")
                    try:
                        image = Image.open(image_path).convert('RGB')
                        batch_images.append(preprocess(image))
                    except Exception as e:
                        print(f"  警告: 无法加载图片 {item_id}.png: {e}")
                        # 使用黑色图片作为占位
                        batch_images.append(preprocess(Image.new('RGB', (224, 224), (0, 0, 0))))

                if len(batch_images) == 0:
                    continue

                # 转换为tensor batch
                images_tensor = torch.stack(batch_images).to(device)

                # 提取视觉特征
                image_features = model.encode_image(images_tensor)
                image_features = image_features.cpu().float()

                # L2归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # 存储特征
                for j, item_id in enumerate(batch_items):
                    if j < len(image_features):
                        clip_features[item_id] = image_features[j]

        # 为没有图片的物品使用随机特征
        missing_items = subset_items - available_images
        if len(missing_items) > 0:
            print(f"  为 {len(missing_items)} 个缺少图片的物品生成随机特征")
            for item_id in missing_items:
                clip_features[item_id] = torch.randn(512) * 0.01

        print(f"✓ CLIP特征提取完成: {clip_features.shape}")
        print(f"  真实图片特征: {len(available_images)}")
        print(f"  随机特征: {len(missing_items)}")

    except ImportError:
        print("⚠️ 警告: CLIP未安装，使用随机初始化特征")
        print("   安装命令: pip install git+https://github.com/openai/CLIP.git")

        # 创建随机特征矩阵
        clip_features = torch.randn(max_item_id + 1, 512) * 0.01

        # 为子集中的物品设置非零特征
        for item_id in subset_items:
            clip_features[item_id] = torch.randn(512) * 0.1

        print(f"✓ 使用随机CLIP特征: {clip_features.shape}")

    return clip_features


def extract_text_features(
    descriptions: Dict[int, str],
    max_item_id: int,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    使用Sentence-BERT提取文本特征

    Args:
        descriptions: 物品文本描述字典
        max_item_id: 最大物品ID
        device: 计算设备

    Returns:
        text_features: [max_item_id+1, 384] 文本特征张量
    """
    print("\n提取Sentence-BERT文本特征...")
    print("正在加载Sentence-BERT模型...")

    try:
        from sentence_transformers import SentenceTransformer

        # 加载Sentence-BERT模型
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("✓ Sentence-BERT模型加载成功 (all-MiniLM-L6-v2)")

        # 创建特征矩阵 [max_item_id+1, 384]
        text_features = torch.zeros(max_item_id + 1, 384, dtype=torch.float32)

        # 批量提取特征
        batch_size = 32
        item_ids = sorted(descriptions.keys())

        for i in tqdm(range(0, len(item_ids), batch_size), desc="提取文本特征"):
            batch_ids = item_ids[i:i+batch_size]
            batch_texts = [descriptions[item_id] for item_id in batch_ids]

            # 编码文本
            embeddings = model.encode(
                batch_texts,
                convert_to_tensor=True,
                show_progress_bar=False,
                device=device
            )
            embeddings = embeddings.cpu().float()

            # L2归一化
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

            # 存储特征
            for j, item_id in enumerate(batch_ids):
                text_features[item_id] = embeddings[j]

        print(f"✓ 文本特征提取完成: {text_features.shape}")

    except ImportError:
        print("⚠️ 警告: Sentence-Transformers未安装，使用随机初始化特征")
        print("   安装命令: pip install sentence-transformers")

        # 创建随机特征矩阵
        text_features = torch.randn(max_item_id + 1, 384) * 0.01

        # 为已知物品设置非零特征
        for item_id in descriptions.keys():
            text_features[item_id] = torch.randn(384) * 0.1

        print(f"✓ 使用随机文本特征: {text_features.shape}")

    return text_features


def main():
    """主函数"""

    print("="*60)
    print("ML-1M子集真实多模态特征提取")
    print("="*60)
    print()

    # 配置路径
    project_root = Path(__file__).parent.parent
    ml1m_dir = project_root / 'data' / 'Multimodal_Datasets' / 'M_ML-1M'
    subset_file = project_root / 'data' / 'ml-1m' / 'subset_ratings.dat'
    output_dir = project_root / 'data' / 'ml-1m'

    image_dir = ml1m_dir / 'image'
    text_file = ml1m_dir / 'text.xls'

    # 检查文件是否存在
    if not subset_file.exists():
        print(f"❌ 错误: 子集文件不存在: {subset_file}")
        print("请先运行: python scripts/preprocess_ml1m_subset.py")
        return 1

    if not image_dir.exists():
        print(f"❌ 错误: 图片目录不存在: {image_dir}")
        return 1

    if not text_file.exists():
        print(f"❌ 错误: 文本描述文件不存在: {text_file}")
        return 1

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print()

    # 1. 加载子集物品列表
    subset_items = load_subset_items(str(subset_file))
    max_item_id = max(subset_items)
    print(f"  最大物品ID: {max_item_id}")

    # 2. 加载文本描述
    descriptions = load_text_descriptions(str(text_file), subset_items)

    # 3. 检查可用图片
    available_images = check_available_images(str(image_dir), subset_items)

    # 4. 提取CLIP视觉特征（从真实图片）
    clip_features = extract_clip_visual_features(
        str(image_dir),
        subset_items,
        available_images,
        max_item_id,
        device
    )

    # 5. 提取Sentence-BERT文本特征
    text_features = extract_text_features(descriptions, max_item_id, device)

    # 6. 保存特征
    print("\n保存特征文件...")
    output_dir.mkdir(parents=True, exist_ok=True)

    clip_output = output_dir / 'clip_features.pt'
    text_output = output_dir / 'text_features.pt'

    torch.save(clip_features, clip_output)
    torch.save(text_features, text_output)

    print(f"✓ CLIP特征已保存: {clip_output}")
    print(f"  形状: {clip_features.shape}")
    print(f"  大小: {clip_output.stat().st_size / 1024 / 1024:.2f} MB")

    print(f"✓ 文本特征已保存: {text_output}")
    print(f"  形状: {text_features.shape}")
    print(f"  大小: {text_output.stat().st_size / 1024 / 1024:.2f} MB")

    # 7. 验证特征
    print("\n验证特征质量...")

    # 检查非零特征数量
    clip_nonzero = (clip_features.abs().sum(dim=1) > 0).sum().item()
    text_nonzero = (text_features.abs().sum(dim=1) > 0).sum().item()

    print(f"  CLIP特征非零数量: {clip_nonzero} / {clip_features.shape[0]}")
    print(f"    - 真实图片特征: {len(available_images)}")
    print(f"    - 随机特征: {len(subset_items - available_images)}")
    print(f"  文本特征非零数量: {text_nonzero} / {text_features.shape[0]}")

    # 统计信息
    print(f"\n  CLIP特征统计:")
    print(f"    min: {clip_features.min():.4f}")
    print(f"    max: {clip_features.max():.4f}")
    print(f"    mean: {clip_features.mean():.4f}")
    print(f"    std: {clip_features.std():.4f}")

    print(f"\n  文本特征统计:")
    print(f"    min: {text_features.min():.4f}")
    print(f"    max: {text_features.max():.4f}")
    print(f"    mean: {text_features.mean():.4f}")
    print(f"    std: {text_features.std():.4f}")

    print("\n" + "="*60)
    print("✓ 特征提取完成！")
    print("="*60)
    print("\n现在可以使用这些真实特征进行训练:")
    print(f"  python scripts/train_fedmem.py \\")
    print(f"    --data_dir data/ml-1m \\")
    print(f"    --data_file subset_ratings.dat \\")
    print(f"    --visual_file clip_features.pt \\")
    print(f"    --text_file text_features.pt")
    print()

    return 0


if __name__ == '__main__':
    exit(main())
