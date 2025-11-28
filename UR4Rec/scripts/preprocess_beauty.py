"""
Amazon Beauty 数据集预处理脚本

自动下载并转换亚马逊美妆产品评论数据集为 UR4Rec 所需格式。
"""
import os
import sys
import argparse
import urllib.request
import gzip
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


# Amazon Beauty 数据集 URL（5-core 版本：用户和物品至少有5次交互）
DATASET_URL = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz'
METADATA_URL = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz'


def download_file(url: str, output_path: Path):
    """
    下载文件

    Args:
        url: 文件 URL
        output_path: 输出路径
    """
    if output_path.exists():
        print(f"文件已存在: {output_path}")
        return

    print(f"正在下载: {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"下载完成: {output_path}")
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载数据集:")
        print(f"  评论数据: {DATASET_URL}")
        print(f"  元数据: {METADATA_URL}")
        print(f"并将文件放置在: {output_path.parent}")
        sys.exit(1)


def parse_json_gz(file_path: Path) -> List[Dict]:
    """
    解析 gzip 压缩的 JSON 文件

    Args:
        file_path: 文件路径

    Returns:
        解析后的数据列表
    """
    print(f"解析文件: {file_path}")
    data = []

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    print(f"解析完成: {len(data)} 条记录")
    return data


def load_beauty_dataset(raw_dir: Path) -> Tuple[List[Dict], Dict]:
    """
    加载 Amazon Beauty 数据集

    Args:
        raw_dir: 原始数据目录

    Returns:
        reviews: 评论数据列表
        metadata: 产品元数据字典
    """
    # 下载评论数据
    reviews_file = raw_dir / 'reviews_Beauty_5.json.gz'
    download_file(DATASET_URL, reviews_file)

    # 下载元数据
    metadata_file = raw_dir / 'meta_Beauty.json.gz'
    download_file(METADATA_URL, metadata_file)

    # 解析评论数据
    print("\n加载评论数据...")
    reviews = parse_json_gz(reviews_file)

    # 解析元数据
    print("\n加载产品元数据...")
    metadata_list = parse_json_gz(metadata_file)

    # 转换元数据为字典
    metadata = {}
    for item in metadata_list:
        if 'asin' in item:
            metadata[item['asin']] = {
                'title': item.get('title', 'Unknown'),
                'brand': item.get('brand', 'Unknown'),
                'categories': item.get('categories', [])
            }

    print(f"\n总计: {len(reviews)} 条评论, {len(metadata)} 个产品")
    return reviews, metadata


def create_item_mapping(reviews: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    创建物品 ID 映射（ASIN -> 整数 ID）

    Args:
        reviews: 评论数据

    Returns:
        asin_to_id: ASIN -> ID 映射
        id_to_asin: ID -> ASIN 映射
    """
    print("创建物品 ID 映射...")

    asins = sorted(set(r['asin'] for r in reviews))

    asin_to_id = {asin: idx + 1 for idx, asin in enumerate(asins)}
    id_to_asin = {idx + 1: asin for idx, asin in enumerate(asins)}

    print(f"物品数量: {len(asin_to_id)}")
    return asin_to_id, id_to_asin


def create_user_mapping(reviews: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    创建用户 ID 映射（reviewerID -> 整数 ID）

    Args:
        reviews: 评论数据

    Returns:
        user_to_id: reviewerID -> ID 映射
        id_to_user: ID -> reviewerID 映射
    """
    print("创建用户 ID 映射...")

    users = sorted(set(r['reviewerID'] for r in reviews))

    user_to_id = {user: idx + 1 for idx, user in enumerate(users)}
    id_to_user = {idx + 1: user for idx, user in enumerate(users)}

    print(f"用户数量: {len(user_to_id)}")
    return user_to_id, id_to_user


def build_user_sequences(
    reviews: List[Dict],
    user_to_id: Dict[str, int],
    asin_to_id: Dict[str, int],
    min_rating: float = 4.0,
    min_seq_len: int = 5
) -> Dict[int, List[Tuple[int, int]]]:
    """
    构建用户交互序列

    Args:
        reviews: 评论数据
        user_to_id: 用户 ID 映射
        asin_to_id: 物品 ID 映射
        min_rating: 最小评分阈值
        min_seq_len: 最小序列长度

    Returns:
        user_sequences: 用户ID -> [(item_id, timestamp), ...]
    """
    print(f"构建用户交互序列（评分 >= {min_rating}）...")

    user_sequences = defaultdict(list)

    for review in reviews:
        if review.get('overall', 0) >= min_rating:
            user_id = user_to_id[review['reviewerID']]
            item_id = asin_to_id[review['asin']]
            timestamp = review.get('unixReviewTime', 0)

            user_sequences[user_id].append((item_id, timestamp))

    # 按时间戳排序
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[1])

    # 过滤序列长度太短的用户
    user_sequences = {
        uid: seq for uid, seq in user_sequences.items()
        if len(seq) >= min_seq_len
    }

    print(f"有效用户数: {len(user_sequences)}")
    return dict(user_sequences)


def generate_candidates(
    user_sequences: Dict[int, List[Tuple[int, int]]],
    all_items: set,
    num_candidates: int = 20,
    positive_ratio: float = 0.2
) -> List[Dict]:
    """
    为每个用户生成候选物品

    Args:
        user_sequences: 用户序列
        all_items: 所有物品集合
        num_candidates: 候选物品数量
        positive_ratio: 正样本比例

    Returns:
        samples: 生成的样本列表
    """
    print("生成候选物品...")

    samples = []

    for user_id, sequence in tqdm(user_sequences.items()):
        if len(sequence) < 2:
            continue

        # 使用前 n-1 个作为历史
        split_point = max(1, len(sequence) - 5)
        history_items = [item for item, _ in sequence[:split_point]]
        future_items = [item for item, _ in sequence[split_point:]]

        if len(future_items) == 0:
            continue

        # 生成候选物品
        num_positive = max(1, int(num_candidates * positive_ratio))
        num_negative = num_candidates - num_positive

        positive_samples = random.sample(
            future_items,
            min(num_positive, len(future_items))
        )

        # 负样本
        interacted_items = set(history_items) | set(future_items)
        negative_pool = list(all_items - interacted_items)

        if len(negative_pool) < num_negative:
            continue

        negative_samples = random.sample(negative_pool, num_negative)

        # 组合
        candidates = positive_samples + negative_samples
        random.shuffle(candidates)

        sample = {
            'user_id': int(user_id),
            'user_history': [int(item) for item in history_items],
            'candidates': [int(item) for item in candidates],
            'ground_truth': [int(item) for item in positive_samples]
        }

        samples.append(sample)

    print(f"生成样本数: {len(samples)}")
    return samples


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """划分数据集"""
    print("划分数据集...")

    random.shuffle(samples)

    n_total = len(samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train+n_val]
    test_samples = samples[n_train+n_val:]

    print(f"训练集: {len(train_samples)}")
    print(f"验证集: {len(val_samples)}")
    print(f"测试集: {len(test_samples)}")

    return train_samples, val_samples, test_samples


def save_data(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    metadata: Dict,
    id_to_asin: Dict[int, str],
    output_dir: Path
):
    """保存处理后的数据"""
    print(f"保存数据到 {output_dir}...")

    dataset_dir = output_dir / 'beauty'
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 保存数据集
    with open(dataset_dir / 'train.json', 'w') as f:
        json.dump(train_samples, f, indent=2)

    with open(dataset_dir / 'val.json', 'w') as f:
        json.dump(val_samples, f, indent=2)

    with open(dataset_dir / 'test.json', 'w') as f:
        json.dump(test_samples, f, indent=2)

    # 保存产品元数据（映射回 ASIN）
    product_metadata = {}
    for item_id, asin in id_to_asin.items():
        if asin in metadata:
            product_metadata[item_id] = metadata[asin]

    with open(dataset_dir / 'products.json', 'w', encoding='utf-8') as f:
        json.dump(product_metadata, f, indent=2, ensure_ascii=False)

    # 保存 ID 映射
    with open(dataset_dir / 'id_mapping.json', 'w') as f:
        json.dump({'id_to_asin': id_to_asin}, f, indent=2)

    # 统计信息
    stats = {
        'dataset': 'Amazon Beauty',
        'num_users': len(set(s['user_id'] for s in train_samples + val_samples + test_samples)),
        'num_items': len(id_to_asin),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples)
    }

    with open(dataset_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"保存完成！")
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main(args):
    """主函数"""
    random.seed(args.seed)

    # 设置路径
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'data' / 'raw' / 'beauty'
    processed_dir = project_root / 'data' / 'processed'

    # 加载数据
    reviews, metadata = load_beauty_dataset(raw_dir)

    # 创建 ID 映射
    asin_to_id, id_to_asin = create_item_mapping(reviews)
    user_to_id, id_to_user = create_user_mapping(reviews)

    # 构建用户序列
    user_sequences = build_user_sequences(
        reviews,
        user_to_id,
        asin_to_id,
        args.min_rating,
        args.min_seq_len
    )

    # 获取所有物品
    all_items = set(asin_to_id.values())

    # 生成候选物品
    samples = generate_candidates(
        user_sequences,
        all_items,
        args.num_candidates,
        args.positive_ratio
    )

    # 划分数据集
    train_samples, val_samples, test_samples = split_dataset(
        samples,
        args.train_ratio,
        args.val_ratio
    )

    # 保存数据
    save_data(
        train_samples,
        val_samples,
        test_samples,
        metadata,
        id_to_asin,
        processed_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预处理 Amazon Beauty 数据集')

    parser.add_argument('--min_rating', type=float, default=4.0,
                        help='最小评分阈值')
    parser.add_argument('--min_seq_len', type=int, default=5,
                        help='最小序列长度')
    parser.add_argument('--num_candidates', type=int, default=20,
                        help='候选物品数量')
    parser.add_argument('--positive_ratio', type=float, default=0.2,
                        help='正样本比例')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()
    main(args)
