"""
MovieLens 数据集预处理脚本

支持 MovieLens-100K 和 MovieLens-1M 数据集。
自动下载、解压并转换为 UR4Rec 所需格式。
"""
import os
import sys
import argparse
import urllib.request
import zipfile
import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


# 数据集下载 URL
DATASET_URLS = {
    'ml-100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
}


def download_dataset(dataset: str, raw_dir: Path) -> Path:
    """
    下载 MovieLens 数据集

    Args:
        dataset: 数据集名称（'ml-100k' 或 'ml-1m'）
        raw_dir: 原始数据存储目录

    Returns:
        解压后的数据集目录路径
    """
    dataset_dir = raw_dir / dataset

    if dataset_dir.exists():
        print(f"数据集已存在: {dataset_dir}")
        return dataset_dir

    print(f"正在下载 {dataset} 数据集...")
    url = DATASET_URLS[dataset]
    zip_path = raw_dir / f"{dataset}.zip"

    # 下载
    raw_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, zip_path)
    print(f"下载完成: {zip_path}")

    # 解压
    print(f"正在解压...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(raw_dir)

    # 删除 zip 文件
    zip_path.unlink()
    print(f"解压完成: {dataset_dir}")

    return dataset_dir


def load_movielens_100k(dataset_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    加载 MovieLens-100K 数据集

    Args:
        dataset_dir: 数据集目录

    Returns:
        ratings: 评分数据 DataFrame
        movies: 电影信息字典
    """
    print("加载 MovieLens-100K 数据...")

    # 加载评分数据
    ratings_file = dataset_dir / 'u.data'
    ratings = pd.read_csv(
        ratings_file,
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )

    # 加载电影信息
    movies_file = dataset_dir / 'u.item'
    movies_df = pd.read_csv(
        movies_file,
        sep='|',
        names=['item_id', 'title', 'release_date', 'video_release_date',
               'imdb_url'] + [f'genre_{i}' for i in range(19)],
        encoding='latin-1'
    )

    # 创建电影字典
    movies = {}
    for _, row in movies_df.iterrows():
        movies[row['item_id']] = {
            'title': row['title'],
            'release_date': row['release_date']
        }

    print(f"加载完成: {len(ratings)} 条评分, {len(movies)} 部电影")
    return ratings, movies


def load_movielens_1m(dataset_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """
    加载 MovieLens-1M 数据集

    Args:
        dataset_dir: 数据集目录

    Returns:
        ratings: 评分数据 DataFrame
        movies: 电影信息字典
    """
    print("加载 MovieLens-1M 数据...")

    # 加载评分数据
    ratings_file = dataset_dir / 'ratings.dat'
    ratings = pd.read_csv(
        ratings_file,
        sep='::',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python',
        encoding='latin-1'
    )

    # 加载电影信息
    movies_file = dataset_dir / 'movies.dat'
    movies_df = pd.read_csv(
        movies_file,
        sep='::',
        names=['item_id', 'title', 'genres'],
        engine='python',
        encoding='latin-1'
    )

    # 创建电影字典
    movies = {}
    for _, row in movies_df.iterrows():
        movies[row['item_id']] = {
            'title': row['title'],
            'genres': row['genres']
        }

    print(f"加载完成: {len(ratings)} 条评分, {len(movies)} 部电影")
    return ratings, movies


def filter_ratings(ratings: pd.DataFrame, min_rating: float = 4.0) -> pd.DataFrame:
    """
    过滤评分数据，只保留高评分（视为正样本）

    Args:
        ratings: 原始评分数据
        min_rating: 最小评分阈值

    Returns:
        过滤后的评分数据
    """
    print(f"过滤评分 >= {min_rating} 的数据...")
    filtered = ratings[ratings['rating'] >= min_rating].copy()
    print(f"过滤后: {len(filtered)} 条评分")
    return filtered


def build_user_sequences(
    ratings: pd.DataFrame,
    min_seq_len: int = 5
) -> Dict[int, List[Tuple[int, int]]]:
    """
    构建用户交互序列

    Args:
        ratings: 评分数据
        min_seq_len: 最小序列长度

    Returns:
        user_sequences: 用户ID -> [(item_id, timestamp), ...] 的字典
    """
    print("构建用户交互序列...")

    user_sequences = defaultdict(list)

    for _, row in ratings.iterrows():
        user_sequences[row['user_id']].append((row['item_id'], row['timestamp']))

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
    为每个用户生成候选物品和 ground truth

    Args:
        user_sequences: 用户序列
        all_items: 所有物品集合
        num_candidates: 候选物品数量
        positive_ratio: 正样本在候选中的比例

    Returns:
        samples: 生成的样本列表
    """
    print("生成候选物品和ground truth...")

    samples = []

    for user_id, sequence in tqdm(user_sequences.items()):
        # 至少需要留一个作为测试
        if len(sequence) < 2:
            continue

        # 使用前 n-1 个作为历史，最后几个作为 ground truth
        split_point = max(1, len(sequence) - 5)
        history_items = [item for item, _ in sequence[:split_point]]
        future_items = [item for item, _ in sequence[split_point:]]

        if len(future_items) == 0:
            continue

        # 生成候选物品
        num_positive = max(1, int(num_candidates * positive_ratio))
        num_negative = num_candidates - num_positive

        # 从 future_items 中随机选择正样本
        positive_samples = random.sample(
            future_items,
            min(num_positive, len(future_items))
        )

        # 从未交互的物品中随机选择负样本
        interacted_items = set(history_items) | set(future_items)
        negative_pool = list(all_items - interacted_items)

        if len(negative_pool) < num_negative:
            continue

        negative_samples = random.sample(negative_pool, num_negative)

        # 组合候选物品
        candidates = positive_samples + negative_samples
        random.shuffle(candidates)

        # 创建样本
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
    """
    划分数据集

    Args:
        samples: 所有样本
        train_ratio: 训练集比例
        val_ratio: 验证集比例

    Returns:
        train_samples, val_samples, test_samples
    """
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
    movies: Dict,
    output_dir: Path,
    dataset_name: str
):
    """
    保存处理后的数据

    Args:
        train_samples: 训练样本
        val_samples: 验证样本
        test_samples: 测试样本
        movies: 电影信息
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    print(f"保存数据到 {output_dir}...")

    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # 保存数据集
    with open(dataset_dir / 'train.json', 'w') as f:
        json.dump(train_samples, f, indent=2)

    with open(dataset_dir / 'val.json', 'w') as f:
        json.dump(val_samples, f, indent=2)

    with open(dataset_dir / 'test.json', 'w') as f:
        json.dump(test_samples, f, indent=2)

    # 保存电影信息（元数据）
    with open(dataset_dir / 'movies.json', 'w', encoding='utf-8') as f:
        json.dump(movies, f, indent=2, ensure_ascii=False)

    # 保存统计信息
    stats = {
        'dataset': dataset_name,
        'num_users': len(set(s['user_id'] for s in train_samples + val_samples + test_samples)),
        'num_items': len(movies),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples)
    }

    with open(dataset_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"保存完成！")
    print(f"训练集: {dataset_dir / 'train.json'}")
    print(f"验证集: {dataset_dir / 'val.json'}")
    print(f"测试集: {dataset_dir / 'test.json'}")
    print(f"电影信息: {dataset_dir / 'movies.json'}")
    print(f"\n统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main(args):
    """主函数"""
    random.seed(args.seed)

    # 设置路径
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'

    # 下载数据集
    dataset_dir = download_dataset(args.dataset, raw_dir)

    # 加载数据
    if args.dataset == 'ml-100k':
        ratings, movies = load_movielens_100k(dataset_dir)
    elif args.dataset == 'ml-1m':
        ratings, movies = load_movielens_1m(dataset_dir)
    else:
        raise ValueError(f"不支持的数据集: {args.dataset}")

    # 过滤评分
    ratings = filter_ratings(ratings, args.min_rating)

    # 构建用户序列
    user_sequences = build_user_sequences(ratings, args.min_seq_len)

    # 获取所有物品
    all_items = set(ratings['item_id'].unique())

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
        movies,
        processed_dir,
        args.dataset
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='预处理 MovieLens 数据集')

    parser.add_argument('--dataset', type=str, default='ml-100k',
                        choices=['ml-100k', 'ml-1m'],
                        help='数据集名称')
    parser.add_argument('--min_rating', type=float, default=4.0,
                        help='最小评分阈值（视为正样本）')
    parser.add_argument('--min_seq_len', type=int, default=5,
                        help='最小序列长度')
    parser.add_argument('--num_candidates', type=int, default=20,
                        help='每个用户的候选物品数量')
    parser.add_argument('--positive_ratio', type=float, default=0.2,
                        help='候选中正样本的比例')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    args = parser.parse_args()
    main(args)
