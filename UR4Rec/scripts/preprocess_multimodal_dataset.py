"""
预处理多模态 MovieLens 数据集

处理 Multimodal_Datasets 目录下的数据：
- movies.dat: 电影元数据
- ratings.dat: 用户评分
- text.xls: 电影文本描述
- image/: 电影图片
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
import shutil


def load_movies(movies_file: str) -> dict:
    """
    加载电影元数据

    格式: movie_id::title::genres
    """
    print(f"\n加载电影元数据: {movies_file}")

    movies = {}
    with open(movies_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 3:
                movie_id = int(parts[0])
                title = parts[1]
                genres = parts[2].split('|')

                movies[movie_id] = {
                    'title': title,
                    'genres': genres
                }

    print(f"加载了 {len(movies)} 部电影")
    return movies


def load_text_descriptions(text_file: str) -> dict:
    """
    加载电影文本描述（从 Excel 文件）

    Args:
        text_file: text.xls 文件路径

    Returns:
        {movie_id: description}
    """
    print(f"\n加载文本描述: {text_file}")

    try:
        # 尝试使用 pandas 读取 Excel
        df = pd.read_excel(text_file)

        # 查看列名
        print(f"Excel 列名: {df.columns.tolist()}")

        descriptions = {}

        # 优先使用列名匹配
        if 'movie-id' in df.columns and 'review' in df.columns:
            print("使用列名: 'movie-id' 和 'review'")
            for idx, row in df.iterrows():
                movie_id = int(row['movie-id'])
                description = str(row['review'])
                if pd.notna(description) and description != 'nan':
                    descriptions[movie_id] = description
        # 回退到位置索引
        elif len(df.columns) >= 2:
            print("使用位置索引: 第1列=movie_id, 第2列=description")
            for idx, row in df.iterrows():
                movie_id = int(row.iloc[0])
                description = str(row.iloc[1])
                if pd.notna(description) and description != 'nan':
                    descriptions[movie_id] = description

        print(f"加载了 {len(descriptions)} 个文本描述")
        return descriptions

    except Exception as e:
        print(f"警告: 无法读取 Excel 文件: {e}")
        print("将使用电影标题和类型作为描述")
        return {}


def load_ratings(ratings_file: str) -> list:
    """
    加载评分数据

    格式: user_id::movie_id::rating::timestamp
    """
    print(f"\n加载评分数据: {ratings_file}")

    ratings = []
    with open(ratings_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 4:
                user_id = int(parts[0])
                movie_id = int(parts[1])
                rating = float(parts[2])
                timestamp = int(parts[3])

                ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp
                })

    print(f"加载了 {len(ratings)} 条评分")
    return ratings


def build_sequences(
    ratings: list,
    min_rating: float = 4.0,
    min_seq_len: int = 5
) -> dict:
    """
    构建用户交互序列

    Args:
        ratings: 评分列表
        min_rating: 最小评分（只保留高评分）
        min_seq_len: 最小序列长度

    Returns:
        {user_id: [movie_ids sorted by timestamp]}
    """
    print(f"\n构建用户序列 (min_rating={min_rating}, min_seq_len={min_seq_len})")

    # 过滤低评分
    filtered_ratings = [r for r in ratings if r['rating'] >= min_rating]
    print(f"过滤后保留 {len(filtered_ratings)} 条高评分记录")

    # 按用户分组
    user_interactions = defaultdict(list)
    for r in filtered_ratings:
        user_interactions[r['user_id']].append((r['movie_id'], r['timestamp']))

    # 按时间排序
    sequences = {}
    for user_id, interactions in user_interactions.items():
        # 排序
        interactions.sort(key=lambda x: x[1])
        movie_seq = [item_id for item_id, _ in interactions]

        # 过滤短序列
        if len(movie_seq) >= min_seq_len:
            sequences[user_id] = movie_seq

    print(f"保留 {len(sequences)} 个用户序列")

    return sequences


def remap_ids(sequences: dict, movies: dict):
    """
    重新映射 ID（从 1 开始连续编号）

    Args:
        sequences: 用户序列
        movies: 电影元数据

    Returns:
        remapped_sequences, remapped_movies, user_map, item_map
    """
    print("\n重新映射 ID...")

    # 收集所有出现的 user_id 和 movie_id
    all_users = sorted(sequences.keys())
    all_movies = set()
    for seq in sequences.values():
        all_movies.update(seq)
    all_movies = sorted(all_movies)

    # 创建映射
    user_map = {old_id: new_id for new_id, old_id in enumerate(all_users, 1)}
    item_map = {old_id: new_id for new_id, old_id in enumerate(all_movies, 1)}

    # 反向映射
    user_map_reverse = {v: k for k, v in user_map.items()}
    item_map_reverse = {v: k for k, v in item_map.items()}

    # 重映射序列
    remapped_sequences = {}
    for old_user_id, movie_seq in sequences.items():
        new_user_id = user_map[old_user_id]
        new_movie_seq = [item_map[movie_id] for movie_id in movie_seq]
        remapped_sequences[new_user_id] = new_movie_seq

    # 重映射电影元数据
    remapped_movies = {}
    for old_movie_id in all_movies:
        if old_movie_id in movies:
            new_movie_id = item_map[old_movie_id]
            remapped_movies[new_movie_id] = movies[old_movie_id]

    print(f"映射了 {len(user_map)} 个用户，{len(item_map)} 个物品")

    return remapped_sequences, remapped_movies, user_map, item_map, user_map_reverse, item_map_reverse


def split_sequences(
    sequences: dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
):
    """
    划分训练/验证/测试集

    策略：leave-one-out
    - 训练集：前 n-2 个交互
    - 验证集：第 n-1 个交互
    - 测试集：第 n 个交互
    """
    print(f"\n划分数据集 (train={train_ratio}, val={val_ratio})")

    train_seqs = {}
    val_seqs = {}
    test_seqs = {}

    for user_id, seq in sequences.items():
        seq_len = len(seq)

        if seq_len < 3:
            # 太短，全部用于训练
            train_seqs[user_id] = seq
        else:
            # 最后一个用于测试
            test_seqs[user_id] = seq

            # 倒数第二个用于验证
            val_seqs[user_id] = seq[:-1]

            # 前面用于训练
            train_seqs[user_id] = seq[:-2]

    print(f"训练集: {len(train_seqs)} 用户")
    print(f"验证集: {len(val_seqs)} 用户")
    print(f"测试集: {len(test_seqs)} 用户")

    return train_seqs, val_seqs, test_seqs


def copy_images(
    source_image_dir: str,
    output_image_dir: str,
    item_map_reverse: dict,
    remapped_movies: dict
):
    """
    复制图片到输出目录，并按新的 ID 命名

    Args:
        source_image_dir: 原始图片目录
        output_image_dir: 输出图片目录
        item_map_reverse: 新ID -> 旧ID 的映射
        remapped_movies: 重映射后的电影元数据
    """
    print(f"\n复制图片...")

    source_path = Path(source_image_dir)
    output_path = Path(output_image_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for new_id in tqdm(remapped_movies.keys(), desc="复制图片"):
        old_id = item_map_reverse[new_id]

        # 原始图片路径
        source_file = source_path / f"{old_id}.png"

        # 新图片路径
        output_file = output_path / f"{new_id}.png"

        if source_file.exists():
            try:
                shutil.copy2(source_file, output_file)
                success_count += 1
            except Exception as e:
                print(f"\n复制失败 {old_id} -> {new_id}: {e}")
                fail_count += 1
        else:
            fail_count += 1

    print(f"成功: {success_count}, 失败: {fail_count}")


def save_outputs(
    output_dir: str,
    train_seqs: dict,
    val_seqs: dict,
    test_seqs: dict,
    movies: dict,
    text_descriptions: dict,
    user_map: dict,
    item_map: dict,
    item_map_reverse: dict
):
    """保存所有输出文件"""
    print(f"\n保存输出到: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 保存序列
    np.save(output_path / 'train_sequences.npy', train_seqs)
    np.save(output_path / 'val_sequences.npy', val_seqs)
    np.save(output_path / 'test_sequences.npy', test_seqs)

    # 2. 保存电影元数据
    item_metadata = {}
    for new_id, movie_info in movies.items():
        old_id = item_map_reverse[new_id]

        # 添加文本描述
        description = text_descriptions.get(old_id, "")
        if not description:
            # 使用标题和类型作为描述
            description = f"{movie_info['title']}. Genres: {', '.join(movie_info['genres'])}"

        item_metadata[str(new_id)] = {
            'title': movie_info['title'],
            'genres': movie_info['genres'],
            'description': description,
            'original_id': old_id
        }

    with open(output_path / 'item_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(item_metadata, f, indent=2, ensure_ascii=False)

    # 3. 保存 ID 映射
    with open(output_path / 'user_map.json', 'w') as f:
        json.dump({str(k): v for k, v in user_map.items()}, f, indent=2)

    with open(output_path / 'item_map.json', 'w') as f:
        json.dump({str(k): v for k, v in item_map.items()}, f, indent=2)

    # 4. 保存统计信息
    stats = {
        'num_users': len(user_map),
        'num_items': len(item_map),
        'num_interactions_train': sum(len(seq) for seq in train_seqs.values()),
        'num_interactions_val': sum(len(seq) for seq in val_seqs.values()),
        'num_interactions_test': sum(len(seq) for seq in test_seqs.values()),
        'avg_seq_len': np.mean([len(seq) for seq in train_seqs.values()]),
        'min_seq_len': min(len(seq) for seq in train_seqs.values()),
        'max_seq_len': max(len(seq) for seq in train_seqs.values())
    }

    with open(output_path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n保存完成:")
    print(f"  - train_sequences.npy")
    print(f"  - val_sequences.npy")
    print(f"  - test_sequences.npy")
    print(f"  - item_metadata.json")
    print(f"  - user_map.json")
    print(f"  - item_map.json")
    print(f"  - stats.json")
    print(f"  - images/ (如果复制了图片)")

    # 打印统计
    print("\n数据统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='预处理多模态 MovieLens 数据集')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['ml-100k', 'ml-1m'],
                        help='数据集名称')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Multimodal_Datasets 目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')

    parser.add_argument('--min_rating', type=float, default=4.0,
                        help='最小评分（默认4.0）')
    parser.add_argument('--min_seq_len', type=int, default=5,
                        help='最小序列长度（默认5）')

    parser.add_argument('--copy_images', action='store_true',
                        help='是否复制图片')

    args = parser.parse_args()

    print("=" * 60)
    print("预处理多模态 MovieLens 数据集")
    print("=" * 60)
    print(f"数据集: {args.dataset}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")

    # 确定数据路径
    if args.dataset == 'ml-100k':
        dataset_dir = Path(args.data_dir) / 'M_ML-100K'
    else:
        dataset_dir = Path(args.data_dir) / 'M_ML-1M'

    movies_file = dataset_dir / 'movies.dat'
    ratings_file = dataset_dir / 'ratings.dat'
    text_file = dataset_dir / 'text.xls'
    image_dir = dataset_dir / 'image'

    # 检查文件是否存在
    if not movies_file.exists():
        print(f"错误: 找不到 {movies_file}")
        return
    if not ratings_file.exists():
        print(f"错误: 找不到 {ratings_file}")
        return

    # 1. 加载数据
    movies = load_movies(str(movies_file))
    ratings = load_ratings(str(ratings_file))

    text_descriptions = {}
    if text_file.exists():
        text_descriptions = load_text_descriptions(str(text_file))

    # 2. 构建序列
    sequences = build_sequences(
        ratings,
        min_rating=args.min_rating,
        min_seq_len=args.min_seq_len
    )

    # 3. 重映射 ID
    remapped_seqs, remapped_movies, user_map, item_map, user_map_reverse, item_map_reverse = remap_ids(
        sequences, movies
    )

    # 4. 划分数据集
    train_seqs, val_seqs, test_seqs = split_sequences(remapped_seqs)

    # 5. 保存输出
    save_outputs(
        args.output_dir,
        train_seqs,
        val_seqs,
        test_seqs,
        remapped_movies,
        text_descriptions,
        user_map,
        item_map,
        item_map_reverse
    )

    # 6. 复制图片（如果需要）
    if args.copy_images and image_dir.exists():
        output_image_dir = Path(args.output_dir) / 'images'
        copy_images(
            str(image_dir),
            str(output_image_dir),
            item_map_reverse,
            remapped_movies
        )

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
