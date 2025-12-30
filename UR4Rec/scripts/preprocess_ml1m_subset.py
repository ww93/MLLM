"""
MovieLens-1M 长序列子集生成脚本

目标: 生成Top-1000活跃用户的长序列子集，用于证明"终身学习"能力
核心功能:
1. 从原始 ratings.dat 加载数据 (UserID::ItemID::Rating::Timestamp)
2. 选择交互次数最多的Top-1000用户 (Active Users)
3. 重新映射 UserID 到 [0, 999] 范围
4. 保持 ItemID 不变 (与预提取的LLM特征保持一致)
5. 按时间戳排序
6. 保存为空格分隔格式: user_id item_id rating timestamp
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm


def load_ratings(ratings_file: Path) -> pd.DataFrame:
    """
    加载 MovieLens-1M ratings.dat 文件

    格式: UserID::ItemID::Rating::Timestamp

    Args:
        ratings_file: ratings.dat 文件路径

    Returns:
        ratings: DataFrame with columns [user_id, item_id, rating, timestamp]
    """
    print(f"正在加载原始数据: {ratings_file}")

    # 读取双冒号分隔的数据
    ratings = pd.read_csv(
        ratings_file,
        sep='::',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python',
        encoding='latin-1'
    )

    print(f"✓ 加载完成: {len(ratings)} 条交互记录")
    print(f"  用户数: {ratings['user_id'].nunique()}")
    print(f"  物品数: {ratings['item_id'].nunique()}")
    print(f"  评分范围: [{ratings['rating'].min()}, {ratings['rating'].max()}]")
    print(f"  时间戳范围: [{ratings['timestamp'].min()}, {ratings['timestamp'].max()}]")

    return ratings


def select_top_users(
    ratings: pd.DataFrame,
    top_k: int = 1000
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    选择交互次数最多的Top-K活跃用户

    Args:
        ratings: 原始评分数据
        top_k: 选择的用户数量（默认1000）

    Returns:
        filtered_ratings: 过滤后的评分数据
        user_mapping: 原始UserID -> 新UserID的映射字典 (0到top_k-1)
    """
    print(f"\n正在选择Top-{top_k}活跃用户...")

    # 计算每个用户的交互次数
    user_counts = ratings.groupby('user_id').size().reset_index(name='count')

    # 按交互次数降序排序，选择Top-K
    top_users = user_counts.nlargest(top_k, 'count')

    print(f"✓ 选择了{len(top_users)}个活跃用户")
    print(f"  最大交互次数: {top_users['count'].max()}")
    print(f"  最小交互次数: {top_users['count'].min()}")
    print(f"  平均交互次数: {top_users['count'].mean():.2f}")
    print(f"  中位数交互次数: {top_users['count'].median():.2f}")

    # 过滤数据，只保留Top-K用户的记录
    filtered_ratings = ratings[ratings['user_id'].isin(top_users['user_id'])].copy()

    print(f"✓ 过滤后保留 {len(filtered_ratings)} 条交互记录")

    # 创建UserID映射: 原始ID -> 新ID (0到top_k-1)
    # 按照交互次数从多到少映射，保证user_id=0是最活跃的用户
    top_users_sorted = top_users.sort_values('count', ascending=False)
    user_mapping = {
        old_id: new_id
        for new_id, old_id in enumerate(top_users_sorted['user_id'])
    }

    print(f"✓ 创建UserID映射: {len(user_mapping)} 个用户")
    print(f"  原始UserID范围: [{min(user_mapping.keys())}, {max(user_mapping.keys())}]")
    print(f"  新UserID范围: [0, {top_k-1}]")

    return filtered_ratings, user_mapping


def remap_user_ids(
    ratings: pd.DataFrame,
    user_mapping: Dict[int, int]
) -> pd.DataFrame:
    """
    重新映射UserID到[0, top_k-1]范围
    注意: ItemID保持不变，以与预提取的LLM特征对齐

    Args:
        ratings: 过滤后的评分数据
        user_mapping: 原始UserID -> 新UserID的映射

    Returns:
        remapped_ratings: 重新映射UserID后的数据
    """
    print("\n正在重新映射UserID...")

    # 应用映射
    ratings_remapped = ratings.copy()
    ratings_remapped['user_id'] = ratings_remapped['user_id'].map(user_mapping)

    # 验证映射
    assert ratings_remapped['user_id'].isna().sum() == 0, "存在未映射的UserID!"

    print(f"✓ UserID重新映射完成")
    print(f"  新UserID范围: [{ratings_remapped['user_id'].min()}, {ratings_remapped['user_id'].max()}]")
    print(f"  ItemID保持不变: [{ratings_remapped['item_id'].min()}, {ratings_remapped['item_id'].max()}]")

    return ratings_remapped


def sort_by_timestamp(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    按时间戳排序交互记录（全局排序）

    Args:
        ratings: 评分数据

    Returns:
        sorted_ratings: 按时间戳排序后的数据
    """
    print("\n正在按时间戳排序...")

    sorted_ratings = ratings.sort_values('timestamp').reset_index(drop=True)

    print(f"✓ 排序完成: {len(sorted_ratings)} 条记录")

    return sorted_ratings


def save_subset(
    ratings: pd.DataFrame,
    output_file: Path,
    user_mapping: Dict[int, int]
):
    """
    保存子集数据到文件

    格式: 空格分隔 - user_id item_id rating timestamp
    同时保存用户映射信息

    Args:
        ratings: 处理后的评分数据
        output_file: 输出文件路径
        user_mapping: UserID映射字典（用于保存映射信息）
    """
    print(f"\n正在保存子集数据到: {output_file}")

    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 保存评分数据 (空格分隔格式)
    ratings.to_csv(
        output_file,
        sep=' ',
        columns=['user_id', 'item_id', 'rating', 'timestamp'],
        index=False,
        header=False  # 不写表头，与原始格式一致
    )

    print(f"✓ 保存完成: {output_file}")
    print(f"  格式: user_id item_id rating timestamp (空格分隔)")
    print(f"  记录数: {len(ratings)}")

    # 保存用户映射 (用于追溯原始UserID)
    mapping_file = output_file.parent / 'user_mapping.txt'
    with open(mapping_file, 'w') as f:
        f.write("# 原始UserID -> 新UserID (0-999)\n")
        for old_id, new_id in sorted(user_mapping.items(), key=lambda x: x[1]):
            f.write(f"{old_id} {new_id}\n")

    print(f"✓ 用户映射保存至: {mapping_file}")

    # 打印统计信息
    print("\n" + "="*60)
    print("数据集统计:")
    print("="*60)
    print(f"用户数: {ratings['user_id'].nunique()}")
    print(f"物品数: {ratings['item_id'].nunique()}")
    print(f"交互数: {len(ratings)}")
    print(f"稀疏度: {1 - len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique()):.4f}")
    print(f"平均序列长度: {len(ratings) / ratings['user_id'].nunique():.2f}")

    # 按用户统计序列长度分布
    seq_lengths = ratings.groupby('user_id').size()
    print(f"\n序列长度分布:")
    print(f"  最小: {seq_lengths.min()}")
    print(f"  最大: {seq_lengths.max()}")
    print(f"  平均: {seq_lengths.mean():.2f}")
    print(f"  中位数: {seq_lengths.median():.2f}")
    print(f"  标准差: {seq_lengths.std():.2f}")
    print("="*60)


def main(args):
    """主函数"""

    # 设置路径
    if args.input_file:
        ratings_file = Path(args.input_file)
    else:
        # 默认路径: data/Multimodal_Datasets/M_ML-1M/ratings.dat
        project_root = Path(__file__).parent.parent
        ratings_file = project_root / 'data' / 'Multimodal_Datasets' / 'M_ML-1M' / 'ratings.dat'

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        # 默认输出路径: data/ml-1m/subset_ratings.dat
        project_root = Path(__file__).parent.parent
        output_file = project_root / 'data' / 'ml-1m' / 'subset_ratings.dat'

    # 检查输入文件是否存在
    if not ratings_file.exists():
        print(f"错误: 输入文件不存在: {ratings_file}")
        print(f"请确保ML-1M数据集已下载到正确位置")
        return

    print("="*60)
    print("MovieLens-1M 长序列子集生成")
    print("="*60)
    print(f"输入文件: {ratings_file}")
    print(f"输出文件: {output_file}")
    print(f"Top-K用户: {args.top_k}")
    print("="*60)

    # 步骤1: 加载原始数据
    ratings = load_ratings(ratings_file)

    # 步骤2: 选择Top-K活跃用户并创建映射
    filtered_ratings, user_mapping = select_top_users(ratings, args.top_k)

    # 步骤3: 重新映射UserID (ItemID保持不变)
    remapped_ratings = remap_user_ids(filtered_ratings, user_mapping)

    # 步骤4: 按时间戳排序
    sorted_ratings = sort_by_timestamp(remapped_ratings)

    # 步骤5: 保存子集
    save_subset(sorted_ratings, output_file, user_mapping)

    print(f"\n✓ 全部完成！")
    print(f"生成的子集可用于FedDMMR的终身学习实验")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='生成MovieLens-1M长序列子集 (用于FedDMMR终身学习实验)'
    )

    parser.add_argument(
        '--input_file',
        type=str,
        default="UR4Rec/data/Multimodal_Datasets/M_ML-1M/ratings.dat",
        help='输入ratings.dat文件路径 (默认: data/Multimodal_Datasets/M_ML-1M/ratings.dat)'
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default="UR4Rec/data/ml-1m/subset_ratings.dat",
        help='输出子集文件路径 (默认: data/ml-1m/subset_ratings.dat)'
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=1000,
        help='选择Top-K活跃用户 (默认: 1000)'
    )

    args = parser.parse_args()
    main(args)
