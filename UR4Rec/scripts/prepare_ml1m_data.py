"""
预处理MovieLens-1M数据，转换为FedMem训练格式
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def preprocess_ml1m(input_file, output_file):
    """
    预处理ML-1M数据

    输入格式: user_id::movie_id::rating::timestamp
    输出格式: user_id movie_id rating timestamp (空格分隔)
    """
    print(f"读取数据: {input_file}")

    user_sequences = defaultdict(list)

    with open(input_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) != 4:
                continue

            user_id = int(parts[0])
            movie_id = int(parts[1])
            rating = int(parts[2])
            timestamp = int(parts[3])

            user_sequences[user_id].append((timestamp, movie_id, rating))

    print(f"总用户数: {len(user_sequences)}")

    # 按时间排序每个用户的序列
    for user_id in user_sequences:
        user_sequences[user_id].sort(key=lambda x: x[0])

    # 写入输出文件
    print(f"写入数据: {output_file}")
    with open(output_file, 'w') as f:
        for user_id, items in user_sequences.items():
            for timestamp, movie_id, rating in items:
                f.write(f"{user_id} {movie_id} {rating} {timestamp}\n")

    print(f"预处理完成！")
    print(f"  用户数: {len(user_sequences)}")
    print(f"  交互数: {sum(len(items) for items in user_sequences.values())}")

    # 统计信息
    seq_lengths = [len(items) for items in user_sequences.values()]
    print(f"  序列长度: min={min(seq_lengths)}, max={max(seq_lengths)}, avg={sum(seq_lengths)/len(seq_lengths):.1f}")


if __name__ == "__main__":
    input_file = "UR4Rec/data/Multimodal_Datasets/M_ML-1M/ratings.dat"
    output_file = "UR4Rec/data/ml1m_ratings_processed.dat"

    preprocess_ml1m(input_file, output_file)
