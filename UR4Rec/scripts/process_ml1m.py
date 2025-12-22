#!/usr/bin/env python3
"""
处理ML-1M数据集为FedMem训练格式

将 M_ML-1M/ratings.dat 转换为 load_user_sequences 可以读取的格式
格式: user_id item_1 item_2 item_3 ...
"""

import os
from collections import defaultdict
from typing import Dict, List

def process_ml1m_ratings(
    input_path: str,
    output_path: str,
    min_rating: int = 4,
    min_seq_len: int = 5
):
    """
    处理ML-1M ratings文件（使用>=4星阈值，从ML-100K的教训中学习）

    Args:
        input_path: 输入文件路径 (ratings.dat)
        output_path: 输出文件路径
        min_rating: 最小评分阈值（默认4星）
        min_seq_len: 最小序列长度（过滤掉交互过少的用户）
    """
    print("=" * 70)
    print(f"处理ML-1M数据集 (min_rating >= {min_rating})")
    print("=" * 70)

    # 读取评分数据
    print(f"\n📖 正在读取: {input_path}")
    user_items = defaultdict(list)

    with open(input_path, 'r') as f:
        for line in f:
            parts = line.strip().split('::')
            if len(parts) >= 4:
                user_id = int(parts[0])
                item_id = int(parts[1])
                rating = int(parts[2])
                timestamp = int(parts[3])

                # 只保留>=4星的明确正样本（从ML-100K实验学到的教训）
                if rating >= min_rating:
                    user_items[user_id].append((timestamp, item_id))

    print(f"✅ 读取完成")
    print(f"   - 原始用户数: {len(user_items)}")

    # 按时间排序
    print("\n⏰ 正在按时间排序...")
    for user_id in user_items:
        user_items[user_id].sort(key=lambda x: x[0])  # 按timestamp排序
        user_items[user_id] = [item_id for _, item_id in user_items[user_id]]

    # 过滤短序列
    print(f"\n🔍 正在过滤序列（最小长度: {min_seq_len}）...")
    filtered_users = {
        uid: items for uid, items in user_items.items()
        if len(items) >= min_seq_len
    }

    print(f"✅ 过滤完成")
    print(f"   - 保留用户数: {len(filtered_users)}")
    print(f"   - 过滤掉: {len(user_items) - len(filtered_users)} 个用户")

    # 统计信息
    all_items = set()
    seq_lengths = []
    for items in filtered_users.values():
        all_items.update(items)
        seq_lengths.append(len(items))

    print(f"\n📊 数据统计:")
    print(f"   - 用户数: {len(filtered_users)}")
    print(f"   - 物品数: {len(all_items)}")
    print(f"   - 最大Item ID: {max(all_items)}")
    print(f"   - 平均序列长度: {sum(seq_lengths) / len(seq_lengths):.2f}")
    print(f"   - 最短序列: {min(seq_lengths)}")
    print(f"   - 最长序列: {max(seq_lengths)}")

    # 写入输出文件
    print(f"\n💾 正在保存到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        for user_id in sorted(filtered_users.keys()):
            items = filtered_users[user_id]
            # 格式: user_id item_1 item_2 item_3 ...
            f.write(f"{user_id} {' '.join(map(str, items))}\n")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ 保存成功！文件大小: {file_size_mb:.2f} MB")

    print("\n" + "=" * 70)
    print("🎉 处理完成！")
    print("=" * 70)

    return len(filtered_users), len(all_items)


def main():
    # 路径配置
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_path = os.path.join(
        project_root,
        'UR4Rec/data/Multimodal_Datasets/M_ML-1M/ratings.dat'
    )
    output_path = os.path.join(
        project_root,
        'UR4Rec/data/ml1m_ratings_processed.dat'
    )

    # 检查输入文件
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ 找不到输入文件: {input_path}")

    # 处理数据
    num_users, num_items = process_ml1m_ratings(
        input_path=input_path,
        output_path=output_path,
        min_rating=4,  # 使用4星阈值（基于ML-100K实验的教训）
        min_seq_len=5
    )

    print(f"\n💡 下一步:")
    print(f"   1. 生成item descriptions (LLM推理)")
    print(f"      python UR4Rec/scripts/generate_llm_data.py \\")
    print(f"          --data_dir UR4Rec/data/ml1m \\")
    print(f"          --output_dir UR4Rec/data/ml1m \\")
    print(f"          --only_items")
    print(f"   2. 抽取CLIP特征")
    print(f"   3. 抽取文本特征")
    print(f"   4. 运行训练")
    print()


if __name__ == "__main__":
    main()
