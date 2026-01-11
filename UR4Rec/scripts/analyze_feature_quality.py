"""
多模态特征质量分析脚本

分析CLIP视觉特征和SBERT文本特征是否适合推荐任务：
1. 用户喜欢的物品之间的特征相似度（应该高）
2. 随机物品对之间的特征相似度（基线）
3. 特征的分布统计
4. 特征多样性分析
5. 与协同过滤的相关性
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_user_sequences(data_path):
    """加载用户交互序列"""
    print("\n[1/6] 加载用户交互序列...")
    user_sequences = defaultdict(list)

    with open(data_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user_id = int(parts[0])
                item_id = int(parts[1])
                user_sequences[user_id].append(item_id)

    print(f"  ✓ 用户数: {len(user_sequences)}")
    print(f"  ✓ 平均序列长度: {np.mean([len(seq) for seq in user_sequences.values()]):.1f}")

    return user_sequences


def load_features(visual_path, text_path):
    """加载多模态特征"""
    print("\n[2/6] 加载多模态特征...")

    visual_feats = torch.load(visual_path, weights_only=True)
    text_feats = torch.load(text_path, weights_only=True)

    print(f"  ✓ 视觉特征: {visual_feats.shape}")
    print(f"  ✓ 文本特征: {text_feats.shape}")

    # L2归一化，用于计算余弦相似度
    visual_feats = visual_feats / (visual_feats.norm(dim=1, keepdim=True) + 1e-8)
    text_feats = text_feats / (text_feats.norm(dim=1, keepdim=True) + 1e-8)

    return visual_feats, text_feats


def compute_similarity_stats(features, item_pairs, desc):
    """计算物品对之间的特征相似度统计"""
    similarities = []

    for item1, item2 in item_pairs:
        if item1 < len(features) and item2 < len(features):
            sim = torch.dot(features[item1], features[item2]).item()
            similarities.append(sim)

    if len(similarities) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}

    return {
        "mean": np.mean(similarities),
        "std": np.std(similarities),
        "min": np.min(similarities),
        "max": np.max(similarities),
        "median": np.median(similarities),
        "count": len(similarities)
    }


def analyze_user_preference_similarity(user_sequences, visual_feats, text_feats):
    """分析用户喜欢的物品之间的特征相似度"""
    print("\n[3/6] 分析用户偏好相似度...")

    # 采样用户和物品对
    num_users = min(100, len(user_sequences))
    sampled_users = np.random.choice(list(user_sequences.keys()), num_users, replace=False)

    user_visual_pairs = []
    user_text_pairs = []

    for user_id in tqdm(sampled_users, desc="  采样用户"):
        items = user_sequences[user_id]
        if len(items) < 2:
            continue

        # 每个用户采样5对物品
        num_pairs = min(5, len(items) * (len(items) - 1) // 2)
        for _ in range(num_pairs):
            idx1, idx2 = np.random.choice(len(items), 2, replace=False)
            user_visual_pairs.append((items[idx1], items[idx2]))
            user_text_pairs.append((items[idx1], items[idx2]))

    print(f"  ✓ 采样了 {len(user_visual_pairs)} 个用户内物品对")

    # 计算相似度
    visual_sim = compute_similarity_stats(visual_feats, user_visual_pairs, "用户偏好-视觉")
    text_sim = compute_similarity_stats(text_feats, user_text_pairs, "用户偏好-文本")

    return {
        "visual": visual_sim,
        "text": text_sim
    }


def analyze_random_similarity(visual_feats, text_feats, num_pairs=1000):
    """分析随机物品对的相似度（基线）"""
    print("\n[4/6] 分析随机物品对相似度（基线）...")

    num_items = len(visual_feats)
    random_pairs = []

    for _ in range(num_pairs):
        idx1, idx2 = np.random.choice(num_items, 2, replace=False)
        random_pairs.append((idx1, idx2))

    visual_sim = compute_similarity_stats(visual_feats, random_pairs, "随机-视觉")
    text_sim = compute_similarity_stats(text_feats, random_pairs, "随机-文本")

    return {
        "visual": visual_sim,
        "text": text_sim
    }


def analyze_feature_diversity(visual_feats, text_feats):
    """分析特征的多样性"""
    print("\n[5/6] 分析特征多样性...")

    # 计算特征的标准差（每个维度）
    visual_std = visual_feats.std(dim=0).mean().item()
    text_std = text_feats.std(dim=0).mean().item()

    # 计算特征的有效秩（衡量多样性）
    visual_svd = torch.linalg.svdvals(visual_feats)
    text_svd = torch.linalg.svdvals(text_feats)

    # 计算归一化的有效秩
    def effective_rank(s):
        s = s / s.sum()
        entropy = -(s * torch.log(s + 1e-10)).sum()
        return torch.exp(entropy).item()

    visual_rank = effective_rank(visual_svd)
    text_rank = effective_rank(text_svd)

    print(f"  ✓ 视觉特征标准差: {visual_std:.4f}")
    print(f"  ✓ 文本特征标准差: {text_std:.4f}")
    print(f"  ✓ 视觉特征有效秩: {visual_rank:.1f} / {visual_feats.shape[1]}")
    print(f"  ✓ 文本特征有效秩: {text_rank:.1f} / {text_feats.shape[1]}")

    return {
        "visual": {
            "std": visual_std,
            "effective_rank": visual_rank,
            "max_rank": visual_feats.shape[1]
        },
        "text": {
            "std": text_std,
            "effective_rank": text_rank,
            "max_rank": text_feats.shape[1]
        }
    }


def analyze_collaborative_correlation(user_sequences, visual_feats, text_feats):
    """分析特征与协同过滤的相关性"""
    print("\n[6/6] 分析特征与协同过滤的相关性...")

    # 计算物品的流行度
    item_popularity = defaultdict(int)
    for items in user_sequences.values():
        for item in items:
            item_popularity[item] += 1

    # 找出热门物品和冷门物品
    sorted_items = sorted(item_popularity.items(), key=lambda x: x[1], reverse=True)
    top_items = [item for item, _ in sorted_items[:100]]  # Top 100
    tail_items = [item for item, _ in sorted_items[-100:]]  # Bottom 100

    # 计算热门物品之间的相似度
    top_pairs = []
    for i in range(len(top_items)):
        for j in range(i+1, min(i+5, len(top_items))):
            top_pairs.append((top_items[i], top_items[j]))

    top_visual = compute_similarity_stats(visual_feats, top_pairs, "热门物品-视觉")
    top_text = compute_similarity_stats(text_feats, top_pairs, "热门物品-文本")

    # 计算冷门物品之间的相似度
    tail_pairs = []
    for i in range(len(tail_items)):
        for j in range(i+1, min(i+5, len(tail_items))):
            tail_pairs.append((tail_items[i], tail_items[j]))

    tail_visual = compute_similarity_stats(visual_feats, tail_pairs, "冷门物品-视觉")
    tail_text = compute_similarity_stats(text_feats, tail_pairs, "冷门物品-文本")

    return {
        "popular_items": {
            "visual": top_visual,
            "text": top_text
        },
        "tail_items": {
            "visual": tail_visual,
            "text": tail_text
        }
    }


def print_results(results):
    """打印分析结果"""
    print("\n" + "="*80)
    print("特征质量分析报告")
    print("="*80)

    # 1. 用户偏好相似度
    print("\n[结果1] 用户喜欢的物品之间的相似度（期望：高相似度）")
    print(f"  视觉特征:")
    for k, v in results["user_preference"]["visual"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"  文本特征:")
    for k, v in results["user_preference"]["text"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 2. 随机基线
    print("\n[结果2] 随机物品对的相似度（基线）")
    print(f"  视觉特征:")
    for k, v in results["random"]["visual"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"  文本特征:")
    for k, v in results["random"]["text"].items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 3. 对比分析
    print("\n[结果3] 用户偏好 vs 随机基线")
    user_visual = results["user_preference"]["visual"]["mean"]
    random_visual = results["random"]["visual"]["mean"]
    user_text = results["user_preference"]["text"]["mean"]
    random_text = results["random"]["text"]["mean"]

    visual_lift = (user_visual - random_visual) / (random_visual + 1e-8) * 100
    text_lift = (user_text - random_text) / (random_text + 1e-8) * 100

    print(f"  视觉特征提升: {visual_lift:+.1f}%")
    print(f"    用户偏好: {user_visual:.4f}")
    print(f"    随机基线: {random_visual:.4f}")
    if visual_lift > 5:
        print(f"    ✅ 视觉特征能区分用户偏好")
    elif visual_lift > 0:
        print(f"    ⚠️  视觉特征略微相关，但信号较弱")
    else:
        print(f"    ❌ 视觉特征与用户偏好不相关")

    print(f"\n  文本特征提升: {text_lift:+.1f}%")
    print(f"    用户偏好: {user_text:.4f}")
    print(f"    随机基线: {random_text:.4f}")
    if text_lift > 5:
        print(f"    ✅ 文本特征能区分用户偏好")
    elif text_lift > 0:
        print(f"    ⚠️  文本特征略微相关，但信号较弱")
    else:
        print(f"    ❌ 文本特征与用户偏好不相关")

    # 4. 特征多样性
    print("\n[结果4] 特征多样性")
    print(f"  视觉特征:")
    print(f"    标准差: {results['diversity']['visual']['std']:.4f}")
    print(f"    有效秩: {results['diversity']['visual']['effective_rank']:.1f} / {results['diversity']['visual']['max_rank']}")
    print(f"  文本特征:")
    print(f"    标准差: {results['diversity']['text']['std']:.4f}")
    print(f"    有效秩: {results['diversity']['text']['effective_rank']:.1f} / {results['diversity']['text']['max_rank']}")

    # 5. 协同过滤相关性
    print("\n[结果5] 与协同过滤的相关性")
    print(f"  热门物品之间:")
    print(f"    视觉相似度: {results['collaborative']['popular_items']['visual']['mean']:.4f}")
    print(f"    文本相似度: {results['collaborative']['popular_items']['text']['mean']:.4f}")
    print(f"  冷门物品之间:")
    print(f"    视觉相似度: {results['collaborative']['tail_items']['visual']['mean']:.4f}")
    print(f"    文本相似度: {results['collaborative']['tail_items']['text']['mean']:.4f}")

    # 6. 总结建议
    print("\n" + "="*80)
    print("诊断建议")
    print("="*80)

    if visual_lift < 2 and text_lift < 2:
        print("\n⚠️  **严重问题**: 多模态特征与推荐任务不相关")
        print("\n可能原因:")
        print("  1. CLIP/SBERT是在通用任务上预训练的，不适合推荐")
        print("  2. 电影的视觉海报和描述文本与用户兴趣关联度低")
        print("  3. 数据集中用户行为主要由协同过滤驱动")

        print("\n建议:")
        print("  ❌ 放弃使用多模态特征")
        print("  ✅ 只使用SASRec (Stage 1)")
        print("  ✅ 或者重新提取更相关的特征（如类型标签、导演、演员）")

    elif visual_lift < 5 or text_lift < 5:
        print("\n⚠️  **中等问题**: 多模态特征信号较弱")
        print("\n建议:")
        print("  ⚠️  使用极小的gating_init (0.0001)")
        print("  ⚠️  降低contrastive_lambda (0.1或更低)")
        print("  ⚠️  考虑使用注意力机制而非固定权重")

    else:
        print("\n✅ 多模态特征与推荐任务相关")
        print("\n建议:")
        print("  ✅ 可以使用Stage 2/3训练")
        print("  ✅ gating_init可以稍大 (0.001 - 0.01)")
        print("  ✅ contrastive_lambda可以保持 (0.5)")


def main():
    # 配置路径
    data_dir = Path("UR4Rec/data/ml-1m")
    data_file = data_dir / "subset_ratings.dat"
    visual_file = data_dir / "clip_features.pt"
    text_file = data_dir / "text_features.pt"

    print("="*80)
    print("多模态特征质量分析")
    print("="*80)
    print(f"数据目录: {data_dir}")
    print(f"交互文件: {data_file}")
    print(f"视觉特征: {visual_file}")
    print(f"文本特征: {text_file}")

    # 检查文件存在性
    if not data_file.exists():
        print(f"\n❌ 错误: 交互文件不存在: {data_file}")
        return 1
    if not visual_file.exists():
        print(f"\n❌ 错误: 视觉特征文件不存在: {visual_file}")
        return 1
    if not text_file.exists():
        print(f"\n❌ 错误: 文本特征文件不存在: {text_file}")
        return 1

    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 执行分析
    user_sequences = load_user_sequences(data_file)
    visual_feats, text_feats = load_features(visual_file, text_file)

    results = {
        "user_preference": analyze_user_preference_similarity(
            user_sequences, visual_feats, text_feats
        ),
        "random": analyze_random_similarity(visual_feats, text_feats),
        "diversity": analyze_feature_diversity(visual_feats, text_feats),
        "collaborative": analyze_collaborative_correlation(
            user_sequences, visual_feats, text_feats
        )
    }

    # 打印结果
    print_results(results)

    # 保存结果
    output_file = Path("UR4Rec/feature_quality_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ 分析结果已保存到: {output_file}")
    print("="*80)

    return 0


if __name__ == "__main__":
    exit(main())
