#!/usr/bin/env python3
"""
从LLM生成的item描述提取文本嵌入特征

这个脚本从 data/llm_generated/item_descriptions.json 读取item描述，
使用Sentence-BERT模型提取文本嵌入，保存为 .pt 文件供FedMem训练使用。

用法:
    python UR4Rec/scripts/generate_text_features.py

输出:
    UR4Rec/data/item_text_features.pt - [num_items, 384] 文本特征tensor
"""

import json
import torch
import os
from typing import Dict
from sentence_transformers import SentenceTransformer


def load_item_descriptions(json_path: str) -> Dict[int, str]:
    """
    从JSON文件加载item描述

    Args:
        json_path: JSON文件路径

    Returns:
        字典 {item_id: description_text}
    """
    print(f"📖 正在加载item描述: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 将字符串键转换为整数
    descriptions = {int(k): v for k, v in data.items()}
    print(f"✅ 加载了 {len(descriptions)} 个item描述")
    return descriptions


def generate_text_embeddings(
    descriptions: Dict[int, str],
    model_name: str = 'all-MiniLM-L6-v2',
    device: str = 'cpu'
) -> torch.Tensor:
    """
    使用Sentence-BERT生成文本嵌入

    Args:
        descriptions: {item_id: description_text}
        model_name: Sentence-BERT模型名称
        device: 'cpu', 'cuda', 或 'mps'

    Returns:
        文本特征tensor [num_items, embedding_dim]
    """
    print(f"\n🤖 正在加载Sentence-BERT模型: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ 模型加载完成，嵌入维度: {embedding_dim}")

    # 确定最大item_id以创建正确大小的tensor
    max_item_id = max(descriptions.keys())
    num_items = max_item_id + 1

    print(f"\n🔢 Item ID范围: 1 - {max_item_id}")
    print(f"📦 创建特征矩阵: [{num_items}, {embedding_dim}]")

    # 初始化特征矩阵（零填充）
    text_features = torch.zeros(num_items, embedding_dim, dtype=torch.float32)

    # 准备批量编码的文本和对应的item_id
    item_ids = []
    texts = []
    for item_id in sorted(descriptions.keys()):
        item_ids.append(item_id)
        texts.append(descriptions[item_id])

    print(f"\n🚀 正在生成 {len(texts)} 个item的文本嵌入...")

    # 批量编码（更高效）
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = item_ids[i:i+batch_size]

        # 生成嵌入
        embeddings = model.encode(
            batch_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=device
        )

        # 填充到特征矩阵
        for j, item_id in enumerate(batch_ids):
            text_features[item_id] = embeddings[j].cpu()

        if (i // batch_size + 1) % 10 == 0:
            print(f"  进度: {i + len(batch_texts)}/{len(texts)} items")

    print(f"✅ 文本嵌入生成完成！")

    # 统计信息
    num_nonzero = (text_features.abs().sum(dim=1) > 0).sum().item()
    print(f"\n📊 统计信息:")
    print(f"  - 有效特征: {num_nonzero}/{num_items}")
    print(f"  - 零填充: {num_items - num_nonzero}")
    print(f"  - 特征形状: {text_features.shape}")
    print(f"  - 特征范围: [{text_features.min():.4f}, {text_features.max():.4f}]")

    return text_features


def main():
    # 路径配置
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_path = os.path.join(project_root, 'data', 'llm_generated', 'item_descriptions.json')
    output_path = os.path.join(project_root, 'UR4Rec', 'data', 'item_text_features.pt')

    print("=" * 70)
    print("📝 从LLM生成的描述提取文本特征")
    print("=" * 70)

    # 检查输入文件
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ 找不到item描述文件: {json_path}")

    # 加载描述
    descriptions = load_item_descriptions(json_path)

    # 选择设备
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"\n🖥️  使用设备: {device.upper()}")

    # 生成嵌入
    text_features = generate_text_embeddings(descriptions, device=device)

    # 保存特征
    print(f"\n💾 正在保存特征到: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(text_features, output_path)

    # 验证保存
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ 保存成功！文件大小: {file_size_mb:.2f} MB")

    # 测试加载
    print(f"\n🧪 验证文件可以正确加载...")
    loaded_features = torch.load(output_path)
    print(f"✅ 加载成功！形状: {loaded_features.shape}")

    print("\n" + "=" * 70)
    print("🎉 文本特征生成完成！")
    print("=" * 70)
    print(f"\n💡 使用方法:")
    print(f"   python UR4Rec/scripts/train_fedmem.py \\")
    print(f"       --visual_file clip_features.pt \\")
    print(f"       --text_file item_text_features.pt \\")
    print(f"       --contrastive_lambda 0.1")
    print()


if __name__ == "__main__":
    main()
