#!/usr/bin/env python3
"""
测试脚本：验证 text.xls 中的描述是否被正确提取和使用
"""
import json
from pathlib import Path

def test_text_extraction():
    """测试文本描述提取"""

    print("=" * 60)
    print("测试 text.xls 描述提取")
    print("=" * 60)

    # 检查预处理后的数据
    metadata_file = Path("data/ml-100k-multimodal/item_metadata.json")

    if not metadata_file.exists():
        print(f"❌ 未找到: {metadata_file}")
        print("请先运行:")
        print("  python scripts/preprocess_multimodal_dataset.py \\")
        print("      --dataset ml-100k \\")
        print("      --data_dir data/Multimodal_Datasets \\")
        print("      --output_dir data/ml-100k-multimodal \\")
        print("      --copy_images")
        return False

    print(f"\n✅ 找到文件: {metadata_file}")

    # 加载元数据
    with open(metadata_file, 'r', encoding='utf-8') as f:
        item_metadata = json.load(f)

    print(f"\n📊 物品总数: {len(item_metadata)}")

    # 统计有描述的物品
    with_desc = 0
    without_desc = 0

    for item_id, meta in item_metadata.items():
        if 'description' in meta and meta['description']:
            # 检查是否不只是标题+类型
            desc = meta['description']
            title = meta.get('title', '')
            if desc != f"{title}. Genres: {', '.join(meta.get('genres', []))}":
                with_desc += 1
            else:
                without_desc += 1
        else:
            without_desc += 1

    print(f"\n描述统计:")
    print(f"  ✅ 有文本描述 (来自 text.xls): {with_desc}")
    print(f"  ⚠️  仅有基础信息 (标题+类型): {without_desc}")
    print(f"  📈 描述覆盖率: {with_desc / len(item_metadata) * 100:.1f}%")

    # 显示前3个示例
    print(f"\n📝 示例 (前3个有完整描述的物品):")
    count = 0
    for item_id, meta in item_metadata.items():
        desc = meta.get('description', '')
        title = meta.get('title', '')

        # 跳过只有基础信息的
        if desc == f"{title}. Genres: {', '.join(meta.get('genres', []))}":
            continue

        print(f"\n物品 {item_id}: {title}")
        print(f"  类型: {', '.join(meta.get('genres', []))}")
        print(f"  描述: {desc[:150]}..." if len(desc) > 150 else f"  描述: {desc}")

        count += 1
        if count >= 3:
            break

    # 检查 LLM 生成的数据
    print("\n" + "=" * 60)
    print("检查 LLM 数据生成")
    print("=" * 60)

    llm_desc_file = Path("data/ml-100k-multimodal/llm_generated/item_descriptions.json")

    if llm_desc_file.exists():
        print(f"✅ 找到 LLM 生成的描述: {llm_desc_file}")

        with open(llm_desc_file, 'r', encoding='utf-8') as f:
            llm_descriptions = json.load(f)

        print(f"   生成的描述数量: {len(llm_descriptions)}")

        # 检查是否使用了原始描述
        match_count = 0
        for item_id in list(llm_descriptions.keys())[:100]:  # 检查前100个
            if item_id in item_metadata:
                original_desc = item_metadata[item_id].get('description', '')
                llm_desc = llm_descriptions[item_id]

                if original_desc and original_desc in llm_desc:
                    match_count += 1

        if match_count > 50:
            print(f"   ✅ 确认：使用了 text.xls 中的原始描述")
        else:
            print(f"   🔄 可能使用了 LLM 重新生成的描述")
    else:
        print(f"⚠️  未找到 LLM 生成的描述")
        print("运行以下命令生成:")
        print("  python scripts/generate_llm_data.py \\")
        print("      --config configs/movielens_100k.yaml \\")
        print("      --data_dir data/ml-100k-multimodal \\")
        print("      --output_dir data/ml-100k-multimodal/llm_generated \\")
        print("      --llm_backend mock")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

    return True


if __name__ == "__main__":
    test_text_extraction()
