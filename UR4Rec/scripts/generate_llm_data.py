"""
生成 LLM 数据

使用 LLM 离线生成：
1. 用户偏好描述（基于历史交互）
2. 物品文本描述（基于元数据）

生成后的数据用于训练轻量级检索器

使用方法：
- 基础用法: python generate_llm_data.py --data_dir data/ --output_dir output/
- 只生成用户偏好: 添加 --only_users
- 只生成物品描述: 添加 --only_items

默认配置：
- 模型: qwen-flash (DashScope)
- API Key: 从环境变量 DASHSCOPE_API_KEY 读取

物品描述 Prompt:
使用专门设计的语义密集型 prompt，关注视觉美学、核心主题、情感基调和受众吸引力。
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import yaml
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, List

from models.llm_generator import LLMPreferenceGenerator


def load_dataset(data_dir: str, split: str = 'train') -> Dict:
    """
    加载数据集

    Args:
        data_dir: 数据目录
        split: 'train' | 'val' | 'test'

    Returns:
        data: 包含用户序列和物品信息的字典
    """
    data_path = Path(data_dir)

    # 加载序列数据
    seq_file = data_path / f'{split}_sequences.npy'
    if not seq_file.exists():
        raise FileNotFoundError(f"序列文件不存在: {seq_file}")

    sequences = np.load(seq_file, allow_pickle=True).item()

    # 加载物品元数据（如果有）
    metadata_file = data_path / 'item_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            item_metadata = json.load(f)
    else:
        print(f"警告: 物品元数据文件不存在: {metadata_file}")
        item_metadata = {}

    # 加载物品映射
    item_map_file = data_path / 'item_map.json'
    if item_map_file.exists():
        with open(item_map_file, 'r', encoding='utf-8') as f:
            item_map = json.load(f)
    else:
        item_map = {}

    return {
        'sequences': sequences,
        'item_metadata': item_metadata,
        'item_map': item_map
    }


def generate_user_preferences(
    generator: LLMPreferenceGenerator,
    sequences: Dict,
    item_metadata: Dict,
    output_path: str,
    max_users: int = None,
    min_history_length: int = 3
):
    """
    生成用户偏好描述

    Args:
        generator: LLM 生成器
        sequences: 用户序列数据
        item_metadata: 物品元数据
        output_path: 输出文件路径
        max_users: 最多生成多少用户（None表示全部）
        min_history_length: 最少历史长度
    """
    user_preferences = {}

    user_ids = list(sequences.keys())
    if max_users is not None:
        user_ids = user_ids[:max_users]

    print(f"\n开始生成用户偏好描述...")
    print(f"总用户数: {len(user_ids)}")

    for user_id in tqdm(user_ids, desc="生成用户偏好"):
        history = sequences[user_id]

        # 过滤历史长度不足的用户
        if len(history) < min_history_length:
            continue

        # 取最近的历史（最多20个）
        recent_history = history[-20:] if len(history) > 20 else history

        # 构造物品信息
        items_info = []
        for item_id in recent_history:
            item_id_str = str(item_id)
            if item_id_str in item_metadata:
                meta = item_metadata[item_id_str]
                items_info.append({
                    'item_id': item_id,
                    'title': meta.get('title', f'Item {item_id}'),
                    'genres': meta.get('genres', []),
                    'categories': meta.get('categories', [])
                })
            else:
                items_info.append({
                    'item_id': item_id,
                    'title': f'Item {item_id}'
                })

        # 生成偏好
        try:
            preference_text = generator.generate_user_preference(
                user_id=user_id,
                user_history=items_info,
                item_metadata=item_metadata
            )

            user_preferences[str(user_id)] = preference_text

        except Exception as e:
            print(f"\n用户 {user_id} 生成失败: {e}")
            # 使用默认偏好
            user_preferences[str(user_id)] = f"该用户有多样化的兴趣爱好"

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(user_preferences, f, ensure_ascii=False, indent=2)

    print(f"\n用户偏好已保存至: {output_path}")
    print(f"成功生成: {len(user_preferences)} 个用户")


def generate_item_descriptions(
    generator: LLMPreferenceGenerator,
    item_metadata: Dict,
    output_path: str,
    max_items: int = None,
    use_existing_descriptions: bool = True
):
    """
    生成物品描述

    Args:
        generator: LLM 生成器
        item_metadata: 物品元数据
        output_path: 输出文件路径
        max_items: 最多生成多少物品（None表示全部）
        use_existing_descriptions: 是否直接使用已有的描述（来自 text.xls）
    """
    item_descriptions = {}

    item_ids = list(item_metadata.keys())
    if max_items is not None:
        item_ids = item_ids[:max_items]

    print(f"\n开始处理物品描述...")
    print(f"总物品数: {len(item_ids)}")

    if use_existing_descriptions:
        print("使用 Multimodal_Datasets 中的物品描述 (text.xls: movie-id + review)")

        existing_count = 0
        generated_count = 0

        for item_id in tqdm(item_ids, desc="提取物品描述"):
            meta = item_metadata[item_id]

            # 优先使用已有的描述
            if 'description' in meta and meta['description']:
                item_descriptions[item_id] = meta['description']
                existing_count += 1
            else:
                # 如果没有描述，构造一个基础描述
                title = meta.get('title', f'Item {item_id}')
                genres = meta.get('genres', [])
                categories = meta.get('categories', [])

                desc_parts = [title]
                if genres:
                    desc_parts.append(f"类型: {', '.join(genres)}")
                if categories:
                    desc_parts.append(f"分类: {', '.join(categories)}")

                item_descriptions[item_id] = ". ".join(desc_parts)
                generated_count += 1

        print(f"  使用已有描述: {existing_count}")
        print(f"  生成默认描述: {generated_count}")

    else:
        print("使用 LLM 生成新的物品描述")

        for item_id in tqdm(item_ids, desc="生成物品描述"):
            meta = item_metadata[item_id]

            try:
                description = generator.generate_item_description(
                    item_id=item_id,
                    item_metadata=meta
                )

                item_descriptions[item_id] = description

            except Exception as e:
                print(f"\n物品 {item_id} 生成失败: {e}")
                # 使用元数据作为默认描述
                title = meta.get('title', f'Item {item_id}')
                genres = meta.get('genres', [])
                categories = meta.get('categories', [])

                desc_parts = [title]
                if genres:
                    desc_parts.append(f"类型: {', '.join(genres)}")
                if categories:
                    desc_parts.append(f"分类: {', '.join(categories)}")

                item_descriptions[item_id] = ". ".join(desc_parts)

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(item_descriptions, f, ensure_ascii=False, indent=2)

    print(f"\n物品描述已保存至: {output_path}")
    print(f"成功生成: {len(item_descriptions)} 个物品")


def main():
    parser = argparse.ArgumentParser(description='使用 LLM 生成用户偏好和物品描述')

    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（可选，当前未使用）')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')

    parser.add_argument('--llm_backend', type=str, default='openai',
                        choices=['openai', 'anthropic', 'local'],
                        help='LLM 后端')
    parser.add_argument('--model_name', type=str, default='qwen-flash',
                        help='LLM 模型名称')
    parser.add_argument('--api_key', type=str, default=None,
                        help='API 密钥（如果为 None，从环境变量读取）')
    parser.add_argument('--base_url', type=str,
                        default='https://dashscope.aliyuncs.com/compatible-mode/v1',
                        help='API 基础 URL（DashScope 或其他兼容 OpenAI 的服务）')

    parser.add_argument('--max_users', type=int, default=None,
                        help='最多生成多少用户（None表示全部）')
    parser.add_argument('--max_items', type=int, default=None,
                        help='最多生成多少物品（None表示全部）')

    parser.add_argument('--min_history_length', type=int, default=3,
                        help='用户最少历史长度')

    # 生成选项：可以选择只生成用户偏好、只生成物品描述，或两者都生成
    generation_group = parser.add_mutually_exclusive_group()
    generation_group.add_argument('--only_users', action='store_true',
                        help='只生成用户偏好（不生成物品描述）')
    generation_group.add_argument('--only_items', action='store_true',
                        help='只生成物品描述（不生成用户偏好）')

    # 保留原有的 skip 参数以兼容旧脚本
    parser.add_argument('--skip_users', action='store_true',
                        help='跳过用户偏好生成')
    parser.add_argument('--skip_items', action='store_true',
                        help='跳过物品描述生成')

    parser.add_argument('--use_existing_descriptions', action='store_true', default=True,
                        help='使用 text.xls 中已有的物品描述（默认开启）')
    parser.add_argument('--regenerate_descriptions', action='store_true',
                        help='使用 LLM 重新生成物品描述（忽略 text.xls）')

    parser.add_argument('--cache_dir', type=str, default='cache/llm_cache',
                        help='LLM 缓存目录')

    args = parser.parse_args()

    # 处理生成选项
    if args.only_users:
        args.skip_items = True
        print("\n模式: 只生成用户偏好")
    elif args.only_items:
        args.skip_users = True
        print("\n模式: 只生成物品描述")
    else:
        print("\n模式: 生成用户偏好和物品描述")

    # 加载配置（如果提供）
    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"\n已加载配置文件: {args.config}")

    print("=" * 60)
    print("LLM 数据生成")
    print("=" * 60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"LLM 后端: {args.llm_backend}")
    print(f"模型: {args.model_name}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    data = load_dataset(args.data_dir, split='train')

    sequences = data['sequences']
    item_metadata = data['item_metadata']

    print(f"  用户数: {len(sequences)}")
    print(f"  物品元数据数: {len(item_metadata)}")

    # 创建 LLM 生成器
    print(f"\n使用 {args.llm_backend} 后端")
    print(f"模型: {args.model_name}")
    print(f"API 端点: {args.base_url}")

    # 检查 API 密钥（如果没有通过参数提供，会从环境变量读取）
    import os
    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n错误: 请提供 API 密钥")
        print("  方法 1: 使用 --api_key 参数")
        print("  方法 2: 设置环境变量:")
        print("    export DASHSCOPE_API_KEY=your-key")
        print("    或")
        print("    export OPENAI_API_KEY=your-key")
        return

    generator = LLMPreferenceGenerator(
        llm_backend=args.llm_backend,
        model_name=args.model_name,
        api_key=args.api_key,  # 可以是 None，生成器会自动从环境变量读取
        base_url=args.base_url,
        cache_dir=args.cache_dir
    )

    # 生成用户偏好
    if not args.skip_users:
        user_pref_path = output_dir / 'user_preferences.json'
        generate_user_preferences(
            generator=generator,
            sequences=sequences,
            item_metadata=item_metadata,
            output_path=str(user_pref_path),
            max_users=args.max_users,
            min_history_length=args.min_history_length
        )
    else:
        print("\n跳过用户偏好生成")

    # 生成物品描述
    if not args.skip_items:
        item_desc_path = output_dir / 'item_descriptions.json'

        # 决定是否使用已有描述
        use_existing = not args.regenerate_descriptions

        generate_item_descriptions(
            generator=generator,
            item_metadata=item_metadata,
            output_path=str(item_desc_path),
            max_items=args.max_items,
            use_existing_descriptions=use_existing
        )
    else:
        print("\n跳过物品描述生成")

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    print("\n生成的文件:")
    if not args.skip_users:
        print(f"  ✓ user_preferences.json (用户偏好)")
    if not args.skip_items:
        print(f"  ✓ item_descriptions.json (物品描述)")

    if args.skip_users and args.skip_items:
        print("  (无文件生成 - 已跳过所有选项)")

    print("\n使用方法:")
    print("  - 生成两者: python generate_llm_data.py --data_dir ... --output_dir ...")
    print("  - 只生成用户偏好: 添加 --only_users")
    print("  - 只生成物品描述: 添加 --only_items")
    print("\n提示: --config 参数现在是可选的（未使用）")


if __name__ == "__main__":
    main()
