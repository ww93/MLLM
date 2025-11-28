"""
下载数据集图片

支持：
1. MovieLens: 从 TMDB API 获取电影海报
2. Amazon Beauty: 从 Amazon 商品图片 URL 下载
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import json
import requests
from tqdm import tqdm
import time
from PIL import Image
from io import BytesIO
import os


def download_movielens_images(
    item_metadata_path: str,
    output_dir: str,
    tmdb_api_key: str = None,
    max_items: int = None
):
    """
    下载 MovieLens 数据集的电影海报

    Args:
        item_metadata_path: 物品元数据文件路径
        output_dir: 输出目录
        tmdb_api_key: TMDB API 密钥（可从 https://www.themoviedb.org/settings/api 获取）
        max_items: 最多下载多少张图片
    """
    print("\n" + "=" * 60)
    print("下载 MovieLens 电影海报")
    print("=" * 60)

    # 加载元数据
    with open(item_metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"加载了 {len(metadata)} 个物品")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 如果没有 API key，使用占位图片
    if not tmdb_api_key:
        print("警告: 未提供 TMDB API 密钥")
        print("将创建占位图片（灰色图片）")
        print("获取 API 密钥: https://www.themoviedb.org/settings/api")
        use_placeholder = True
    else:
        use_placeholder = False

    # TMDB API 配置
    TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
    TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

    # 下载计数
    success_count = 0
    fail_count = 0

    # 记录下载结果
    download_log = {}

    items = list(metadata.items())
    if max_items:
        items = items[:max_items]

    for item_id, item_info in tqdm(items, desc="下载图片"):
        image_path = output_path / f"{item_id}.jpg"

        # 如果已存在，跳过
        if image_path.exists():
            success_count += 1
            download_log[item_id] = {"status": "exists", "path": str(image_path)}
            continue

        if use_placeholder:
            # 创建占位图片（纯灰色）
            img = Image.new('RGB', (500, 750), color=(128, 128, 128))
            img.save(image_path, 'JPEG')
            success_count += 1
            download_log[item_id] = {"status": "placeholder", "path": str(image_path)}
            continue

        # 获取电影标题
        title = item_info.get('title', '')
        if not title:
            fail_count += 1
            download_log[item_id] = {"status": "no_title"}
            continue

        try:
            # 搜索电影
            params = {
                'api_key': tmdb_api_key,
                'query': title,
                'language': 'en-US'
            }
            response = requests.get(TMDB_SEARCH_URL, params=params, timeout=10)
            response.raise_for_status()
            results = response.json().get('results', [])

            if not results:
                # 未找到，使用占位图片
                img = Image.new('RGB', (500, 750), color=(128, 128, 128))
                img.save(image_path, 'JPEG')
                download_log[item_id] = {"status": "not_found", "path": str(image_path)}
                fail_count += 1
                continue

            # 获取第一个结果的海报
            poster_path = results[0].get('poster_path')
            if not poster_path:
                img = Image.new('RGB', (500, 750), color=(128, 128, 128))
                img.save(image_path, 'JPEG')
                download_log[item_id] = {"status": "no_poster", "path": str(image_path)}
                fail_count += 1
                continue

            # 下载图片
            image_url = TMDB_IMAGE_BASE + poster_path
            img_response = requests.get(image_url, timeout=10)
            img_response.raise_for_status()

            # 保存图片
            img = Image.open(BytesIO(img_response.content))
            img = img.convert('RGB')
            img.save(image_path, 'JPEG')

            success_count += 1
            download_log[item_id] = {
                "status": "success",
                "path": str(image_path),
                "url": image_url
            }

            # 避免请求过快
            time.sleep(0.1)

        except Exception as e:
            print(f"\n下载失败 (item_id={item_id}): {e}")
            # 创建占位图片
            img = Image.new('RGB', (500, 750), color=(128, 128, 128))
            img.save(image_path, 'JPEG')
            download_log[item_id] = {"status": "error", "error": str(e), "path": str(image_path)}
            fail_count += 1

    # 保存下载日志
    log_path = output_path / 'download_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(download_log, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"下载完成！")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"输出目录: {output_dir}")
    print(f"下载日志: {log_path}")
    print("=" * 60)


def download_amazon_images(
    item_metadata_path: str,
    output_dir: str,
    max_items: int = None
):
    """
    下载 Amazon 数据集的商品图片

    Args:
        item_metadata_path: 物品元数据文件路径
        output_dir: 输出目录
        max_items: 最多下载多少张图片
    """
    print("\n" + "=" * 60)
    print("下载 Amazon 商品图片")
    print("=" * 60)

    # 加载元数据
    with open(item_metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print(f"加载了 {len(metadata)} 个物品")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 下载计数
    success_count = 0
    fail_count = 0

    # 记录下载结果
    download_log = {}

    items = list(metadata.items())
    if max_items:
        items = items[:max_items]

    for item_id, item_info in tqdm(items, desc="下载图片"):
        image_path = output_path / f"{item_id}.jpg"

        # 如果已存在，跳过
        if image_path.exists():
            success_count += 1
            download_log[item_id] = {"status": "exists", "path": str(image_path)}
            continue

        # 获取图片 URL
        image_url = item_info.get('image', None)
        if not image_url:
            # 创建占位图片
            img = Image.new('RGB', (500, 500), color=(128, 128, 128))
            img.save(image_path, 'JPEG')
            download_log[item_id] = {"status": "no_url", "path": str(image_path)}
            fail_count += 1
            continue

        try:
            # 下载图片
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()

            # 保存图片
            img = Image.open(BytesIO(response.content))
            img = img.convert('RGB')

            # 调整大小（统一尺寸）
            img = img.resize((500, 500), Image.Resampling.LANCZOS)
            img.save(image_path, 'JPEG')

            success_count += 1
            download_log[item_id] = {
                "status": "success",
                "path": str(image_path),
                "url": image_url
            }

            # 避免请求过快
            time.sleep(0.05)

        except Exception as e:
            print(f"\n下载失败 (item_id={item_id}): {e}")
            # 创建占位图片
            img = Image.new('RGB', (500, 500), color=(128, 128, 128))
            img.save(image_path, 'JPEG')
            download_log[item_id] = {"status": "error", "error": str(e), "path": str(image_path)}
            fail_count += 1

    # 保存下载日志
    log_path = output_path / 'download_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(download_log, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"下载完成！")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"输出目录: {output_dir}")
    print(f"下载日志: {log_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='下载数据集图片')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=['movielens', 'amazon'],
                        help='数据集类型')
    parser.add_argument('--item_metadata', type=str, required=True,
                        help='物品元数据文件路径')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')

    parser.add_argument('--tmdb_api_key', type=str, default=None,
                        help='TMDB API 密钥（MovieLens 需要）')

    parser.add_argument('--max_items', type=int, default=None,
                        help='最多下载多少张图片（None表示全部）')

    args = parser.parse_args()

    if args.dataset == 'movielens':
        download_movielens_images(
            item_metadata_path=args.item_metadata,
            output_dir=args.output_dir,
            tmdb_api_key=args.tmdb_api_key,
            max_items=args.max_items
        )
    elif args.dataset == 'amazon':
        download_amazon_images(
            item_metadata_path=args.item_metadata,
            output_dir=args.output_dir,
            max_items=args.max_items
        )


if __name__ == "__main__":
    main()
