#!/usr/bin/env python3
"""
测试 qwen-flash API 连接

使用方法：
  export DASHSCOPE_API_KEY="your-api-key"
  python UR4Rec/scripts/test_llm_connection.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
from openai import OpenAI


def test_dashscope_connection():
    """测试 DashScope API 连接"""
    print("=" * 60)
    print("测试 DashScope qwen-flash 连接")
    print("=" * 60)

    # 检查 API key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("\n❌ 错误: 未设置 DASHSCOPE_API_KEY")
        print("\n请设置环境变量:")
        print("  export DASHSCOPE_API_KEY='your-api-key'")
        print("\n获取 API 密钥: https://dashscope.aliyuncs.com/")
        return False

    print(f"\n✓ API Key: {api_key[:10]}...{api_key[-4:]}")

    # 创建客户端
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        print("✓ 客户端创建成功")
    except Exception as e:
        print(f"❌ 客户端创建失败: {e}")
        return False

    # 测试简单调用
    print("\n正在测试 API 调用...")
    try:
        response = client.chat.completions.create(
            model="qwen-flash",
            messages=[{"role": "user", "content": "Hello, please respond with 'OK'"}],
            max_tokens=50
        )

        content = response.choices[0].message.content
        print(f"✓ API 调用成功")
        print(f"  响应: {content}")

    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        return False

    # 测试电影描述生成
    print("\n正在测试电影描述生成...")
    try:
        prompt = """Analyze the movie 'The Shawshank Redemption'. Do NOT summarize the plot. Instead, construct a dense semantic profile focusing on:
1. Visual Aesthetics (e.g., color palette, lighting, cinematography style).
2. Core Themes (e.g., existentialism, betrayal, coming-of-age).
3. Emotional Tone & Atmosphere (e.g., melancholic, gritty, whimsical).
4. Target Audience Appeal (Why fans love it).
Output a concise, high-density paragraph using distinctive adjectives.

Movie Title: The Shawshank Redemption
Genres: Drama, Crime

Description:"""

        response = client.chat.completions.create(
            model="qwen-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )

        description = response.choices[0].message.content
        print(f"✓ 电影描述生成成功")
        print(f"\n生成的描述:")
        print("-" * 60)
        print(description)
        print("-" * 60)

    except Exception as e:
        print(f"❌ 电影描述生成失败: {e}")
        return False

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
    print("\n你现在可以运行:")
    print("  python UR4Rec/scripts/generate_llm_data.py \\")
    print("    --data_dir UR4Rec/data/Multimodal_Datasets \\")
    print("    --output_dir data/llm_generated \\")
    print("    --max_users 10 --max_items 20")

    return True


if __name__ == "__main__":
    success = test_dashscope_connection()
    sys.exit(0 if success else 1)
