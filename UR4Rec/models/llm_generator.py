"""
离线 LLM 生成模块

用于离线生成用户偏好描述和物品描述，不参与在线推理。
"""
import torch
import json
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMPreferenceGenerator:
    """
    使用 LLM 离线生成用户偏好描述和物品描述
    """

    def __init__(
        self,
        llm_backend: str = "openai",
        model_name: str = "qwen-flash",
        api_key: Optional[str] = "sk-de8b84b8aca743cfa6fb42ec2776280b",
        base_url: Optional[str] = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        cache_dir: str = "data/llm_cache",
        enable_thinking: bool = False
    ):
        """
        Args:
            llm_backend: LLM 后端 ('openai', 'anthropic', 'local')
            model_name: 模型名称
            api_key: API 密钥
            base_url: 自定义API地址（用于OpenAI兼容API，如vLLM, LocalAI等）
            cache_dir: 缓存目录
            enable_thinking: 是否启用深度思考模式（仅 DashScope qwen-flash 等支持）
        """
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 LLM 客户端
        self._init_llm(api_key, base_url)

        # 加载缓存
        self.cache = self._load_cache()

    def _init_llm(self, api_key: Optional[str], base_url: Optional[str] = None):
        """初始化 LLM 客户端

        Args:
            api_key: API密钥
            base_url: 自定义API地址（用于OpenAI兼容API）
        """
        if self.llm_backend == "openai":
            try:
                import openai
                if base_url:
                    self.client = openai.OpenAI(
                        api_key=api_key or "dummy-key",
                        base_url=base_url
                    )
                else:
                    self.client = openai.OpenAI(api_key=api_key)
            except ImportError:
                logger.error("请安装 openai: pip install openai")
                self.client = None

        elif self.llm_backend == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                logger.error("请安装 anthropic: pip install anthropic")
                self.client = None

        elif self.llm_backend == "local":
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            except ImportError:
                logger.error("请安装 transformers: pip install transformers")
                self.client = None

    def _load_cache(self) -> Dict:
        """加载缓存"""
        cache_file = self.cache_dir / "llm_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"user_preferences": {}, "item_descriptions": {}}

    def _save_cache(self):
        """保存缓存"""
        cache_file = self.cache_dir / "llm_cache.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _call_llm(
        self,
        prompt: str,
        max_tokens: int = 500,
        extra_body: Optional[Dict] = None
    ) -> str:
        """调用 LLM

        Args:
            prompt: 提示词
            max_tokens: 最大token数
            extra_body: 额外参数（如阿里云的{"enable_thinking": True}）
        """
        if self.llm_backend == "openai" and self.client:
            request_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }

            # 添加思考模式参数
            if self.enable_thinking or extra_body:
                merged_extra = {}
                if self.enable_thinking:
                    merged_extra["enable_thinking"] = True
                if extra_body:
                    merged_extra.update(extra_body)
                request_params["extra_body"] = merged_extra

            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content

        elif self.llm_backend == "anthropic" and self.client:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.llm_backend == "local":
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()

        else:
            return "LLM not available"

    def generate_user_preference(
        self,
        user_id: int,
        user_history: List[int],
        item_metadata: Dict[int, Dict]
    ) -> str:
        """
        生成用户偏好描述

        Args:
            user_id: 用户ID
            user_history: 用户历史交互物品ID列表
            item_metadata: 物品元数据字典

        Returns:
            用户偏好的文本描述
        """
        cache_key = f"user_{user_id}"

        # 检查缓存
        if cache_key in self.cache["user_preferences"]:
            return self.cache["user_preferences"][cache_key]

        # 构建提示
        items_info = []
        for item_id in user_history[-20:]:  # 最近20个物品
            if item_id in item_metadata:
                meta = item_metadata[item_id]
                items_info.append(f"- {meta.get('title', f'Item {item_id}')}")

        items_str = "\n".join(items_info)

        prompt = f"""Based on the user's historical interactions, summarize the user's preference characteristics.

User's historical interactions:
{items_str}

Please summarize the user's preferences in 2-3 sentences, including:
1. Preferred types/genres
2. Main features of interest
3. Potential interest directions

User preference summary:"""

        # 调用 LLM（带错误处理）
        try:
            preference_text = self._call_llm(prompt, max_tokens=200)
        except Exception as e:
            # 检查是否是内容审核错误
            error_str = str(e).lower()
            if ("400" in error_str or "bad request" in error_str) and \
               ("data_inspection_failed" in error_str or "inappropriate content" in error_str):
                logger.warning(f"用户 {user_id} 触发内容审核，使用兜底文本")
                preference_text = "User has no obvious preferences."
            else:
                # 其他错误继续抛出
                raise

        # 缓存结果
        self.cache["user_preferences"][cache_key] = preference_text
        self._save_cache()

        return preference_text

    def generate_item_description(
        self,
        item_id: int,
        item_metadata: Dict
    ) -> str:
        """
        生成物品描述

        Args:
            item_id: 物品ID
            item_metadata: 物品元数据

        Returns:
            物品的文本描述
        """
        cache_key = f"item_{item_id}"

        # 检查缓存
        if cache_key in self.cache["item_descriptions"]:
            return self.cache["item_descriptions"][cache_key]

        # 构建提示
        title = item_metadata.get('title', f'Item {item_id}')
        genres = item_metadata.get('genres', 'Unknown')

        prompt = f"""Generate a concise description for the following item for a recommendation system.

Item information:
- Title: {title}
- Genres: {genres}

Please describe the item's core features and target audience in 1-2 sentences.

Item description:"""

        # 调用 LLM（带错误处理）
        try:
            description = self._call_llm(prompt, max_tokens=150)
        except Exception as e:
            # 检查是否是内容审核错误
            error_str = str(e).lower()
            if ("400" in error_str or "bad request" in error_str) and \
               ("data_inspection_failed" in error_str or "inappropriate content" in error_str):
                logger.warning(f"物品 {item_id} ({title}) 触发内容审核，使用兜底文本")
                description = "No description available."
            else:
                # 其他错误继续抛出
                raise

        # 缓存结果
        self.cache["item_descriptions"][cache_key] = description
        self._save_cache()

        return description

    def batch_generate_user_preferences(
        self,
        users_data: List[Dict],
        item_metadata: Dict[int, Dict],
        save_path: str
    ):
        """
        批量生成用户偏好描述

        Args:
            users_data: 用户数据列表 [{"user_id": ..., "user_history": [...]}, ...]
            item_metadata: 物品元数据
            save_path: 保存路径
        """
        logger.info(f"开始生成 {len(users_data)} 个用户的偏好描述...")

        preferences = {}

        for user_data in tqdm(users_data, desc="生成用户偏好"):
            user_id = user_data['user_id']
            user_history = user_data['user_history']

            pref_text = self.generate_user_preference(
                user_id, user_history, item_metadata
            )

            preferences[user_id] = pref_text

        # 保存
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, ensure_ascii=False, indent=2)

        logger.info(f"用户偏好描述已保存到 {save_path}")

    def batch_generate_item_descriptions(
        self,
        item_metadata: Dict[int, Dict],
        save_path: str
    ):
        """
        批量生成物品描述

        Args:
            item_metadata: 物品元数据字典
            save_path: 保存路径
        """
        logger.info(f"开始生成 {len(item_metadata)} 个物品的描述...")

        descriptions = {}

        for item_id, meta in tqdm(item_metadata.items(), desc="生成物品描述"):
            desc = self.generate_item_description(item_id, meta)
            descriptions[item_id] = desc

        # 保存
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)

        logger.info(f"物品描述已保存到 {save_path}")


if __name__ == "__main__":
    """
    直接运行此脚本生成 ML-100K 数据集的用户偏好和物品描述

    使用方法:
        1. 设置 API 密钥:
           export DASHSCOPE_API_KEY="your-api-key"

        2. 运行脚本:
           python UR4Rec/models/llm_generator.py

        3. (可选) 自定义参数:
           python UR4Rec/models/llm_generator.py --num_users 100 --num_items 500
    """
    import sys
    import os
    import argparse
    from pathlib import Path

    # 添加项目路径
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from UR4Rec.data.dataset_loader import load_ml_100k

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成 ML-100K 数据集的 LLM 偏好')
    parser.add_argument('--data_dir', type=str,
                       default='UR4Rec/data/Multimodal_Datasets',
                       help='数据集目录')
    parser.add_argument('--output_dir', type=str,
                       default='data/llm_generated',
                       help='输出目录')
    parser.add_argument('--num_users', type=int, default=None,
                       help='生成的用户数量（None=全部）')
    parser.add_argument('--num_items', type=int, default=None,
                       help='生成的物品数量（None=全部）')
    parser.add_argument('--model_name', type=str, default='qwen-flash',
                       help='LLM 模型名称')
    parser.add_argument('--enable_thinking', action='store_true',
                       help='启用深度思考模式（会消耗更多 tokens）')
    parser.add_argument('--skip_users', action='store_true',
                       help='跳过用户偏好生成')
    parser.add_argument('--skip_items', action='store_true',
                       help='跳过物品描述生成')

    args = parser.parse_args()

    print("=" * 60)
    print("ML-100K 数据集 LLM 偏好生成")
    print("=" * 60)

    # 检查 API 密钥
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("\n❌ 错误: 未设置 API 密钥")
        print("\n请设置环境变量:")
        print("  export DASHSCOPE_API_KEY='your-api-key'")
        print("  或")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\n获取阿里云 DashScope API 密钥:")
        print("  https://dashscope.aliyuncs.com/")
        sys.exit(1)

    # 确定使用哪个后端
    if os.getenv("DASHSCOPE_API_KEY"):
        llm_backend = "openai"
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        print(f"\n使用 DashScope API (模型: {args.model_name})")
    else:
        llm_backend = "openai"
        base_url = None
        print(f"\n使用 OpenAI API (模型: {args.model_name})")

    # 加载数据
    print(f"\n[1/4] 加载数据集...")
    try:
        item_metadata, user_sequences, users = load_ml_100k(
            data_dir=args.data_dir,
            min_rating=4.0,
            min_seq_len=5
        )
        print(f"✓ 数据加载完成")
        print(f"  - 物品数: {len(item_metadata)}")
        print(f"  - 用户序列数: {len(user_sequences)}")
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print(f"\n请确保数据集目录存在: {args.data_dir}/M_ML-100K/")
        sys.exit(1)

    # 创建 LLM 生成器
    print(f"\n[2/4] 创建 LLM 生成器...")
    generator = LLMPreferenceGenerator(
        llm_backend=llm_backend,
        model_name=args.model_name,
        api_key=api_key,
        base_url=base_url,
        cache_dir=f"{args.output_dir}/llm_cache",
        enable_thinking=args.enable_thinking
    )

    if args.enable_thinking:
        print("  ⚠️  已启用深度思考模式，会消耗更多 tokens")

    print(f"✓ 生成器创建完成")

    # 准备输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成用户偏好
    if not args.skip_users:
        print(f"\n[3/4] 生成用户偏好...")

        # 选择用户子集（如果指定）
        users_to_generate = list(user_sequences.keys())
        if args.num_users:
            users_to_generate = users_to_generate[:args.num_users]
            print(f"  生成前 {args.num_users} 个用户的偏好")

        # 准备用户数据
        users_data = [
            {
                'user_id': user_id,
                'user_history': user_sequences[user_id]
            }
            for user_id in users_to_generate
        ]

        # 批量生成
        user_pref_path = output_dir / "user_preferences.json"
        generator.batch_generate_user_preferences(
            users_data=users_data,
            item_metadata=item_metadata,
            save_path=str(user_pref_path)
        )

        print(f"✓ 用户偏好已保存到: {user_pref_path}")
    else:
        print(f"\n[3/4] 跳过用户偏好生成 (--skip_users)")

    # 生成物品描述
    if not args.skip_items:
        print(f"\n[4/4] 生成物品描述...")

        # 选择物品子集（如果指定）
        items_to_generate = dict(item_metadata)
        if args.num_items:
            items_to_generate = {
                k: v for k, v in list(item_metadata.items())[:args.num_items]
            }
            print(f"  生成前 {args.num_items} 个物品的描述")

        # 批量生成
        item_desc_path = output_dir / "item_descriptions.json"
        generator.batch_generate_item_descriptions(
            item_metadata=items_to_generate,
            save_path=str(item_desc_path)
        )

        print(f"✓ 物品描述已保存到: {item_desc_path}")
    else:
        print(f"\n[4/4] 跳过物品描述生成 (--skip_items)")

    # 显示缓存统计
    print(f"\n缓存位置: {generator.cache_dir}/llm_cache.json")
    print(f"  - 用户偏好缓存: {len(generator.cache['user_preferences'])} 条")
    print(f"  - 物品描述缓存: {len(generator.cache['item_descriptions'])} 条")

    print("\n" + "=" * 60)
    print("生成完成！")
    print("=" * 60)

    print("\n生成的文件:")
    if not args.skip_users:
        print(f"  ✓ 用户偏好: {output_dir}/user_preferences.json")
    if not args.skip_items:
        print(f"  ✓ 物品描述: {output_dir}/item_descriptions.json")
    print(f"  ✓ 缓存文件: {generator.cache_dir}/llm_cache.json")

    print("\n下一步:")
    print("  1. 检查生成的文件内容")
    print("  2. 使用生成的偏好进行训练:")
    print("     python UR4Rec/scripts/train.py --use_llm_features")

    print("\n💡 提示:")
    print("  - 如果 API 请求失败，脚本会从缓存中恢复")
    print("  - 重复运行脚本不会重复调用 API（使用缓存）")
    print("  - 使用 --num_users 和 --num_items 可以先生成小批量测试")
