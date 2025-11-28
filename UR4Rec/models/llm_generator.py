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
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        cache_dir: str = "data/llm_cache"
    ):
        """
        Args:
            llm_backend: LLM 后端 ('openai', 'anthropic', 'local')
            model_name: 模型名称
            api_key: API 密钥
            cache_dir: 缓存目录
        """
        self.llm_backend = llm_backend
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 LLM 客户端
        self._init_llm(api_key)

        # 加载缓存
        self.cache = self._load_cache()

    def _init_llm(self, api_key: Optional[str]):
        """初始化 LLM 客户端"""
        if self.llm_backend == "openai":
            try:
                import openai
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

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """调用 LLM"""
        if self.llm_backend == "openai" and self.client:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
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

        prompt = f"""基于用户的历史交互物品，总结该用户的偏好特征。

用户历史交互的物品：
{items_str}

请用2-3句话总结该用户的偏好，包括：
1. 偏好的类型/风格
2. 关注的主要特征
3. 可能的兴趣方向

用户偏好总结："""

        # 调用 LLM
        preference_text = self._call_llm(prompt, max_tokens=200)

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

        prompt = f"""请为以下物品生成一个简洁的描述，用于推荐系统。

物品信息：
- 标题：{title}
- 类型：{genres}

请用1-2句话描述该物品的核心特征和适合的用户群体。

物品描述："""

        # 调用 LLM
        description = self._call_llm(prompt, max_tokens=150)

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


class MockLLMGenerator:
    """
    模拟 LLM 生成器（用于没有 API 密钥时的测试）
    """

    def __init__(self):
        pass

    def generate_user_preference(
        self,
        user_id: int,
        user_history: List[int],
        item_metadata: Dict[int, Dict]
    ) -> str:
        """生成模拟的用户偏好"""
        genres = set()
        for item_id in user_history[-10:]:
            if item_id in item_metadata:
                meta = item_metadata[item_id]
                if 'genres' in meta:
                    genres.update(meta['genres'].split('|'))

        if genres:
            genres_str = ', '.join(list(genres)[:3])
            return f"该用户偏好观看 {genres_str} 类型的内容，注重内容质量和情节发展。"
        else:
            return "该用户具有多样化的兴趣，喜欢探索不同类型的内容。"

    def generate_item_description(
        self,
        item_id: int,
        item_metadata: Dict
    ) -> str:
        """生成模拟的物品描述"""
        title = item_metadata.get('title', f'Item {item_id}')
        genres = item_metadata.get('genres', 'Unknown')

        return f"{title} 是一部 {genres} 类型的作品，适合喜欢该类型的观众。"

    def batch_generate_user_preferences(
        self,
        users_data: List[Dict],
        item_metadata: Dict[int, Dict],
        save_path: str
    ):
        """批量生成模拟偏好"""
        preferences = {}
        for user_data in tqdm(users_data, desc="生成用户偏好（模拟）"):
            user_id = user_data['user_id']
            user_history = user_data['user_history']
            pref = self.generate_user_preference(user_id, user_history, item_metadata)
            preferences[user_id] = pref

        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, ensure_ascii=False, indent=2)

        logger.info(f"用户偏好描述（模拟）已保存到 {save_path}")

    def batch_generate_item_descriptions(
        self,
        item_metadata: Dict[int, Dict],
        save_path: str
    ):
        """批量生成模拟描述"""
        descriptions = {}
        for item_id, meta in tqdm(item_metadata.items(), desc="生成物品描述（模拟）"):
            desc = self.generate_item_description(item_id, meta)
            descriptions[item_id] = desc

        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)

        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(descriptions, f, ensure_ascii=False, indent=2)

        logger.info(f"物品描述（模拟）已保存到 {save_path}")


if __name__ == "__main__":
    # 测试模拟生成器
    print("测试 LLM 生成器...")

    generator = MockLLMGenerator()

    # 测试用户偏好生成
    item_metadata = {
        101: {"title": "Toy Story (1995)", "genres": "Animation|Children's|Comedy"},
        205: {"title": "Jumanji (1995)", "genres": "Adventure|Children's|Fantasy"},
        303: {"title": "Heat (1995)", "genres": "Action|Crime|Thriller"}
    }

    user_pref = generator.generate_user_preference(
        user_id=1,
        user_history=[101, 205, 303],
        item_metadata=item_metadata
    )

    print(f"\n用户偏好: {user_pref}")

    # 测试物品描述生成
    item_desc = generator.generate_item_description(
        item_id=101,
        item_metadata=item_metadata[101]
    )

    print(f"\n物品描述: {item_desc}")
