"""
MoE-enhanced Text Preference Retriever

Combines:
1. Text-based preference retrieval (from LLM-generated descriptions)
2. MoE mechanism for better fusion
3. User memory for dynamic preference tracking
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from .retriever_moe_memory import RetrieverMoEMemory, MemoryConfig, UpdateTrigger


class TextEncoder(nn.Module):
    """
    文本编码器：将文本描述转换为向量表示

    使用预训练的句子编码器（如 Sentence-BERT）
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        output_dim: int = 256,
        dropout: float = 0.1,
        freeze_encoder: bool = True
    ):
        """
        Args:
            model_name: 预训练模型名称
            embedding_dim: 预训练模型的嵌入维度
            output_dim: 输出维度
            dropout: Dropout 率
            freeze_encoder: 是否冻结预训练编码器
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_encoder = SentenceTransformer(model_name)
            if freeze_encoder:
                for param in self.sentence_encoder.parameters():
                    param.requires_grad = False
        except ImportError:
            print("警告: sentence-transformers 未安装")
            print("请安装: pip install sentence-transformers")
            self.sentence_encoder = None

        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本列表

        Args:
            texts: 文本列表

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        if self.sentence_encoder:
            with torch.no_grad():
                embeddings = self.sentence_encoder.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False
                )
            return embeddings
        else:
            # Fallback: 随机嵌入（仅用于测试）
            batch_size = len(texts)
            device = next(self.parameters()).device
            return torch.randn(batch_size, self.embedding_dim, device=device)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            text_embeddings: [batch_size, embedding_dim]

        Returns:
            output: [batch_size, output_dim]
        """
        return self.projection(text_embeddings)


class TextPreferenceRetrieverMoE(nn.Module):
    """
    MoE-enhanced Text Preference Retriever

    Features:
    1. Encodes LLM-generated user preferences and item descriptions
    2. Uses MoE mechanism to fuse multiple information sources
    3. Maintains user memory for dynamic preference tracking
    4. Supports both training and inference modes
    """

    def __init__(
        self,
        num_items: int,
        text_model_name: str = "all-MiniLM-L6-v2",
        text_embedding_dim: int = 384,
        output_dim: int = 256,
        # MoE settings
        num_heads: int = 8,
        dropout: float = 0.1,
        num_proxies: int = 4,
        # Memory settings
        memory_config: Optional[MemoryConfig] = None,
        temperature: float = 0.07,
        # CLIP settings
        clip_features_path: Optional[str] = None,
        use_clip: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            num_items: 物品总数
            text_model_name: 预训练文本模型名称
            text_embedding_dim: 文本嵌入维度
            output_dim: 输出维度
            num_heads: MoE 注意力头数
            dropout: Dropout 率
            num_proxies: 代理 token 数量
            memory_config: 记忆机制配置
            temperature: 对比学习温度
            clip_features_path: CLIP特征文件路径
            use_clip: 是否使用CLIP特征替代可训练embedding
            device: 设备
        """
        super().__init__()

        self.num_items = num_items
        self.output_dim = output_dim
        self.temperature = temperature
        self.device = device
        self.use_clip = use_clip

        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=text_model_name,
            embedding_dim=text_embedding_dim,
            output_dim=output_dim,
            dropout=dropout
        )

        # MoE retriever with memory
        self.moe_retriever = RetrieverMoEMemory(
            embedding_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            num_proxies=num_proxies,
            memory_config=memory_config or MemoryConfig(memory_dim=output_dim)
        )

        # Item embeddings (可训练) - 作为fallback或补充特征
        self.item_embeddings = nn.Embedding(
            num_items + 1,  # +1 for padding
            output_dim,
            padding_idx=0
        )
        nn.init.xavier_uniform_(self.item_embeddings.weight[1:])

        # CLIP图片特征 (预提取的)
        self.clip_features: Optional[torch.Tensor] = None
        if use_clip and clip_features_path:
            self.load_clip_features(clip_features_path)

        # LLM 生成的文本存储
        self.user_preference_texts: Dict[int, str] = {}
        self.item_description_texts: Dict[int, str] = {}

        # 预编码的文本向量（用于加速推理）
        self.user_preference_embeddings: Optional[torch.Tensor] = None
        self.item_description_embeddings: Optional[torch.Tensor] = None

        self.to(device)

    def load_clip_features(self, clip_features_path: str):
        """
        加载预提取的CLIP图片特征

        Args:
            clip_features_path: CLIP特征文件路径 (.pt文件)
        """
        print(f"加载 CLIP 特征从: {clip_features_path}")
        clip_features = torch.load(clip_features_path, map_location=self.device)

        # 确保特征维度正确
        if clip_features.size(1) != self.output_dim:
            print(f"警告: CLIP特征维度 ({clip_features.size(1)}) 与output_dim ({self.output_dim}) 不匹配")
            print("将使用线性投影调整维度")
            projection = nn.Linear(clip_features.size(1), self.output_dim).to(self.device)
            with torch.no_grad():
                clip_features = projection(clip_features)

        self.clip_features = clip_features
        print(f"✓ 已加载 CLIP 特征: {clip_features.shape}")
        print(f"  - 物品数量: {clip_features.size(0) - 1}")
        print(f"  - 特征维度: {clip_features.size(1)}")

    def load_llm_generated_data(
        self,
        user_preferences_path: str,
        item_descriptions_path: str
    ):
        """
        加载 LLM 生成的用户偏好和物品描述

        Args:
            user_preferences_path: 用户偏好 JSON 文件路径
            item_descriptions_path: 物品描述 JSON 文件路径
        """
        # 加载用户偏好
        with open(user_preferences_path, 'r', encoding='utf-8') as f:
            user_prefs = json.load(f)
            self.user_preference_texts = {int(k): v for k, v in user_prefs.items()}

        # 加载物品描述
        with open(item_descriptions_path, 'r', encoding='utf-8') as f:
            item_descs = json.load(f)
            self.item_description_texts = {int(k): v for k, v in item_descs.items()}

        print(f"✓ 加载 LLM 数据: {len(self.user_preference_texts)} 个用户偏好, "
              f"{len(self.item_description_texts)} 个物品描述")

        # 预编码所有物品描述（加速推理）
        self._precompute_item_embeddings()

    def _precompute_item_embeddings(self):
        """预编码所有物品的文本描述"""
        print("预编码物品描述...")

        # 创建物品ID到描述的映射
        item_texts = []
        item_ids = []

        for item_id in range(1, self.num_items + 1):
            if item_id in self.item_description_texts:
                item_texts.append(self.item_description_texts[item_id])
                item_ids.append(item_id)
            else:
                item_texts.append("No description available.")
                item_ids.append(item_id)

        # 批量编码
        with torch.no_grad():
            text_embeds = self.text_encoder.encode_text(item_texts)
            text_embeds = text_embeds.to(self.device)
            item_vectors = self.text_encoder(text_embeds)

        # 存储 [num_items, output_dim]
        self.item_description_embeddings = torch.zeros(
            self.num_items + 1, self.output_dim, device=self.device
        )
        self.item_description_embeddings[item_ids] = item_vectors

        print(f"✓ 预编码完成: {len(item_ids)} 个物品")

    def encode_user_preferences(
        self,
        user_ids: List[int]
    ) -> torch.Tensor:
        """
        编码用户偏好文本

        Args:
            user_ids: 用户ID列表

        Returns:
            preference_vectors: [batch_size, output_dim]
        """
        # 获取用户偏好文本
        user_texts = []
        for uid in user_ids:
            if uid in self.user_preference_texts:
                user_texts.append(self.user_preference_texts[uid])
            else:
                user_texts.append("User has no obvious preferences.")

        # 编码
        with torch.no_grad():
            text_embeds = self.text_encoder.encode_text(user_texts)
            text_embeds = text_embeds.to(self.device)

        # Clone to allow gradient flow to downstream layers
        text_embeds = text_embeds.clone().detach().requires_grad_(self.training)

        preference_vectors = self.text_encoder(text_embeds)

        # L2 归一化
        preference_vectors = F.normalize(preference_vectors, p=2, dim=-1)

        return preference_vectors

    def encode_item_descriptions(
        self,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        编码物品描述

        Args:
            item_ids: [batch_size] 或 [batch_size, num_items]

        Returns:
            item_vectors: [batch_size, output_dim] 或 [batch_size, num_items, output_dim]
        """
        if self.item_description_embeddings is not None:
            # 使用预编码的嵌入
            return self.item_description_embeddings[item_ids]
        else:
            # 实时编码（训练时或未预编码时）
            original_shape = item_ids.shape
            item_ids_flat = item_ids.flatten().cpu().tolist()

            item_texts = []
            for iid in item_ids_flat:
                if iid in self.item_description_texts:
                    item_texts.append(self.item_description_texts[iid])
                else:
                    item_texts.append("No description available.")

            with torch.no_grad():
                text_embeds = self.text_encoder.encode_text(item_texts)
                text_embeds = text_embeds.to(self.device)

            # Clone to allow gradient flow to downstream layers
            text_embeds = text_embeds.clone().detach().requires_grad_(self.training)

            item_vectors = self.text_encoder(text_embeds)

            # 恢复原始形状
            if len(original_shape) == 2:
                item_vectors = item_vectors.view(original_shape[0], original_shape[1], -1)

            return F.normalize(item_vectors, p=2, dim=-1)

    def forward(
        self,
        user_ids: List[int],
        item_ids: torch.Tensor,
        user_history: Optional[torch.Tensor] = None,
        update_memory: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播

        Args:
            user_ids: 用户ID列表 [batch_size]
            item_ids: 候选物品ID [batch_size, num_candidates]
            user_history: 用户历史序列 [batch_size, seq_len] (可选)
            update_memory: 是否更新用户记忆

        Returns:
            scores: [batch_size, num_candidates] 匹配分数
            info: 额外信息字典
        """
        batch_size = len(user_ids)
        num_candidates = item_ids.size(1)

        # 1. 编码用户偏好（来自 LLM）
        user_pref_vectors = self.encode_user_preferences(user_ids)  # [B, D]

        # 2. 编码物品描述（来自 LLM）
        item_desc_vectors = self.encode_item_descriptions(item_ids)  # [B, N, D]

        # 3. 获取物品视觉特征（CLIP或可训练embedding）
        if self.use_clip and self.clip_features is not None:
            # 使用预提取的CLIP图片特征
            item_embed_vectors = self.clip_features[item_ids]  # [B, N, D]
            item_embed_vectors = F.normalize(item_embed_vectors, p=2, dim=-1)
        else:
            # 使用可训练的物品嵌入（fallback）
            item_embed_vectors = self.item_embeddings(item_ids)  # [B, N, D]
            item_embed_vectors = F.normalize(item_embed_vectors, p=2, dim=-1)

        # 4. 为每个候选物品使用 MoE 融合
        # RetrieverMoEMemory 期望每次处理一个物品，所以我们需要循环或者重构输入
        all_scores = []
        all_routing_weights = []
        memory_info_list = []

        for i in range(num_candidates):
            # 准备当前候选物品的输入
            # user_pref: [B, 1, D] - 用户偏好作为序列
            # item_desc: [B, 1, D] - 当前物品描述作为序列
            # item_image: [B, 1, D] - 使用物品嵌入替代图像
            # target_item: [B, D] - 当前物品的嵌入

            current_item_desc = item_desc_vectors[:, i:i+1, :]  # [B, 1, D]
            current_item_embed = item_embed_vectors[:, i:i+1, :]  # [B, 1, D]
            target_item = (item_desc_vectors[:, i, :] + item_embed_vectors[:, i, :]) / 2  # [B, D]

            # 调用 MoE retriever
            item_scores, routing_weights, memory_info = self.moe_retriever(
                user_ids=user_ids,
                user_pref=user_pref_vectors.unsqueeze(1),  # [B, 1, D]
                item_desc=current_item_desc,  # [B, 1, D]
                item_image=current_item_embed,  # [B, 1, D]
                target_item=target_item,  # [B, D]
                update_memory=update_memory and (i == 0)  # 只在第一个物品时更新记忆
            )

            all_scores.append(item_scores)
            all_routing_weights.append(routing_weights)
            if i == 0:
                memory_info_list.append(memory_info)

        # 合并所有候选物品的分数
        scores = torch.stack(all_scores, dim=1)  # [B, N]

        # 温度缩放
        scores = scores / self.temperature

        info = {
            'user_pref_vectors': user_pref_vectors,
            'item_desc_vectors': item_desc_vectors,
            'routing_weights': torch.stack(all_routing_weights, dim=1),  # [B, N, num_experts]
            'memory_info': memory_info_list[0] if memory_info_list else {}
        }

        return scores, info

    def get_top_k_items(
        self,
        user_ids: List[int],
        k: int = 10,
        exclude_items: Optional[List[List[int]]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为用户推荐 top-k 物品

        Args:
            user_ids: 用户ID列表
            k: 推荐物品数量
            exclude_items: 每个用户要排除的物品列表

        Returns:
            top_k_items: [batch_size, k] top-k 物品ID
            top_k_scores: [batch_size, k] top-k 分数
        """
        batch_size = len(user_ids)

        # 获取所有物品ID
        all_item_ids = torch.arange(1, self.num_items + 1, device=self.device)
        all_item_ids = all_item_ids.unsqueeze(0).expand(batch_size, -1)  # [B, num_items]

        # 计算分数
        scores, _ = self.forward(
            user_ids=user_ids,
            item_ids=all_item_ids,
            update_memory=False
        )

        # 排除物品
        if exclude_items is not None:
            for i, exclude_list in enumerate(exclude_items):
                if exclude_list:
                    exclude_mask = torch.tensor(exclude_list, device=self.device)
                    scores[i, exclude_mask - 1] = float('-inf')

        # Top-k
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
        top_k_items = all_item_ids.gather(1, top_k_indices)

        return top_k_items, top_k_scores

    def get_memory_stats(self) -> Dict:
        """获取记忆统计信息"""
        return self.moe_retriever.get_memory_stats()

    def save_memories(self, save_path: str):
        """保存用户记忆"""
        self.moe_retriever.save_memories(save_path)

    def load_memories(self, load_path: str):
        """加载用户记忆"""
        self.moe_retriever.load_memories(load_path)
