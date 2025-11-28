"""
轻量级文本偏好检索器

将 LLM 生成的文本描述编码为向量，用于快速检索匹配。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class TextEncoder(nn.Module):
    """
    文本编码器：将文本描述转换为向量表示

    使用预训练的句子编码器（如 Sentence-BERT）
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        hidden_dim: int = 512,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            model_name: 预训练模型名称
            embedding_dim: 预训练模型的嵌入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            dropout: Dropout 率
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_encoder = SentenceTransformer(model_name)
            # 冻结预训练参数（可选）
            for param in self.sentence_encoder.parameters():
                param.requires_grad = False
        except ImportError:
            print("警告: sentence-transformers 未安装，使用简单编码器")
            self.sentence_encoder = None

        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
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
            # 简单编码器：使用随机投影
            batch_size = len(texts)
            return torch.randn(batch_size, self.embedding_dim)

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            text_embeddings: [batch_size, embedding_dim]

        Returns:
            output: [batch_size, output_dim]
        """
        return self.projection(text_embeddings)


class TextPreferenceRetriever(nn.Module):
    """
    文本偏好检索器

    基于文本描述进行偏好匹配和物品检索
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        num_items: int,
        embedding_dim: int = 256,
        temperature: float = 0.07
    ):
        """
        Args:
            text_encoder: 文本编码器
            num_items: 物品数量
            embedding_dim: 嵌入维度
            temperature: 温度参数（用于对比学习）
        """
        super().__init__()

        self.text_encoder = text_encoder
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # 物品嵌入（从文本描述初始化，可训练）
        self.item_embeddings = nn.Embedding(
            num_items + 1,  # +1 for padding
            embedding_dim,
            padding_idx=0
        )

        # 初始化
        nn.init.xavier_uniform_(self.item_embeddings.weight[1:])

    def encode_preferences(self, preference_texts: List[str]) -> torch.Tensor:
        """
        编码用户偏好文本

        Args:
            preference_texts: 用户偏好文本列表

        Returns:
            preference_vectors: [batch_size, embedding_dim]
        """
        # 使用文本编码器
        text_embeds = self.text_encoder.encode_text(preference_texts)

        # 投影到偏好空间
        preference_vectors = self.text_encoder(text_embeds)

        # L2 归一化
        preference_vectors = F.normalize(preference_vectors, p=2, dim=-1)

        return preference_vectors

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        编码物品

        Args:
            item_ids: [batch_size, num_items]

        Returns:
            item_vectors: [batch_size, num_items, embedding_dim]
        """
        item_embeds = self.item_embeddings(item_ids)

        # L2 归一化
        item_embeds = F.normalize(item_embeds, p=2, dim=-1)

        return item_embeds

    def compute_similarity(
        self,
        preference_vectors: torch.Tensor,
        item_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        计算偏好-物品相似度

        Args:
            preference_vectors: [batch_size, embedding_dim]
            item_vectors: [batch_size, num_items, embedding_dim]

        Returns:
            similarity_scores: [batch_size, num_items]
        """
        # 余弦相似度
        preference_vectors = preference_vectors.unsqueeze(1)  # [batch, 1, dim]

        similarity = torch.matmul(
            preference_vectors,
            item_vectors.transpose(1, 2)
        ).squeeze(1)  # [batch, num_items]

        return similarity

    def forward(
        self,
        preference_texts: List[str],
        candidate_ids: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：检索匹配候选物品

        Args:
            preference_texts: 用户偏好文本列表
            candidate_ids: [batch_size, num_candidates]
            candidate_mask: [batch_size, num_candidates]

        Returns:
            similarity_scores: [batch_size, num_candidates]
            preference_vectors: [batch_size, embedding_dim]
        """
        # 编码偏好
        preference_vectors = self.encode_preferences(preference_texts)

        # 编码候选物品
        item_vectors = self.encode_items(candidate_ids)

        # 计算相似度
        similarity_scores = self.compute_similarity(preference_vectors, item_vectors)

        # 应用掩码
        if candidate_mask is not None:
            similarity_scores = similarity_scores.masked_fill(
                ~candidate_mask, -1e9
            )

        return similarity_scores, preference_vectors

    def retrieve_top_k(
        self,
        preference_text: str,
        k: int = 20,
        candidate_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检索 Top-K 物品

        Args:
            preference_text: 用户偏好文本
            k: 返回的物品数量
            candidate_ids: 候选物品ID（如果为None，则从所有物品中检索）

        Returns:
            top_k_ids: [k] - Top-K 物品ID
            top_k_scores: [k] - 对应的相似度分数
        """
        # 编码偏好
        preference_vector = self.encode_preferences([preference_text])

        if candidate_ids is None:
            # 从所有物品中检索
            all_item_ids = torch.arange(1, self.num_items + 1).unsqueeze(0)
            item_vectors = self.encode_items(all_item_ids)
        else:
            # 从候选物品中检索
            item_vectors = self.encode_items(candidate_ids.unsqueeze(0))

        # 计算相似度
        similarity_scores = self.compute_similarity(
            preference_vector, item_vectors
        ).squeeze(0)

        # 获取 Top-K
        top_k_scores, top_k_indices = torch.topk(similarity_scores, k=min(k, len(similarity_scores)))

        if candidate_ids is None:
            top_k_ids = top_k_indices + 1  # +1 因为物品ID从1开始
        else:
            top_k_ids = candidate_ids[top_k_indices]

        return top_k_ids, top_k_scores

    def update_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_text_descriptions: List[str]
    ):
        """
        使用文本描述更新物品嵌入

        Args:
            item_ids: 物品ID列表
            item_text_descriptions: 物品文本描述列表
        """
        with torch.no_grad():
            # 编码文本描述
            text_embeds = self.text_encoder.encode_text(item_text_descriptions)
            item_vectors = self.text_encoder(text_embeds)
            item_vectors = F.normalize(item_vectors, p=2, dim=-1)

            # 更新嵌入
            for item_id, vector in zip(item_ids, item_vectors):
                self.item_embeddings.weight[item_id] = vector


class PreferenceRetrievalLoss(nn.Module):
    """
    偏好检索损失函数
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        similarity_scores: torch.Tensor,
        labels: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算检索损失

        Args:
            similarity_scores: [batch_size, num_candidates] - 相似度分数
            labels: [batch_size, num_candidates] - 标签（1表示正样本，0表示负样本）
            candidate_mask: [batch_size, num_candidates] - 掩码

        Returns:
            loss: 标量损失
            metrics: 指标字典
        """
        # 二元交叉熵损失
        bce_loss = F.binary_cross_entropy_with_logits(
            similarity_scores / self.temperature,
            labels.float(),
            reduction='none'
        )

        # 应用掩码
        if candidate_mask is not None:
            bce_loss = bce_loss * candidate_mask.float()
            loss = bce_loss.sum() / candidate_mask.sum().clamp(min=1)
        else:
            loss = bce_loss.mean()

        # 计算指标
        with torch.no_grad():
            # 预测
            preds = (similarity_scores > 0).long()

            # 准确率
            if candidate_mask is not None:
                correct = ((preds == labels) & candidate_mask).sum().float()
                total = candidate_mask.sum().float()
            else:
                correct = (preds == labels).sum().float()
                total = labels.numel()

            accuracy = correct / total.clamp(min=1)

            # 正样本召回率
            positive_mask = (labels == 1)
            if candidate_mask is not None:
                positive_mask = positive_mask & candidate_mask

            if positive_mask.sum() > 0:
                recall = (preds[positive_mask] == 1).sum().float() / positive_mask.sum()
            else:
                recall = torch.tensor(0.0)

        metrics = {
            'bce_loss': loss.item(),
            'accuracy': accuracy.item(),
            'recall': recall.item()
        }

        return loss, metrics


if __name__ == "__main__":
    print("测试文本偏好检索器...")

    # 创建模型
    text_encoder = TextEncoder(embedding_dim=384, output_dim=128)
    retriever = TextPreferenceRetriever(
        text_encoder=text_encoder,
        num_items=1000,
        embedding_dim=128
    )

    print(f"模型参数: {sum(p.numel() for p in retriever.parameters()):,}")

    # 测试数据
    preference_texts = [
        "该用户喜欢动作和科幻类型的电影",
        "该用户偏好浪漫喜剧和家庭电影"
    ]

    candidate_ids = torch.randint(1, 1000, (2, 10))
    candidate_mask = torch.ones(2, 10, dtype=torch.bool)

    # 前向传播
    with torch.no_grad():
        similarity_scores, preference_vectors = retriever(
            preference_texts,
            candidate_ids,
            candidate_mask
        )

    print(f"\n相似度分数形状: {similarity_scores.shape}")
    print(f"偏好向量形状: {preference_vectors.shape}")
    print(f"\n示例相似度分数: {similarity_scores[0]}")

    # 测试检索
    top_k_ids, top_k_scores = retriever.retrieve_top_k(
        preference_texts[0],
        k=5
    )

    print(f"\nTop-5 物品ID: {top_k_ids}")
    print(f"Top-5 分数: {top_k_scores}")

    # 测试损失函数
    loss_fn = PreferenceRetrievalLoss()
    labels = torch.randint(0, 2, (2, 10))

    loss, metrics = loss_fn(similarity_scores, labels, candidate_mask)

    print(f"\n损失: {loss.item():.4f}")
    print(f"指标: {metrics}")
