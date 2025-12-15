"""
UR4Rec V2: 正确的架构实现

结合 SASRec 基础推荐器和文本偏好检索器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path

from .sasrec import SASRec
from .text_preference_retriever import TextPreferenceRetriever, TextEncoder


class UR4RecV2(nn.Module):
    """
    UR4Rec V2: 用户偏好检索推荐系统（正确版本）

    架构:
    1. 离线阶段：LLM 生成用户偏好和物品描述
    2. 在线阶段：
       - SASRec 生成候选物品排序
       - 文本偏好检索器匹配并重排序
       - 融合两路结果
    """

    def __init__(
        self,
        num_items: int,
        # SASRec 参数
        sasrec_hidden_dim: int = 256,
        sasrec_num_blocks: int = 2,
        sasrec_num_heads: int = 4,
        sasrec_dropout: float = 0.1,
        # 文本检索器参数
        text_model_name: str = "all-MiniLM-L6-v2",
        text_embedding_dim: int = 384,
        retriever_output_dim: int = 256,
        # 融合参数
        fusion_method: str = "weighted",  # 'weighted', 'rank', 'cascade'
        sasrec_weight: float = 0.5,
        retriever_weight: float = 0.5,
        # 其他
        max_seq_len: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            num_items: 物品总数
            sasrec_*: SASRec 相关参数
            text_*: 文本编码器参数
            retriever_*: 检索器参数
            fusion_method: 融合方法
            *_weight: 融合权重
            max_seq_len: 最大序列长度
            device: 设备
        """
        super().__init__()

        self.num_items = num_items
        self.fusion_method = fusion_method
        self.sasrec_weight = sasrec_weight
        self.retriever_weight = retriever_weight
        self.device = device

        # SASRec 基础推荐模型
        self.sasrec = SASRec(
            num_items=num_items,
            hidden_dim=sasrec_hidden_dim,
            num_blocks=sasrec_num_blocks,
            num_heads=sasrec_num_heads,
            dropout=sasrec_dropout,
            max_seq_len=max_seq_len
        )

        # 文本偏好检索器
        text_encoder = TextEncoder(
            model_name=text_model_name,
            embedding_dim=text_embedding_dim,
            output_dim=retriever_output_dim
        )

        self.preference_retriever = TextPreferenceRetriever(
            text_encoder=text_encoder,
            num_items=num_items,
            embedding_dim=retriever_output_dim
        )

        # 用户偏好缓存（离线生成的）
        self.user_preferences: Dict[int, str] = {}

        # 物品描述缓存
        self.item_descriptions: Dict[int, str] = {}

        self.to(device)

    def load_llm_generated_data(
        self,
        user_preferences_path: str,
        item_descriptions_path: str
    ):
        """
        加载 LLM 离线生成的数据

        Args:
            user_preferences_path: 用户偏好文件路径
            item_descriptions_path: 物品描述文件路径
        """
        # 加载用户偏好
        if Path(user_preferences_path).exists():
            with open(user_preferences_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.user_preferences = {int(k): v for k, v in data.items()}
            print(f"加载了 {len(self.user_preferences)} 个用户偏好")
        else:
            print(f"警告: 用户偏好文件不存在: {user_preferences_path}")

        # 加载物品描述
        if Path(item_descriptions_path).exists():
            with open(item_descriptions_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.item_descriptions = {int(k): v for k, v in data.items()}
            print(f"加载了 {len(self.item_descriptions)} 个物品描述")

            # 使用物品描述初始化物品嵌入
            self._init_item_embeddings_from_text()
        else:
            print(f"警告: 物品描述文件不存在: {item_descriptions_path}")

    def _init_item_embeddings_from_text(self):
        """使用文本描述初始化物品嵌入"""
        print("正在使用文本描述初始化物品嵌入...")

        item_ids = list(self.item_descriptions.keys())
        item_texts = [self.item_descriptions[iid] for iid in item_ids]

        # 批量处理
        batch_size = 64
        for i in range(0, len(item_ids), batch_size):
            batch_ids = item_ids[i:i+batch_size]
            batch_texts = item_texts[i:i+batch_size]

            self.preference_retriever.update_item_embeddings(
                torch.tensor(batch_ids),
                batch_texts
            )

        print("物品嵌入初始化完成")

    def forward_sasrec(
        self,
        input_seq: torch.Tensor,
        candidate_items: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        SASRec 前向传播

        Args:
            input_seq: [batch_size, seq_len]
            candidate_items: [batch_size, num_candidates]
            padding_mask: [batch_size, seq_len]

        Returns:
            sasrec_scores: [batch_size, num_candidates]
        """
        sasrec_scores = self.sasrec.predict(
            input_seq, candidate_items, padding_mask
        )
        return sasrec_scores

    def forward_retriever(
        self,
        user_ids: List[int],
        candidate_items: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        文本偏好检索器前向传播

        Args:
            user_ids: 用户ID列表
            candidate_items: [batch_size, num_candidates]
            candidate_mask: [batch_size, num_candidates]

        Returns:
            retriever_scores: [batch_size, num_candidates]
        """
        # 获取用户偏好文本
        preference_texts = []
        for uid in user_ids:
            if uid in self.user_preferences:
                preference_texts.append(self.user_preferences[uid])
            else:
                # 默认偏好
                preference_texts.append("该用户具有多样化的兴趣。")

        # 检索匹配
        retriever_scores, _ = self.preference_retriever(
            preference_texts,
            candidate_items,
            candidate_mask
        )

        return retriever_scores

    def fuse_scores(
        self,
        sasrec_scores: torch.Tensor,
        retriever_scores: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        融合两路分数

        Args:
            sasrec_scores: [batch_size, num_candidates]
            retriever_scores: [batch_size, num_candidates]
            candidate_mask: [batch_size, num_candidates]

        Returns:
            fused_scores: [batch_size, num_candidates]
        """
        if self.fusion_method == "weighted":
            # 加权融合
            # 先归一化
            sasrec_scores_norm = F.softmax(sasrec_scores, dim=-1)
            retriever_scores_norm = F.softmax(retriever_scores, dim=-1)

            fused_scores = (
                self.sasrec_weight * sasrec_scores_norm +
                self.retriever_weight * retriever_scores_norm
            )

        elif self.fusion_method == "rank":
            # 基于排名的融合
            sasrec_ranks = torch.argsort(
                torch.argsort(sasrec_scores, dim=-1, descending=True),
                dim=-1
            )
            retriever_ranks = torch.argsort(
                torch.argsort(retriever_scores, dim=-1, descending=True),
                dim=-1
            )

            # 排名越小越好，转换为分数
            sasrec_rank_scores = 1.0 / (sasrec_ranks.float() + 1)
            retriever_rank_scores = 1.0 / (retriever_ranks.float() + 1)

            fused_scores = (
                self.sasrec_weight * sasrec_rank_scores +
                self.retriever_weight * retriever_rank_scores
            )

        elif self.fusion_method == "cascade":
            # 级联：先用 SASRec 筛选，再用检索器重排
            # 使用 SASRec 的 Top-K，然后用检索器重排这 K 个
            fused_scores = sasrec_scores + 0.5 * retriever_scores

        else:
            # 默认：简单相加
            fused_scores = sasrec_scores + retriever_scores

        # 应用掩码
        if candidate_mask is not None:
            fused_scores = fused_scores.masked_fill(~candidate_mask, -1e9)

        return fused_scores

    def forward(
        self,
        user_ids: List[int],
        input_seq: torch.Tensor,
        candidate_items: torch.Tensor,
        seq_padding_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        完整的前向传播

        Args:
            user_ids: 用户ID列表
            input_seq: [batch_size, seq_len]
            candidate_items: [batch_size, num_candidates]
            seq_padding_mask: [batch_size, seq_len]
            candidate_mask: [batch_size, num_candidates]

        Returns:
            final_scores: [batch_size, num_candidates]
            scores_dict: 包含各个模块分数的字典
        """
        # SASRec 分数
        sasrec_scores = self.forward_sasrec(
            input_seq, candidate_items, seq_padding_mask
        )

        # 检索器分数
        retriever_scores = self.forward_retriever(
            user_ids, candidate_items, candidate_mask
        )

        # 融合
        final_scores = self.fuse_scores(
            sasrec_scores, retriever_scores, candidate_mask
        )

        scores_dict = {
            'sasrec_scores': sasrec_scores,
            'retriever_scores': retriever_scores,
            'final_scores': final_scores
        }

        return final_scores, scores_dict

    def predict(
        self,
        user_ids: List[int],
        input_seq: torch.Tensor,
        candidate_items: torch.Tensor,
        seq_padding_mask: Optional[torch.Tensor] = None,
        candidate_mask: Optional[torch.Tensor] = None,
        return_scores: bool = False
    ) -> Union[List[List[int]], Tuple[List[List[int]], Dict]]:
        """
        预测并返回排序后的物品列表

        Args:
            user_ids: 用户ID列表
            input_seq: [batch_size, seq_len]
            candidate_items: [batch_size, num_candidates]
            seq_padding_mask: [batch_size, seq_len]
            candidate_mask: [batch_size, num_candidates]
            return_scores: 是否返回分数

        Returns:
            ranked_items: 排序后的物品ID列表
            scores_dict: (可选) 分数字典
        """
        final_scores, scores_dict = self.forward(
            user_ids, input_seq, candidate_items,
            seq_padding_mask, candidate_mask
        )

        # 排序
        _, sorted_indices = torch.sort(final_scores, dim=-1, descending=True)

        # 获取排序后的物品ID
        ranked_items = []
        for i in range(candidate_items.size(0)):
            if candidate_mask is not None:
                valid_indices = sorted_indices[i][candidate_mask[i][sorted_indices[i]]]
            else:
                valid_indices = sorted_indices[i]

            ranked_item_ids = candidate_items[i][valid_indices].tolist()
            ranked_items.append(ranked_item_ids)

        if return_scores:
            return ranked_items, scores_dict
        else:
            return ranked_items


if __name__ == "__main__":
    print("测试 UR4Rec V2...")

    # 创建模型
    model = UR4RecV2(
        num_items=1000,
        sasrec_hidden_dim=128,
        sasrec_num_blocks=2,
        text_embedding_dim=384,
        retriever_output_dim=128
    )

    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟数据
    batch_size = 4
    seq_len = 10
    num_candidates = 20

    user_ids = [1, 2, 3, 4]
    input_seq = torch.randint(1, 1000, (batch_size, seq_len))
    candidate_items = torch.randint(1, 1000, (batch_size, num_candidates))
    seq_padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    candidate_mask = torch.ones(batch_size, num_candidates, dtype=torch.bool)

    # 设置模拟偏好
    for uid in user_ids:
        model.user_preferences[uid] = f"用户{uid}喜欢动作和科幻类型"

    # 前向传播
    with torch.no_grad():
        final_scores, scores_dict = model(
            user_ids, input_seq, candidate_items,
            seq_padding_mask, candidate_mask
        )

    print(f"\nSASRec 分数形状: {scores_dict['sasrec_scores'].shape}")
    print(f"检索器分数形状: {scores_dict['retriever_scores'].shape}")
    print(f"最终分数形状: {final_scores.shape}")

    # 预测
    ranked_items = model.predict(
        user_ids, input_seq, candidate_items,
        seq_padding_mask, candidate_mask
    )

    print(f"\n排序结果:")
    for i, items in enumerate(ranked_items):
        print(f"  用户 {user_ids[i]}: Top-5 = {items[:5]}")
