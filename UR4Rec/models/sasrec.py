"""
SASRec: Self-Attentive Sequential Recommendation

基于 Transformer 的序列推荐模型，用作 UR4Rec 的基础推荐器。

参考论文:
Kang, W. C., & McAuley, J. (2018). Self-attentive sequential recommendation. ICDM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PointWiseFeedForward(nn.Module):
    """
    Point-wise Feed-Forward Network
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]

        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        x = x.transpose(-1, -2)  # [batch, hidden, seq_len]
        x = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(x)))))
        x = x.transpose(-1, -2)  # [batch, seq_len, hidden]
        return x


class SASRecBlock(nn.Module):
    """
    SASRec Transformer Block
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = PointWiseFeedForward(hidden_dim, dropout)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            attn_mask: [seq_len, seq_len] - 注意力掩码（因果掩码）
            key_padding_mask: [batch_size, seq_len] - 填充掩码

        Returns:
            output: [batch_size, seq_len, hidden_dim]
        """
        # Self-attention
        attn_output, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )

        # Add & Norm
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)

        # Add & Norm
        x = self.norm2(x + self.dropout(ff_output))

        return x


class SASRec(nn.Module):
    """
    SASRec: Self-Attentive Sequential Recommendation Model
    """

    def __init__(
        self,
        num_items: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 50
    ):
        """
        Args:
            num_items: 物品总数
            hidden_dim: 隐藏层维度
            num_blocks: Transformer 块数量
            num_heads: 注意力头数
            dropout: Dropout 率
            max_seq_len: 最大序列长度
        """
        super().__init__()

        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Item Embedding
        self.item_embedding = nn.Embedding(
            num_items + 1,  # +1 for padding
            hidden_dim,
            padding_idx=0
        )

        # Positional Embedding
        self.positional_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """
        创建因果掩码（下三角矩阵）

        Args:
            seq_len: 序列长度

        Returns:
            mask: [seq_len, seq_len]
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool),
            diagonal=1
        )
        return mask

    def forward(
        self,
        input_seq: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_seq: [batch_size, seq_len] - 输入序列
            padding_mask: [batch_size, seq_len] - 填充掩码（True表示有效位置）

        Returns:
            output: [batch_size, seq_len, hidden_dim] - 序列表示
        """
        batch_size, seq_len = input_seq.shape

        # Item Embedding
        seq_emb = self.item_embedding(input_seq)  # [batch, seq_len, hidden]

        # Positional Embedding
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0)
        pos_emb = self.positional_embedding(positions)  # [1, seq_len, hidden]

        # Add embeddings
        x = seq_emb + pos_emb
        x = self.dropout(x)

        # 创建因果掩码
        causal_mask = self._create_causal_mask(seq_len).to(input_seq.device)

        # Key padding mask（MultiheadAttention 使用的格式：True表示需要忽略）
        if padding_mask is not None:
            key_padding_mask = ~padding_mask  # 反转：True表示填充位置
        else:
            key_padding_mask = None

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        # Layer norm
        x = self.layer_norm(x)

        return x

    def predict(
        self,
        input_seq: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测下一个物品的分数

        Args:
            input_seq: [batch_size, seq_len]
            candidate_items: [batch_size, num_candidates] - 候选物品（如果为None，预测所有物品）
            padding_mask: [batch_size, seq_len]

        Returns:
            scores: [batch_size, num_candidates] 或 [batch_size, num_items]
        """
        # 获取序列表示
        seq_output = self.forward(input_seq, padding_mask)  # [batch, seq_len, hidden]

        # 使用最后一个位置的表示
        if padding_mask is not None:
            # 找到每个序列的最后一个有效位置
            seq_lengths = padding_mask.sum(dim=1) - 1  # [batch]
            batch_indices = torch.arange(seq_output.size(0), device=seq_output.device)
            last_output = seq_output[batch_indices, seq_lengths]  # [batch, hidden]
        else:
            last_output = seq_output[:, -1, :]  # [batch, hidden]

        # 计算分数
        if candidate_items is not None:
            # 只计算候选物品的分数
            candidate_emb = self.item_embedding(candidate_items)  # [batch, num_cand, hidden]
            scores = torch.matmul(
                last_output.unsqueeze(1),  # [batch, 1, hidden]
                candidate_emb.transpose(1, 2)  # [batch, hidden, num_cand]
            ).squeeze(1)  # [batch, num_cand]
        else:
            # 计算所有物品的分数
            item_emb = self.item_embedding.weight[1:]  # [num_items, hidden] (exclude padding)
            scores = torch.matmul(last_output, item_emb.T)  # [batch, num_items]

        return scores

    def compute_loss(
        self,
        input_seq: torch.Tensor,
        target_items: torch.Tensor,
        negative_items: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算 BPR 损失

        Args:
            input_seq: [batch_size, seq_len]
            target_items: [batch_size] - 正样本物品
            negative_items: [batch_size, num_negatives] - 负样本物品
            padding_mask: [batch_size, seq_len]

        Returns:
            loss: 标量损失
            metrics: 指标字典
        """
        # 获取序列表示
        seq_output = self.forward(input_seq, padding_mask)

        # 使用最后一个位置
        if padding_mask is not None:
            seq_lengths = padding_mask.sum(dim=1) - 1
            batch_indices = torch.arange(seq_output.size(0), device=seq_output.device)
            last_output = seq_output[batch_indices, seq_lengths]
        else:
            last_output = seq_output[:, -1, :]

        # 正样本分数
        pos_emb = self.item_embedding(target_items)  # [batch, hidden]
        pos_scores = (last_output * pos_emb).sum(dim=-1)  # [batch]

        # 负样本分数
        neg_emb = self.item_embedding(negative_items)  # [batch, num_neg, hidden]
        neg_scores = torch.matmul(
            last_output.unsqueeze(1),  # [batch, 1, hidden]
            neg_emb.transpose(1, 2)  # [batch, hidden, num_neg]
        ).squeeze(1)  # [batch, num_neg]

        # BPR 损失
        # loss = -log(sigmoid(pos_score - neg_score))
        diff = pos_scores.unsqueeze(1) - neg_scores  # [batch, num_neg]
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()

        # 指标
        with torch.no_grad():
            # 计算排名
            all_scores = torch.cat([
                pos_scores.unsqueeze(1),
                neg_scores
            ], dim=1)  # [batch, 1+num_neg]

            ranks = (neg_scores >= pos_scores.unsqueeze(1)).sum(dim=1) + 1  # [batch]
            mrr = (1.0 / ranks.float()).mean()

            # 准确率（正样本分数是否最高）
            accuracy = (pos_scores > neg_scores.max(dim=1)[0]).float().mean()

        metrics = {
            'bpr_loss': loss.item(),
            'mrr': mrr.item(),
            'accuracy': accuracy.item()
        }

        return loss, metrics


if __name__ == "__main__":
    print("测试 SASRec 模型...")

    # 创建模型
    model = SASRec(
        num_items=1000,
        hidden_dim=128,
        num_blocks=2,
        num_heads=4,
        max_seq_len=50
    )

    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 测试数据
    batch_size = 4
    seq_len = 10
    num_candidates = 20

    input_seq = torch.randint(1, 1000, (batch_size, seq_len))
    padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    candidate_items = torch.randint(1, 1000, (batch_size, num_candidates))

    # 前向传播
    with torch.no_grad():
        seq_output = model(input_seq, padding_mask)
        print(f"\n序列输出形状: {seq_output.shape}")

        # 预测候选物品
        scores = model.predict(input_seq, candidate_items, padding_mask)
        print(f"候选物品分数形状: {scores.shape}")
        print(f"示例分数: {scores[0][:5]}")

    # 测试损失
    target_items = torch.randint(1, 1000, (batch_size,))
    negative_items = torch.randint(1, 1000, (batch_size, 10))

    loss, metrics = model.compute_loss(
        input_seq, target_items, negative_items, padding_mask
    )

    print(f"\n损失: {loss.item():.4f}")
    print(f"指标: {metrics}")
